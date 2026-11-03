"""X1 — mini Jarvis local: microfone -> Silero VAD + smart-turn -> Whisper -> LLM (Ollama) -> Piper -> alto-falante."""
import asyncio
import ctypes
import json
import os
import re
import shutil
import subprocess
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path

import nvidia.cublas.lib
import nvidia.cudnn.lib

# ctranslate2 needs CUDA 12 cuBLAS/cuDNN; load the pip wheels' copies so no system CUDA or LD_LIBRARY_PATH is needed
for lib in ("cublas/lib/libcublasLt.so.12", "cublas/lib/libcublas.so.12", "cudnn/lib/libcudnn.so.9"):
    ctypes.CDLL(str(Path(nvidia.cublas.lib.__path__[0]).parents[1] / lib), mode=ctypes.RTLD_GLOBAL)

import httpx
import numpy as np
import onnxruntime as ort
import faster_whisper
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from faster_whisper.feature_extractor import FeatureExtractor
from huggingface_hub import hf_hub_download
from piper import PiperVoice

LLM = "qwen3:1.7b"
OLLAMA = "http://127.0.0.1:11434"
PORT = 8731
URL = f"http://x1.localhost:{PORT}"  # its own host, so the window gets its own app id (and the X1 icon in the dock)
DATA = Path.home() / ".local/share/x1"  # install.sh puts the app, Ollama and the browser profile here
SR, WIN = 16000, 512                  # Silero VAD window = 32 ms
PAUSE = round(0.3 * SR / WIN)         # first short silence: judge the turn and start drafting the reply
PARTIAL_EVERY = round(0.6 * SR / WIN) # live transcript refresh while the user talks
# while X1 talks, the user cuts in with ~0.4 s of speech within 0.8 s: "não, não, não" (with gaps) counts, a quick "uhum"
# doesn't. Measured on laptop speakers: X1's own voice after Chrome's echo canceller peaks at VAD 0.05 (p99 0.017).
BARGE_P, BARGE, BARGE_WIN = 0.6, round(0.4 * SR / WIN), round(0.8 * SR / WIN)
SYSTEM = (
    "Você é o X1, um assistente pessoal no estilo do J.A.R.V.I.S.: educado, calmo, eficiente e com um humor seco e sutil. "
    "Você conversa por voz, em português do Brasil. Responda curto, como numa conversa falada: uma a três frases, "
    "sem listas, sem markdown, sem emojis. Escreva números e siglas do jeito que se fala. "
    "O que o usuário diz chega por transcrição automática e pode ter erros: entenda a intenção. "
    "Se não souber algo, diga com franqueza."
)
# Whisper's classic Portuguese hallucinations on silence/noise
JUNK = re.compile(r"obrigad[oa] por assistir|legendas? (pela|por)|amara\.org|inscreva-se|^\W*$", re.I)
# a thought that ends on a connective, a filler or "..." is not finished yet
OPEN = re.compile(r"(\.\.\.|…|,|\b(e|ou|mas|que|porque|então|tipo|né|de|do|da|dos|das|com|pra|para|um|uma|o|a|os|as|no|na|em|se|"
                  r"quando|como|mais|é|ahn*|ãh+|hã+|hum+|hm+|éh+|eh+|uh+))\W*$", re.I)
# makes Whisper keep the hesitations (so OPEN can see them) and spell the assistant's name
FILLERS = "Ahn... então, X1, eu acho que... hmm, tipo, sabe?"
SENTENCE = re.compile(r"(?<=[.!?…:;])\s+|\n+")
CLAUSE = re.compile(r"(?<=[,;:.!?…—])\s+|\n+")  # whitespace after the comma, so "3,5" never splits

WEB = Path(__file__).parent / "web"
READY = threading.Event()
whisper = vad = turn = mel = voice = LOAD_ERROR = None


def load():
    """Load the speech models (~15 s; downloads them on first run). The window opens before this finishes."""
    global whisper, vad, turn, mel, voice, LOAD_ERROR
    try:
        whisper = WhisperModel("small", device="cuda", compute_type="int8_float16")
        vad = ort.InferenceSession(str(Path(faster_whisper.__file__).parent / "assets/silero_vad_v6.onnx"))
        turn = ort.InferenceSession(hf_hub_download("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx"))
        mel = FeatureExtractor(feature_size=80, chunk_length=8)
        voice = PiperVoice.load(hf_hub_download("rhasspy/piper-voices", "pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx"),
                                hf_hub_download("rhasspy/piper-voices", "pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx.json"))
    except Exception as e:  # e.g. a CUDA/driver problem: the window shows it instead of "carregando…" forever
        LOAD_ERROR = f"{type(e).__name__}: {e}"
        raise
    finally:
        READY.set()


def system_prompt():
    now = datetime.now()
    return f"{SYSTEM} Agora são {now:%H:%M} de {now:%d/%m/%Y}."


def transcribe(audio, beam=1):
    segs, _ = whisper.transcribe(audio, language="pt", beam_size=beam, condition_on_previous_text=False, without_timestamps=True,
                                 initial_prompt=FILLERS)
    return " ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.6 and not JUNK.search(s.text)).strip()


def turn_done(audio):
    """smart-turn v3: probability that the speaker finished the thought, from the last 8 s of audio (prosody, not words)."""
    a = audio[-8 * SR:]
    a = np.pad(a, (8 * SR - len(a), 0))  # left-pad to 8 s like smart-turn's reference inference
    a = (a - a.mean()) / np.sqrt(a.var() + 1e-7)
    return float(turn.run(None, {"input_features": mel(a, padding=0)[None, :, :800].astype(np.float32)})[0].ravel()[0])


def wait_for(p, text):
    """Seconds of silence before X1 may speak. Tuned on real NURC-SP dialogue, where smart-turn alone says
    "done" on 69% of mid-thought pauses: the words veto it, and its confidence only shortens the wait."""
    if not text or OPEN.search(text):
        return 3.0  # "e…", "tipo", "ahn", "né", or a mumble Whisper couldn't read: still thinking out loud, keep the audio
    return 1.0 if p > 0.7 else 2.0


def next_chunk(text, first):
    """Next speakable piece of the streamed reply, or None to wait for more tokens. The first piece may end at a
    comma so X1 starts talking sooner; the rest are whole sentences, which Piper intones better."""
    for m in (CLAUSE if first else SENTENCE).finditer(text):
        if len(text[:m.start()].strip()) >= (12 if first else 20):
            return text[:m.start()].strip(), text[m.end():]
    return None


def tts(text):
    return b"".join(c.audio_int16_bytes for c in voice.synthesize(text))


class Session:
    def __init__(s, ws):
        s.ws, s.history = ws, []
        s.h = s.c = np.zeros((1, 1, 128), np.float32)
        s.ctx = np.zeros(64, np.float32)
        s.pre = deque(maxlen=10)  # ~320 ms of pre-roll so the first syllable isn't clipped
        s.buf, s.run, s.silence, s.since_partial = [], 0, 0, 0
        s.task, s.spoke, s.busy, s.prev, s.go, s.need, s.text = None, False, False, None, asyncio.Event(), PAUSE, ""
        s.playing, s.voiced, s.recent = False, deque(maxlen=BARGE_WIN), deque(maxlen=BARGE_WIN + 10)  # browser reports playback
        s.spoken, s.t_play = [], 0.0  # pieces sent to the speaker (text, seconds) and when the first one started

    async def send(s, **m):
        await s.ws.send_text(json.dumps(m))

    def vad(s, x):
        p, s.h, s.c = vad.run(None, {"input": np.concatenate([s.ctx, x])[None], "h": s.h, "c": s.c})
        s.ctx = x[-64:]
        return float(p.ravel()[0])

    async def on_audio(s, x):
        p = s.vad(x)
        if s.playing or (s.task and s.spoke):  # X1 is talking (echo-cancelled by the browser): the user has priority
            s.recent.append(x)
            s.voiced.append(p > BARGE_P)
            if sum(s.voiced) >= BARGE:
                await s.interrupt()
                s.buf, s.run, s.silence, s.since_partial = list(s.recent), BARGE, 0, 0  # what they said over X1 starts their turn
                s.recent.clear()
                await s.send(type="interrupt")
                await s.send(type="state", state="hearing")
            return
        if s.recent:  # X1 just finished: what overlapped its last words becomes pre-roll, and the barge-in count restarts
            s.pre.extend(s.recent)
            s.recent.clear()
            s.voiced.clear()
        s.run = s.run + 1 if p > 0.5 else 0
        if p > 0.5:
            resumed = not s.buf or s.silence >= PAUSE
            if not s.buf:
                s.buf = list(s.pre)
            if s.task and s.run >= 4:  # they kept talking before X1 said a word: the user has priority, drop the draft
                s.task.cancel()
                s.task, resumed = None, True
                if s.go.is_set() and s.prev is not None:  # a committed voice turn: same thought, glue its audio back
                    s.buf = [s.prev, *s.buf]
            if resumed:
                await s.send(type="state", state="hearing")
            s.silence = 0
        elif s.buf:
            s.silence += 1
        else:
            s.pre.append(x)
            return
        s.buf.append(x)
        s.since_partial += 1
        if p > 0.5 and s.since_partial >= PARTIAL_EVERY and not s.busy:
            s.since_partial = 0
            asyncio.create_task(s.partial())
        if s.silence == PAUSE:  # first short pause: hear the intonation, read the words, start drafting the reply
            audio = np.concatenate(s.buf)
            p_done, s.text = await asyncio.gather(asyncio.to_thread(turn_done, audio), asyncio.to_thread(transcribe, audio, 5))
            s.need = round(wait_for(p_done, s.text) * SR / WIN)
            if s.text and not s.task:
                await s.send(type="partial", text=s.text)
                s.go = asyncio.Event()
                s.task = asyncio.create_task(s.reply(s.text))
                if s.need > PAUSE:
                    await s.send(type="state", state="waiting")
        if s.silence >= max(PAUSE, s.need):  # silence lasted as long as the policy asked: commit, X1 may speak
            audio, s.buf, s.silence, s.since_partial = np.concatenate(s.buf), [], 0, 0
            if s.task and not s.go.is_set():
                s.prev = audio
                s.go.set()
                await s.send(type="final", text=s.text)
                await s.send(type="state", state="thinking")
            elif not s.task:  # noise or a cough: nothing to answer
                await s.send(type="state", state="listening")

    async def partial(s):
        s.busy = True
        try:
            text = await asyncio.to_thread(transcribe, np.concatenate(s.buf))
            if s.buf and text:
                await s.send(type="partial", text=text)
        finally:
            s.busy = False

    async def interrupt(s):
        """Stop X1 mid-reply; its history keeps only what the user actually heard."""
        task, talking = s.task, s.playing or (s.task and s.spoke)
        s.task, s.playing = None, False
        s.voiced.clear()
        if task:
            task.cancel()
            await asyncio.wait([task])  # let it settle its history first
        if talking and s.history and s.history[-1]["role"] == "assistant":
            left, heard = time.monotonic() - s.t_play, []
            for text, secs in s.spoken:
                if left <= 0:
                    break
                heard.append(text)
                left -= secs
            s.history[-1]["content"] = " ".join(heard) + "…"

    async def on_text(s, m):
        if m.get("type") == "playing":
            s.playing = bool(m.get("on"))
            return
        if m.get("type") not in ("stop", "text"):
            return
        await s.interrupt()
        if m.get("type") == "stop":
            await s.send(type="state", state="listening")
        elif m.get("type") == "text" and str(m.get("text", "")).strip():
            text = str(m["text"]).strip()[:2000]
            await s.send(type="final", text=text)
            await s.send(type="state", state="thinking")
            s.go, s.prev = asyncio.Event(), None  # typed: there's no audio to glue back if they then speak
            s.go.set()
            s.task = asyncio.create_task(s.reply(text))

    async def reply(s, text):
        s.history.append({"role": "user", "content": text})
        s.spoke, said, pending, s.spoken = False, "", "", []
        body = {"model": LLM, "think": False, "keep_alive": -1, "options": {"num_ctx": 4096},
                "messages": [{"role": "system", "content": system_prompt()}, *s.history[-12:]]}
        try:
            async with httpx.AsyncClient(timeout=None) as http, http.stream("POST", f"{OLLAMA}/api/chat", json=body) as r:
                if r.is_error:
                    raise httpx.HTTPError(f"{r.status_code} {(await r.aread()).decode()[:200]}")
                async for line in r.aiter_lines():
                    if line:
                        m = json.loads(line)
                        if "error" in m:
                            raise httpx.HTTPError(m["error"])
                        pending += m.get("message", {}).get("content", "")
                        while chunk := next_chunk(pending, not said):  # speak as soon as a piece is ready
                            piece, pending = chunk
                            said += await s.say(piece)
            said += await s.say(pending)
            await s.send(type="state", state="listening")
        except asyncio.CancelledError:
            if not s.spoke:  # nothing was said: drop the turn, it will be merged or replaced
                s.history.pop()
                said = ""
            raise
        except httpx.HTTPError as e:
            if not s.spoke:
                s.history.pop()
            await s.send(type="reply", text=f"[erro no Ollama: {e}]")
            await s.send(type="state", state="listening")
        finally:
            if said:
                s.history.append({"role": "assistant", "content": said.strip()})
            if s.task is asyncio.current_task():
                s.task = None

    async def say(s, text):
        text = re.sub(r"[*#_`~]+", "", text).strip()
        if not re.search(r"\w", text):
            return ""
        pcm = await asyncio.to_thread(tts, text)
        await s.go.wait()  # drafted early: speak only once the silence was long enough
        if not s.spoke:
            s.spoke, s.t_play = True, time.monotonic()
            await s.send(type="state", state="speaking")
        await s.send(type="reply", text=text)
        await s.ws.send_bytes(pcm)
        s.spoken.append((text, len(pcm) / 2 / voice.config.sample_rate))
        return text + " "


app = FastAPI()


@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})  # always serve the latest UI


active = None  # one session at a time: two windows would both listen and both answer


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global active
    await ws.accept()
    await asyncio.to_thread(READY.wait)
    if LOAD_ERROR:
        await ws.send_text(json.dumps({"type": "error", "text": f"falha ao carregar os modelos: {LOAD_ERROR}"}))
        return
    if active:
        await active.close(code=4001)  # the newest window takes over
    active = ws
    s = Session(ws)
    await s.send(type="hello", rate=voice.config.sample_rate)
    try:
        while (m := await ws.receive())["type"] != "websocket.disconnect":
            if m.get("bytes") and len(m["bytes"]) == WIN * 2:  # the browser sends exactly one 512-sample Int16 frame
                await s.on_audio(np.frombuffer(m["bytes"], np.int16).astype(np.float32) / 32768)
            elif m.get("text"):
                await s.on_text(json.loads(m["text"]))
    finally:
        if s.task:
            s.task.cancel()
        if active is ws:
            active = None



def ollama_exe():
    return shutil.which("ollama") or str(DATA / "ollama/bin/ollama")


def start_ollama():
    """Start Ollama unless it's already up; returns the process we started, if any."""
    try:
        httpx.get(f"{OLLAMA}/api/version", timeout=1)
        return None
    except httpx.HTTPError:
        pass
    env = {**os.environ, "OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"}  # smaller KV cache: room for Whisper in 4 GB
    return subprocess.Popen([ollama_exe(), "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def allow_mic(profile):
    """Pre-grant the mic to X1's page in its private Chrome profile: no prompt, and no warning-bar flags needed."""
    prefs = profile / "Default/Preferences"
    p = json.loads(prefs.read_text()) if prefs.exists() else {}
    (p.setdefault("profile", {}).setdefault("content_settings", {}).setdefault("exceptions", {})
     .setdefault("media_stream_mic", {}))[f"{URL},*"] = {"setting": 1}
    prefs.parent.mkdir(parents=True, exist_ok=True)
    prefs.write_text(json.dumps(p))


def running():
    try:
        return httpx.get(f"http://127.0.0.1:{PORT}", timeout=1).is_success
    except httpx.HTTPError:
        return False


if __name__ == "__main__":  # the desktop app: server + Ollama + an app window; closing the window shuts it all down
    if Path("/etc/ssl/certs/ca-certificates.crt").exists():  # trust the system CAs (TLS-inspecting proxies) for model downloads
        os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
    if running():  # already open (clicking the icon focuses that window); a second one would be a second listener
        raise SystemExit("O X1 já está aberto.")
    chrome = next(filter(None, map(shutil.which, ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"))), None)
    profile = Path.home() / "snap/chromium/common/x1" if chrome and chrome.startswith("/snap/") else DATA / "browser"  # snaps can't write ~/.local
    window = [chrome, f"--app={URL}", f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
              "--autoplay-policy=no-user-gesture-required"]  # its own profile: a separate app, and the call returns when it closes
    threading.Thread(target=load, daemon=True).start()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    ollama = start_ollama()
    try:
        if chrome:
            allow_mic(profile)
            subprocess.run(window)
        else:  # shortcut: plain browser tab and no auto-shutdown, until we ship a webview for Chrome-less machines
            webbrowser.open(URL)
            threading.Event().wait()
    finally:
        try:  # free the GPU: unload the LLM (Ollama may be shared with other apps)
            httpx.post(f"{OLLAMA}/api/generate", json={"model": LLM, "keep_alive": 0}, timeout=5)
        except httpx.HTTPError:
            pass
        if ollama:
            ollama.terminate()
        server.should_exit = True
    os._exit(0)  # don't wait on worker threads (e.g. one still waiting for models that failed to load)
