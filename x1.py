"""X1 — mini Jarvis local: microfone -> Silero VAD + smart-turn -> Whisper -> LLM (Ollama) -> Piper -> alto-falante."""
import asyncio
import ctypes
import json
import re
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
OLLAMA = "http://127.0.0.1:11434/api/chat"
SR, WIN = 16000, 512                  # Silero VAD window = 32 ms
PAUSE = round(0.3 * SR / WIN)         # first short silence: judge the turn and start drafting the reply
PARTIAL_EVERY = round(0.6 * SR / WIN) # live transcript refresh while the user talks
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

WEB = Path(__file__).parent / "web"
whisper = WhisperModel("small", device="cuda", compute_type="int8_float16")
vad = ort.InferenceSession(str(Path(faster_whisper.__file__).parent / "assets/silero_vad_v6.onnx"))
turn = ort.InferenceSession(hf_hub_download("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx"))
mel = FeatureExtractor(feature_size=80, chunk_length=8)
voice = PiperVoice.load(hf_hub_download("rhasspy/piper-voices", "pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx"),
                        hf_hub_download("rhasspy/piper-voices", "pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx.json"))


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
    if OPEN.search(text):
        return 3.0  # "e…", "tipo", "ahn", "né": still thinking out loud
    return 1.0 if p > 0.7 else 2.0


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

    async def send(s, **m):
        await s.ws.send_text(json.dumps(m))

    def vad(s, x):
        p, s.h, s.c = vad.run(None, {"input": np.concatenate([s.ctx, x])[None], "h": s.h, "c": s.c})
        s.ctx = x[-64:]
        return float(p.ravel()[0])

    async def on_audio(s, x):
        p = s.vad(x)
        if s.task and s.spoke:
            return  # X1 is talking; the browser mutes the mic too
        s.run = s.run + 1 if p > 0.5 else 0
        if p > 0.5:
            resumed = not s.buf or s.silence >= PAUSE
            if not s.buf:
                s.buf = list(s.pre)
            if s.task and s.run >= 4:  # they kept talking before X1 said a word: same thought, drop the draft
                s.task.cancel()
                s.task, resumed = None, True
                if s.go.is_set():  # we had already committed and cleared the buffer: glue the audio back
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

    async def on_text(s, m):
        if s.task:
            s.task.cancel()
            s.task = None
        if m.get("type") == "stop":
            await s.send(type="state", state="listening")
        elif m.get("type") == "text" and str(m.get("text", "")).strip():
            text = str(m["text"]).strip()[:2000]
            await s.send(type="final", text=text)
            await s.send(type="state", state="thinking")
            s.go = asyncio.Event()
            s.go.set()
            s.task = asyncio.create_task(s.reply(text))

    async def reply(s, text):
        s.history.append({"role": "user", "content": text})
        s.spoke, said, pending = False, "", ""
        system = f"{SYSTEM} Agora são {datetime.now():%H:%M} de {datetime.now():%d/%m/%Y}."
        body = {"model": LLM, "think": False, "keep_alive": -1, "options": {"num_ctx": 4096},
                "messages": [{"role": "system", "content": system}, *s.history[-12:]]}
        try:
            async with httpx.AsyncClient(timeout=None) as http, http.stream("POST", OLLAMA, json=body) as r:
                async for line in r.aiter_lines():
                    if line:
                        pending += json.loads(line).get("message", {}).get("content", "")
                        *done, pending = SENTENCE.split(pending)  # speak each sentence as soon as it's complete
                        for sentence in done:
                            said += await s.say(sentence)
            said += await s.say(pending)
            await s.send(type="state", state="listening")
        except asyncio.CancelledError:
            if not s.spoke:  # nothing was said: drop the turn, it will be merged or replaced
                s.history.pop()
                said = ""
            raise
        except httpx.HTTPError as e:
            await s.send(type="reply", text=f"[sem conexão com o Ollama: {e}]")
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
            s.spoke = True
            await s.send(type="state", state="speaking")
        await s.send(type="reply", text=text)
        await s.ws.send_bytes(pcm)
        return text + " "


app = FastAPI()


@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})  # always serve the latest UI


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
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


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
