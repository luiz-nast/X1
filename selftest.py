"""Pipeline self-check: Piper speaks, X1 listens. Needs `ollama serve` running. Run: uv run selftest.py"""
import json
import subprocess
import time

import numpy as np
from fastapi.testclient import TestClient

import x1

x1.load()

assert x1.wait_for(0.99, "eu queria saber se dá pra, tipo") == 3.0 and x1.wait_for(0.99, "e aí eu pensei em ahn") == 3.0 and x1.wait_for(0.99, "") == 3.0
assert x1.wait_for(0.99, "que horas são?") == 1.0 and x1.wait_for(0.2, "que horas são?") == 2.5
assert x1.next_chunk("Claro, senhor, já está feito. E o res", True) == ("Claro, senhor,", "já está feito. E o res")
assert x1.next_chunk("Já está feito, chefe. E o res", False) == ("Já está feito, chefe.", "E o res") and not x1.next_chunk("São 3,5 litros", True)


def speak(text):  # Piper 22.05 kHz -> 16 kHz float (ffmpeg low-passes; naive interpolation aliases and fools Whisper)
    cmd = ["ffmpeg", "-v", "quiet", "-f", "s16le", "-ar", str(x1.voice.config.sample_rate), "-i", "-", "-ar", "16000", "-f", "s16le", "-"]
    return np.frombuffer(subprocess.run(cmd, input=x1.tts(text), capture_output=True).stdout, np.int16).astype(np.float32) / 32768


def silence(secs):
    return np.zeros(int(secs * x1.SR), np.float32)


def timed(label, f, *a):
    t = time.perf_counter()
    r = f(*a)
    print(f"{label:<26} {1000 * (time.perf_counter() - t):6.0f} ms  {r!r}")
    return r


done = np.concatenate([silence(.5), speak("Olá, X1. Que horas são agora em Brasília?"), silence(.5)])
timed("smart-turn (finished)", x1.turn_done, done)
timed("stt warmup", x1.transcribe, done)
assert "bras" in timed("stt", x1.transcribe, done).lower()  # Piper's synthetic voice trips "que horas"; check a stable word


def stream(ws, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    for i in range(0, len(pcm) - x1.WIN + 1, x1.WIN):
        ws.send_bytes(pcm[i:i + x1.WIN].tobytes())


def events(ws, until):
    """Read the server's events (audio as "audio") until `until(events)` is true."""
    ev = []
    while not until(ev):
        m = ws.receive()
        if m.get("bytes"):
            ev.append(("audio", None))
        else:
            e = json.loads(m["text"])
            ev.append((e["type"], e.get("text") or e.get("state")))
    return ev


answered = lambda ev: bool(ev) and ev[-1] == ("state", "listening") and ("audio", None) in ev


def converse(audio):
    """Stream audio to the real server like the browser does; return the events until X1 finishes answering."""
    with TestClient(x1.app).websocket_connect("/ws") as ws:
        ws.receive_json()  # hello
        stream(ws, audio)
        return events(ws, answered)


# a thought with a filler and a 1.2 s pause in the middle must come out as ONE turn
mid = np.concatenate([speak("Então, eu estava pensando em, ahn,"), silence(1.2), speak("comprar um café amanhã cedo."), silence(3.5)])
ev = converse(mid)
finals = [v for k, v in ev if k == "final"]
print("mid-thought pause ->", finals, "| reply:", " ".join(v for k, v in ev if k == "reply"))
assert len(finals) == 1 and "caf" in finals[0].lower(), ev

ev = converse(np.concatenate([done, silence(2.5)]))
print("question ->", [v for k, v in ev if k == "final"], "| reply:", " ".join(v for k, v in ev if k == "reply"))
assert ("audio", None) in ev

# talking over X1 makes it stop, and what was said becomes the next turn
with TestClient(x1.app).websocket_connect("/ws") as ws:
    ws.receive_json()
    stream(ws, np.concatenate([done, silence(2.5)]))
    events(ws, lambda ev: ("audio", None) in ev)  # X1 started talking
    ws.send_text(json.dumps({"type": "playing", "on": True}))  # as the browser reports
    stream(ws, np.concatenate([speak("Não, não, para. Espera aí, eu quero outra coisa."), silence(3.5)]))
    ev = events(ws, lambda ev: any(k == "final" for k, _ in ev))
    print("talked over X1 ->", [e for e in ev if e[0] in ("interrupt", "final")])
    assert ("interrupt", None) in ev and "espera" in [v for k, v in ev if k == "final"][-1].lower(), ev

# speaking right after typing must not crash (it once glued a missing audio back) and the voice takes over
with TestClient(x1.app).websocket_connect("/ws") as ws:
    ws.receive_json()
    ws.send_text(json.dumps({"type": "text", "text": "Me conta uma curiosidade sobre o espaço."}))
    stream(ws, np.concatenate([speak("Esquece, me fala de Brasília."), silence(3.5)]))
    ev = events(ws, answered)
    print("typed then spoke ->", [v for k, v in ev if k == "final"])
    assert "bras" in [v for k, v in ev if k == "final"][-1].lower(), ev
print("ok")
