"""Pipeline self-check: Piper speaks, X1 listens. Needs `ollama serve` running. Run: uv run selftest.py"""
import json
import subprocess
import time

import numpy as np
from fastapi.testclient import TestClient

import x1

x1.load()

assert x1.wait_for(0.99, "eu queria saber se dá pra, tipo") == 3.0 and x1.wait_for(0.99, "e aí eu pensei em ahn") == 3.0 and x1.wait_for(0.99, "") == 3.0
assert x1.wait_for(0.99, "que horas são?") == 1.0 and x1.wait_for(0.2, "que horas são?") == 2.0
assert x1.SENTENCE.split("Claro, senhor. Já está feito! E o res") == ["Claro, senhor.", "Já está feito!", "E o res"]


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
timed("whisper warmup", x1.transcribe, done)
assert "bras" in timed("whisper beam=5", x1.transcribe, done, 5).lower()  # Piper's synthetic voice trips "que horas"; check a stable word


def converse(audio):
    """Stream audio to the real server like the browser does; return the events until X1 finishes answering."""
    events, pcm = [], (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with TestClient(x1.app).websocket_connect("/ws") as ws:
        ws.receive_json()  # hello
        for i in range(0, len(pcm) - x1.WIN + 1, x1.WIN):
            ws.send_bytes(pcm[i:i + x1.WIN].tobytes())
        t = time.perf_counter()
        while True:
            m = ws.receive()
            if m.get("bytes"):
                events.append(("audio", round(time.perf_counter() - t, 2)))
            else:
                e = json.loads(m["text"])
                events.append((e["type"], e.get("text") or e.get("state")))
                if e == {"type": "state", "state": "listening"} and any(k == "audio" for k, _ in events):
                    return events


# a thought with a filler and a 1.2 s pause in the middle must come out as ONE turn
mid = np.concatenate([speak("Então, eu estava pensando em, ahn,"), silence(1.2), speak("comprar um café amanhã cedo."), silence(3.5)])
ev = converse(mid)
finals = [v for k, v in ev if k == "final"]
print("mid-thought pause ->", finals, "| reply:", " ".join(v for k, v in ev if k == "reply"))
assert len(finals) == 1 and "caf" in finals[0].lower(), ev

ev = converse(np.concatenate([done, silence(2.5)]))
print("question ->", [v for k, v in ev if k == "final"], "| reply:", " ".join(v for k, v in ev if k == "reply"))
assert any(k == "audio" for k, _ in ev)
print("ok")
