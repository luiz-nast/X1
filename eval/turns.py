"""How long should X1 wait before answering? Replays real pauses of spontaneous Brazilian Portuguese.

Data: one dialogue (SP_DID_200) of the NURC-SP corpus. For each speech segment we know what came next:
  hold  = the same speaker went on after `gap` seconds: answering before that is talking over them
  shift = the other speaker took the floor: a real end of turn
Run: uv run --with pyarrow eval/turns.py   (first run downloads 123 MB and transcribes ~1000 segments, ~15 min)
"""
import io
import json
import statistics as st
import sys
import wave
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parents[1]))
import x1  # noqa: E402

CACHE = Path(__file__).with_name("nurc.json")
L = 1.5  # seconds X1 needs for its first audio anyway (transcribe + first LLM sentence + TTS), measured on a GTX 1650
PAUSE = x1.PAUSE * x1.WIN / x1.SR

if not CACHE.exists():
    x1.load()
    f = hf_hub_download("nilc-nlp/CORAA-NURC-SP-Audio-Corpus", "data/filtered/test-00002.parquet", repo_type="dataset")
    segs = sorted(pq.read_table(f).to_pylist(), key=lambda r: (r["audio_name"], float(r["start_time"])))
    rows = []
    for a, b in zip(segs, segs[1:]):
        gap = float(b["start_time"]) - float(a["end_time"])
        if a["audio_name"] != b["audio_name"] or not 0 <= gap <= 3:
            continue  # a segment is missing in between: we don't know what happened
        with wave.open(io.BytesIO(a["audio"]["bytes"])) as w:
            assert w.getframerate() == x1.SR and w.getnchannels() == 1
            audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        rows.append(dict(hold=a["speaker_id"] == b["speaker_id"], gap=gap, p=x1.turn_done(audio), text=x1.transcribe(audio)))
    CACHE.write_text(json.dumps(rows, ensure_ascii=False))
rows = json.loads(CACHE.read_text())


def punct(t):  # RealtimeVoiceChat's pause per ending punctuation
    t = t.strip()
    return 2.3 if t.endswith(("...", "…")) else .33 if t.endswith("?") else .39 if t.endswith((".", "!")) else 1.25


POLICIES = {
    "silêncio fixo 0,5 s": lambda p, t: .5,
    "silêncio fixo 2,0 s": lambda p, t: 2.0,
    "smart-turn sozinho (p>0,5, senão 2,5 s)": lambda p, t: PAUSE if p > .5 else 2.5,
    "média pontuação + modelo (RealtimeVoiceChat)": lambda p, t: max(PAUSE, .65 * punct(t) + .35 * (1 - p) * 2.5),
    "degraus por confiança (LiveKit)": lambda p, t: PAUSE if p > .7 else 1.2 if p > .3 else 3.0,
    "X1 (wait_for)": x1.wait_for,
}
holds = [r for r in rows if r["hold"] and r["gap"] > PAUSE]
shifts = [r for r in rows if not r["hold"]]
print(f"{len(holds)} pausas no meio da fala (mediana {st.median(r['gap'] for r in holds):.2f} s), {len(shifts)} fins de turno, L = {L} s\n")
print(f"{'política':46} {'falou por cima':>14} {'resposta média':>15}")
for name, f in POLICIES.items():
    over = st.mean(bool(r["text"]) and max(f(r["p"], r["text"]), L) < r["gap"] for r in holds)  # nothing transcribed: no answer
    print(f"{name:46} {over:14.0%} {st.mean(max(f(r['p'], r['text']), L) for r in shifts):14.2f}s")
print(f"\nsmart-turn diz 'terminou' (p>0,5): {st.mean(r['p'] > .5 for r in holds):.0%} das pausas no meio, "
      f"{st.mean(r['p'] > .5 for r in shifts):.0%} dos fins de turno")
print(f"texto 'aberto': {st.mean(bool(x1.OPEN.search(r['text'])) for r in holds):.0%} das pausas no meio, "
      f"{st.mean(bool(x1.OPEN.search(r['text'])) for r in shifts):.0%} dos fins de turno")
