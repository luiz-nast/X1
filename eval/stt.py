"""Speech-to-text bake-off for X1: which model hears Brazilian Portuguese best on this machine.

  uv run --with num2words --with pyarrow --with edge-tts eval/stt.py build   # test audio -> eval/stt_data/
  uv run eval/stt.py noisy                                                   # `cmd` with room reverb + noise (silent, digital)
  uv run --with num2words --with nvidia-cublas-cu12 --with "nvidia-cudnn-cu12>=9,<10" eval/stt.py run fw:small onnx:...
                                                                             # score models (each loads in turn)

Sets: cmd = assistant phrases (numbers, "valeu", slang, hesitation) in natural neural voices;
      noisy = the same phrases with room reverb and background noise at 15 dB SNR (made digitally, nothing is played);
      nurc = real spontaneous speech from São Paulo (NURC-SP) with human transcripts.
"""
import asyncio
import gc
import io
import json
import re
import subprocess
import sys
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import x1  # noqa: E402

DATA = Path(__file__).with_name("stt_data")
RESULTS = Path(__file__).with_name("stt_results.json")
VOICES = ["pt-BR-AntonioNeural", "pt-BR-FranciscaNeural", "pt-BR-ThalitaMultilingualNeural"]
CMD = [
    "Que horas são agora?", "Valeu, X1.", "Valeu, pode parar.", "Sete.", "Seis.", "Dezessete.", "Oitenta e nove.",
    "Me acorda às sete da manhã.", "Quanto é sete vezes oito?", "Coloca um timer de quinze minutos.",
    "Liga a luz da sala.", "Qual a previsão do tempo pra amanhã em Brasília?", "Beleza, entendi.",
    "Não, não, para. Espera aí.", "Toca uma música do Djavan.",
    "Manda uma mensagem pro João dizendo que eu vou atrasar dez minutos.", "Quantos dias faltam pro Natal?",
    "Tô cansado hoje, cara.", "Fala sério, mano.", "Show de bola, muito obrigado.",
    "Ahn, deixa eu pensar, pode ser às seis e meia.", "Qual é a raiz quadrada de cento e quarenta e quatro?",
    "Converte cinquenta dólares pra reais.", "A reunião é na quarta-feira às catorze horas.",
    "Quem ganhou a Copa do Mundo de dois mil e dois?", "Aumenta o volume.", "Pesquisa receita de pão de queijo.",
    "Sim.", "Não.", "Pode repetir, por favor?", "Tá bom, valeu, tchau.", "Me lembra de ligar pra minha mãe amanhã.",
    "X1, você está me ouvindo?", "Seis ou sete, tanto faz.", "Vinte e três graus.", "Liga pro Luiz Felipe.",
]


def norm(t):
    """Compare words, not spelling: digits become words, no case, accents, punctuation or hesitations."""
    from num2words import num2words
    t = t.lower().replace("%", " por cento")
    t = re.sub(r"(\d+)h(\d*)\b", r"\1 horas \2", t)  # "7h30"
    t = re.sub(r"(\d+),(\d+)", r"\1 vírgula \2", t)
    t = re.sub(r"(?<=[^\W\d_])(?=\d)", " ", t)  # "X1" -> "X 1", same as "X um"
    t = re.sub(r"\d+", lambda m: num2words(int(m.group()), lang="pt_BR"), t)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-z ]", " ", t.replace("-", " "))
    t = re.sub(r"\b(a+h+n*|e+h+|h+u+m+|h+m+|u+h+|u+h+u+m+|a+h+a+m+|mhm)\b", " ", t)  # ãh, éh, hum, ahn...
    t = re.sub(r"\bquatorze\b", "catorze", t)
    t = re.sub(r"\bpara\b", "pra", t)  # said "pra", often written "para"
    return t.split()


def edits(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1]


assert norm("Me acorda às 7h da manhã!") == ["me", "acorda", "as", "sete", "horas", "da", "manha"] == norm("me acorda às 7 horas da manhã")
assert edits("a b c".split(), "a x c d".split()) == 2 and norm("Valeu, X1.") == norm("valeu, x um") and norm("Ãh, é pra 14%") == ["e", "pra", "catorze", "por", "cento"]


def save(path, pcm16k):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(pcm16k, -1, 1) * 32767).astype(np.int16).tobytes())


def to16k(data):  # any audio file's bytes -> 16 kHz mono float
    out = subprocess.run(["ffmpeg", "-v", "quiet", "-i", "-", "-ar", "16000", "-ac", "1", "-f", "s16le", "-"],
                         input=data, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768


def load_set(name):
    refs = json.loads((DATA / f"{name}.json").read_text())
    for r in refs:
        with wave.open(str(DATA / name / f"{r['id']}.wav")) as w:
            r["audio"] = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return refs


def build():
    import truststore
    truststore.inject_into_ssl()  # edge-tts pins certifi; trust the system store too (TLS-inspecting proxies)
    import edge_tts
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    refs = []
    for i, text in enumerate(CMD):
        for v in VOICES:
            uid = f"{i:02d}-{v.split('-')[2][:4].lower()}"
            refs.append({"id": uid, "text": text})
            if (DATA / "cmd" / f"{uid}.wav").exists():
                continue

            async def synth():
                return b"".join([c["data"] async for c in edge_tts.Communicate(text, v).stream() if c["type"] == "audio"])
            for attempt in range(4):  # the free service drops a request now and then
                try:
                    mp3 = asyncio.run(synth())
                    break
                except edge_tts.exceptions.NoAudioReceived:
                    time.sleep(2 + 2 * attempt)
            else:
                raise SystemExit(f"edge-tts failed for {uid}")
            save(DATA / "cmd" / f"{uid}.wav", np.concatenate([np.zeros(4000, np.float32), to16k(mp3), np.zeros(4000, np.float32)]))
    (DATA / "cmd.json").write_text(json.dumps(refs, ensure_ascii=False, indent=0))

    f = hf_hub_download("nilc-nlp/CORAA-NURC-SP-Audio-Corpus", "data/filtered/test-00002.parquet", repo_type="dataset")
    rows = [r for r in pq.read_table(f).to_pylist() if 1.5 <= float(r["duration"]) <= 10]
    refs = []
    for r in rows[:: max(1, len(rows) // 150)][:150]:  # spread over the recordings
        uid = f"{r['audio_name']}-{float(r['start_time']):07.1f}"
        save(DATA / "nurc" / f"{uid}.wav", to16k(r["audio"]["bytes"]))
        refs.append({"id": uid, "text": r["text"]})
    (DATA / "nurc.json").write_text(json.dumps(refs, ensure_ascii=False, indent=0))
    print("built", {s: len(json.loads((DATA / f"{s}.json").read_text())) for s in ("cmd", "nurc")})


def noisy():
    """`cmd` as a laptop mic would hear it in a room: reverb + background noise at 15 dB SNR. Built digitally (nothing is
    played out loud: the default output may be headphones, and the room would leak into the mic)."""
    rng = np.random.default_rng(0)
    t = np.arange(int(0.4 * 16000)) / 16000
    rir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / 0.4)  # RT60 = 0.4 s, a small room
    rir[0] = 8.0  # direct sound
    rir /= np.abs(rir).sum() / 4
    refs = json.loads((DATA / "cmd.json").read_text())
    for r in load_set("cmd"):
        wet = np.convolve(r["audio"], rir)[:len(r["audio"])]
        pink = np.cumsum(rng.standard_normal(len(wet))) * 0.02  # brown-ish hum + hiss, like a fan
        pink = pink - np.convolve(pink, np.ones(400) / 400, "same") + rng.standard_normal(len(wet)) * 0.3
        noise = pink / np.sqrt(np.mean(pink ** 2)) * np.sqrt(np.mean(wet[wet != 0] ** 2)) / 10 ** (15 / 20)
        save(DATA / "noisy" / f"{r['id']}.wav", (wet + noise) * 0.9 / max(np.abs(wet + noise).max(), 1e-6))
    (DATA / "noisy.json").write_text(json.dumps(refs, ensure_ascii=False, indent=0))
    print("noisy:", len(refs))


def loader(spec):
    """'fw:<whisper model or HF repo>[:noprompt]' = faster-whisper on the GPU, decoded the way X1 used to;
    'onnx:<onnx-asr model>[@<HF repo>][:int8]' = onnx-asr on the CPU (what X1 uses now)."""
    kind, name, *flags = spec.split(":")
    if kind == "fw":
        import ctypes
        import nvidia.cublas.lib
        for lib in ("cublas/lib/libcublasLt.so.12", "cublas/lib/libcublas.so.12", "cudnn/lib/libcudnn.so.9"):  # CUDA 12 for ctranslate2
            ctypes.CDLL(str(Path(nvidia.cublas.lib.__path__[0]).parents[1] / lib), mode=ctypes.RTLD_GLOBAL)
        from faster_whisper import WhisperModel
        model = WhisperModel(name, device="cuda", compute_type="int8_float16")
        prompt = None if "noprompt" in flags else "Ahn... então, X1, eu acho que... hmm, tipo, sabe?"

        def hear(a):
            segs, _ = model.transcribe(a, language="pt", beam_size=5, condition_on_previous_text=False,
                                       without_timestamps=True, initial_prompt=prompt)
            return " ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.6)
        return hear
    if kind == "onnx":  # 'onnx:<onnx-asr model>[@<HF repo>][:int8]'
        import onnx_asr
        model_name, _, repo = name.partition("@")
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo, local_dir=Path.home() / ".cache/x1-stt" / repo.replace("/", "--")) if repo else None  # real files:
        model = onnx_asr.load_model(model_name, path, quantization="int8" if "int8" in flags else None)  # ORT refuses symlinked .onnx.data
        return lambda a: model.recognize(a, sample_rate=16000)
    raise SystemExit(f"unknown model spec {spec}")


def vram():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split()[0])


def run(specs):
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    sets = {s: load_set(s) for s in ("cmd", "noisy", "nurc") if (DATA / f"{s}.json").exists()}
    for spec in specs:
        base = vram()
        hear = loader(spec)
        hear(np.zeros(16000, np.float32))  # warm up
        res = {"vram_mb": vram() - base}
        for name, refs in sets.items():
            errs = words = 0
            lat, worst, hyps = [], [], []
            for r in refs:
                t = time.perf_counter()
                hyp = hear(r["audio"])
                lat.append(time.perf_counter() - t)
                hyps.append(hyp)
                ref_w, hyp_w = norm(r["text"]), norm(hyp)
                e = edits(ref_w, hyp_w)
                errs, words = errs + e, words + len(ref_w)
                if e:
                    worst.append((e / max(len(ref_w), 1), r["text"], hyp))
            lat.sort()
            res[name] = {"wer": errs / words, "n": len(refs), "lat_s": sum(lat) / len(lat), "lat_p90_s": lat[int(.9 * len(lat))]}
            if name != "nurc":  # NURC-SP is CC-BY-NC-ND: keep its scores, not its (or derived) text
                res[name] |= {"errors": [w[1:] for w in sorted(worst, reverse=True)[:40]], "hyps": hyps}
            print(f"{spec:40} {name:5} WER {errs / words:6.1%}  {res[name]['lat_s']:.2f}s/frase", flush=True)
        results[spec] = res
        RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=1))
        hear = None
        gc.collect()


if __name__ == "__main__":
    {"build": lambda: build(), "noisy": lambda: noisy(), "run": lambda: run(sys.argv[2:])}[sys.argv[1]]()
