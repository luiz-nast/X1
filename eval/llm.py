"""Quick bake-off of local LLMs for X1: speed on this GPU + a battery of pt-BR sentences, saved for judging.
Run: uv run eval/llm.py qwen3:1.7b llama3.2:3b ...   (pulls missing models; tests one at a time, the GPU fits one)
"""
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parents[1]))
import x1  # noqa: E402  (no models are loaded: only the prompt and the Ollama settings are used)

OUT = Path(__file__).with_name("llm_results.json")
# (case, user turns, what a good answer does), judged on the LAST reply
CASES = [
    ("identidade", ["Quem é você?"], "Se apresenta como X1, assistente de voz; curto."),
    ("hora", ["Que horas são?"], "Diz a hora de asked_at (ela vem no prompt de sistema)."),
    ("data", ["Que dia é hoje?"], "Diz a data de asked_at (vem no prompt de sistema)."),
    ("conta", ["Quanto é dezessete vezes vinte e três?"], "391."),
    ("conversão", ["Quantos minutos tem em três horas e meia?"], "210 minutos."),
    ("raciocínio", ["Se eu tenho três maçãs e como uma, quantas sobram?"], "Duas."),
    ("fato", ["Qual é a capital da Austrália?"], "Canberra."),
    ("fato-br", ["Quem escreveu Dom Casmurro?"], "Machado de Assis."),
    ("fato-ano", ["Em que ano o homem pisou na Lua pela primeira vez?"], "1969."),
    ("armadilha", ["Me conta sobre o prêmio Nobel de física que o Pelé ganhou."], "Corrige: Pelé nunca ganhou Nobel. Não inventa."),
    ("tempo-real", ["Qual foi o placar do jogo do Flamengo ontem?"], "Diz que não tem acesso a resultados em tempo real. Não inventa placar."),
    ("pessoal", ["Qual é o telefone do meu dentista?"], "Diz que não sabe. Não inventa número."),
    ("capacidade", ["Liga a luz da sala."], "Diz que ainda não controla dispositivos. Não finge que ligou."),
    ("lembrete", ["Me lembra de tomar o remédio às oito da noite."], "Diz que ainda não cria lembretes. Não finge que agendou."),
    ("estilo", ["Me explica o que é inflação."], "Correto, 1 a 3 frases, sem markdown nem listas."),
    ("lista", ["Me fala cinco frutas."], "Cinco frutas em fala corrida, sem markdown, bullets ou numeração."),
    ("pensando-alto", ["Ahn, então, eu tava pensando, tipo, se vale a pena, sei lá, aprender piano depois dos trinta?"],
     "Entende a pergunta apesar das hesitações; responde curto e encorajador."),
    ("transcrição-ruim", ["qual a previsão do tempo pra amanha em brasilha"],
     "Entende Brasília; diz que não tem acesso à previsão do tempo. Não inventa temperatura."),
    ("memória", ["Meu nome é Luiz e eu moro em Brasília.", "Onde eu moro e qual é o meu nome?"], "Luiz, Brasília."),
    ("contexto", ["Quero fazer um bolo de cenoura.", "E quanto tempo vai no forno?"], "Entende que é o bolo: uns 35 a 45 minutos, ~180 °C."),
    ("gíria", ["E aí, beleza? Tô meio cansado hoje."], "Natural, empático, curto, português do Brasil."),
    ("piada", ["Me conta uma piada curta."], "Uma piada curta em português que faça sentido."),
    ("técnico", ["Como eu faço um commit no git?"], "git add e git commit -m com mensagem; curto, sem markdown."),
]


def chat(model, messages):
    body = {"model": model, "think": False, "keep_alive": -1, "options": x1.OPTIONS, "messages": messages}
    t, first, sentence, text, done = time.perf_counter(), None, None, "", {}
    with httpx.stream("POST", f"{x1.OLLAMA}/api/chat", json=body, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                m = json.loads(line)
                text += m.get("message", {}).get("content", "")
                first = first or (text and time.perf_counter() - t)
                sentence = sentence or (x1.next_chunk(text, True) and time.perf_counter() - t)  # first piece X1 can speak
                done = m if m.get("done") else done
    total = time.perf_counter() - t
    return text.strip(), dict(first=first, sentence=sentence or total, total=total, hit_limit=done.get("done_reason") == "length",
                              tps=done.get("eval_count", 0) / max(done.get("eval_duration", 1) / 1e9, 1e-9))


results = json.loads(OUT.read_text()) if OUT.exists() else {}
for model in sys.argv[1:]:
    if subprocess.run([x1.ollama_exe(), "show", model], capture_output=True).returncode:
        subprocess.run([x1.ollama_exe(), "pull", model], check=True)
    chat(model, [{"role": "user", "content": "oi"}])  # load + warm up
    used = next(m["size_vram"] for m in httpx.get(f"{x1.OLLAMA}/api/ps").json()["models"] if m["name"] == model) // 2**20
    rows = []
    for case, turns, expect in CASES:
        msgs, asked_at = [{"role": "system", "content": x1.system_prompt()}], f"{datetime.now():%H:%M %d/%m/%Y}"
        for u in turns:
            msgs.append({"role": "user", "content": u})
            reply, speed = chat(model, msgs)
            msgs.append({"role": "assistant", "content": reply})
        rows.append(dict(case=case, turns=turns, expect=expect, asked_at=asked_at, reply=reply, **speed))
        print(f"{model:28} {case:17} {speed['sentence']:5.2f}s {'LOOP ' if speed['hit_limit'] else ''}{reply[:90]!r}", flush=True)
    httpx.post(f"{x1.OLLAMA}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)  # free the GPU for the next one
    med = lambda k: sorted(r[k] for r in rows)[len(rows) // 2]
    results[model] = dict(vram_mb=used, first_sentence_s=med("sentence"), tokens_per_s=med("tps"), rows=rows)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1))

print(f"\n{'modelo':28} {'VRAM':>7} {'1ª frase':>9} {'tokens/s':>9}")
for model, r in results.items():
    print(f"{model:28} {r['vram_mb']:6}M {r['first_sentence_s']:8.2f}s {r['tokens_per_s']:9.0f}")
