# X1

Um mini Jarvis local, com a voz como canal principal. Tudo roda na sua máquina: nenhum áudio ou texto vai pra nuvem.

Testado numa GTX 1650 (4 GB) com 7 GB de RAM, no Linux. Com tudo carregado, usa ~1,9 GB de VRAM.

```
navegador (microfone com cancelamento de eco + bolinha)  ⇄ WebSocket ⇄  x1.py
  mic 16 kHz → Silero VAD → pausa de 0,3 s → smart-turn v3.2 (entonação) + Whisper small (palavras)
  → política de espera → Qwen3 1.7B (Ollama), frase a frase → Piper pt_BR (voz) → navegador
```

## Rodar

Precisa de GPU NVIDIA (driver ≥ 550), [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com), Chrome e ffmpeg (este só pro selftest).

```bash
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve   # num terminal à parte
ollama pull qwen3:1.7b
uv run x1.py                                  # na primeira vez, baixa o Whisper, o smart-turn e a voz
google-chrome --app=http://localhost:8000     # abre como aplicativo
```

- Para ativar, toque na bolinha ou aperte Espaço.
- Enquanto o X1 fala, Espaço ou um clique interrompe.
- Também dá pra digitar.

`uv run selftest.py` testa a cadeia inteira. Um dos casos é uma fala com "ahn" e uma pausa de 1,2 s no meio, que precisa sair como um turno só.

## Esperar você terminar de pensar

Gente não fala a ideia de uma vez: pensa em voz alta, faz "ahn", para e continua. O X1 decide quando responder em quatro passos:

1. O **Silero VAD** marca fala e silêncio a cada 32 ms.
2. **Na primeira pausa de 0,3 s**, ele olha duas coisas:
   - a **entonação**, com o [smart-turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3), que dá a probabilidade de você ter terminado;
   - as **palavras**, com o Whisper. Um prompt de contexto faz o Whisper manter os "ahn", "éh" e "tipo" em vez de limpá-los.
3. Com isso, ele decide **quanto silêncio esperar** (`wait_for` em `x1.py`):
   - frase terminando em "e", "que", "de", "tipo", "né", "ahn", "…" ou vírgula: **3 s**;
   - senão, se o smart-turn está confiante (p > 0,7): **1 s**;
   - senão: **2 s**.
4. **Execução especulativa.** Na própria pausa de 0,3 s ele já começa a gerar a resposta, mas só solta a voz quando o silêncio atinge a espera escolhida.
   - Se você volta a falar antes disso, ele descarta a resposta e continua ouvindo a mesma ideia.
   - Nesta máquina, a primeira frase de áudio leva ~1,5 s para ficar pronta de qualquer jeito. Então esperas de até ~1,5 s não custam latência.

A bolinha mostra "pode continuar…" quando ele acha que você está no meio de um pensamento.

### Por que essa política (medido)

Avaliei as políticas em **fala espontânea real**: o diálogo SP_DID_200 do [NURC-SP](https://huggingface.co/datasets/nilc-nlp/CORAA-NURC-SP-Audio-Corpus), com 155 pausas no meio da fala (mediana de 1,22 s) e 25 trocas de falante. "Falou por cima" é quando o X1 começaria a responder antes de a pessoa voltar a falar. Considerei 1,5 s até o primeiro áudio. Para reproduzir: `uv run --with pyarrow eval/turns.py`.

| política | falou por cima | resposta média |
|---|---|---|
| silêncio fixo (o que a maioria faz) | 37% | 1,50 s |
| smart-turn sozinho (p > 0,5, senão 2,5 s) | 27% | 1,82 s |
| média pontuação + modelo ([RealtimeVoiceChat](https://github.com/KoljaB/RealtimeVoiceChat)) | 31% | 1,60 s |
| degraus por confiança (estilo LiveKit) | 30% | 1,80 s |
| **X1: texto aberto → 3 s, p > 0,7 → 1 s, senão 2 s** | **14%** | **2,16 s** |

O que os dados mostraram:

- **O smart-turn sozinho não separa os casos em português espontâneo.** Ele diz "terminou" em 69% das pausas no meio da fala e em 68% dos fins de turno reais. Bate com [este estudo](https://huggingface.co/marcosremar2/turn-taking-study).
- **Nas amostras sintéticas do próprio smart-turn, o quadro é outro:** ele erra só 5%. Voz de TTS engana.
- **As palavras é que vetam a resposta precipitada.**

## Por que esses modelos (medido nesta máquina)

| peça | escolha | medido | descartado |
|---|---|---|---|
| fala → texto | faster-whisper `small` int8 (GPU) | ~0,5 s por frase, 390 MB | `large-v3-turbo`: 1,8 s por frase e estourou a RAM ao carregar |
| cérebro | `qwen3:1.7b` sem thinking (Ollama) | 85 tokens/s, 1ª frase em ~0,45 s | `qwen3.5:2b`: 17 tokens/s nesta GPU, respostas piores |
| voz | Piper `pt_BR-faber-medium` (CPU) | ~0,2 s por frase | Kokoro: vozes pt-BR fracas e mais RAM |
| fim de turno | Silero VAD + smart-turn v3.2 (CPU) | ~0,1 s | LiveKit turn-detector: 400 MB e licença restrita |

## Limites deste kickoff

- **Meia-duplex:** o microfone fica mudo enquanto o X1 fala, senão ele ouve a si mesmo pelo alto-falante do notebook. Para interromper, use Espaço ou clique.
- **Ainda não tem:**
  - wake word;
  - ferramentas (timer, busca, apps);
  - memória entre sessões.
- **A política foi calibrada num único diálogo de entrevista.** Ele tem mais monólogo que um uso de assistente, e só 25 fins de turno. Quando você usar de verdade, as constantes de `wait_for` são o lugar de ajustar.
