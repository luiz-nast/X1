# X1

Um mini Jarvis local, com a voz como canal principal. Tudo roda na sua máquina: nenhum áudio ou texto vai pra nuvem.

Testado numa GTX 1650 (4 GB) com 7 GB de RAM, no Ubuntu. Com tudo carregado, usa ~1,9 GB de VRAM.

```
navegador (microfone com cancelamento de eco + bolinha)  ⇄ WebSocket ⇄  x1.py
  mic 16 kHz → Silero VAD → pausa de 0,3 s → smart-turn v3.2 (entonação) + Whisper small (palavras)
  → política de espera → Qwen3 1.7B (Ollama), pedaço a pedaço → Piper pt_BR (voz) → navegador
```

## Instalar

```bash
curl -fsSL https://raw.githubusercontent.com/luiz-nast/X1/master/install.sh | bash
```

Depois é só abrir o **X1** pelo menu de aplicativos. Ele abre numa janela própria e começa a ouvir sozinho. Fechar a janela desliga tudo e libera a GPU.

O instalador:
- coloca tudo em `~/.local/share/x1`;
- usa o seu Ollama, se você já tiver um; senão, instala um ali dentro;
- baixa o LLM e os modelos de voz e testa a GPU;
- cria o atalho no menu.

Rodar de novo atualiza. Precisa de GPU NVIDIA, Chrome (ou Chromium), `git`, `curl` e `zstd`.

Para desinstalar:

```bash
rm -rf ~/.local/share/x1 ~/.local/share/applications/x1.desktop ~/.local/share/icons/hicolor/scalable/apps/x1.svg
```

- **Interromper:** fale por cima dele. Sua fala tem prioridade: ele para e passa a te ouvir. Espaço ou clique na bolinha também interrompe.
- **Digitar:** o campo embaixo da bolinha também funciona.
- **Testar a cadeia inteira (para desenvolvimento):** `uv run selftest.py`. Inclui uma fala com "ahn" e uma pausa de 1,2 s no meio, que precisa sair como um turno só.

## Esperar você terminar de pensar

Gente não fala a ideia de uma vez: pensa em voz alta, faz "ahn", para e continua. O X1 decide quando responder em quatro passos:

1. O **Silero VAD** marca fala e silêncio a cada 32 ms.
2. **Na primeira pausa de 0,3 s**, ele olha duas coisas:
   - a **entonação**, com o [smart-turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3), que dá a probabilidade de você ter terminado;
   - as **palavras**, com o Whisper. Um prompt de contexto faz o Whisper manter os "ahn", "éh" e "tipo" em vez de limpá-los.
3. Com isso, ele decide **quanto silêncio esperar** (`wait_for` em `x1.py`):
   - frase terminando em "e", "que", "de", "tipo", "né", "ahn", "…" ou vírgula, ou um murmúrio que o Whisper não entendeu: **3 s**;
   - senão, se o smart-turn está confiante (p > 0,7): **1 s**;
   - senão: **2 s**.
4. **Execução especulativa.** Na própria pausa de 0,3 s ele já começa a gerar a resposta, mas só solta a voz quando o silêncio atinge a espera escolhida.
   - Se você volta a falar antes disso, ele descarta a resposta e continua ouvindo a mesma ideia.
   - Nesta máquina, a primeira frase de áudio leva ~1,5 s para ficar pronta de qualquer jeito. Então esperas de até ~1,5 s não custam latência.

A bolinha mostra "pode continuar…" quando ele acha que você está no meio de um pensamento.

### O que vale a pena aqui (medido)

**Teste 1: fala espontânea real.** O diálogo SP_DID_200 do [NURC-SP](https://huggingface.co/datasets/nilc-nlp/CORAA-NURC-SP-Audio-Corpus) tem 155 pausas no meio da fala (mediana de 1,22 s) e 25 trocas de falante.
- "Falou por cima" é quando o X1 começaria a responder antes de a pessoa voltar a falar.
- Considerei 1,5 s até o primeiro áudio, como nesta máquina.
- Para reproduzir: `uv run --with pyarrow eval/turns.py`.

| política | falou por cima | resposta média |
|---|---|---|
| silêncio fixo de 0,5 s (o que a maioria faz) | 32% | 1,50 s |
| smart-turn sozinho (p > 0,5, senão 2,5 s) | 24% | 1,82 s |
| média pontuação + modelo ([RealtimeVoiceChat](https://github.com/KoljaB/RealtimeVoiceChat)) | 26% | 1,57 s |
| degraus por confiança (estilo LiveKit) | 26% | 1,80 s |
| silêncio fixo de 2,0 s | 13% | 2,00 s |
| **X1** (`wait_for`) | **14%** | **2,22 s** |

**Teste 2: fala clara**, com 281 amostras em português do teste do smart-turn.

| política | resposta em fala completa | responderia fala incompleta em ≤ 1,5 s |
|---|---|---|
| silêncio fixo de 2,0 s | 2,00 s | 0% |
| **X1** | **1,56 s** | 3% |

O que os dados mostraram:

- **O que mais reduz cortes é simplesmente esperar mais (~2 s).** Em fala espontânea, o smart-turn diz "terminou" em 69% das pausas no meio da fala e em 68% dos fins de turno reais, ou seja, não separa os dois casos. Bate com [este estudo](https://huggingface.co/marcosremar2/turn-taking-study). As palavras abertas também não separam: 32% contra 32%.
- **A execução especulativa é o que deixa essa espera barata.** A resposta vai sendo gerada enquanto ele espera, então ~1,5 s dos 2 s já estariam gastos de qualquer jeito.
- **O smart-turn e as regras de texto pagam em comandos claros.** O X1 responde ~0,45 s mais rápido que a espera fixa de 2 s, sem cortar mais na fala espontânea.
- **Os dados de calibração têm limites:**
  - **voz de TTS engana:** nas amostras sintéticas o smart-turn sozinho erra só 5%;
  - **amostra pequena:** um único diálogo de entrevista, com mais monólogo que o uso de um assistente e só 25 fins de turno.

## Sua fala tem prioridade

Enquanto o X1 fala, o microfone continua ligado. O cancelamento de eco do Chrome tira a voz dele do microfone; isso cobre o áudio que a própria página toca. Medido nos alto-falantes deste notebook, o eco que sobra fica com probabilidade de fala de no máximo 0,05 no Silero (p99 de 0,017).

- **Quando ele para:** se você fala uns 0,4 s dentro de 0,8 s (probabilidade acima de 0,6). Um "não, não, não", com as pausinhas entre as palavras, conta. Um "uhum" rápido, não.
- **O que acontece com sua fala:** o que você disse por cima vira o começo do seu turno.
- **O que ele lembra:** o histórico guarda só o que você chegou a ouvir dele, com "…".

## Resposta em streaming

O LLM gera token a token. A voz sai em pedaços:
- o **primeiro pedaço** sai na primeira vírgula (com pelo menos 12 caracteres), para ele começar a falar mais cedo;
- **o resto sai frase a frase**, porque o Piper entoa melhor frases inteiras.

Uma vírgula sem espaço depois (como em "3,5") não corta.

## Por que esses modelos (medido nesta máquina)

| peça | escolha | medido | descartado |
|---|---|---|---|
| fala → texto | faster-whisper `small` int8 (GPU) | ~0,5 s por frase, 390 MB | `large-v3-turbo`: 1,8 s por frase e estourou a RAM ao carregar |
| cérebro | `qwen3:1.7b` sem thinking (Ollama) | 85 tokens/s, 1ª frase em ~0,45 s | `qwen3.5:2b`: 17 tokens/s nesta GPU, respostas piores |
| voz | Piper `pt_BR-faber-medium` (CPU) | ~0,2 s por frase | Kokoro: vozes pt-BR fracas e mais RAM |
| fim de turno | Silero VAD + smart-turn v3.2 (CPU) | ~0,1 s | LiveKit turn-detector: 400 MB e licença restrita |

## Limites deste kickoff

- **Outros áudios do computador:** o cancelamento de eco só remove a voz do próprio X1. Um vídeo tocando ao lado é ouvido como se fosse você (aconteceu na calibração). Isso só se resolve com uma wake word.
- **Interromper falando foi calibrado só contra o eco.** Uma voz de "usuário" tocada pelo mesmo alto-falante foi suprimida junto com o eco, então só um teste com uma pessoa de verdade confirma os limiares. Se ele não parar quando você fala, os números são `BARGE_P`/`BARGE` em `x1.py`.
- **Ainda não tem:**
  - wake word;
  - ferramentas (timer, busca, apps);
  - memória entre sessões.
- **A calibração é de um único diálogo de entrevista.** Quando você usar de verdade, as constantes de `wait_for` são o lugar de ajustar.
