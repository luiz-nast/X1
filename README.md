# X1

Um mini Jarvis local, com a voz como canal principal. Tudo roda na sua máquina: nenhum áudio ou texto vai pra nuvem.

Testado numa GTX 1650 (4 GB), no Ubuntu. Com tudo carregado: ~2,9 GB de VRAM (o LLM) e ~3 GB de RAM (a fala vira texto na CPU).

```
navegador (microfone com cancelamento de eco + bolinha)  ⇄ WebSocket ⇄  x1.py
  mic 16 kHz → Silero VAD → pausa de 0,3 s → smart-turn v3.2 (entonação) + Parakeet pt-BR (palavras, CPU)
  → política de espera → Qwen3 4B Instruct (Ollama), pedaço a pedaço → Piper pt_BR (voz) → navegador
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

Rodar de novo atualiza. Precisa de GPU NVIDIA, Chrome (ou Chromium), `git`, `curl` e `zstd`, e ~3 GB de RAM livres para o modelo de fala.

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
   - as **palavras**, com o Parakeet. Ele pontua, e marca com "…" a frase que fica no ar.
3. Com isso, ele decide **quanto silêncio esperar** (`wait_for` em `x1.py`):
   - frase terminando em "e", "que", "de", "tipo", "né", "ahn", "…" ou vírgula, ou um murmúrio que ele não entendeu: **3 s**;
   - senão, se o smart-turn está muito confiante (p > 0,9): **1 s**;
   - senão: **2,5 s**.
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
| silêncio fixo de 0,5 s (o que a maioria faz) | 37% | 1,50 s |
| smart-turn sozinho (p > 0,5, senão 2,5 s) | 27% | 1,82 s |
| média pontuação + modelo ([RealtimeVoiceChat](https://github.com/KoljaB/RealtimeVoiceChat)) | 35% | 1,51 s |
| degraus por confiança (estilo LiveKit) | 30% | 1,80 s |
| silêncio fixo de 2,0 s | 17% | 2,00 s |
| **X1** (`wait_for`) | **14%** | **2,06 s** |

**Teste 2: fala clara**, com 281 amostras em português do teste do smart-turn.

| política | resposta em fala completa | responderia fala incompleta em ≤ 1,5 s |
|---|---|---|
| silêncio fixo de 2,0 s | 2,00 s | 0% |
| **X1** | **1,67 s** | 3% |

O que os dados mostraram:

- **O que mais reduz cortes é simplesmente esperar mais (~2 s).** Em fala espontânea, o smart-turn diz "terminou" em 69% das pausas no meio da fala e em 68% dos fins de turno reais, ou seja, não separa os dois casos. Bate com [este estudo](https://huggingface.co/marcosremar2/turn-taking-study).
- **As palavras separam melhor com um STT melhor.** Com o texto do Parakeet, frase "aberta" aparece em 21% das pausas no meio e só em 8% dos fins de turno. Com o Whisper small era 32% contra 32%. Como o Parakeet não escreve "ahn", a espera foi reajustada: confiança acima de 0,9 para responder rápido e 2,5 s no resto.
- **A execução especulativa é o que deixa essa espera barata.** A resposta vai sendo gerada enquanto ele espera, então ~1,5 s dos 2 s já estariam gastos de qualquer jeito.
- **O smart-turn e as regras de texto pagam em comandos claros.** O X1 responde ~0,33 s mais rápido que a espera fixa de 2 s, e corta menos que ela na fala espontânea (14% contra 17%).
- **Os dados de calibração têm limites:**
  - **voz de TTS engana:** nas amostras sintéticas o smart-turn sozinho erra só 5%;
  - **amostra pequena:** um único diálogo de entrevista, com mais monólogo que o uso de um assistente e só 25 fins de turno.

## Ouvir direito (comparação de modelos de fala → texto)

O Whisper `small` errava "sete" (virava "6") e "valeu". Comparei 7 modelos em três conjuntos (`eval/stt.py`), sem tocar som nenhum:

- **comandos:** 36 frases de assistente (números, "valeu", gíria, hesitação) em 3 vozes neurais naturais (edge-tts), 108 clipes;
- **com ruído:** os mesmos clipes com reverberação de sala e ruído a 15 dB, aplicados digitalmente;
- **fala real:** 150 trechos de conversa espontânea do [NURC-SP](https://huggingface.co/datasets/nilc-nlp/CORAA-NURC-SP-Audio-Corpus), com transcrição humana.

O WER compara palavras, não grafia: "7" conta igual a "sete", e caixa, acento, pontuação e hesitação são ignorados.

| modelo | comandos | com ruído | fala real | por frase | onde |
|---|---|---|---|---|---|
| Whisper small (o anterior) | 5,2% | 12,1% | 42,9% | 0,5 s | GPU |
| Whisper small com fine-tune pt | 7,3% | 14,7% | 37,0% | 0,5 s | GPU |
| Whisper large-v3-turbo | 2,2% | 6,7% | 27,7% | 1,8 s | GPU, mas não cabe junto do LLM |
| Parakeet TDT 0.6B v3 (base) | 5,8% | 12,1% | 39,5% | 0,4 s | CPU |
| Parakeet v3 pt-BR TAGARELA (int8) | 3,0% | 5,0% | 20,2% | 0,3 s | CPU |
| **Parakeet v3 pt-BR ([alefiury](https://huggingface.co/alefiury/parakeet-tdt-v3-ptBR-30s-onnx), CC-BY-4.0)** | **2,6%** | **4,1%** | **15,2%** | **0,3 s** | **CPU** |

- **Escolhido:** o Parakeet pt-BR do alefiury. Em fala real ele erra um terço do que o Whisper small errava, é 6 vezes mais rápido que o turbo e não usa VRAM. Custa ~3 GB de RAM.
- **O prompt de hesitações do Whisper saiu:** não melhorava nada e dobrava a latência.
- **Microfone:** o Chrome agora captura sem supressão de ruído e sem ganho automático, e mantém só o cancelamento de eco. O supressor do Chrome roda no nível mais agressivo e apaga consoantes como o "s" e o "t" de "sete"; o ganho automático mexe sozinho no volume do microfone do sistema. É o que Google, Deepgram e AssemblyAI recomendam para reconhecimento de fala.

## Sua fala tem prioridade

Enquanto o X1 fala, o microfone continua ligado. O cancelamento de eco do Chrome tira a voz dele do microfone; isso cobre o áudio que a própria página toca. Na calibração, o eco que sobrou ficou com probabilidade de fala de no máximo 0,05 no Silero (p99 de 0,017). Mas a saída de som padrão desta máquina hoje é o fone, e a calibração pode ter rodado com o som no fone, o que não prova nada sobre o alto-falante.

- **Quando ele para:** se você fala uns 0,4 s dentro de 0,8 s (probabilidade acima de 0,6). Um "não, não, não", com as pausinhas entre as palavras, conta. Um "uhum" rápido, não.
- **O que acontece com sua fala:** o que você disse por cima vira o começo do seu turno.
- **O que ele lembra:** o histórico guarda só o que você chegou a ouvir dele, com "…".

## Resposta em streaming

O LLM gera token a token. A voz sai em pedaços:
- o **primeiro pedaço** sai na primeira vírgula (com pelo menos 12 caracteres), para ele começar a falar mais cedo;
- **o resto sai frase a frase**, porque o Piper entoa melhor frases inteiras.

Uma vírgula sem espaço depois (como em "3,5") não corta.

## Qual cérebro (comparação de LLMs)

Testei 6 modelos que cabem na GTX 1650 com uma bateria de 23 frases em português (`eval/llm.py`), um por vez na GPU. Os casos cobrem:
- hora, data, contas e fatos;
- armadilhas de alucinação: o "Nobel do Pelé", o placar de ontem, o telefone do dentista;
- pedidos que ele não sabe cumprir (ligar a luz, criar lembrete);
- estilo falado, fala "pensando alto", transcrição com erro e memória da conversa.

Dois juízes independentes avaliaram cada resposta, e um terceiro desempatou as discordâncias (`eval/llm_judged*.json`).

**Rodada 1** (prompt original):

| modelo | acertos | alucinações | fala natural | 1º pedaço | tokens/s | VRAM |
|---|---|---|---|---|---|---|
| qwen3:1.7b (o anterior) | 13/23 | 7 | 13/23 | 0,24 s | 91 | 1,4 GB |
| **qwen3:4b-instruct-2507** | **18/23** | **4** | **20/23** | 0,60 s | 35¹ | 2,6 GB |
| granite4.1:3b | 17/23 | 4 | 18/23 | 0,41 s | 55 | 2,2 GB |
| llama3.2:3b | 13/23 | 7 | 16/23 | 0,30 s | 60 | 2,2 GB |
| phi4-mini:3.8b | 12/23 | 8 | 16/23 | 0,36 s | 39 | 2,6 GB |
| lfm3:1.7b | 13/23 | 9 | 5/23 | 0,65 s | 95 | 1,1 GB |

**Todos erraram nas mesmas coisas:** fingiam ligar a luz ou criar lembrete e inventavam placar e previsão do tempo. A causa era o prompt, que não dizia o que ele não sabe fazer. Uma frase no prompt resolveu quase tudo.

**Rodada 2** (prompt com os limites), com os três melhores:

| modelo | acertos | alucinações | fala natural |
|---|---|---|---|
| **qwen3:4b-instruct-2507** | **19/23** | 3 | 16/23 |
| granite4.1:3b | 19/23 | 2 | 19/23 |
| qwen3:1.7b | 13/23 | 6 | 17/23 |

**Escolhido: `qwen3:4b-instruct-2507`.** Ele empata com o granite, mas os erros dele são leves: chamou Armstrong de "piloto" em vez de comandante e ofereceu avisar um horário. O granite erra conta simples nas duas rodadas (17 × 23 = 294). A diferença de velocidade some na voz, porque o TTS fala mais devagar que os dois geram texto.

- **O que ainda escapa:** "me lembra às oito" (ele finge que vai lembrar), piada sem graça e um ou outro erro de concordância.
- **O que a bateria pegou de quebra:** um modelo entrou em loop e gerou 38 mil tokens. Agora a resposta tem um teto (`MAX_TOKENS`).

¹ Com todas as camadas na GPU, como o X1 roda (`num_gpu`), dá 45 tokens/s. Na bateria, o ajuste automático do Ollama tinha deixado 3 de 37 camadas na CPU.

## Por que esses modelos (medido nesta máquina)

| peça | escolha | medido | descartado |
|---|---|---|---|
| fala → texto | Parakeet TDT v3 pt-BR (alefiury), ONNX na CPU | ~0,3 s por frase; 15% de WER em fala real | Whisper small (43%), turbo (1,8 s e sem espaço na GPU) e mais 4, na tabela acima |
| cérebro | `qwen3:4b-instruct-2507` (Ollama, todas as camadas na GPU) | 45 tokens/s; 19/23 na bateria | os outros 5 da tabela acima; `qwen3.5:2b`: 17 tokens/s nesta GPU |
| voz | Piper `pt_BR-faber-medium` (CPU) | ~0,2 s por frase | Kokoro: vozes pt-BR fracas e mais RAM |
| fim de turno | Silero VAD + smart-turn v3.2 (CPU) | ~0,1 s | LiveKit turn-detector: 400 MB e licença restrita |

## Limites deste kickoff

- **Outros áudios do computador:** o cancelamento de eco só remove a voz do próprio X1. Um vídeo tocando ao lado é ouvido como se fosse você (aconteceu na calibração). Isso só se resolve com uma wake word.
- **Interromper falando foi calibrado só contra o eco, talvez com o som no fone.** Só um teste com uma pessoa de verdade, no alto-falante, confirma os limiares. Se ele não parar quando você fala, os números são `BARGE_P`/`BARGE` em `x1.py`.
- **O STT ainda não foi testado com a sua voz.** Os testes usam vozes neurais e gravações do NURC-SP. Gravar umas 20 frases suas e rodar `eval/stt.py` nelas é o próximo passo para medir o que você de fato ouve.
- **Ainda não tem:**
  - wake word;
  - ferramentas (timer, busca, apps);
  - memória entre sessões.
- **A calibração é de um único diálogo de entrevista.** Quando você usar de verdade, as constantes de `wait_for` são o lugar de ajustar.
