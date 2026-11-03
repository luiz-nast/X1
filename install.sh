#!/usr/bin/env bash
# X1 installer: puts everything in ~/.local/share/x1 and adds X1 to the app menu. Run it again to update.
#   curl -fsSL https://raw.githubusercontent.com/luiz-nast/X1/master/install.sh | bash
set -euo pipefail

X1="$HOME/.local/share/x1"
APP="$X1/app"
REPO="https://github.com/luiz-nast/X1"
export UV_SYSTEM_CERTS=1  # trust the system CA store too (TLS-inspecting proxies)
[ -f /etc/ssl/certs/ca-certificates.crt ] && export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }
die() { printf '\n\033[1;31mX1: %s\033[0m\n' "$1" >&2; exit 1; }

command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1 || die "precisa de uma GPU NVIDIA com o driver instalado."
for cmd in git curl zstd; do command -v "$cmd" >/dev/null || die "falta '$cmd': sudo apt install $cmd"; done
command -v google-chrome >/dev/null || command -v chromium >/dev/null || command -v chromium-browser >/dev/null \
  || echo "aviso: sem Chrome/Chromium o X1 abre numa aba do navegador, sem cancelamento de eco nem janela própria."

step "Baixando o X1"
if [ -d "$APP/.git" ]; then git -C "$APP" pull --ff-only; else git clone --depth 1 "$REPO" "$APP"; fi

step "Python e dependências (uv)"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --project "$APP" --frozen

step "Ollama"
OLLAMA="$(command -v ollama || echo "$X1/ollama/bin/ollama")"
if [ ! -x "$OLLAMA" ]; then  # unpack aside and swap in only when complete: an interrupted download never looks installed
  rm -rf "$X1/ollama.new" && mkdir -p "$X1/ollama.new"
  curl -fL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst | zstd -d | tar -x -C "$X1/ollama.new"
  rm -rf "$X1/ollama" && mv "$X1/ollama.new" "$X1/ollama"
fi
STARTED=""
if ! curl -fs http://127.0.0.1:11434/api/version >/dev/null; then
  OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 "$OLLAMA" serve >/dev/null 2>&1 &
  STARTED=$!
  trap '[ -n "$STARTED" ] && kill "$STARTED"' EXIT
  until curl -fs http://127.0.0.1:11434/api/version >/dev/null; do sleep 1; done
fi
MODEL="$(sed -n 's/^LLM = "\(.*\)"/\1/p' "$APP/x1.py")"
"$OLLAMA" pull "$MODEL"

step "Modelos de voz: Whisper, smart-turn e Piper (e um teste da GPU)"
(cd "$APP" && .venv/bin/python -c "import x1; x1.load()")

step "Atalho no menu de aplicativos"
install -Dm644 "$APP/web/icon.svg" "$HOME/.local/share/icons/hicolor/scalable/apps/x1.svg"
mkdir -p "$HOME/.local/share/applications"
cat > "$HOME/.local/share/applications/x1.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=X1
Comment=Assistente de voz local
Exec="$APP/.venv/bin/python" "$APP/x1.py"
Icon=x1
Terminal=false
Categories=Utility;AudioVideo;
StartupNotify=true
StartupWMClass=chrome-x1.localhost__-Default
SingleMainWindow=true
EOF
command -v update-desktop-database >/dev/null && update-desktop-database "$HOME/.local/share/applications" || true

step "Pronto: abra o X1 pelo menu de aplicativos."
