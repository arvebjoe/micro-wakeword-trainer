#!/usr/bin/env bash
# One-time environment setup: creates .venv and installs all dependencies.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

version_ok=$("$PYTHON" -c 'import sys; print(int((3,10) <= sys.version_info[:2] <= (3,12)))')
if [[ "$version_ok" != "1" ]]; then
    echo "ERROR: Python 3.10–3.12 required (TensorFlow 2.18 has no wheels for newer versions)." >&2
    echo "Found: $("$PYTHON" --version)" >&2
    exit 1
fi

if [[ ! -d .venv ]]; then
    "$PYTHON" -m venv .venv
fi
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo
echo "Setup complete. Next steps:"
echo "  source .venv/bin/activate"
echo "  export OPENAI_API_KEY=sk-..."
echo "  # edit config.yaml (wake word, author, ...)"
echo "  python -m trainer generate"
echo "  python -m trainer download"
echo "  python -m trainer features"
echo "  python -m trainer train"
