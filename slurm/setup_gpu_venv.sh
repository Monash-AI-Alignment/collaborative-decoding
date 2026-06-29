#!/usr/bin/env bash
# Create the lean GPU venv for Phase 1 collaborative-decoding experiments.
#
# Installs ONLY what Phase 1 needs (no training stack): vLLM (which pins a
# compatible torch), transformers (white-box weak model), datasets (benchmark
# loading), plus the sympy/pylatexenc the math grader imports. The full
# pyproject (unsloth/sglang/verl/flash-attn) is intentionally NOT installed.
#
# Idempotent: exits early if the venv already imports the key packages.
# Best run inside a GPU allocation (per M3 guidance: install GPU packages from
# a GPU node). Installing only downloads prebuilt CUDA wheels, but a GPU node
# lets the final import check confirm CUDA is actually visible.
set -euo pipefail

VENV="${W2S_GPU_VENV:-$HOME/venvs/w2s-decode-gpu}"
PY_VER="3.12"

if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import torch, vllm, transformers, datasets, sympy, pylatexenc" 2>/dev/null; then
    echo "[setup] GPU venv already complete at $VENV — skipping install."
    exit 0
fi

command -v uv >/dev/null 2>&1 || { echo "[setup] ERROR: uv not on PATH"; exit 1; }

echo "[setup] Creating lean GPU venv at $VENV (Python $PY_VER)"
uv venv --python "$PY_VER" "$VENV"

echo "[setup] Installing vLLM (pulls a compatible torch) + HF libs + grader deps ..."
uv pip install --python "$VENV/bin/python" \
    "vllm==0.11.0" \
    "transformers>=4.51.0" \
    "datasets==3.6.0" \
    "accelerate>=1.0.0" \
    "hf_transfer" \
    "sympy" \
    "pylatexenc"

echo "[setup] Verifying imports + CUDA visibility ..."
"$VENV/bin/python" - <<'PY'
import torch, transformers, datasets, vllm
print("torch       ", torch.__version__, "| cuda", torch.version.cuda, "| available:", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("datasets    ", datasets.__version__)
print("vllm        ", vllm.__version__)
if not torch.cuda.is_available():
    print("[setup] WARNING: torch.cuda.is_available() is False — run this on a GPU node.")
PY

echo "[setup] GPU venv ready: $VENV"
