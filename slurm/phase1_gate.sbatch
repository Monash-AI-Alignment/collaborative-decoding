#!/bin/bash
# Phase 1 Gate 1a/1b for collaborative decoding: measure U_weak / U_strong and
# the entropy-deferral frontier on GSM8K. Run on one GPU.
#
#   sbatch --export=ALL,HF_TOKEN=hf_xxx slurm/phase1_gate.sbatch
#
# Adjust the SBATCH lines below to match your allocation. The defaults follow
# the project's CLAUDE.md GPU example (fit / fitq / ml23 / A100).
#
#SBATCH --job-name=w2s-phase1-gate
#SBATCH --partition=fit
#SBATCH --qos=fitq
#SBATCH --account=ml23
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

# ---- knobs (override via env: W2S_REPO, W2S_GPU_VENV, W2S_EVAL_SIZE, W2S_BENCH, W2S_OUT) ----
REPO="${W2S_REPO:-/fs04/ml23/smur0075/automated-w2s-research}"
export W2S_GPU_VENV="${W2S_GPU_VENV:-$HOME/venvs/w2s-decode-gpu}"
EVAL_SIZE="${W2S_EVAL_SIZE:-50}"          # start small to validate the pipeline + gap; raise to 100+ later
BENCH="${W2S_BENCH:-gsm8k}"               # gsm8k (safest for the 1B weak) | math
OUT="${W2S_OUT:-/scratch2/ml23/smur0075/w2s_decode_runs/phase1_${SLURM_JOB_ID:-local}}"

# ---- environment ----
export HF_HOME=/scratch2/ml23/smur0075/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
: "${HF_TOKEN:?Set HF_TOKEN — Llama-3.2-1B-Instruct is GATED; accept its license on huggingface.co first}"
# torch/vLLM wheels bundle their CUDA runtime, so the cuda module is optional (driver is enough).
module load cuda/12.2.0 2>/dev/null || echo "[gate] note: cuda module not loaded; relying on driver + wheel-bundled runtime"

mkdir -p "$OUT"
cd "$REPO"

echo "===== ensure lean GPU venv ====="
bash slurm/setup_gpu_venv.sh
PY="$W2S_GPU_VENV/bin/python"

echo "===== GPU + model load smoke (fail fast on auth/OOM) ====="
nvidia-smi -L || true
$PY scripts/smoke_weak_model.py
$PY scripts/smoke_strong_model.py

echo "===== Gate 1a/1b (single model load via run_gate.py) ====="
# Loads weak+strong ONCE and runs weak_only / strong_only(free-running) / entropy sweep,
# printing the U_weak/U_strong/frontier summary. Much faster than per-policy decode_cli.
$PY scripts/run_gate.py --benchmark "$BENCH" --eval-size "$EVAL_SIZE" --out "$OUT"
echo ""
echo "Results JSON in: $OUT"
