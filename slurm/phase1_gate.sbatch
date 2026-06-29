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

echo "===== Gate 1a: baselines (strong_only free-running for clean U_strong) ====="
$PY -m w2s_research.decode_cli --idea weak_only   --benchmark "$BENCH" --eval-size "$EVAL_SIZE" --out "$OUT/weak_only.json"
$PY -m w2s_research.decode_cli --idea strong_only --benchmark "$BENCH" --eval-size "$EVAL_SIZE" --span-stop none --span-max-tokens 1024 --out "$OUT/strong_only.json"

echo "===== Gate 1b: entropy-threshold frontier (span-level handoff) ====="
for T in 0.3 0.6 1.0 1.5 2.0; do
  $PY -m w2s_research.decode_cli --idea entropy_threshold --benchmark "$BENCH" --eval-size "$EVAL_SIZE" --tau "$T" --out "$OUT/entropy_tau_${T}.json"
done

echo "===== SUMMARY (Gate 1a/1b) ====="
$PY - "$OUT" <<'PY'
import json, sys, glob, os
out = sys.argv[1]
load = lambda p: json.load(open(p))
def util(name):
    p = os.path.join(out, name)
    return load(p)["utility"] if os.path.exists(p) else None
uw, us = util("weak_only.json"), util("strong_only.json")
print(f"U_weak   = {uw}")
print(f"U_strong = {us}")
gap = (us - uw) if (uw is not None and us is not None) else None
if gap is not None:
    print(f"gap (U_strong - U_weak) = {gap:.4f}   <-- GATE 1a: want this clearly positive (~0.4+ on GSM8K)")
print("\ntau\tutility\tf_weak\tutility_recovery")
for p in sorted(glob.glob(os.path.join(out, "entropy_tau_*.json")), key=lambda x: float(x.split('entropy_tau_')[1][:-5])):
    d = load(p); tau = p.split("entropy_tau_")[1][:-5]
    rec = ((d["utility"] - uw) / gap) if (gap and gap > 0) else float("nan")
    print(f"{tau}\t{d['utility']:.4f}\t{d['weak_token_fraction']:.4f}\t{rec:.4f}")
print("\nGATE 1b: f_weak should rise and utility_recovery should fall as tau grows,")
print("with intermediate points BETWEEN the two baselines (coherent collaborative text).")
PY
echo ""
echo "Results JSON in: $OUT"
