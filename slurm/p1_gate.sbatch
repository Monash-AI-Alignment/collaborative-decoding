#!/bin/bash
# P1 gate (GPU, no Claude usage): build the canonical alpaca_eval reference artifact,
# then verify eval_idea reproduces the policy-search winner (margin_threshold tau=0.05,
# span 64 -> f_weak ~0.44 at recovery >= 0.98 per docs/alpaca_eval_results.md).
#
#   sbatch --export=ALL,ORCHESTRATOR_API_URL=http://<server-node>:8000 slurm/p1_gate.sbatch
#
#SBATCH --job-name=w2s-p1-gate
#SBATCH --partition=fit
#SBATCH --qos=fitq
#SBATCH --account=ml23
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

REPO="${W2S_REPO:-/fs04/ax74/smur0075/automated-w2s-research}"
: "${HF_TOKEN:?set HF_TOKEN — Llama-3.2-1B-Instruct is gated}"

export HF_HOME=/scratch2/ml23/smur0075/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export STRONG_GEN_TIMEOUT="${STRONG_GEN_TIMEOUT:-300}"
export W2S_BASELINES_DIR="${W2S_BASELINES_DIR:-/scratch2/ml23/smur0075/w2s_decode_runs/baselines}"

# The judge endpoint moves nodes when its always-on server job rolls — resolve it from
# the registry at job start (blocks through a handoff) instead of baking in a node:port.
if [ -z "${JUDGE_URL:-}" ]; then
  JUDGE_URL="$("$HOME/bin/vllm-endpoint" gemma4 --wait)"
fi
export JUDGE_URL

module load cuda/12.2.0 2>/dev/null || true
cd "$REPO"
echo "[p1] judge=$JUDGE_URL server=${ORCHESTRATOR_API_URL:-<none>} baselines_dir=$W2S_BASELINES_DIR"

if [ -f "$W2S_BASELINES_DIR/alpaca_eval.json" ] && [ -z "${P1_FORCE_BOOTSTRAP:-}" ]; then
  echo "[p1] step 1/2: SKIP — canonical artifact exists (set P1_FORCE_BOOTSTRAP=1 to rebuild)"
else
  echo "[p1] step 1/2: bootstrap canonical baselines (n=${P1_BOOTSTRAP_N:-100})"
  PYTHONPATH="$REPO" "$REPO/.venv/bin/python" scripts/bootstrap_baselines.py \
    --benchmark alpaca_eval --eval-size "${P1_BOOTSTRAP_N:-100}"
fi

echo "[p1] step 2/2: eval_idea margin_threshold (tau=0.05, span 64, n=${P1_EVAL_N:-60})"
MARGIN_TAU=0.05 SPAN_MAX_TOKENS=64 \
PYTHONPATH="$REPO" "$REPO/.venv/bin/python" -m w2s_research.core.eval_idea_cli \
  --idea margin_threshold --benchmark alpaca_eval --eval-size "${P1_EVAL_N:-60}" \
  --out "$W2S_BASELINES_DIR/p1_margin_threshold.json"

echo "[p1] GATE COMPLETE"
