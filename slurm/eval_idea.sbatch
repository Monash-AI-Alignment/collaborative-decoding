#!/bin/bash
# Evaluate ONE idea against the canonical reference on a GPU (no Claude usage).
# Useful for anchor checks (strong_only should score recovery~1.0, weak_only ~0.0)
# and for re-confirming agent-proposed policies out-of-band.
#
#   sbatch --export=ALL,IDEA=strong_only slurm/eval_idea.sbatch
#
#SBATCH --job-name=w2s-eval-idea
#SBATCH --partition=fit
#SBATCH --qos=fitq
#SBATCH --account=ml23
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

REPO="${W2S_REPO:-/fs04/ax74/smur0075/automated-w2s-research}"
: "${HF_TOKEN:?set HF_TOKEN — Llama-3.2-1B-Instruct is gated}"
IDEA="${IDEA:?set IDEA=<idea dir name under w2s_research/ideas/>}"

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
echo "[eval] idea=$IDEA n=${EVAL_N:-60} judge=$JUDGE_URL"
PYTHONPATH="$REPO" "$REPO/.venv/bin/python" -m w2s_research.core.eval_idea_cli \
  --idea "$IDEA" --benchmark alpaca_eval --eval-size "${EVAL_N:-60}" \
  --out "$W2S_BASELINES_DIR/eval_${IDEA}_${SLURM_JOB_ID}.json"
echo "[eval] DONE"
