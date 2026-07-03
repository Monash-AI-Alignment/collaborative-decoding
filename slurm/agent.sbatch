#!/bin/bash
# One autonomous collaborative-decoding research agent (GPU). Scale out by sbatch'ing
# N copies — they queue and start staggered, sharing findings through the server. The
# agent NEVER sbatch's sub-jobs; it runs the engine in-process on this allocation.
#
#   sbatch --export=ALL,ORCHESTRATOR_API_URL=http://<server-node>:8000,HF_TOKEN=hf_xxx slurm/agent.sbatch
#
#SBATCH --job-name=collab-agent
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

REPO="${W2S_REPO:-/fs04/ml23/smur0075/automated-w2s-research}"
: "${ORCHESTRATOR_API_URL:?set ORCHESTRATOR_API_URL=http://<server-node>:8000 (see collab-server job output)}"
: "${HF_TOKEN:?set HF_TOKEN — Llama-3.2-1B-Instruct is gated}"

export HF_HOME=/scratch2/ml23/smur0075/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
# The judge endpoint moves nodes when its always-on server job rolls — resolve it from
# the registry at job start (blocks through a handoff) instead of baking in a node:port.
# Plain assignment (not `export VAR=$(...)`) so set -e aborts if resolution fails.
if [ -z "${JUDGE_URL:-}" ]; then
  JUDGE_URL="$("$HOME/bin/vllm-endpoint" gemma4 --wait)"
fi
export JUDGE_URL
export BENCHMARK="${BENCHMARK:-alpaca_eval}"
export R_BAR="${R_BAR:-0.98}"
export WEAK_MODEL="${WEAK_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
export STRONG_MODEL="${STRONG_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export W2S_BASELINES_DIR="${W2S_BASELINES_DIR:-/scratch2/ml23/smur0075/w2s_decode_runs/baselines}"
export FULL_AUTO_MAX_RUNTIME_SECONDS="${FULL_AUTO_MAX_RUNTIME_SECONDS:-25200}"   # 7h walltime guard
export MAX_CONSECUTIVE_ERRORS="${MAX_CONSECUTIVE_ERRORS:-4}"                      # stop if rate-limited
export IDEA_UID="${IDEA_UID:-collab-${SLURM_JOB_ID:-local}}"
export IDEA_NAME="${IDEA_NAME:-collaborative_decoding}"

module load cuda/12.2.0 2>/dev/null || true
cd "$REPO"
echo "[agent] server=$ORCHESTRATOR_API_URL judge=$JUDGE_URL benchmark=$BENCHMARK r_bar=$R_BAR"
# The agent loop (claude_agent_sdk) AND the engine (torch/vLLM) run in this GPU venv. vLLM/torch/
# httpx are present; the agent SDK is installed on demand if missing.
~/venvs/w2s-decode-gpu/bin/python -c 'import claude_agent_sdk' 2>/dev/null || \
  ~/venvs/w2s-decode-gpu/bin/python -m pip install --quiet claude-agent-sdk
~/venvs/w2s-decode-gpu/bin/python -c 'import claude_agent_sdk, vllm; print("[agent] env ok")'
PYTHONPATH="$REPO" ~/venvs/w2s-decode-gpu/bin/python -m w2s_research.research_loop.agent
