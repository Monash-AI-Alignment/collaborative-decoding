#!/bin/bash
# Watermark-removal probe in TWO sequential processes on one GPU allocation:
#   (1) generate + save completions (vLLM engine + weak model); process exits -> GPU freed
#   (2) detect on the saved completions (detector alone, full GPU) -> p-values
# Generic over the watermark method via WATERMARK={inference_kgw|eth_french}.
#
#   sbatch --export=ALL,HF_TOKEN=hf_xxx,WATERMARK=inference_kgw slurm/probe_watermark.sbatch
#   WATERMARK=eth_french FP_MODEL=<ckpt> FP_EMB=<yaml> sbatch --export=ALL,HF_TOKEN=hf_xxx slurm/probe_watermark.sbatch
#
#SBATCH --job-name=wm-probe
#SBATCH --partition=fit
#SBATCH --qos=fitq
#SBATCH --account=ml23
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

REPO="${W2S_REPO:-/fs04/ax74/smur0075/automated-w2s-research}"
FP_REPO="${FP_REPO:-/scratch2/ml23/smur0075/robust-llm-fingerprints}"
: "${HF_TOKEN:?set HF_TOKEN — Llama-3.2-1B-Instruct is gated}"

export HF_HOME=/scratch2/ml23/smur0075/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export STRONG_GEN_TIMEOUT="${STRONG_GEN_TIMEOUT:-300}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CG_TAU="${CG_TAU:-0.16}"
export SPAN_MAX_TOKENS="${SPAN_MAX_TOKENS:-64}"

WATERMARK="${WATERMARK:-inference_kgw}"
POLICIES="${POLICIES:-strong_only,weak_only,sota}"   # 'sota' = live leaderboard #1 (dynamic, not hardcoded)
PROBE_N="${PROBE_N:-100}"
RUNDIR=/scratch2/ml23/smur0075/w2s_decode_runs
GENS="$RUNDIR/wm_gens_${WATERMARK}_${SLURM_JOB_ID}.json"
OUT="$RUNDIR/wm_probe_${WATERMARK}_${SLURM_JOB_ID}.json"

# method-specific args
case "$WATERMARK" in
  eth_french)
    FP_MODEL="${FP_MODEL:-$FP_REPO/models/Qwen2.5-3B-Instruct_French}"
    FP_EMB="${FP_EMB:-$FP_REPO/configs/embedding/qwen2.5-3B/main/french.yaml}"
    METHOD_ARGS=(--fingerprinted-model "$FP_MODEL" --embedding-config "$FP_EMB") ;;
  inference_kgw)
    STRONG_MODEL="${STRONG_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
    METHOD_ARGS=(--strong-model "$STRONG_MODEL" --kgw-gamma "${KGW_GAMMA:-0.25}"
                 --kgw-delta "${KGW_DELTA:-2.0}" --kgw-seeding "${KGW_SEEDING:-simple_1}") ;;
  *) echo "unknown WATERMARK=$WATERMARK" >&2; exit 1 ;;
esac

if [ -z "${JUDGE_URL:-}" ]; then JUDGE_URL="$("$HOME/bin/vllm-endpoint" gemma4 --wait)"; fi
export JUDGE_URL

module load cuda/12.2.0 2>/dev/null || true
cd "$REPO"
echo "[wm-probe] watermark=$WATERMARK n=$PROBE_N policies=$POLICIES tau=$CG_TAU judge=$JUDGE_URL"
echo "[wm-probe] method-args: ${METHOD_ARGS[*]}"

echo "[wm-probe] === phase 1: generate ==="
PYTHONPATH="$REPO" "$REPO/.venv/bin/python" scripts/probe_watermark.py \
  --watermark "$WATERMARK" --n "$PROBE_N" --tau "$CG_TAU" --span-max "$SPAN_MAX_TOKENS" \
  --policies "$POLICIES" --out "$GENS" "${METHOD_ARGS[@]}"

echo "[wm-probe] === phase 2: detect (fresh process, GPU freed) ==="
PYTHONPATH="$REPO" "$REPO/.venv/bin/python" scripts/detect_watermark.py \
  --watermark "$WATERMARK" --gens "$GENS" --out "$OUT" "${METHOD_ARGS[@]}"

echo "[wm-probe] === regenerate living site data (docs/watermarks.json) from latest probes ==="
PYTHONPATH="$REPO" "$REPO/.venv/bin/python" scripts/build_watermark_site_data.py \
  --auto "$RUNDIR" --out "$REPO/docs/watermarks.json" || echo "[wm-probe] site-data regen failed (non-fatal)"
echo "[wm-probe] DONE -> $OUT ; site data refreshed. (run 'wrangler deploy' from docs/ to publish)"
