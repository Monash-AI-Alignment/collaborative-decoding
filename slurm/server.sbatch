#!/bin/bash
# Persistent CPU job: the collaborative-decoding findings/leaderboard server.
# Must be up before agents. Prints its host:port so you can set ORCHESTRATOR_API_URL.
#
#   sbatch slurm/server.sbatch
#   # then, from the printed line: export ORCHESTRATOR_API_URL=http://<node>:8000
#
#SBATCH --job-name=collab-server
#SBATCH --partition=comp
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

REPO="${W2S_REPO:-/fs04/ax74/smur0075/automated-w2s-research}"
export W2S_REPO="$REPO"   # the server's watermark trigger sbatch's $W2S_REPO/slurm/probe_watermark.sbatch
export W2S_SERVER_DB="${W2S_SERVER_DB:-/scratch2/ml23/smur0075/w2s_decode_runs/server.db}"
export W2S_SITE_DATA="${W2S_SITE_DATA:-$REPO/docs/data.json}"   # at-source public site export
export PORT="${PORT:-8000}"

cd "$REPO"
echo "[server] node=$(hostname) port=$PORT db=$W2S_SERVER_DB"
echo "[server] -> set on agents:  export ORCHESTRATOR_API_URL=http://$(hostname):$PORT"
PYTHONPATH="$REPO" "$REPO/.venv/bin/python" -m w2s_research.server.app
