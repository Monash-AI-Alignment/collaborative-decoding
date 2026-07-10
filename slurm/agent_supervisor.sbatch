#!/bin/bash
# Self-sustaining watchdog for the collaborative-decoding research loop — the "cron"
# that survives your terminal session. Modeled on the always-on vLLM supervisor.
#
# Each tick it: (1) keeps the orchestrator server up; (2) keeps N_AGENTS GPU agents
# alive (resubmitting as their 8h jobs end), launched against the CURRENT server url;
# (3) publishes fresh docs/data.json + watermarks.json to monash/main (the public
# site feed). Self-chains before its own wall so it runs indefinitely.
#
#   sbatch --export=ALL,HF_TOKEN=$HF_TOKEN,N_AGENTS=1 slurm/agent_supervisor.sbatch
#   scancel -n collab-supervisor   # stop it (running agents finish their walls)
#
#SBATCH --job-name=collab-supervisor
#SBATCH --partition=comp
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out
#SBATCH --signal=B:USR1@600

set -uo pipefail

REPO="${W2S_REPO:-/fs04/ax74/smur0075/automated-w2s-research}"
AGENT_VENV="${AGENT_VENV:-/fs04/scratch2/ax74/smur0075/venvs/w2s-tl}"
N_AGENTS="${N_AGENTS:-1}"
INTERVAL="${SUPERVISOR_INTERVAL:-600}"          # 10-min tick
: "${HF_TOKEN:?export HF_TOKEN — agents need it (gated Llama-3.2-1B)}"

stamp() { date -Iseconds; }

chain_and_exit() {
    echo "$(stamp) supervisor: wall approaching — chaining successor and exiting"
    sbatch --export=ALL,HF_TOKEN="$HF_TOKEN",N_AGENTS="$N_AGENTS",AGENT_VENV="$AGENT_VENV" \
        "$REPO/slurm/agent_supervisor.sbatch" || echo "$(stamp) supervisor: WARNING self-chain failed" >&2
    exit 0
}
trap chain_and_exit USR1

server_url() {   # current running server node -> url, else empty
    local node
    node=$(squeue -u "$USER" -n collab-server -t R -h -o "%N" 2>/dev/null | head -1)
    [ -n "$node" ] && echo "http://${node}:8000" || echo ""
}

ensure_server() {
    local n; n=$(squeue -u "$USER" -n collab-server -t R,PD -h 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
        echo "$(stamp) supervisor: server down -> resubmit"
        timeout 30 sbatch "$REPO/slurm/server.sbatch" || echo "$(stamp) supervisor: server sbatch failed" >&2
    fi
}

ensure_agents() {
    local url="$1"
    [ -z "$url" ] && { echo "$(stamp) supervisor: no running server yet — hold agents"; return; }
    local n; n=$(squeue -u "$USER" -n collab-agent -t R,PD -h 2>/dev/null | wc -l)
    local need=$(( N_AGENTS - n ))
    local i
    for (( i=0; i<need; i++ )); do
        echo "$(stamp) supervisor: launch agent ($url)"
        timeout 30 sbatch --export=ALL,ORCHESTRATOR_API_URL="$url",HF_TOKEN="$HF_TOKEN",AGENT_VENV="$AGENT_VENV" \
            "$REPO/slurm/agent.sbatch" || echo "$(stamp) supervisor: agent sbatch failed" >&2
    done
}

publish() {
    cd "$REPO" || return
    if ! git diff --quiet -- docs/data.json docs/watermarks.json 2>/dev/null; then
        git add docs/data.json docs/watermarks.json
        if git commit -q -m "data: publish latest leaderboard + findings" 2>/dev/null; then
            if timeout 120 git push monash collaborative-decoding:main >/dev/null 2>&1; then
                echo "$(stamp) supervisor: published data.json -> monash/main"
            else
                echo "$(stamp) supervisor: push FAILED (will retry next tick)" >&2
            fi
        fi
    fi
}

WM_STATE="${WM_STATE_FILE:-/fs04/scratch2/ax74/smur0075/venvs/.wm_last_sota}"

current_sota() {   # leaderboard #1 idea name, or empty
    python3 -c "import json;L=(json.load(open('$REPO/docs/data.json')).get('leaderboard') or []);print(L[0].get('idea_name','') if L else '')" 2>/dev/null
}

maybe_refresh_watermark() {   # re-run the KGW watermark probe when the SOTA changes so the site graph tracks it
    local cur running last
    cur="$(current_sota)"; [ -z "$cur" ] && return
    running=$(squeue -u "$USER" -n wm-probe -t R,PD -h 2>/dev/null | wc -l)
    last=$(cat "$WM_STATE" 2>/dev/null || echo "")
    if [ "$cur" != "$last" ] && [ "$running" -eq 0 ]; then
        echo "$(stamp) supervisor: SOTA changed ($last -> $cur) — refreshing watermark graph (inference_kgw)"
        if timeout 30 sbatch --export=ALL,HF_TOKEN="$HF_TOKEN",WATERMARK=inference_kgw "$REPO/slurm/probe_watermark.sbatch"; then
            echo "$cur" > "$WM_STATE"
        else
            echo "$(stamp) supervisor: wm-probe sbatch failed (retry next SOTA change)" >&2
        fi
    fi
}

echo "=== collab-supervisor job ${SLURM_JOB_ID:-?} @ $(stamp) N_AGENTS=$N_AGENTS interval=${INTERVAL}s venv=$AGENT_VENV ==="
while true; do
    ensure_server
    URL="$(server_url)"
    ensure_agents "$URL"
    publish                    # push data.json/watermarks.json when they change
    maybe_refresh_watermark    # re-run the watermark probe when a new SOTA is certified
    # background sleep + wait so the USR1 self-chain trap fires promptly
    sleep "$INTERVAL" & wait $!
done
