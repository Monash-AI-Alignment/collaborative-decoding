# Running Phase 1 on Monash M3 (GPU)

Phase 1 = the collaborative-decoding engine + baselines + CLI. These scripts run the
**deferred GPU Gate 1a/1b**: measure the weak-only / strong-only utility band and the
entropy-deferral frontier on GSM8K, with weak = `Llama-3.2-1B-Instruct` (white-box, HF) and
strong = `Qwen/Qwen2.5-7B-Instruct` (black-box, vLLM).

## One-time prerequisites (you must do these)

1. **Accept the Llama license + get an HF token.** Llama-3.2-1B-Instruct is gated:
   - Visit https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct and accept the license
     with the same HF account your token belongs to.
   - Create a token at https://huggingface.co/settings/tokens (read scope is enough).
2. (Optional) Pre-build the GPU venv once in an interactive GPU session, so the first batch
   job doesn't spend ~15–20 min installing:
   ```bash
   # from a GPU node (e.g. smux/srun with a GPU)
   cd /fs04/ml23/smur0075/automated-w2s-research
   bash slurm/setup_gpu_venv.sh          # creates ~/venvs/w2s-decode-gpu
   ```

## Launch the gate

```bash
cd /fs04/ml23/smur0075/automated-w2s-research
sbatch --export=ALL,HF_TOKEN=hf_your_token_here slurm/phase1_gate.sbatch
```

That single job will: ensure the lean GPU venv exists → smoke-test loading both models →
run `weak_only` and `strong_only` (free-running) → sweep `entropy_threshold` over τ ∈
{0.3,0.6,1.0,1.5,2.0} → print a summary table.

Watch it: `squeue -u $USER`; logs stream to `slurm-w2s-phase1-gate-<jobid>.out`.

## Reading the result

The job prints, and writes JSON to `/scratch2/ml23/smur0075/w2s_decode_runs/phase1_<jobid>/`:

- **GATE 1a** — `U_strong − U_weak` should be clearly positive (~0.4+ on GSM8K). If it's tiny
  or `weak_only` utility is ~0 (incoherent), stop and revisit model sizes (see spec Risk #1).
- **GATE 1b** — as τ grows, `f_weak` (weak-token fraction) should rise and `utility_recovery`
  should fall, with intermediate points lying *between* the two baselines — confirming the
  cross-tokenizer span handoff produces coherent collaborative text.

## Knobs (env overrides)

| Var | Default | Meaning |
|-----|---------|---------|
| `W2S_EVAL_SIZE` | `50` | examples per run (raise to 100+ once validated) |
| `W2S_BENCH` | `gsm8k` | `gsm8k` or `math` |
| `W2S_GPU_VENV` | `~/venvs/w2s-decode-gpu` | GPU venv location |
| `W2S_OUT` | `/scratch2/.../phase1_<jobid>` | results dir |

Example: `sbatch --export=ALL,HF_TOKEN=...,W2S_EVAL_SIZE=100,W2S_BENCH=math slurm/phase1_gate.sbatch`

## Allocation notes

The `#SBATCH` headers default to `--partition=fit --qos=fitq --account=ml23 --gres=gpu:A100:1`
(from the project CLAUDE.md). Edit them if your group's GPU allocation differs (e.g. `m3h` +
`--qos=m3h` for H100, or `gpu`). One A100 (40 or 80 GB) fits both models — vLLM is capped at
`gpu_memory_utilization=0.6` to leave room for the HF weak model.

## Notes / known limits (Phase 1)

- The HF weak decode loop re-decodes context each step (O(N²)); fine at this scale, optimize
  later if slow. `run_decode` loads both models even when a baseline only needs one — harmless
  on an 80 GB card, mild waste on 40 GB.
- `ANTHROPIC_API_KEY` is **not** needed here — that's for the autonomous researcher (Phase 3).
- AlpacaEval is not wired into this gate yet (needs an LLM judge); GSM8K/MATH are CPU-scored
  exact-match.
