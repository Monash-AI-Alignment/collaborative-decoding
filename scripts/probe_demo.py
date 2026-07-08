"""End-to-end demo of WHITE-BOX access to the weak model (Option B).

Proves the full loop the research prompt now enables:
  1. OFFLINE: load the weak model with full activation access (TransformerLens),
     sweep instructions, harvest the layer-L residual-stream activation at every
     decoding step, and fit a linear probe on those activations.
  2. GENERALIZATION: on HELD-OUT instructions, check the offline-trained probe
     predicts the weak model's live uncertainty from activations alone.
  3. DECODE: run the real collaborative-decoding engine with a TL-backed weak
     model whose per-step activations flow onto WeakStepState, and the probe
     policy (ideas/probe_demo) driving DEFER/CONTINUE from them.

CPU-friendly (float32); strong model is a stub, so no vllm endpoint is needed —
the point is to prove the white-box MECHANISM, not to produce a leaderboard row.

Run:  PYTHONPATH=<repo> python scripts/probe_demo.py --train --layer 8
"""
import argparse
import os

import numpy as np

from w2s_research.core import signals
from w2s_research.core.collab_decode import CollaborativeDecoder, aggregate_weak_fraction
from w2s_research.core.decode_config import DecodeConfig
from w2s_research.core.interfaces import StrongOutput
from w2s_research.core.white_box import TLWhiteBoxWeakModel

PROBE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "w2s_research", "ideas", "probe_demo", "probe.npz",
)

TRAIN_INSTRUCTIONS = [
    "Explain why the sky is blue.",
    "What is the capital of France, and why is it famous?",
    "Compute 17 * 23 and show your working.",
    "Give three tips for writing clear code.",
    "Describe how a bicycle stays upright when moving.",
    "What causes the seasons on Earth?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "How does a refrigerator keep food cold?",
    "List the first five prime numbers and explain what prime means.",
    "Why does bread rise when baked?",
    "Explain the difference between weather and climate.",
    "What is photosynthesis and why does it matter?",
]
EVAL_INSTRUCTIONS = [  # held out from training
    "Explain how vaccines help the immune system.",
    "What is 48 divided by 6, and how would you check the answer?",
    "Describe why the ocean appears to have tides.",
]


class StubStrongModel:
    """Black-box stand-in: returns a short neutral span so defers make progress.

    Faithful to the StrongModel Protocol (text in, text out) — the demo does not
    need a real strong model to exercise the white-box weak path + probe policy.
    """
    SPAN = " In short, the key mechanism explains the observed effect."

    def generate(self, instruction, assistant_text, *, stop, max_tokens, temperature):
        return StrongOutput(text=self.SPAN, finished=False)


class RecordingPolicy:
    """Wraps the real policy to record, at every LIVE decode step, the probe score
    and the actual entropy of the activation the policy saw — including the
    post-defer mixed (weak+strong) contexts the offline probe was NOT trained on.
    This measures the probe on the ON-POLICY distribution, not just pure-weak.
    """
    def __init__(self, inner, thr):
        self.inner = inner
        self.thr = thr
        self.hook = inner.required_hooks[0]
        self.required_hooks = inner.required_hooks
        self.rows = []   # (probe_score, entropy)

    def decide(self, state):
        decision = self.inner.decide(state)
        acts = state.activations
        if acts and self.hook in acts and "logits" in acts:
            self.rows.append((self.inner.probe_score(acts[self.hook]),
                              signals.entropy(acts["logits"])))
        return decision


def harvest_dataset(weak, instructions, hook, max_new_tokens):
    X, ent = [], []
    for instr in instructions:
        for r in weak.harvest(instr, max_new_tokens=max_new_tokens, hooks=[hook, "logits"]):
            X.append(np.asarray(r["activations"][hook], dtype=np.float32))
            ent.append(signals.entropy(r["activations"]["logits"]))
    return np.stack(X), np.asarray(ent, dtype=np.float32)


def train_probe(weak, layer, out_path, max_new_tokens=40):
    import torch
    hook = f"blocks.{layer}.hook_resid_post"
    print(f"[train] harvesting {len(TRAIN_INSTRUCTIONS)} instructions @ {hook} ...", flush=True)
    X, ent = harvest_dataset(weak, TRAIN_INSTRUCTIONS, hook, max_new_tokens)
    thr = float(np.median(ent))
    y = (ent > thr).astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    print(f"[train] N={len(y)} d={X.shape[1]} pos_rate={y.mean():.3f} ent_thr={thr:.3f}", flush=True)

    Xt, yt = torch.tensor(Xs), torch.tensor(y)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    lossf = torch.nn.BCEWithLogitsLoss()
    for it in range(400):
        opt.zero_grad()
        loss = lossf(Xt @ w + b.squeeze(), yt) + 1e-2 * (w * w).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = ((torch.sigmoid(Xt @ w + b.squeeze()) >= 0.5).float() == yt).float().mean().item()
    np.savez(out_path, hook=hook, w=w.detach().numpy().astype(np.float32),
             b=float(b.item()), mu=mu.astype(np.float32), sd=sd.astype(np.float32),
             threshold=thr)
    print(f"[train] probe saved -> {out_path}  train_acc={acc:.3f}", flush=True)
    return hook, thr


def _report(tag, scores, ent, thr):
    scores, ent = np.asarray(scores), np.asarray(ent)
    y = (ent > thr).astype(np.float32)
    pred = (scores >= 0.5).astype(np.float32)
    acc = float((pred == y).mean()) if len(y) else float("nan")
    majority = float(max(y.mean(), 1 - y.mean())) if len(y) else float("nan")
    corr = float(np.corrcoef(scores, ent)[0, 1]) if len(ent) > 1 else float("nan")
    print(f"[val] {tag}: N={len(y)}  probe-vs-entropy acc={acc:.3f} "
          f"(majority-class baseline={majority:.3f})  corr(score,entropy)={corr:.3f}", flush=True)
    return acc, corr


def validate_probe(weak, policy, hook, thr, max_new_tokens=40):
    # NOTE: this is the PURE-WEAK (pre-defer) distribution — activations from a
    # context of only weak-model text. It is NOT the on-policy distribution: once
    # the policy defers, resync() mixes in strong-model text and later activations
    # are drawn from a distribution the probe never saw. The on-policy number is
    # reported separately by run_decode(). Also: the N steps below are
    # autoregressively correlated within only a few instructions, so effective-N
    # is small — treat these as indicative, not a rigorous evaluation.
    print(f"\n[val] PURE-WEAK held-out (pre-defer) on {len(EVAL_INSTRUCTIONS)} instructions ...", flush=True)
    X, ent = harvest_dataset(weak, EVAL_INSTRUCTIONS, hook, max_new_tokens)
    scores = np.array([policy.probe_score(x) for x in X])
    return _report("pure-weak", scores, ent, thr)


def run_decode(weak, policy, hook, thr):
    print(f"\n[decode] running collab_decode with TL weak model + probe policy ...", flush=True)
    cfg = DecodeConfig(benchmark="alpaca_eval", eval_size=len(EVAL_INSTRUCTIONS))
    cfg.max_steps = 50
    cfg.max_chars = 500
    cfg.span_stop = ["\n"]
    # capture what the probe reads + "logits" so we can log live entropy on-policy
    weak.capture_hooks = list(dict.fromkeys(list(policy.required_hooks) + ["logits"]))
    rec = RecordingPolicy(policy, thr)
    dec = CollaborativeDecoder(weak, StubStrongModel(), rec, cfg)
    results = dec.run_dataset(EVAL_INSTRUCTIONS)
    for i, r in enumerate(results):
        print(f"  ex{i}: weak_steps={r.num_weak_steps} defers={r.num_defers} "
              f"weak_chars={r.weak_chars} strong_chars={r.strong_chars} "
              f"f_weak={ (r.weak_chars/(r.weak_chars+r.strong_chars)) if (r.weak_chars+r.strong_chars) else 0:.3f}",
              flush=True)
    fw = aggregate_weak_fraction(results)
    print(f"[decode] overall f_weak={fw:.3f}  (activations flowed to policy: "
          f"{'YES' if any(r.num_weak_steps for r in results) else 'n/a'})", flush=True)
    # ON-POLICY probe evaluation: the live steps actually seen, incl. post-defer
    # mixed contexts. This is the distribution the probe is deployed on.
    if rec.rows:
        s, e = zip(*rec.rows)
        _report("on-policy (live, incl. post-defer)", s, e, thr)
    return fw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    args = ap.parse_args()

    print("=== [0] loading TL white-box weak model (CPU/float32) ===", flush=True)
    weak = TLWhiteBoxWeakModel(
        os.getenv("WEAK_MODEL", "meta-llama/Llama-3.2-1B-Instruct"),
        device="cpu", dtype="float32",
    )
    print(f"loaded: n_layers={weak.n_layers} d_model={weak.d_model}", flush=True)

    if args.train or not os.path.exists(PROBE_PATH):
        hook, thr = train_probe(weak, args.layer, PROBE_PATH, args.max_new_tokens)
    else:
        z = np.load(PROBE_PATH)
        hook, thr = str(z["hook"]), float(z["threshold"])
        print(f"[train] reusing existing probe {PROBE_PATH} ({hook})", flush=True)

    import w2s_research.ideas.probe_demo.run as probe_idea
    policy = probe_idea.ProbePolicy(probe_path=PROBE_PATH)

    validate_probe(weak, policy, hook, thr, args.max_new_tokens)
    run_decode(weak, policy, hook, thr)
    print("\n=== DEMO COMPLETE: offline probe -> live activations -> deferral, end to end ===", flush=True)


if __name__ == "__main__":
    main()
