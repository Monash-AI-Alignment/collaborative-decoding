"""probe_demo — deferral driven by a LINEAR PROBE on the weak model's
residual stream, not by a logit scalar.

This is the white-box-access demo. A probe is trained OFFLINE (see
`scripts/probe_demo.py --train`): we sweep instructions, harvest the layer-L
residual-stream activation at every decoding step, and fit a logistic-regression
probe that reads that activation. At decode time the policy applies the probe to
the LIVE activation exposed on `state.activations` and defers when the probe
predicts the weak model is uncertain.

The probe artefact (`probe.npz`, produced by the trainer) stores: the hook name
it reads, the weight vector `w` + bias `b`, and the feature mean/std used to
standardize activations. It is loaded lazily so importing this module needs only
numpy (no torch / transformer_lens), keeping it importable anywhere.
"""
import os

import numpy as np

from w2s_research.core.policy import Decision, DeferralPolicy

IDEA_NAME = "probe_demo"

_PROBE_PATH = os.path.join(os.path.dirname(__file__), "probe.npz")


class ProbePolicy(DeferralPolicy):
    name = "probe_demo"

    def __init__(self, probe_path=_PROBE_PATH, threshold=0.5):
        z = np.load(probe_path)
        self.hook = str(z["hook"])
        self.w = z["w"].astype(np.float32)          # (d_model,)
        self.b = float(z["b"])
        self.mu = z["mu"].astype(np.float32)        # feature mean (d_model,)
        self.sd = z["sd"].astype(np.float32)        # feature std  (d_model,)
        self.threshold = threshold
        # hooks the weak model must capture for this policy to work
        self.required_hooks = [self.hook]

    def probe_score(self, activation) -> float:
        a = np.asarray(activation, dtype=np.float32)
        a = (a - self.mu) / self.sd
        logit = float(a @ self.w + self.b)
        return 1.0 / (1.0 + np.exp(-logit))         # P(uncertain)

    def decide(self, state):
        acts = state.activations
        if not acts or self.hook not in acts:
            # Probe hook unavailable (e.g. the scalar-only hf backend): can't score,
            # so keep the weak token. Run with `--weak-backend tl` to enable the probe.
            return Decision.CONTINUE
        return Decision.DEFER if self.probe_score(acts[self.hook]) >= self.threshold \
            else Decision.CONTINUE


def build_policy(config):
    threshold = float(os.getenv("PROBE_THRESHOLD", "0.5"))
    config.span_max_tokens = int(os.getenv("SPAN_MAX_TOKENS", str(config.span_max_tokens)))
    return ProbePolicy(threshold=threshold)
