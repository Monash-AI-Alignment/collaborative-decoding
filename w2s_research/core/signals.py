"""Pure, read-only helpers over the weak model's next-token logits.

Nothing here is privileged plumbing — a policy that wants a distributional signal
lists ``"logits"`` in its ``required_hooks`` and derives whatever it needs from
``state.activations["logits"]`` (these helpers, or its own math). A policy that
doesn't care about the distribution never requests "logits" and never pays for it.

All values match the definitions the engine used to precompute: entropy in nats,
margin = p(top1) - p(top2), everything upcast to fp32 first.
"""
import torch


def probs(logits) -> "torch.Tensor":
    return torch.softmax(logits.float(), dim=-1)


def top_token_id(logits) -> int:
    return int(torch.as_tensor(logits).argmax().item())


def top1_prob(logits) -> float:
    return float(probs(logits).max().item())


def entropy(logits) -> float:
    p = probs(logits)
    return float(-(p * torch.log(p.clamp_min(1e-12))).sum().item())


def margin(logits) -> float:
    p = probs(logits)
    v = torch.topk(p, k=min(2, p.shape[-1])).values
    return float((v[0] - v[1]).item()) if v.numel() > 1 else float(v[0].item())
