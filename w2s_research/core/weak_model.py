"""White-box weak model adapter (HuggingFace Transformers).

Stateful greedy decoder that maintains its OWN token-id sequence (like
model.generate), so the weak model's continuation is faithful. We deliberately
do NOT reconstruct the context from text on every step: that round-trip through
the chat template strips trailing whitespace and corrupts generation (the model
degenerates into endless newlines). The text round-trip happens ONLY at a defer
handback (`resync`), which re-encodes the accumulated assistant text via raw
tokenization (no whitespace-stripping template) — preserving cross-tokenizer
composition with the strong model while keeping weak generation faithful.

Note: `peek` runs a full forward over the current context each step (O(N) per
step, O(N^2) per example). Correct but not fast; a KV-cache incremental forward
is the obvious next optimization.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .interfaces import WeakStep

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class HFWeakModel:
    def __init__(self, model_name: str, max_model_len: int = 4096,
                 device: str = "cuda", dtype: str = "bfloat16"):
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=_DTYPES[dtype],
        ).to(device).eval()
        self.eos_token_id = self.tokenizer.eos_token_id
        self._ids = None  # [1, T] tensor: the current context token ids

    def _prefix_ids(self, instruction: str):
        """Chat-templated user turn + assistant generation prompt (no assistant content)."""
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
        if not torch.is_tensor(ids):          # transformers>=5 returns a BatchEncoding
            ids = ids["input_ids"]
        return ids.to(self.device)

    def begin(self, instruction: str) -> None:
        self._ids = self._prefix_ids(instruction)[:, -self.max_model_len:]

    def commit(self, token_id: int) -> None:
        tok = torch.tensor([[int(token_id)]], device=self.device)
        self._ids = torch.cat([self._ids, tok], dim=1)[:, -self.max_model_len:]

    def resync(self, instruction: str, assistant_text: str) -> None:
        prefix = self._prefix_ids(instruction)
        if assistant_text:
            # raw tokenize (NO chat template) so trailing whitespace is preserved
            cont = self.tokenizer(
                assistant_text, add_special_tokens=False, return_tensors="pt",
            ).input_ids.to(self.device)
            ids = torch.cat([prefix, cont], dim=1)
        else:
            ids = prefix
        self._ids = ids[:, -self.max_model_len:]

    @torch.no_grad()
    def peek(self) -> WeakStep:
        logits = self.model(self._ids).logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        top_id = int(probs.argmax().item())

        ent = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
        top2 = torch.topk(probs, k=min(2, probs.numel()))
        top1_prob = float(top2.values[0].item())
        margin = float((top2.values[0] - top2.values[1]).item()) if top2.values.numel() > 1 else top1_prob

        is_eos = self.eos_token_id is not None and top_id == self.eos_token_id
        if is_eos:
            text_piece = ""
        else:
            # marginal text contributed by this token, robust to BPE leading spaces
            prev = self.tokenizer.decode(self._ids[0], skip_special_tokens=True)
            after = self.tokenizer.decode(
                torch.cat([self._ids[0], torch.tensor([top_id], device=self.device)]),
                skip_special_tokens=True,
            )
            text_piece = after[len(prev):]

        return WeakStep(top_token_id=top_id, text_piece=text_piece, entropy=ent,
                        top1_prob=top1_prob, margin=margin, is_eos=is_eos)
