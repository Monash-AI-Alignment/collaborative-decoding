"""White-box weak model backed by TransformerLens (HookedTransformer).

Same faithful greedy-decode + resync contract as HFWeakModel (see weak_model.py),
but at each step it also exposes the weak model's INTERNAL activations (residual
stream, attention, MLP, per-head — any TransformerLens hook), so a policy can run
a linear probe / mechinterp readout on the model's state, not just its logits.

The weak model is genuinely white-box: full read access to activations here, and
the SAME class can be loaded offline (see `harvest`) to sweep data and train a
probe BEFORE evaluation. The strong model stays black-box (text in, text out).

Faithfulness contract — mirrors HFWeakModel's decode recipe EXACTLY, but note the
forward is TransformerLens's OWN re-implementation (its rotary/gelu/LayerNorm ops,
fp32 attention-score upcast, different reduction ordering), so logits are NOT
bit-identical to AutoModelForCausalLM and the greedy argmax can differ near ties —
especially in bf16. A float32 CPU smoke test matched HF greedy 12/12 on one prompt,
but that is a spot check, not a proof. `scripts/check_tl_hf_faithfulness` runs a
multi-prompt token-by-token comparison; run it (and ideally a bf16-GPU version)
before treating TL-produced leaderboard entries as comparable to HF baselines.
The recipe this class reproduces:
  * greedy argmax over the fp32-upcast final-position logits;
  * the model keeps its OWN token-id sequence across CONTINUE steps (never
    re-tokenizes accumulated text mid-run);
  * resync() re-encodes from text ONLY at a defer handback, as chat-templated
    prefix + RAW-tokenized assistant text (no template) — preserving trailing
    whitespace so continuation does not degenerate into newlines;
  * every state mutation left-truncates to max_model_len.

Like HFWeakModel, `peek` runs a full forward each step (O(N) per step). A
KV-cache incremental forward is the obvious later optimization.
"""
from typing import Dict, List, Optional

import torch

from .interfaces import WeakStep

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class TLWhiteBoxWeakModel:
    def __init__(self, model_name: str, max_model_len: int = 4096,
                 device: Optional[str] = None, dtype: str = "bfloat16",
                 capture_hooks: Optional[List[str]] = None, fold_ln: bool = True):
        # transformer_lens/transformers are imported lazily so the rest of the
        # engine (and the test suite) never require them just to import this file.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformer_lens import HookedTransformer

        self.model_name = model_name
        self.max_model_len = max_model_len
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # bf16 on CPU is slow/partly-unsupported; fp32 there also makes argmax exact.
        if device == "cpu" and dtype == "bfloat16":
            dtype = "float32"
        self.device = device
        self.dtype = _DTYPES[dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf = AutoModelForCausalLM.from_pretrained(model_name, dtype=self.dtype)
        loader = HookedTransformer.from_pretrained if fold_ln \
            else HookedTransformer.from_pretrained_no_processing
        self.model = loader(model_name, hf_model=hf, tokenizer=self.tokenizer,
                            device=device, dtype=self.dtype)
        del hf
        self.model.eval()

        self.eos_token_id = self.tokenizer.eos_token_id
        self.n_layers = self.model.cfg.n_layers
        self.d_model = self.model.cfg.d_model
        # Default capture: the last layer's residual stream (post-block).
        self.capture_hooks = list(capture_hooks) if capture_hooks else \
            [f"blocks.{self.n_layers - 1}.hook_resid_post"]
        self._ids = None  # [1, T] LongTensor: current context token ids

    # -- context management (identical recipe to HFWeakModel) ------------------
    def _prefix_ids(self, instruction: str):
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
            cont = self.tokenizer(
                assistant_text, add_special_tokens=False, return_tensors="pt",
            ).input_ids.to(self.device)
            ids = torch.cat([prefix, cont], dim=1)
        else:
            ids = prefix
        self._ids = ids[:, -self.max_model_len:]

    # -- the white-box step ----------------------------------------------------
    @torch.no_grad()
    def _forward_capture(self, ids) -> tuple:
        """Full forward; return (fp32 last-position logits, {hook: last-pos activation}).

        Internal hooks come from run_with_cache; the pseudo-hook "logits" (if the
        policy requested it) carries the raw next-token distribution.
        """
        names = {h for h in self.capture_hooks if h != "logits"}
        logits, cache = self.model.run_with_cache(
            ids, return_type="logits", names_filter=lambda n: n in names,
        )
        last_logits = logits[0, -1, :].float()
        acts: Dict[str, torch.Tensor] = {
            n: cache[n][0, -1].detach().float().cpu() for n in names if n in cache
        }
        if "logits" in self.capture_hooks:
            acts["logits"] = last_logits.detach().cpu()
        return last_logits, acts

    @torch.no_grad()
    def peek(self) -> WeakStep:
        logits, acts = self._forward_capture(self._ids)
        top_id = int(logits.argmax().item())

        is_eos = self.eos_token_id is not None and top_id == self.eos_token_id
        if is_eos:
            text_piece = ""
        else:
            prev = self.tokenizer.decode(self._ids[0], skip_special_tokens=True)
            after = self.tokenizer.decode(
                torch.cat([self._ids[0], torch.tensor([top_id], device=self.device)]),
                skip_special_tokens=True,
            )
            text_piece = after[len(prev):]

        return WeakStep(top_token_id=top_id, text_piece=text_piece,
                        is_eos=is_eos, activations=acts)

    # -- offline exploration ---------------------------------------------------
    @torch.no_grad()
    def harvest(self, instruction: str, max_new_tokens: int = 40,
                hooks: Optional[List[str]] = None) -> List[dict]:
        """Free-run greedy for up to `max_new_tokens`, returning per-step records
        {activations, top_token_id, text_piece}. Request "logits" among `hooks` if
        you want the distribution (derive entropy/margin via core.signals).

        For OFFLINE probe training: sweep a dataset, collect (activation -> label)
        pairs, fit a probe, then apply it live in a policy via `peek().activations`.
        """
        saved = self.capture_hooks
        if hooks is not None:
            self.capture_hooks = list(hooks)
        try:
            self.begin(instruction)
            records = []
            for _ in range(max_new_tokens):
                step = self.peek()
                if step.is_eos:
                    break
                records.append(dict(
                    activations=step.activations,
                    top_token_id=step.top_token_id, text_piece=step.text_piece,
                ))
                self.commit(step.top_token_id)
            return records
        finally:
            self.capture_hooks = saved
