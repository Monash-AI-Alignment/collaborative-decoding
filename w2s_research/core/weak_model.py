"""White-box weak model adapter (HuggingFace Transformers).

Exposes per-step next-token uncertainty by running a forward pass and reading the
final-position logits. Chat-templates the (instruction, assistant_text) using the
weak model's OWN tokenizer with assistant-continuation, so it composes with a
different-tokenizer strong model at the text level.
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
            model_name, torch_dtype=_DTYPES[dtype],
        ).to(device).eval()
        self.eos_token_id = self.tokenizer.eos_token_id

    def _build_ids(self, instruction: str, assistant_text: str):
        messages = [{"role": "user", "content": instruction}]
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
            ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, continue_final_message=True,
                add_generation_prompt=False, return_tensors="pt",
            )
        else:
            ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            )
        return ids[:, -self.max_model_len:].to(self.device)

    @torch.no_grad()
    def next_step(self, instruction: str, assistant_text: str) -> WeakStep:
        ids = self._build_ids(instruction, assistant_text)
        logits = self.model(ids).logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        top_id = int(probs.argmax().item())

        # entropy in nats and top1-top2 margin, computed in torch then summarised
        ent = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
        top2 = torch.topk(probs, k=min(2, probs.numel()))
        top1_prob = float(top2.values[0].item())
        margin = float((top2.values[0] - top2.values[1]).item()) if top2.values.numel() > 1 else top1_prob

        is_eos = self.eos_token_id is not None and top_id == self.eos_token_id
        if is_eos:
            text_piece = ""
        else:
            # marginal text contributed by this token, robust to BPE leading spaces:
            prev = self.tokenizer.decode(ids[0], skip_special_tokens=True)
            after = self.tokenizer.decode(
                torch.cat([ids[0], torch.tensor([top_id], device=self.device)]),
                skip_special_tokens=True,
            )
            text_piece = after[len(prev):]

        return WeakStep(top_token_id=top_id, text_piece=text_piece, entropy=ent,
                        top1_prob=top1_prob, margin=margin, is_eos=is_eos)
