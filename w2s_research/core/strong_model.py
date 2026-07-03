"""Black-box strong model adapter (vLLM).

Wraps a local vLLM engine but the public surface is text-in / text-out only — no
logits or token ids are returned, enforcing the black-box constraint. Applies the
strong model's OWN chat template with assistant-continuation so it composes with a
different-tokenizer weak model at the text level.
"""
import os
from typing import List, Optional

from vllm import LLM, SamplingParams

from .interfaces import StrongOutput
from .span_heal import heal_span_to_token_boundary
from .timeout_guard import timeout


class VLLMStrongModel:
    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.6,
                 max_model_len: int = 4096):
        self.model_name = model_name
        # Fail-fast if a single generation hangs (e.g. the vLLM engine core died and the
        # client blocks forever). Generous default (~50x a normal call); override via env.
        self.gen_timeout = int(os.environ.get("STRONG_GEN_TIMEOUT", "300"))
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=1,
            enforce_eager=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _build_prompt(self, instruction: str, assistant_text: str) -> str:
        messages = [{"role": "user", "content": instruction}]
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, continue_final_message=True,
                add_generation_prompt=False,
            )
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def generate(self, instruction: str, assistant_text: str, *,
                 stop: Optional[List[str]], max_tokens: int, temperature: float) -> StrongOutput:
        prompt = self._build_prompt(instruction, assistant_text)
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            include_stop_str_in_output=True,   # keep the "\n" so assistant_text stays well-formed
        )
        with timeout(self.gen_timeout,
                     "strong-model generation timed out — vLLM engine may have died"):
            out = self.llm.generate([prompt], params)[0].outputs[0]
        text = out.text
        if out.stop_reason is not None:
            # The stop-string match can land inside a multi-char token (':\n\n' is one
            # Qwen token) — extend to the token boundary so the assistant text stays on
            # the model's own tokenization path across span handoffs.
            text = heal_span_to_token_boundary(
                text, out.token_ids,
                lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True))
        # finished on EOS only when vLLM stopped without matching a stop string and not on length
        finished = (out.finish_reason == "stop") and (out.stop_reason is None)
        return StrongOutput(text=text, finished=finished)
