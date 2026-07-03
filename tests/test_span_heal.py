"""Span healing: stop-string truncation must not cut inside a multi-char token.

vLLM truncates a deferred span's text at the stop-string match. Qwen encodes
':\n\n' as ONE token, so stop='\n' keeps ':\n' and silently drops the second
newline — the re-encoded context then falls off the model's greedy tokenization
path and chunked decoding diverges from free-running (measured: strong_only
scored utility 0.407 vs its own reference instead of ~0.5).
"""
from w2s_research.core.span_heal import heal_span_to_token_boundary

# Fake vocab: token 1447 = ':\n\n' (a real Qwen merge), 198 = '\n'.
VOCAB = {1447: ":\n\n", 198: "\n", 10: "examples", 16: "1", 13: "."}


def _decode(ids):
    return "".join(VOCAB[i] for i in ids)


def test_mid_token_cut_is_extended_to_token_boundary():
    # vLLM matched stop='\n' inside token ':\n\n' and returned 'examples:\n'.
    healed = heal_span_to_token_boundary("examples:\n", [10, 1447], _decode)
    assert healed == "examples:\n\n"


def test_clean_cut_at_token_boundary_is_unchanged():
    healed = heal_span_to_token_boundary("examples\n", [10, 198], _decode)
    assert healed == "examples\n"


def test_detokenizer_mismatch_falls_back_to_vllm_text():
    # If the full decode does not extend the truncated text, keep vLLM's text.
    healed = heal_span_to_token_boundary("something else", [10, 1447], _decode)
    assert healed == "something else"


def test_empty_token_ids_is_unchanged():
    assert heal_span_to_token_boundary("abc", [], _decode) == "abc"
