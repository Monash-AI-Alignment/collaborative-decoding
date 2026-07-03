"""Heal stop-string-truncated span text to a token boundary (pure, no vLLM dep).

vLLM truncates generated text at the stop-string match, which can land INSIDE a
multi-char token: Qwen encodes ':\n\n' as one token, so stop='\n' returns ':\n'
and silently drops the second newline. The concatenated assistant text then
re-encodes to a token sequence the model never generated, greedy continuation
falls off the free-running path, and every deferred span degrades formatting
(measured: strong_only vs its own free-running reference scored utility 0.407
instead of ~0.5). Extending the text to the end of its final generated token
keeps chunked decoding prefix-consistent with free-running generation.
"""


def heal_span_to_token_boundary(text, token_ids, decode):
    """Return `text` extended to the end of the last token in `token_ids`.

    `decode` maps a list of token ids to text (pass the model tokenizer's decode,
    with special tokens skipped). Falls back to `text` unchanged when there are
    no token ids or the full decode does not extend it (detokenizer edge cases).
    """
    if not token_ids:
        return text
    full = decode(list(token_ids))
    if len(full) > len(text) and full.startswith(text):
        return full
    return text
