from w2s_research.core.decode_config import DecodeConfig


def test_defaults_match_locked_models():
    cfg = DecodeConfig()
    assert cfg.weak_model == "meta-llama/Llama-3.2-1B-Instruct"
    assert cfg.strong_model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.benchmark == "gsm8k"
    assert cfg.r_bar == 0.98
    assert cfg.span_stop == ["\n"]


def test_eval_size_override():
    cfg = DecodeConfig(eval_size=32, benchmark="math")
    assert cfg.eval_size == 32
    assert cfg.benchmark == "math"
