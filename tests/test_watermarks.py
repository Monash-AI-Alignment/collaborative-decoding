"""Watermark framework: registry, method reconstruction, and the KGW generator/detector
green-list mechanism (CPU, small vocab — the full detector forces CUDA)."""
from types import SimpleNamespace

import pytest


def test_registry_names_and_lookup():
    from w2s_research.core.watermarks import WATERMARKS, get_watermark_cls
    assert set(WATERMARKS) == {"eth_french", "inference_kgw"}
    assert get_watermark_cls("inference_kgw").name == "inference_kgw"
    assert get_watermark_cls("eth_french").name == "eth_french"
    with pytest.raises(ValueError):
        get_watermark_cls("does_not_exist")


def test_methods_subclass_and_reconstruct_from_cli():
    from w2s_research.core.watermarks.base import WatermarkMethod
    from w2s_research.core.watermarks.eth_french import ETHFrenchWatermark
    from w2s_research.core.watermarks.inference_kgw import InferenceKGWWatermark
    assert issubclass(ETHFrenchWatermark, WatermarkMethod)
    assert issubclass(InferenceKGWWatermark, WatermarkMethod)

    e = ETHFrenchWatermark.from_cli_args(
        SimpleNamespace(fingerprinted_model="ckpt", embedding_config="key.yaml", weak_model=None))
    assert e.fingerprinted_model == "ckpt" and e.embedding_config == "key.yaml"
    assert e.benchmark == "alpaca_eval" and e.alpha == 1e-3

    k = InferenceKGWWatermark.from_cli_args(
        SimpleNamespace(strong_model="Qwen/Q", kgw_gamma=0.25, kgw_delta=4.0,
                        kgw_seeding="simple_1", weak_model="w/1B"))
    assert k.strong_model == "Qwen/Q" and k.gamma == 0.25 and k.delta == 4.0
    assert k.weak_model == "w/1B"


def test_kgw_v1_processor_boosts_only_greenlist(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("robust_fp")
    pytest.importorskip("vllm")
    from types import SimpleNamespace
    monkeypatch.setenv("KGW_GAMMA", "0.25")
    monkeypatch.setenv("KGW_DELTA", "5.0")
    monkeypatch.setenv("KGW_SEEDING", "simple_1")
    from w2s_research.core.watermarks.kgw_v1_processor import KGWV1LogitsProcessor
    V = 64
    cfg = SimpleNamespace(model_config=SimpleNamespace(get_vocab_size=lambda: V))
    proc = KGWV1LogitsProcessor(cfg, torch.device("cpu"), False)
    proc._reqs = {0: ([3], [9])}                        # (prompt, output); cw=1 -> seed = last = 9
    out = proc.apply(torch.zeros(1, V))
    green = set(proc.base._get_greenlist_ids(torch.tensor([9])).tolist())
    assert len(green) == int(V * 0.25)
    for i in range(V):
        assert out[0, i].item() == pytest.approx(5.0 if i in green else 0.0)


def test_kgw_green_sequence_is_detectable():
    """Build a fully-green sequence and confirm it reads ~100% green while an unwatermarked
    sequence reads ~gamma — using the SAME WatermarkBase the real generator-processor and the
    detector's masks are built from (KGWWatermark fills its masks via _get_greenlist_ids)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("robust_fp")
    from robust_fp.watermarks.kgw.watermark_processor import WatermarkBase
    V, gamma = 128, 0.25
    base = WatermarkBase(vocab=list(range(V)), gamma=gamma, delta=4.0,
                         seeding_scheme="simple_1", device="cpu")

    def green_of(prev):
        return set(base._get_greenlist_ids(torch.tensor([prev])).tolist())

    seq = [5]
    for _ in range(300):
        seq.append(min(green_of(seq[-1])))              # always emit a green token
    unwm = [(i * 7 + 3) % V for i in range(len(seq))]   # structure-free-ish, unwatermarked

    def green_frac(s):
        return sum(s[t] in green_of(s[t - 1]) for t in range(1, len(s))) / (len(s) - 1)

    wm, base_frac = green_frac(seq), green_frac(unwm)
    assert wm > 0.95                # watermarked -> nearly all green
    assert base_frac < 0.5          # unwatermarked -> near the gamma chance rate
    assert wm - base_frac > 0.5     # clear separation the z-test turns into p<<alpha
