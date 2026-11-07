import torch

from library import rocm_utils
from library.attention import sdpa_auto


def test_parse_gfx_arch_strips_target_features():
    assert rocm_utils.parse_gfx_arch("gfx1030:sramecc+:xnack-") == "gfx1030"
    assert rocm_utils.parse_gfx_arch("gfx90a") == "gfx90a"
    assert rocm_utils.parse_gfx_arch(None) is None
    assert rocm_utils.parse_gfx_arch("") is None


def test_arch_classification():
    # RDNA2 (RX 6000 series): no matrix cores, bf16 is emulated.
    assert rocm_utils.is_rdna2("gfx1030")
    assert rocm_utils.is_rdna("gfx1030")
    # RDNA3.
    assert rocm_utils.is_rdna("gfx1100")
    assert not rocm_utils.is_rdna2("gfx1100")
    # CDNA (Instinct).
    assert rocm_utils.is_cdna("gfx90a")
    assert not rocm_utils.is_rdna("gfx90a")
    assert not rocm_utils.is_cdna("gfx1030")
    # An explicit unknown arch must not be treated as RDNA. (None means "detect the
    # current device", so it is not asserted here: it depends on the machine.)
    assert not rocm_utils.is_rdna("gfx0000")
    assert not rocm_utils.is_rdna2("gfx0000")


def test_sdpa_auto_is_not_chunked_below_threshold_and_matches():
    """Long sequences are only chunked when no fused kernel exists; results must match either way."""
    torch.manual_seed(0)
    q = torch.randn(1, 2, 128, 8)
    k = torch.randn(1, 2, 128, 8)
    v = torch.randn(1, 2, 128, 8)
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(sdpa_auto(q, k, v), reference, atol=1e-6)


def test_vae_dtype_defaults_to_fp32_on_rdna(monkeypatch):
    """On RDNA the VAE runs in fp32: bf16 convs are emulated and fp32 matches the reference."""
    from library import anima_train_utils

    monkeypatch.setattr(rocm_utils, "is_rdna", lambda arch=None: True)
    args = type("Args", (), {"vae_dtype": "auto"})()
    assert anima_train_utils.resolve_vae_dtype(args, torch.bfloat16) is torch.float32

    # Explicit choice always wins.
    args_fp16 = type("Args", (), {"vae_dtype": "fp16"})()
    assert anima_train_utils.resolve_vae_dtype(args_fp16, torch.bfloat16) is torch.float16


def test_vae_dtype_keeps_training_dtype_off_rdna(monkeypatch):
    from library import anima_train_utils

    monkeypatch.setattr(rocm_utils, "is_rdna", lambda arch=None: False)
    args = type("Args", (), {"vae_dtype": "auto"})()
    assert anima_train_utils.resolve_vae_dtype(args, torch.bfloat16) is torch.bfloat16
