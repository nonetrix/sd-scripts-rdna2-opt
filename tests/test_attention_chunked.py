import torch

from library.attention import sdpa_auto


def _qkv(seqlen=64, heads=2, dim=8, batch=2):
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seqlen, dim)
    k = torch.randn(batch, heads, seqlen, dim)
    v = torch.randn(batch, heads, seqlen, dim)
    return q, k, v


def test_chunked_sdpa_matches_unchunked():
    q, k, v = _qkv()
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    for chunk_size in (8, 16, 32, 64, 1000):
        out = sdpa_auto(q, k, v, chunk_size=chunk_size)
        assert out.shape == reference.shape
        assert torch.allclose(out, reference, atol=1e-6), f"chunk_size={chunk_size}"


def test_chunked_sdpa_matches_unchunked_with_mask():
    q, k, v = _qkv()
    mask = torch.ones(2, 1, 1, 64, dtype=torch.bool)
    mask[0, :, :, 40:] = False
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    for chunk_size in (16, 32):
        out = sdpa_auto(q, k, v, attn_mask=mask, chunk_size=chunk_size)
        assert torch.allclose(out, reference, atol=1e-6), f"chunk_size={chunk_size}"


def test_chunked_sdpa_matches_unchunked_with_dropout_eval():
    q, k, v = _qkv()
    # dropout_p=0.0 is the training-used path (Anima attention has no dropout by default).
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    out = sdpa_auto(q, k, v, dropout_p=0.0, chunk_size=16)
    assert torch.allclose(out, reference, atol=1e-6)


def test_chunked_sdpa_handles_5d_shaped_vae_tensors():
    """The Qwen-Image VAE attention block calls this with [B, 1, L, C] tensors."""
    q, k, v = _qkv(seqlen=256, heads=1, dim=16, batch=1)
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = sdpa_auto(q, k, v, chunk_size=32)
    assert torch.allclose(out, reference, atol=1e-6)
