import torch

from library.qwen_image_autoencoder_kl import AutoencoderKLQwenImage


def _build_vae(disable_cache: bool = False) -> AutoencoderKLQwenImage:
    kwargs = dict(
        base_dim=4,
        z_dim=2,
        dim_mult=[1, 2],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[True, False],
        dropout=0.0,
        latents_mean=[0.0, 0.0],
        latents_std=[1.0, 1.0],
        input_channels=3,
        disable_cache=disable_cache,
    )
    torch.manual_seed(0)
    return AutoencoderKLQwenImage(**kwargs).float().eval()


def _capture_feat_cache(module) -> dict:
    """Record the feat_cache the encoder/decoder is called with."""
    captured = {}

    def hook(_module, _args, kwargs):
        captured["feat_cache"] = kwargs.get("feat_cache")

    module.register_forward_pre_hook(hook, with_kwargs=True)
    return captured


def test_single_frame_matches_cache_disabled_vae_bitwise():
    vae = _build_vae()
    vae_no_cache = _build_vae(disable_cache=True)

    pixels = torch.randn(1, 3, 1, 8, 8)
    latents = torch.randn(1, 2, 1, 4, 4)

    with torch.no_grad():
        encoded = vae.encode(pixels, return_dict=False)[0].mode()
        encoded_ref = vae_no_cache.encode(pixels, return_dict=False)[0].mode()
        decoded = vae.decode(latents, return_dict=False)[0]
        decoded_ref = vae_no_cache.decode(latents, return_dict=False)[0]

    assert torch.equal(encoded, encoded_ref)
    assert torch.equal(decoded, decoded_ref)


def test_single_frame_skips_the_temporal_cache():
    vae = _build_vae()
    enc_capture = _capture_feat_cache(vae.encoder)
    dec_capture = _capture_feat_cache(vae.decoder)

    with torch.no_grad():
        vae.encode(torch.randn(1, 3, 1, 8, 8), return_dict=False)
        vae.decode(torch.randn(1, 2, 1, 4, 4), return_dict=False)

    assert enc_capture["feat_cache"] is None
    assert dec_capture["feat_cache"] is None
    # The caches are neither allocated nor filled for a single image.
    assert getattr(vae, "_enc_feat_map", None) is None


def test_multi_frame_still_uses_the_causal_cache():
    vae = _build_vae()
    enc_capture = _capture_feat_cache(vae.encoder)

    with torch.no_grad():
        encoded = vae.encode(torch.randn(1, 3, 5, 8, 8), return_dict=False)[0].mode()

    # 5 frames -> 3 latent frames with a single causal temporal downsampling stage.
    assert encoded.shape[2] == 3
    assert enc_capture["feat_cache"] is not None
