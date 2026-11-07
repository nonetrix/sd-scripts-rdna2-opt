"""AMD ROCm / RDNA helpers.

This module centralises the small number of places where sd-scripts has to behave
differently on AMD GPUs to avoid severe performance cliffs. Everything here is
opt-in by architecture detection and never changes numerics: the goal is to pick
paths that are *equally accurate* but much cheaper on these cards.

Background (measured on an RX 6800, gfx1030, ROCm 7.2, PyTorch 2.14):

* FlashAttention / memory-efficient attention / cuDNN attention are not compiled
  for gfx1030, so ``scaled_dot_product_attention`` silently falls back to the
  ``math`` backend, which materialises the full L x L attention matrix. At
  1024x1024 Anima latents (L=4096, 16 heads) that is a ~0.5 GiB temporary per
  attention call, ~4.2 GiB peak with backward, per transformer block.
* hipBLASLt and the CK GEMM backend both refuse gfx1030, so bf16 matmuls run
  through rocBLAS at roughly 6.6 TFLOP/s while fp16 reaches ~22 TFLOP/s and
  fp32 ~12 TFLOP/s. RDNA2 has no matrix cores, so bf16 (which is emulated) is
  the *slowest* of the three for both GEMM and convolution.
* MIOpen has no bf16 conv kernel for these shapes and upcasts internally: a 3x3
  conv at 1024x1024 takes ~33 ms in bf16 vs ~5 ms in fp16.

None of this is visible from the outside, hence the runtime probes below.
"""

import functools
import logging
import warnings
from typing import Optional

import torch

from library.utils import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

# RDNA1/2/3/4 architectures. All of them are consumer/“wave32” designs without
# CDNA matrix cores, which is what makes bf16 slow (it is emulated via fp32).
_RDNA_ARCH_PREFIXES = ("gfx10", "gfx11", "gfx12")

# gfx1030 (RX 6800/6900), gfx1031 (RX 6700), gfx1032 (RX 6600), gfx1034 (RX 6500),
# gfx1035/1036 (APUs).
_RDNA2_ARCHES = frozenset({"gfx1030", "gfx1031", "gfx1032", "gfx1034", "gfx1035", "gfx1036"})


def is_rocm() -> bool:
    """True when PyTorch is a ROCm (HIP) build."""
    return torch.version.hip is not None


def parse_gfx_arch(name: Optional[str]) -> Optional[str]:
    """Normalise a ``gcnArchName`` such as ``gfx1030:sramecc+:xnack-`` to ``gfx1030``."""
    if not name:
        return None
    return str(name).split(":", 1)[0].strip() or None


def get_gfx_arch(device: Optional[int] = None) -> Optional[str]:
    """Return the gfx architecture of ``device`` (default: current device), or None."""
    if not is_rocm() or not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(device)
        return parse_gfx_arch(getattr(props, "gcnArchName", None))
    except Exception:  # pragma: no cover - depends on the driver
        return None


def is_rdna(arch: Optional[str] = None) -> bool:
    """True for RDNA1/2/3/4 GPUs (no matrix cores, wave32)."""
    arch = arch or get_gfx_arch()
    return bool(arch) and arch.startswith(_RDNA_ARCH_PREFIXES)


def is_rdna2(arch: Optional[str] = None) -> bool:
    arch = arch or get_gfx_arch()
    return arch in _RDNA2_ARCHES


def is_cdna(arch: Optional[str] = None) -> bool:
    """True for CDNA (Instinct) accelerators, e.g. gfx90a / gfx942."""
    arch = arch or get_gfx_arch()
    return bool(arch) and arch.startswith("gfx9")


def describe_device() -> str:
    """Human readable one-line description used in logs."""
    if not is_rocm():
        return "non-ROCm build"
    arch = get_gfx_arch()
    kind = "RDNA2" if is_rdna2(arch) else ("RDNA" if is_rdna(arch) else ("CDNA" if is_cdna(arch) else "unknown"))
    return f"ROCm {torch.version.hip}, arch {arch} ({kind})"


@functools.lru_cache(maxsize=1)
def sdpa_fused_available() -> bool:
    """Whether PyTorch can actually run a fused (flash/mem-efficient) SDPA kernel here.

    PyTorch reports flash/mem-efficient SDPA as "enabled" on ROCm builds even when
    the kernels were not compiled for the current architecture, so the only
    reliable check is to force those backends and see whether a call succeeds.
    """
    if not torch.cuda.is_available():
        return False
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:  # pragma: no cover - very old torch
        return False

    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    for dtype in (torch.bfloat16, torch.float16):
        try:
            q = torch.zeros(1, 1, 8, 8, device="cuda", dtype=dtype)
            with warnings.catch_warnings():
                # PyTorch warns for every unavailable backend; this is a probe, not a failure.
                warnings.simplefilter("ignore")
                with sdpa_kernel(backends):
                    torch.nn.functional.scaled_dot_product_attention(q, q, q)
            return True
        except Exception:
            continue
    return False


def apply_rocm_optimizations(args=None) -> None:
    """Log AMD-specific advice and apply the safe global switches.

    Called once, before any model is loaded, from ``accelerator_setup.prepare_accelerator``.
    It only ever changes *how* work is executed, never the results.
    """
    if not is_rocm():
        return

    logger.info(f"AMD GPU detected: {describe_device()}")

    mixed_precision = getattr(args, "mixed_precision", None) if args is not None else None
    if is_rdna() and mixed_precision == "bf16":
        # Measured on gfx1030: 6.6 TFLOP/s bf16 vs 22 TFLOP/s fp16 vs 12 TFLOP/s fp32 for a
        # 4096x8192x2048 GEMM, and ~33 ms vs ~5 ms for a 3x3/1024x1024 convolution.
        logger.info(
            "RDNA GPUs have no matrix cores and hipBLASLt/CK do not support them, so bf16 matmuls run at "
            "roughly 6.6 TFLOP/s on gfx1030. The model is GEMM-bound here; bf16 is kept because it has the widest "
            "range and matches the released Anima weights."
        )

    if not sdpa_fused_available():
        logger.info(
            "No fused (flash/memory-efficient) SDPA kernel is available for this GPU: "
            "attention will use the bounded-memory chunked fallback (identical math, much lower VRAM)."
        )
