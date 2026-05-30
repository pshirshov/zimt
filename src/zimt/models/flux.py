"""Black Forest Labs FLUX loaders.

FLUX.1-dev is a 12B guidance-distilled flow-matching DiT with two text
encoders (CLIP-L + T5-XXL). At bf16 the transformer alone is ~24 GB, so
the whole pipeline overflows a 32 GB card; we load the transformer as
**fp8 via optimum-quanto** (qfloat8). Benchmarked on the Arc Pro B70 at
1024²/28 steps:

    backend                    warm    peak VRAM
    quanto qfloat8             40.3s   22.32 GiB   ← chosen
    torchao float8 weight-only 45.8s   33.53 GiB   (spills past 32 GB)
    torchao float8 dynamic-act 53.2s   22.32 GiB

quanto won on both axes. torchao's fp8 ships CUDA cutlass kernels that
don't load on XPU, so it falls back to a slower/heavier path. The T5
encoder stays bf16 (quality) — 22 GB peak leaves comfortable headroom.

FLUX.2-dev proper is huge (~32B transformer + a ~24B Mistral-3 encoder)
and doesn't fit 32 GB without slow offload. Instead we ship **FLUX.2
[klein] 9B** — BFL's guidance-distilled open variant — which uses a
compact **Qwen3** text encoder (~15 GB) and a 16.9 GB transformer. bf16
all-resident is ~34 GB (just over the card), so the transformer loads as
**fp8 via quanto** and Qwen3 stays bf16 → ~24 GB peak, same proven recipe
as FLUX.1.

Both families are guidance-distilled: ``guidance_scale`` is the distilled
guidance embedding (not classifier-free CFG), and the FLUX.2 pipelines
have no ``negative_prompt`` parameter at all — see the ``family != "flux2"``
guard in :func:`zimt.generate.generate`.
"""

from __future__ import annotations

from typing import Any

import torch

from ..memory import MemStrategy, finalize_pipe, from_pretrained_kwargs
from ..tokenize_report import tokenize_one

FLUX1_REPO = "black-forest-labs/FLUX.1-dev"
FLUX2_KLEIN_REPO = "black-forest-labs/FLUX.2-klein-9B"


def _fuse_loras(pipe: Any, loras: tuple[tuple[str, float], ...]) -> None:
    """Bake each registry LoRA into ``pipe.transformer`` (bf16) at its weight.

    Fused sequentially — load → fuse at the LoRA's weight → unload — so the
    per-LoRA weight is honoured and nothing remains as a live PEFT adapter
    (which would not survive the subsequent quanto quantization).
    """
    from .loras import LORAS
    for name, weight in loras:
        spec = LORAS.get(name)
        if spec is None:
            raise ValueError(f"unknown lora {name!r}")
        kwargs: dict[str, Any] = {}
        if spec.weight_name:
            kwargs["weight_name"] = spec.weight_name
        pipe.load_lora_weights(spec.repo_id, **kwargs)
        pipe.fuse_lora(lora_scale=weight)
        pipe.unload_lora_weights()


def _load_flux_fp8(pipe_cls: Any, transformer_cls: Any, repo: str,
                   device: str, mem: MemStrategy,
                   loras: tuple[tuple[str, float], ...]) -> Any:
    """Build a FLUX pipeline with an fp8 transformer.

    Without LoRAs we quantize during ``from_pretrained`` (QuantoConfig) — the
    fast path that never materialises the full bf16 transformer. With LoRAs we
    must load bf16, fuse the adapters (PEFT can't inject into quantized
    layers), then quantize the fused weights via optimum-quanto.
    """
    if not loras:
        from diffusers import QuantoConfig
        transformer = transformer_cls.from_pretrained(
            repo, subfolder="transformer",
            quantization_config=QuantoConfig(weights_dtype="float8"),
            torch_dtype=torch.bfloat16,
        )
        pipe = pipe_cls.from_pretrained(
            repo, transformer=transformer, torch_dtype=torch.bfloat16,
            **from_pretrained_kwargs(mem),
        )
        finalize_pipe(pipe, device, mem)
        return pipe

    # Fuse-at-load path: bf16 transformer → fuse LoRAs → quantize to fp8.
    from optimum.quanto import freeze, qfloat8, quantize
    transformer = transformer_cls.from_pretrained(
        repo, subfolder="transformer", torch_dtype=torch.bfloat16,
    )
    pipe = pipe_cls.from_pretrained(
        repo, transformer=transformer, torch_dtype=torch.bfloat16,
    )
    _fuse_loras(pipe, loras)
    quantize(pipe.transformer, weights=qfloat8)
    freeze(pipe.transformer)
    finalize_pipe(pipe, device, mem)
    return pipe


def load(device: str, mem: MemStrategy,
         loras: tuple[tuple[str, float], ...] = ()) -> Any:
    """FLUX.1-dev — fp8 (quanto qfloat8) transformer, T5 bf16.

    ``loras`` (if any) are fused into the transformer at load time — see
    :func:`_load_flux_fp8`.
    """
    from diffusers import FluxPipeline, FluxTransformer2DModel
    return _load_flux_fp8(FluxPipeline, FluxTransformer2DModel,
                          FLUX1_REPO, device, mem, loras)


def load_flux2(device: str, mem: MemStrategy,
               loras: tuple[tuple[str, float], ...] = ()) -> Any:
    """FLUX.2 [klein] 9B — fp8 (quanto) transformer, Qwen3 bf16."""
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
    return _load_flux_fp8(Flux2KleinPipeline, Flux2Transformer2DModel,
                          FLUX2_KLEIN_REPO, device, mem, loras)


def tokenize_report(pipe: Any, text: str) -> None:
    """FLUX.1 has CLIP-L (77 tok, pooled) + T5-XXL (512 tok, sequence)."""
    tokenize_one(pipe.tokenizer, text, max_ctx=77, label="CLIP-L (text_encoder)")
    print()
    tokenize_one(pipe.tokenizer_2, text, max_ctx=512, label="T5-XXL (text_encoder_2)")


def tokenize_report_flux2(pipe: Any, text: str) -> None:
    """FLUX.2 [klein] uses a single Qwen3 text encoder."""
    tokenize_one(pipe.tokenizer, text, max_ctx=512, label="Qwen3 (single encoder)")
