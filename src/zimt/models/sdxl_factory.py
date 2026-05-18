"""Shared SDXL pipeline loader.

Every SDXL-based model in the registry uses the same wiring:
  * load the diffusers-format mirror from HF
  * substitute ``madebyollin/sdxl-vae-fp16-fix`` for the bundled VAE
    (the original SDXL VAE has the SD-1.x ``scaling_factor`` and isn't
    fp16-stable — pastel output is the classic symptom)
  * load everything in bfloat16

The scheduler is *not* set here — that's the responsibility of the
``_ensure_sampler`` step in :mod:`zimt.generate`, which honours the
``ModelSpec.default_sampler`` and any per-model ``scheduler_overrides``.

We use ``from_pretrained`` rather than ``from_single_file`` because the
latter rebuilds CLIPTextModel by reading ``model.text_model.embeddings…``,
an attribute path that transformers 5.x removed.
"""

from __future__ import annotations

from typing import Any, Callable

import torch


def make_sdxl_loader(
    repo_id: str,
    *,
    use_fp16_fix_vae: bool = True,
) -> Callable[[str], Any]:
    """Return a ``load(device) -> pipeline`` callable for the given HF repo."""

    def load(device: str) -> Any:
        from diffusers import (  # type: ignore[import-not-found]
            AutoencoderKL,
            StableDiffusionXLPipeline,
        )
        kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        if use_fp16_fix_vae:
            kwargs["vae"] = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=torch.bfloat16,
            )
        pipe = StableDiffusionXLPipeline.from_pretrained(repo_id, **kwargs)
        pipe.to(device)
        return pipe

    load.__name__ = f"load_sdxl[{repo_id}]"
    load.__doc__ = (
        f"Load {repo_id} via diffusers StableDiffusionXLPipeline.from_pretrained, "
        f"with the madebyollin fp16-fix VAE substituted in bf16."
    )
    setattr(load, "repo_id", repo_id)
    return load
