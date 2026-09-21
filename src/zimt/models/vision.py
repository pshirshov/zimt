"""Qwen-Image 2.1, Krea 2, and Ideogram 4 Diffusers loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from ..memory import MemStrategy, finalize_pipe, from_pretrained_kwargs
from ..tokenize_report import tokenize_one

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline

IDEOGRAM_REPO = "ideogram-ai/ideogram-4-nf4-diffusers"
IDEOGRAM_PROMPT_HEAD_REPO = "diffusers/qwen3-vl-8b-instruct-lm-head"


def make_loader(repo_id: str) -> Callable[..., DiffusionPipeline]:
    def load(device: str, mem: MemStrategy,
             loras: tuple[tuple[str, float], ...] = ()) -> DiffusionPipeline:
        from diffusers import DiffusionPipeline

        if loras:
            raise ValueError("LoRAs are not supported for this model in Zimt")
        pipe = DiffusionPipeline.from_pretrained(
            repo_id, torch_dtype=torch.bfloat16,
            **from_pretrained_kwargs(mem),
        )
        finalize_pipe(pipe, device, mem)
        return pipe

    return load


def load_ideogram(device: str, mem: MemStrategy,
                  loras: tuple[tuple[str, float], ...] = ()) -> DiffusionPipeline:
    if torch.device(device).type != "cuda" or torch.version.hip is not None:
        raise ValueError("Ideogram 4 NF4 requires NVIDIA CUDA; XPU/ROCm/CPU are not supported")
    if loras:
        raise ValueError("LoRAs are not supported for Ideogram 4 in Zimt")
    from diffusers import Ideogram4Pipeline, Ideogram4PromptEnhancerHead

    head = Ideogram4PromptEnhancerHead.from_pretrained(
        IDEOGRAM_PROMPT_HEAD_REPO, torch_dtype=torch.bfloat16,
    )
    pipe = Ideogram4Pipeline.from_pretrained(
        IDEOGRAM_REPO, prompt_enhancer_head=head, torch_dtype=torch.bfloat16,
        **from_pretrained_kwargs(mem),
    )
    finalize_pipe(pipe, device, mem)
    return pipe


def tokenize_qwen(pipe: DiffusionPipeline, text: str) -> None:
    # Qwen 2.1 has no fixed text-token limit in its pipeline call.
    tok = pipe.processor.tokenizer
    ids = tok(text, add_special_tokens=False)["input_ids"]
    print(f"  Qwen3-VL: {len(ids)} raw tokens (no fixed pipeline text-token limit)")


def tokenize_krea(pipe: DiffusionPipeline, text: str) -> None:
    tokenize_one(pipe.tokenizer, text, max_ctx=512,
                 label="Qwen3-VL (raw prompt; pipeline adds its chat template)")


def tokenize_ideogram(pipe: DiffusionPipeline, text: str) -> None:
    tokenize_one(pipe.tokenizer, text, max_ctx=2048,
                 label="Qwen3-VL (before local caption expansion)")
