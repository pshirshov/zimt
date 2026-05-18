"""Tongyi-MAI Z-Image-Turbo loader + tokenize report.

Z-Image-Turbo is a 6B-parameter flow-matching DiT with a Qwen3-4B text
encoder. README pins ``guidance_scale=0.0`` (so the negative prompt is a
no-op until you raise CFG), 8 DiT forwards via ``num_inference_steps=9``.
"""

from __future__ import annotations

from typing import Any

import torch

from ..tokenize_report import tokenize_one


def load(device: str) -> Any:
    from diffusers import ZImagePipeline  # type: ignore[attr-defined]

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipe.to(device)
    return pipe


def tokenize_report(pipe: Any, text: str) -> None:
    tok = pipe.tokenizer
    templated = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    tokenize_one(tok, text, max_ctx=512, label="qwen3 (single encoder)",
                 templated=templated)
