"""SDXL has two text encoders (CLIP-L, OpenCLIP-G), both 77-tok. We report
both side-by-side so users can see which one drops a concept."""

from __future__ import annotations

from typing import Any

from ..tokenize_report import tokenize_one


def tokenize_report_sdxl(pipe: Any, text: str) -> None:
    tokenize_one(pipe.tokenizer, text, max_ctx=77, label="CLIP-L (text_encoder)")
    print()
    tokenize_one(pipe.tokenizer_2, text, max_ctx=77, label="OpenCLIP-G (text_encoder_2)")
