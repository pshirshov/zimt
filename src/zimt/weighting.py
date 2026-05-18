"""Prompt-weighting (compel) integration for SDXL pipelines.

Compel parses A1111-style weighting syntax — ``(word:1.3)``, ``(word)+``,
``[word]-`` — and produces per-token-weighted embeddings that drop into
the pipeline's ``prompt_embeds`` / ``pooled_prompt_embeds`` slots
instead of ``prompt`` / ``negative_prompt``.

Z-Image isn't supported by compel (different text-encoder architecture);
:func:`build_compel` returns ``None`` for that family and callers fall
back to passing prompts as strings.
"""

from __future__ import annotations

import re
from typing import Any

_WEIGHTING_PAT = re.compile(r"\([^)]+:[\-\d.]+\)|[()\[\]][+\-]+")


def has_weighting(text: str) -> bool:
    """Cheap precheck: does this prompt contain compel-style weighting syntax?

    If False, callers can skip the compel encode and let the pipeline
    tokenize the prompt directly — that's faster and works for any model
    family.
    """
    return bool(text) and bool(_WEIGHTING_PAT.search(text))


def build_compel(pipe: Any, family: str) -> Any | None:
    """Return a Compel instance for an SDXL ``pipe``, ``None`` otherwise.

    Raises only if compel is installed but the construction itself fails;
    a missing ``compel`` import returns ``None`` so the rest of the app
    keeps working without prompt weighting.
    """
    if family != "sdxl":
        return None
    try:
        from compel import Compel, ReturnedEmbeddingsType  # type: ignore[import-not-found]
    except ImportError:
        return None
    return Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
    )


def encode_sdxl(
    compel: Any, prompt: str, negative_prompt: str | None,
) -> dict[str, Any]:
    """Run compel on the prompt(s) and return kwargs for the SDXL pipe call."""
    cond, pooled = compel([prompt])
    out: dict[str, Any] = {
        "prompt_embeds": cond,
        "pooled_prompt_embeds": pooled,
    }
    if negative_prompt:
        neg_cond, neg_pooled = compel([negative_prompt])
        out["negative_prompt_embeds"] = neg_cond
        out["negative_pooled_prompt_embeds"] = neg_pooled
    return out
