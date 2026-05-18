"""Tokenizer analysis used by the ``/tokenize`` command.

Surfaces three heuristics for whether the text encoder "knows" each word:
  1. Token count per whitespace-delimited word. ≥3 = yellow (likely OOD for
     the tokenizer); ≥5 = red.
  2. ``⚠ byte-frag`` flag when a single token's decode contains ``U+FFFD``
     (a partial UTF-8 piece) — the closest analogue to the old word-level
     "UNK".
  3. Encoder-input length against the model's ``max_sequence_length`` (or 77
     for SDXL's two CLIP encoders), so users see when they're about to be
     truncated.

The tokenizer interface is intentionally loose — the diffusers pipelines
expose HuggingFace ``PreTrainedTokenizer`` instances, but the surface we use
(``__call__(text, return_offsets_mapping=True)`` and ``decode``) is stable.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from .ansi import CYAN, DIM, RED, RESET, YELLOW


class _Tokenizer(Protocol):
    def __call__(self, text: str, /, **kwargs: Any) -> Any: ...
    def decode(self, ids: list[int], /, **kwargs: Any) -> str: ...


def tokenize_one(tok: Any, text: str, max_ctx: int, label: str,
                 templated: str | None = None) -> None:
    """Print a single tokenizer's report.

    ``templated`` is the actual text that reaches the encoder (after chat
    template / special tokens are applied). If ``None``, the budget is
    measured against ``tok(text, add_special_tokens=True)``.
    """
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids: list[int] = list(enc["input_ids"])
    offsets: list[tuple[int, int]] = list(enc["offset_mapping"])

    budget_ids = (tok(templated, add_special_tokens=False)["input_ids"]
                  if templated is not None
                  else tok(text, add_special_tokens=True)["input_ids"])
    budget_used = len(budget_ids)
    over = budget_used > max_ctx
    colour_b = RED if over else (YELLOW if budget_used > max_ctx * 0.8 else "")
    print(f"  {CYAN}{label}{RESET}  "
          f"raw: {len(ids)} tok   "
          f"encoder input: {colour_b}{budget_used}{RESET}/{max_ctx}"
          f"{'  [TRUNCATED]' if over else ''}")

    rows: list[tuple[str, list[tuple[int, str, bool]]]] = []
    for m in re.finditer(r"\S+", text):
        ws, we = m.span()
        word = m.group()
        pieces: list[tuple[int, str, bool]] = []
        for tid, (s, e) in zip(ids, offsets):
            if e <= s:
                continue
            if e > ws and s < we:
                piece = text[s:e]
                decoded = tok.decode([tid], skip_special_tokens=False)
                is_frag = "�" in decoded
                pieces.append((tid, piece, is_frag))
        rows.append((word, pieces))

    if not rows:
        return
    width = min(40, max(len(w) for w, _ in rows))
    for word, pieces in rows:
        n = len(pieces)
        any_frag = any(f for _, _, f in pieces)
        colour = RED if (n >= 5 or any_frag) else (YELLOW if n >= 3 else "")
        piece_repr = ", ".join(repr(p) for _, p, _ in pieces)
        flag = " ⚠ byte-frag" if any_frag else ""
        wcell = word if len(word) <= width else word[: width - 1] + "…"
        print(f"    {colour}{wcell:<{width}}{RESET}  {n:>3}  "
              f"{DIM}[{piece_repr}]{RESET}{flag}")
