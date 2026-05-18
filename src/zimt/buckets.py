"""Resolution bucket lists per model family + a parser for the `/res` command.

Two curated sets are exported:
  * :data:`SDXL_BUCKETS` — the canonical aspect buckets SDXL was trained on.
    Going off-bucket is exactly where SDXL/Pony quality falls off.
  * :data:`ZIMAGE_BUCKETS` — Z-Image was trained with variable resolution
    rather than fixed buckets (docs: 512² to 2048², any aspect ratio). These
    anchor on the README's two worked examples (1024² and 1280×720) and add
    common aspect ratios near 1 MP with 16-px-aligned dimensions.

Each entry is ``(width, height, label)``. The first entry is the default.
"""

from __future__ import annotations

from typing import Any

Resolution = tuple[int, int, str]

SDXL_BUCKETS: list[Resolution] = [
    (1024, 1024, "1:1 square"),
    (1152,  896, "9:7 landscape"),
    ( 896, 1152, "7:9 portrait"),
    (1216,  832, "3:2 landscape"),
    ( 832, 1216, "2:3 portrait"),
    (1344,  768, "16:9 widescreen"),
    ( 768, 1344, "9:16 tall"),
    (1536,  640, "12:5 ultrawide"),
    ( 640, 1536, "5:12 ultra-tall"),
]

ZIMAGE_BUCKETS: list[Resolution] = [
    (1024, 1024, "1:1 square (1.05 MP, README default)"),
    (1280,  720, "16:9 landscape (0.92 MP, README example)"),
    ( 720, 1280, "9:16 portrait (0.92 MP)"),
    (1152,  896, "9:7 landscape (1.03 MP)"),
    ( 896, 1152, "7:9 portrait (1.03 MP)"),
    (1216,  832, "3:2 landscape (1.01 MP)"),
    ( 832, 1216, "2:3 portrait (1.01 MP)"),
    (1344,  768, "16:9 hi-res landscape (1.03 MP)"),
    ( 768, 1344, "16:9 hi-res portrait (1.03 MP)"),
]


# Orientation keywords accepted by `/res`. Each picks the largest preset
# (by total pixel area) whose aspect matches the requested orientation.
_ORIENTATION_PREDICATES: dict[str, Any] = {
    "square":    lambda w, h: w == h,
    "landscape": lambda w, h: w > h,
    "portrait":  lambda w, h: w < h,
}


def parse_res(arg: str, presets: list[Resolution]) -> tuple[int, int] | None:
    """Parse one of: ``<N>`` (1-based preset index), ``WxH``, ``W H``, or
    one of ``square`` / ``landscape`` / ``portrait``.

    Orientation keywords pick the largest matching preset by area. Returns
    ``(w, h)`` or ``None`` for unparseable input. Warns on stdout when W
    or H isn't a multiple of 16 — Z-Image silently rounds those down,
    SDXL just degrades.
    """
    s = arg.strip()
    lower = s.lower()
    if lower in _ORIENTATION_PREDICATES:
        pred = _ORIENTATION_PREDICATES[lower]
        candidates = [(w, h) for w, h, _ in presets if pred(w, h)]
        if not candidates:
            return None
        return max(candidates, key=lambda wh: wh[0] * wh[1])
    if s.isdigit():
        idx = int(s)
        if 1 <= idx <= len(presets):
            w, h, _ = presets[idx - 1]
            return (w, h)
    for sep in ("x", "X", "*", " "):
        if sep in s:
            parts = s.split(sep)
            if len(parts) != 2:
                continue
            try:
                w, h = int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                return None
            if w % 16 or h % 16:
                print(f"warning: {w}x{h} not divisible by 16 — "
                      "Z-Image silently rounds, SDXL may degrade")
            return (w, h)
    return None
