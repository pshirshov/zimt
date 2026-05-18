#!/usr/bin/env python3
"""Populate ``lib.fakeHash`` placeholders in ``nix/wheels-*.nix`` using
already-cached wheel files from pip's http-v2 cache. No network.

Strategy:
  1. Walk every ``.body`` file under ``~/.cache/pip/http-v2/``.
  2. Open it as a wheel (zip) and read its ``*.dist-info/METADATA`` to
     extract the package name + version.
  3. Build an index ``{(pname, version): (body_path, sri_hash)}``.
  4. Walk ``nix/wheels-*.nix``, parsing each ``(wheel { pname=…; version=…; })``
     block. For every block whose pname+version is in the index, replace its
     ``lib.fakeHash`` (or insert ``hash = …;`` if missing) with the SRI hash.

The wheel filename is *not* useful for matching: pip stores the cached body
by URL-derived hash and the original URL isn't in the sidecar. Reading the
wheel's METADATA is the deterministic identifier.

Run from anywhere::

    .venv/bin/python scripts/seed-from-pip-cache.py
"""

from __future__ import annotations

import base64
import hashlib
import os
import pathlib
import re
import sys
import zipfile

CACHE_ROOT = pathlib.Path(os.path.expanduser("~/.cache/pip/http-v2"))


def sha256_sri(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return "sha256-" + base64.b64encode(h.digest()).decode()


def wheel_pname_version(body: pathlib.Path) -> tuple[str, str] | None:
    """Open ``body`` as a wheel and return (pname, version) from its METADATA.

    Returns None if the file isn't a valid wheel.
    """
    try:
        with zipfile.ZipFile(body) as zf:
            meta_name = next(
                (n for n in zf.namelist() if n.endswith(".dist-info/METADATA")),
                None,
            )
            if not meta_name:
                return None
            data = zf.read(meta_name).decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, OSError):
        return None

    name = ver = None
    for line in data.splitlines():
        if not line:
            break  # end of headers
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Version:"):
            ver = line.split(":", 1)[1].strip()
        if name and ver:
            break
    if not name or not ver:
        return None
    # Normalise pname like pip does: lowercase, '-'/'_' collapsed.
    pname = re.sub(r"[-_.]+", "-", name).lower()
    return (pname, ver)


def index_cache() -> dict[tuple[str, str], tuple[pathlib.Path, str]]:
    idx: dict[tuple[str, str], tuple[pathlib.Path, str]] = {}
    if not CACHE_ROOT.is_dir():
        return idx
    for body in CACHE_ROOT.rglob("*.body"):
        key = wheel_pname_version(body)
        if not key:
            continue
        # Prefer the first hit; collisions are unlikely.
        if key not in idx:
            idx[key] = (body, sha256_sri(body))
    return idx


BLOCK_PAT = re.compile(
    r'\(wheel\s*\{(?P<body>[^()]*?)\}\s*\)',
    re.DOTALL,
)
PNAME_PAT = re.compile(r'pname\s*=\s*"(?P<v>[^"]+)"\s*;')
VERSION_PAT = re.compile(r'version\s*=\s*"(?P<v>[^"]+)"\s*;')
URL_PAT = re.compile(r'url\s*=\s*"(?P<v>[^"]+)"\s*;')


def patch_nix_file(
    path: pathlib.Path,
    idx: dict[tuple[str, str], tuple[pathlib.Path, str]],
) -> tuple[int, int]:
    text = path.read_text()
    replaced = 0
    missing = 0
    for m in list(BLOCK_PAT.finditer(text)):
        block_body = m.group("body")
        pn = PNAME_PAT.search(block_body)
        vn = VERSION_PAT.search(block_body)
        if not pn or not vn:
            continue
        # Same normalisation as the indexer.
        pname_norm = re.sub(r"[-_.]+", "-", pn.group("v")).lower()
        key = (pname_norm, vn.group("v"))
        if key not in idx:
            print(f"  ! no cached wheel for {pname_norm}=={vn.group('v')}")
            missing += 1
            continue
        _path, sri = idx[key]
        block = m.group(0)
        if "lib.fakeHash" in block:
            new = block.replace("lib.fakeHash", f'"{sri}"', 1)
        elif re.search(r'hash\s*=\s*"', block):
            continue
        else:
            url_m = URL_PAT.search(block)
            if not url_m:
                continue
            new = block.replace(
                url_m.group(0),
                url_m.group(0) + f'\n    hash = "{sri}";',
                1,
            )
        text = text.replace(block, new)
        replaced += 1
    if replaced:
        path.write_text(text)
    return replaced, missing


def main() -> int:
    print(f"indexing pip http cache at {CACHE_ROOT}...")
    idx = index_cache()
    print(f"  found {len(idx)} cached wheels")
    for (n, v), (p, _) in sorted(idx.items())[:10]:
        print(f"    {n} {v}  <- {p.name[:16]}…")
    if len(idx) > 10:
        print(f"    … and {len(idx) - 10} more")

    nix_dir = pathlib.Path(__file__).resolve().parent.parent / "nix"
    total_r, total_m = 0, 0
    for nix_file in sorted(nix_dir.glob("wheels-*.nix")):
        r, m = patch_nix_file(nix_file, idx)
        if r or m:
            print(f"  {nix_file.name}: {r} populated, {m} missing")
        total_r += r
        total_m += m
    print(f"\ndone. populated={total_r}  missing={total_m}")
    return 0 if total_r else 1


if __name__ == "__main__":
    sys.exit(main())
