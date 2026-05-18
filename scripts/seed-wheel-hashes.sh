#!/usr/bin/env bash
# Populate every `lib.fakeHash` placeholder in nix/wheels-*.nix by fetching
# the wheel URL and computing the SRI sha256.
#
# Requires: nix-prefetch-url, jq, sed.
set -euo pipefail

cd "$(dirname "$0")/.."

shopt -s nullglob
files=(nix/wheels-*.nix)
if (( ${#files[@]} == 0 )); then
  echo "no nix/wheels-*.nix files found" >&2
  exit 1
fi

LIB_FAKE='lib.fakeHash'

for f in "${files[@]}"; do
  echo "== $f =="
  # Pull every `url = "…"` line that's followed (within the same wheel
  # block) by a `lib.fakeHash` placeholder. We use a small Python helper
  # to keep the line-pairing exact.
  python3 - "$f" <<'PY'
import re, subprocess, sys

path = sys.argv[1]
text = open(path).read()

# Match blocks shaped like:
#   url = "<URL>";
#   <maybe other lines>
#   hash = lib.fakeHash;
pattern = re.compile(
    r'url\s*=\s*"(?P<url>[^"]+)"\s*;[^}]*?hash\s*=\s*lib\.fakeHash',
    re.DOTALL,
)

# Easier path: iterate wheel by wheel, fetch and substitute.
matches = list(pattern.finditer(text))
if not matches:
    print(f"  {path}: no placeholders to populate (already seeded?)")
    sys.exit(0)

for m in matches:
    url = m.group("url")
    print(f"  fetching {url}")
    out = subprocess.run(
        ["nix-prefetch-url", "--type", "sha256", "--unpack=false", url],
        check=True, capture_output=True, text=True,
    )
    sha_b32 = out.stdout.strip()
    sri = subprocess.run(
        ["nix", "hash", "to-sri", "--type", "sha256", sha_b32],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    # Replace the *first* remaining lib.fakeHash; we mutate the file as we go.
    cur = open(path).read()
    cur = cur.replace("lib.fakeHash", f'"{sri}"', 1)
    open(path, "w").write(cur)
    print(f"    -> {sri}")
PY
done

echo
echo "all hashes seeded. Re-run `nix build` to confirm."
