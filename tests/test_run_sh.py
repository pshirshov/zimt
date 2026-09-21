"""Behavioral/Blackbox-Group checks of the development launch environment."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


class RunScriptTests(unittest.TestCase):
    def cache_from_launcher(self, cache: str | None) -> str:
        env = dict(os.environ)
        env.pop("HF_HOME", None)
        if cache is not None:
            env["HF_HOME"] = cache
        completed = subprocess.run(
            ["bash", "-c", '''
nix() { printf '/nix/store/test-gcc\\n'; }
exec() { printf '%s\\n' "$HF_HOME"; }
source ./run.sh
'''],
            cwd=Path(__file__).resolve().parents[1], env=env,
            check=True, text=True, capture_output=True,
        )
        return completed.stdout.strip()

    def test_default_cache_uses_zimt_dataset(self):
        # regression: run.sh directed downloads into an obsolete home checkout.
        self.assertEqual(self.cache_from_launcher(None), "/srv/nvme/zimt/hf_cache")

    def test_explicit_cache_is_preserved(self):
        self.assertEqual(self.cache_from_launcher("/tmp/custom-model-cache"),
                         "/tmp/custom-model-cache")
