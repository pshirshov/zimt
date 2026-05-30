from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import zimt.paths as paths
from zimt.paths import profile_fav_dir, profile_out_dir, valid_profile_name
from zimt.webui import outputs as outputs_mod

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _png(path: str) -> None:
    with open(path, "wb") as f:
        f.write(PNG_MAGIC)


class NameValidationCase(unittest.TestCase):
    def test_valid_names(self) -> None:
        for ok in ("Default", "Work 2", "my-profile", "a_b.c", "X"):
            self.assertTrue(valid_profile_name(ok), ok)

    def test_invalid_names(self) -> None:
        for bad in ("", ".", "..", ".hidden", "a/b", "a\\b", "emoji✨",
                    "trailing ", "  ", None):
            self.assertFalse(valid_profile_name(bad), repr(bad))


class DirHelpersCase(unittest.TestCase):
    def test_dir_helpers_use_out_dir(self) -> None:
        with patch.object(paths, "OUT_DIR", "/tmp/zout"):
            self.assertEqual(profile_out_dir("Work"), "/tmp/zout/Work")
            self.assertEqual(profile_fav_dir("Work"), "/tmp/zout/Work/fav")


class MigrationCase(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.out = os.path.join(self._dir.name, "out")
        os.makedirs(self.out)
        self._patches = [
            patch.object(paths, "OUT_DIR", self.out),
            patch.object(paths, "FAV_DIR", os.path.join(self.out, "fav")),
            patch.object(paths, "_LAYOUT_MARKER", os.path.join(self.out, ".profile-layout")),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        self._dir.cleanup()

    def test_migrates_loose_and_legacy_fav(self) -> None:
        _png(os.path.join(self.out, "a.png"))
        _png(os.path.join(self.out, "b.png"))
        os.makedirs(os.path.join(self.out, "fav"))
        _png(os.path.join(self.out, "fav", "c.png"))

        paths.migrate_outputs_layout()

        default = os.path.join(self.out, "Default")
        self.assertTrue(os.path.isfile(os.path.join(default, "a.png")))
        self.assertTrue(os.path.isfile(os.path.join(default, "b.png")))
        self.assertTrue(os.path.isfile(os.path.join(default, "fav", "c.png")))
        # Loose top-level files were moved, not copied.
        self.assertFalse(os.path.isfile(os.path.join(self.out, "a.png")))
        # Empty legacy fav dir removed.
        self.assertFalse(os.path.isdir(os.path.join(self.out, "fav")))
        # Sentinel written.
        self.assertTrue(os.path.exists(os.path.join(self.out, ".profile-layout")))

    def test_idempotent(self) -> None:
        _png(os.path.join(self.out, "a.png"))
        paths.migrate_outputs_layout()
        # A new loose file after the first run must NOT be swept again.
        _png(os.path.join(self.out, "late.png"))
        paths.migrate_outputs_layout()
        self.assertTrue(os.path.isfile(os.path.join(self.out, "late.png")))
        self.assertFalse(os.path.isfile(os.path.join(self.out, "Default", "late.png")))

    def test_does_not_follow_symlink(self) -> None:
        sentinel = os.path.join(self._dir.name, "outside.png")
        _png(sentinel)
        os.symlink(sentinel, os.path.join(self.out, "link.png"))
        paths.migrate_outputs_layout()
        # The symlink target outside OUT_DIR must survive untouched.
        self.assertTrue(os.path.isfile(sentinel))
        self.assertFalse(os.path.isfile(os.path.join(self.out, "Default", "link.png")))


class ProfileScopedListingCase(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.out = os.path.join(self._dir.name, "out")
        os.makedirs(os.path.join(self.out, "Work", "fav"))
        os.makedirs(os.path.join(self.out, "Other"))
        _png(os.path.join(self.out, "Work", "main.png"))
        _png(os.path.join(self.out, "Work", "fav", "starred.png"))
        _png(os.path.join(self.out, "Other", "elsewhere.png"))
        self._p = patch.object(paths, "OUT_DIR", self.out)
        self._p.start()

    def tearDown(self) -> None:
        self._p.stop()
        self._dir.cleanup()

    def test_list_scoped_to_profile(self) -> None:
        names = {e["name"] for e in outputs_mod.list_outputs("Work", tab="all")["entries"]}
        self.assertEqual(names, {"main.png", "starred.png"})
        # The other profile's image is not visible.
        self.assertNotIn("elsewhere.png", names)

    def test_list_favs_tab(self) -> None:
        entries = outputs_mod.list_outputs("Work", tab="favs")["entries"]
        self.assertEqual([e["name"] for e in entries], ["starred.png"])
        self.assertTrue(entries[0]["fav"])

    def test_resolve_within_profile(self) -> None:
        self.assertIsNotNone(outputs_mod.resolve_output("Work", "main.png"))
        self.assertEqual(outputs_mod.resolve_output("Work", "starred.png")[1], True)
        # Wrong profile / traversal / unknown → None.
        self.assertIsNone(outputs_mod.resolve_output("Work", "elsewhere.png"))
        self.assertIsNone(outputs_mod.resolve_output("../etc", "main.png"))
        self.assertIsNone(outputs_mod.resolve_output("Work", "../Other/elsewhere.png"))


if __name__ == "__main__":
    unittest.main()
