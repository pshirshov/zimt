from __future__ import annotations

import os
import tempfile
import unittest

from zimt.webui.store import (
    DEFAULT_PROFILE_NAME,
    NONFAV_CAP,
    ProfileNotFound,
    SqliteProfileStore,
)


class StoreCase(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.store = SqliteProfileStore(os.path.join(self._dir.name, "zimt.db"))

    def tearDown(self) -> None:
        self.store.close()
        self._dir.cleanup()

    # ----- profiles -------------------------------------------------------

    def test_list_autocreates_default(self) -> None:
        profiles = self.store.list_profiles()
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0]["name"], DEFAULT_PROFILE_NAME)
        self.assertEqual(profiles[0]["prompt_count"], 0)

    def test_create_and_list_counts(self) -> None:
        a = self.store.create_profile("alpha")
        self.store.push_prompt(a["id"], "one")
        self.store.push_prompt(a["id"], "two")
        names = {p["name"]: p["prompt_count"] for p in self.store.list_profiles()}
        self.assertEqual(names["alpha"], 2)

    def test_create_rejects_blank(self) -> None:
        with self.assertRaises(ValueError):
            self.store.create_profile("   ")

    def test_create_rejects_duplicate_name(self) -> None:
        self.store.create_profile("dup")
        with self.assertRaises(ValueError):
            self.store.create_profile("dup")
        # whitespace-trimmed duplicate also collides
        with self.assertRaises(ValueError):
            self.store.create_profile("  dup  ")

    def test_create_rejects_unsafe_name(self) -> None:
        for bad in ("a/b", "..", ".", ".hidden", "name\\x", "emoji✨"):
            with self.assertRaises(ValueError):
                self.store.create_profile(bad)

    def test_ensure_profile_idempotent(self) -> None:
        a = self.store.ensure_profile("Shared")
        b = self.store.ensure_profile("Shared")
        self.assertEqual(a["id"], b["id"])
        names = [p["name"] for p in self.store.list_profiles()]
        self.assertEqual(names.count("Shared"), 1)

    def test_import_rejects_duplicate_name(self) -> None:
        self.store.create_profile("taken")
        with self.assertRaises(ValueError):
            self.store.import_profile("taken", "", [])

    def test_delete_profile(self) -> None:
        base = self.store.list_profiles()[0]["id"]
        extra = self.store.create_profile("extra")["id"]
        self.store.delete_profile(extra)
        ids = {p["id"] for p in self.store.list_profiles()}
        self.assertNotIn(extra, ids)
        self.assertIn(base, ids)

    def test_delete_last_profile_refused(self) -> None:
        only = self.store.list_profiles()[0]["id"]
        with self.assertRaises(ValueError):
            self.store.delete_profile(only)

    def test_delete_cascades_prompts(self) -> None:
        keep = self.store.list_profiles()[0]["id"]
        gone = self.store.create_profile("gone")["id"]
        self.store.push_prompt(gone, "x")
        self.store.delete_profile(gone)
        with self.assertRaises(ProfileNotFound):
            self.store.get_profile(gone)
        # The surviving profile is unaffected.
        self.assertEqual(self.store.get_profile(keep)["prompts"], [])

    def test_unknown_profile_raises(self) -> None:
        for call in (
            lambda: self.store.get_profile(999),
            lambda: self.store.set_globals(999, "x"),
            lambda: self.store.push_prompt(999, "x"),
            lambda: self.store.fav_prompt(999, "x", True),
            lambda: self.store.delete_prompt(999, "x"),
            lambda: self.store.clear_prompts(999),
            lambda: self.store.delete_profile(999),
        ):
            with self.assertRaises(ProfileNotFound):
                call()

    # ----- prompts --------------------------------------------------------

    def test_push_newest_first_and_dedup(self) -> None:
        pid = self.store.list_profiles()[0]["id"]
        self.store.push_prompt(pid, "a")
        self.store.push_prompt(pid, "b")
        res = self.store.push_prompt(pid, "a")  # re-push moves to front
        texts = [p["text"] for p in res["prompts"]]
        self.assertEqual(texts, ["a", "b"])

    def test_push_preserves_fav_on_repush(self) -> None:
        pid = self.store.list_profiles()[0]["id"]
        self.store.push_prompt(pid, "keep")
        self.store.fav_prompt(pid, "keep", True)
        self.store.push_prompt(pid, "other")
        self.store.push_prompt(pid, "keep")
        kept = next(p for p in self.store.get_profile(pid)["prompts"] if p["text"] == "keep")
        self.assertTrue(kept["fav"])

    def test_nonfav_cap_enforced_favs_spared(self) -> None:
        pid = self.store.list_profiles()[0]["id"]
        # One favourite that must survive the cap.
        self.store.push_prompt(pid, "fav-prompt")
        self.store.fav_prompt(pid, "fav-prompt", True)
        for i in range(NONFAV_CAP + 10):
            self.store.push_prompt(pid, f"p{i}")
        prompts = self.store.get_profile(pid)["prompts"]
        favs = [p for p in prompts if p["fav"]]
        nonfavs = [p for p in prompts if not p["fav"]]
        self.assertEqual(len(nonfavs), NONFAV_CAP)
        self.assertEqual([p["text"] for p in favs], ["fav-prompt"])
        # Oldest non-favs were trimmed; newest survive.
        texts = {p["text"] for p in nonfavs}
        self.assertIn(f"p{NONFAV_CAP + 9}", texts)
        self.assertNotIn("p0", texts)

    def test_fav_toggle(self) -> None:
        pid = self.store.list_profiles()[0]["id"]
        self.store.push_prompt(pid, "t")
        self.store.fav_prompt(pid, "t", True)
        self.assertTrue(self.store.get_profile(pid)["prompts"][0]["fav"])
        self.store.fav_prompt(pid, "t", False)
        self.assertFalse(self.store.get_profile(pid)["prompts"][0]["fav"])

    def test_delete_prompt(self) -> None:
        pid = self.store.list_profiles()[0]["id"]
        self.store.push_prompt(pid, "a")
        self.store.push_prompt(pid, "b")
        self.store.delete_prompt(pid, "a")
        self.assertEqual([p["text"] for p in self.store.get_profile(pid)["prompts"]], ["b"])

    def test_clear_keeps_favs(self) -> None:
        pid = self.store.list_profiles()[0]["id"]
        self.store.push_prompt(pid, "keep")
        self.store.fav_prompt(pid, "keep", True)
        self.store.push_prompt(pid, "drop1")
        self.store.push_prompt(pid, "drop2")
        res = self.store.clear_prompts(pid)
        self.assertEqual(res["removed"], 2)
        self.assertEqual(
            [p["text"] for p in self.store.get_profile(pid)["prompts"]], ["keep"]
        )

    # ----- globals --------------------------------------------------------

    def test_globals_get_set(self) -> None:
        pid = self.store.list_profiles()[0]["id"]
        self.assertEqual(self.store.get_profile(pid)["globals"], "")
        self.store.set_globals(pid, "${x=[a|b]}")
        self.assertEqual(self.store.get_profile(pid)["globals"], "${x=[a|b]}")

    # ----- migration / import --------------------------------------------

    def test_import_preserves_order_and_favs(self) -> None:
        prof = self.store.import_profile(
            "migrated",
            "${g=1}",
            [
                {"text": "favo", "fav": True},
                {"text": "newest", "fav": False},
                {"text": "older", "fav": False},
            ],
        )
        got = self.store.get_profile(prof["id"])
        self.assertEqual(got["globals"], "${g=1}")
        self.assertEqual([p["text"] for p in got["prompts"]], ["favo", "newest", "older"])
        self.assertTrue(got["prompts"][0]["fav"])

    def test_import_dedups_and_skips_blanks(self) -> None:
        prof = self.store.import_profile(
            "m",
            "",
            [
                {"text": "dup", "fav": False},
                {"text": "dup", "fav": True},
                {"text": "", "fav": False},
                {"fav": False},
            ],
        )
        got = self.store.get_profile(prof["id"])
        self.assertEqual([p["text"] for p in got["prompts"]], ["dup"])


if __name__ == "__main__":
    unittest.main()
