from __future__ import annotations

import unittest

from zimt.models.registry import MODELS
from zimt.webui.exec_api import _MAX_PIXELS_BY_FAMILY


class RegistryCase(unittest.TestCase):
    def test_every_model_is_well_formed(self) -> None:
        self.assertTrue(MODELS, "registry must not be empty")
        for name, spec in MODELS.items():
            with self.subTest(model=name):
                # Key matches the spec's own name.
                self.assertEqual(spec.name, name)
                # default_sampler must be one of the model's samplers.
                self.assertIn(spec.default_sampler, spec.samplers,
                              f"{name}: default_sampler not in samplers")
                # At least one resolution preset (default_w/h read [0]).
                self.assertTrue(spec.resolutions, f"{name}: no resolutions")
                # load must be callable (the loader entry point).
                self.assertTrue(callable(spec.load), f"{name}: load not callable")
                # repo_id is the source of truth for loader + metadata.
                self.assertTrue(spec.repo_id, f"{name}: empty repo_id")

    def test_every_family_has_a_pixel_cap(self) -> None:
        # _validate_size() in exec_api indexes _MAX_PIXELS_BY_FAMILY by
        # family — a missing entry is a KeyError at generation time.
        for name, spec in MODELS.items():
            with self.subTest(model=name):
                self.assertIn(spec.family, _MAX_PIXELS_BY_FAMILY,
                              f"{name}: family {spec.family!r} missing a pixel cap")


if __name__ == "__main__":
    unittest.main()
