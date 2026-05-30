from __future__ import annotations

import unittest

from zimt.models.loras import LORAS
from zimt.models.registry import MODELS
from zimt.models.spec import LORA_FAMILIES
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


class LoraRegistryCase(unittest.TestCase):
    def test_every_lora_is_well_formed(self) -> None:
        # A LoRA only loads on a base whose family supports LoRAs AND whose
        # compatibility_tags overlap the adapter's compatible_with. Assert
        # each adapter targets a LoRA-capable family and could match at least
        # one registered base — else it's dead weight in the UI.
        base_tags_by_family: dict[str, set[str]] = {}
        for spec in MODELS.values():
            base_tags_by_family.setdefault(spec.family, set()).update(
                spec.compatibility_tags)
        for name, lora in LORAS.items():
            with self.subTest(lora=name):
                self.assertEqual(lora.name, name)
                self.assertTrue(lora.repo_id, f"{name}: empty repo_id")
                self.assertIn(lora.family, LORA_FAMILIES,
                              f"{name}: family {lora.family!r} can't apply LoRAs")
                reachable = base_tags_by_family.get(lora.family, set())
                self.assertTrue(
                    any(t in reachable for t in lora.compatible_with),
                    f"{name}: compatible_with {lora.compatible_with} matches no "
                    f"{lora.family} base (tags {reachable})")


if __name__ == "__main__":
    unittest.main()
