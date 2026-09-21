"""Behavioral/Blackbox-Group checks; no model weights or network required."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    Ideogram4Pipeline,
    Krea2Pipeline,
    QwenImage21Pipeline,
)
from PIL import Image

from zimt.backend import CancelledByUser, LocalBackend, RenderRequest
from zimt.generate import GenConfig, generate
from zimt.memory import MemStrategy
from zimt.models import vision
from zimt.models.registry import MODELS
from zimt.webui import exec_api, models_info
from zimt.webui.state import STATE


class RecordingPipeline:
    """Image producer that checks calls against the installed upstream API."""

    def __init__(self, pipeline_class):
        self.pipeline_class = pipeline_class
        self.scheduler = FlowMatchEulerDiscreteScheduler()
        self.calls = []
        self.expansions = []
        self.caption = json.dumps({
            "high_level_description": "A fox in snow",
            "compositional_deconstruction": {"background": "Snow", "elements": []},
        })

    def upsample_prompt(self, prompt, *, height, width, generator):
        self.expansions.append((prompt, height, width, generator.initial_seed()))
        return [self.caption]

    def __call__(self, **kwargs):
        inspect.signature(self.pipeline_class.__call__).bind(self, **kwargs)
        if self.pipeline_class is Ideogram4Pipeline:
            # Use upstream's validation as well: signature binding alone cannot
            # catch mutually exclusive guidance arguments or schedule lengths.
            validation = SimpleNamespace(vae_scale_factor=8, patch_size=2)
            Ideogram4Pipeline.check_inputs(
                validation, kwargs["prompt"], kwargs["height"], kwargs["width"],
                kwargs["num_inference_steps"], kwargs["guidance_scale"],
                kwargs["guidance_schedule"],
            )
        self.calls.append(kwargs)
        callback = kwargs.get("callback_on_step_end")
        if callback is not None:
            for step in range(kwargs["num_inference_steps"]):
                callback(self, step, 0, {})
        return SimpleNamespace(images=[Image.new("RGBA", (32, 32), (20, 30, 40, 100))])


class VisionModelsTests(unittest.TestCase):
    def setUp(self):
        self.device = patch("zimt.backend.DEVICE", "cpu")
        self.device.start()
        self.addCleanup(self.device.stop)

    def render(self, name, pipeline_class, prompt, cfg):
        spec = MODELS[name]
        g = GenConfig.from_spec(spec)
        g.cfg = cfg
        g.steps = 2
        pipe = RecordingPipeline(pipeline_class)
        progress = []
        req = RenderRequest(spec, g, prompt, "blur", 123,
                            lambda step, total: progress.append((step, total)))
        result = LocalBackend(pipe, spec).render(req)
        self.assertEqual(progress, [(1, 2), (2, 2)])
        self.assertEqual(pipe.calls[0]["generator"].initial_seed(), 123)
        return pipe, result

    def test_qwen_cfg_maps_to_true_cfg_and_disables_negative_at_one(self):
        for cfg, negative in ((1.0, None), (3.0, "blur")):
            with self.subTest(cfg=cfg):
                pipe, _ = self.render("qwen-image-2.1", QwenImage21Pipeline, "fox", cfg)
                self.assertEqual(pipe.calls[0]["true_cfg_scale"], cfg)
                self.assertEqual(pipe.calls[0]["negative_prompt"], negative)

    def test_krea_raw_and_turbo_keep_their_guidance_and_step_defaults(self):
        for name, cfg, steps in (("krea-2-turbo", 0.0, 8), ("krea-2-raw", 3.5, 52)):
            with self.subTest(model=name):
                g = GenConfig.from_spec(MODELS[name])
                self.assertEqual((g.cfg, g.steps), (cfg, steps))
                pipe, _ = self.render(name, Krea2Pipeline, "fox", cfg)
                self.assertEqual(pipe.calls[0]["guidance_scale"], cfg)

    def test_ideogram_expands_locally_and_supports_nondefault_steps(self):
        pipe, result = self.render("ideogram-4", Ideogram4Pipeline, "fox", 5.0)
        self.assertEqual(pipe.expansions, [("fox", 1024, 1024, 123)])
        self.assertEqual(pipe.calls[0]["prompt"], pipe.caption)
        self.assertEqual(result.metadata["revised_prompt"], pipe.caption)
        self.assertEqual(result.revised_prompt, pipe.caption)

    def test_ideogram_accepts_json_without_rewriting(self):
        caption = RecordingPipeline(Ideogram4Pipeline).caption
        pipe, _ = self.render("ideogram-4", Ideogram4Pipeline, caption, 7.0)
        self.assertEqual(pipe.expansions, [])
        self.assertEqual(pipe.calls[0]["prompt"], caption)

    def test_ideogram_rejects_unsupported_devices_before_loading(self):
        for device in ("cpu", "mps"):
            with self.subTest(device=device), self.assertRaisesRegex(ValueError, "NVIDIA CUDA"):
                vision.load_ideogram(device, MemStrategy())

    def test_ideogram_loads_on_xpu_and_nvidia_cuda(self):
        # Regression: the loader rejected XPU despite native NF4 support.
        for device in ("xpu", "xpu:0", "cuda", "cuda:0"):
            placements = []
            pipe = SimpleNamespace(to=placements.append)
            head = object()

            def load_head(repo_id, *, torch_dtype):
                self.assertEqual(repo_id, vision.IDEOGRAM_PROMPT_HEAD_REPO)
                return head

            def load_pipeline(repo_id, *, prompt_enhancer_head, torch_dtype):
                self.assertEqual(repo_id, vision.IDEOGRAM_REPO)
                self.assertIs(prompt_enhancer_head, head)
                return pipe

            with (self.subTest(device=device),
                  patch("torch.version.hip", None),
                  patch("diffusers.Ideogram4PromptEnhancerHead.from_pretrained", load_head),
                  patch("diffusers.Ideogram4Pipeline.from_pretrained", load_pipeline)):
                self.assertIs(vision.load_ideogram(device, MemStrategy(mode="off")), pipe)
                self.assertEqual(placements, [device])

    def test_ideogram_still_rejects_rocm(self):
        with patch("torch.version.hip", "7.0"), self.assertRaisesRegex(ValueError, "ROCm"):
            vision.load_ideogram("cuda", MemStrategy(mode="off"))

    def test_ideogram_still_rejects_loras_on_xpu(self):
        with self.assertRaisesRegex(ValueError, "LoRAs are not supported"):
            vision.load_ideogram("xpu", MemStrategy(mode="off"), (("adapter", 1.0),))

    def test_generation_keeps_qwen_alpha_and_png_provenance(self):
        spec = MODELS["qwen-image-2.1"]
        pipe = RecordingPipeline(QwenImage21Pipeline)
        with tempfile.TemporaryDirectory() as tmp:
            path = generate(LocalBackend(pipe, spec), GenConfig.from_spec(spec),
                            "fox", 123, raw=True, out_dir=tmp)
            with Image.open(path) as image:
                self.assertEqual(image.mode, "RGBA")
                self.assertEqual(image.getpixel((0, 0)), (20, 30, 40, 100))
                self.assertEqual(image.info["model"], "qwen-image-2.1")
                self.assertEqual(image.info["seed"], "123")

    def test_step_callback_still_cancels_new_models(self):
        spec = MODELS["qwen-image-2.1"]
        pipe = RecordingPipeline(QwenImage21Pipeline)

        def cancel(_step, _total):
            raise CancelledByUser()

        request = RenderRequest(spec, GenConfig.from_spec(spec), "fox", None, 42, cancel)
        with self.assertRaises(CancelledByUser):
            LocalBackend(pipe, spec).render(request)

    def test_every_registered_resolution_passes_web_validation(self):
        for spec in MODELS.values():
            with patch.object(STATE, "g", GenConfig.from_spec(spec)):
                for width, height, _label in spec.resolutions:
                    with self.subTest(model=spec.name, width=width, height=height):
                        self.assertIsNone(exec_api._validate_size(width, height))

    def test_qwen_rejects_dimensions_that_would_be_silently_rounded(self):
        with patch.object(STATE, "g", GenConfig.from_spec(MODELS["qwen-image-2.1"])):
            self.assertEqual(exec_api._validate_size(1040, 1024),
                             "width and height must be multiples of 32")

    def test_repl_render_also_rejects_qwen_dimension_rounding(self):
        # regression: the REPL bypasses web size validation.
        spec = MODELS["qwen-image-2.1"]
        g = GenConfig.from_spec(spec)
        g.width = 1040
        pipe = RecordingPipeline(QwenImage21Pipeline)
        req = RenderRequest(spec, g, "fox", None, 42)
        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            LocalBackend(pipe, spec).render(req)
        self.assertEqual(pipe.calls, [])

    def test_ideogram_install_status_requires_prompt_head(self):
        cache = {vision.IDEOGRAM_REPO: {"size_bytes": 100, "last_modified": 1.0}}
        with patch.object(models_info, "_scan_cache", lambda: cache):
            entry = next(e for e in models_info.models_info()["bases"] if e["name"] == "ideogram-4")
            self.assertFalse(entry["installed"])
            cache[vision.IDEOGRAM_PROMPT_HEAD_REPO] = {"size_bytes": 20, "last_modified": 1.0}
            entry = next(e for e in models_info.models_info()["bases"] if e["name"] == "ideogram-4")
            self.assertTrue(entry["installed"])
            self.assertEqual(entry["size_bytes"], 120)
