"""Remote (hosted-API) image backend: discovery, request mapping, decoding.

Live xAI generation can't be exercised in CI (the endpoint needs a funded
team), so these drive the backend against an httpx ``MockTransport`` and the
discovery/merge logic against mocked listings — covering the request shape,
b64 decoding, knob-dropping, error surfacing, and the env-gated on/off
behaviour.
"""

from __future__ import annotations

import base64
import io
import json
import os
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from zimt.backend import (
    RemoteBackend,
    RemoteBackendError,
    RenderRequest,
    _decode_image_response,
)
from zimt.generate import GenConfig, load_spec
from zimt.memory import DEFAULT as DEFAULT_MEM
from zimt.models.registry import MODELS
from zimt.models.remote import (
    XAI_ASPECT_RATIOS,
    discover_remote_models,
    merge_remote_into_registry,
)
from zimt.models.spec import ModelSpec, RemoteImageConfig


def _remote_spec(name: str = "grok-imagine-image-quality") -> ModelSpec:
    return ModelSpec(
        name=name,
        description="test remote model",
        repo_id="",
        family="remote",
        default_steps=0, default_cfg=0.0, default_negative="",
        resolutions=[], samplers={}, default_sampler="",
        remote=RemoteImageConfig(
            api_model=name,
            aspect_ratios=("1:1", "16:9", "9:16"),
            default_aspect_ratio="1:1",
            resolutions=("1k", "2k"),
            default_resolution="1k",
            price_per_image="$0.05",
        ),
    )


def _png_b64(color: tuple[int, int, int] = (10, 20, 30)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _backend_with(handler) -> RemoteBackend:
    spec = _remote_spec()
    client = httpx.Client(transport=httpx.MockTransport(handler),
                          base_url="https://api.x.ai/v1")
    return RemoteBackend(spec, client)


class RemoteRenderTests(unittest.TestCase):
    def test_render_builds_expected_payload_and_decodes_b64(self) -> None:
        captured: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            captured["auth"] = req.headers.get("authorization")
            return httpx.Response(200, json={
                "data": [{"b64_json": _png_b64()}],
                "revised_prompt": "a revised prompt",
            })

        backend = _backend_with(handler)
        g = GenConfig.from_spec(backend.spec)
        g.aspect_ratio = "16:9"
        g.resolution_tier = "2k"
        req = RenderRequest(spec=backend.spec, g=g, full_prompt="a cat",
                            negative=None, seed=7)
        result = backend.render(req)

        self.assertTrue(captured["path"].endswith("/images/generations"))
        self.assertEqual(captured["body"], {
            "model": "grok-imagine-image-quality",
            "prompt": "a cat",
            "n": 1,
            "response_format": "b64_json",
            "aspect_ratio": "16:9",
            "resolution": "2k",
        })
        self.assertIsInstance(result.image, Image.Image)
        self.assertEqual(result.image.size, (8, 8))
        self.assertEqual(result.revised_prompt, "a revised prompt")
        self.assertEqual(result.metadata["device"], "remote:xai")
        self.assertEqual(result.metadata["aspect_ratio"], "16:9")
        self.assertEqual(result.metadata["resolution"], "2k")
        self.assertEqual(result.metadata["revised_prompt"], "a revised prompt")

    def test_render_uses_spec_defaults_when_knobs_unset(self) -> None:
        captured: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(req.content)
            return httpx.Response(200, json={"data": [{"b64_json": _png_b64()}]})

        backend = _backend_with(handler)
        g = GenConfig.from_spec(backend.spec)  # aspect_ratio="1:1", tier="1k"
        req = RenderRequest(spec=backend.spec, g=g, full_prompt="x",
                            negative=None, seed=1)
        backend.render(req)
        self.assertEqual(captured["body"]["aspect_ratio"], "1:1")
        self.assertEqual(captured["body"]["resolution"], "1k")

    def test_render_ignores_local_only_knobs(self) -> None:
        """steps/cfg/sampler/negative/seed must never reach the API payload."""
        captured: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(req.content)
            return httpx.Response(200, json={"data": [{"b64_json": _png_b64()}]})

        backend = _backend_with(handler)
        g = GenConfig.from_spec(backend.spec)
        # Pollute with local knobs as if a stale config carried them.
        g.steps, g.cfg, g.sampler, g.clip_skip = 50, 9.0, "euler-a", 2
        g.negative_prompt = "ugly"
        req = RenderRequest(spec=backend.spec, g=g, full_prompt="x",
                            negative="ugly", seed=99)
        backend.render(req)
        for forbidden in ("steps", "num_inference_steps", "cfg", "guidance_scale",
                          "sampler", "scheduler", "negative_prompt", "seed",
                          "clip_skip"):
            self.assertNotIn(forbidden, captured["body"])

    def test_render_fires_on_step_once(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"b64_json": _png_b64()}]})

        backend = _backend_with(handler)
        g = GenConfig.from_spec(backend.spec)
        calls: list[tuple[int, int]] = []
        req = RenderRequest(spec=backend.spec, g=g, full_prompt="x",
                            negative=None, seed=1,
                            on_step=lambda s, t: calls.append((s, t)))
        backend.render(req)
        self.assertEqual(calls, [(1, 1)])

    def test_render_raises_with_provider_message_on_error(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={
                "code": "permission-denied",
                "error": "team has no credits",
            })

        backend = _backend_with(handler)
        g = GenConfig.from_spec(backend.spec)
        req = RenderRequest(spec=backend.spec, g=g, full_prompt="x",
                            negative=None, seed=1)
        with self.assertRaises(RemoteBackendError) as ctx:
            backend.render(req)
        msg = str(ctx.exception)
        self.assertIn("403", msg)
        self.assertIn("team has no credits", msg)

    def test_unload_closes_client(self) -> None:
        closed = {"v": False}

        class FakeClient:
            def close(self) -> None:
                closed["v"] = True

        backend = RemoteBackend(_remote_spec(), FakeClient())
        backend.unload()
        self.assertTrue(closed["v"])


class DecodeResponseTests(unittest.TestCase):
    def test_url_branch_fetches_image(self) -> None:
        png = Image.new("RGB", (4, 4), (1, 2, 3))
        buf = io.BytesIO(); png.save(buf, "PNG")
        body = {"data": [{"url": "https://cdn.example/img.png"}],
                "revised_prompt": "rp"}
        with patch("httpx.get", return_value=httpx.Response(200, content=buf.getvalue())):
            img, revised = _decode_image_response(body)
        self.assertEqual(img.size, (4, 4))
        self.assertEqual(revised, "rp")

    def test_empty_data_raises(self) -> None:
        with self.assertRaises(RemoteBackendError):
            _decode_image_response({"data": []})

    def test_item_without_image_raises(self) -> None:
        with self.assertRaises(RemoteBackendError):
            _decode_image_response({"data": [{"foo": "bar"}]})


class DiscoveryTests(unittest.TestCase):
    def _client(self, payload, status: int = 200) -> httpx.Client:
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.url.path.endswith("/image-generation-models")
            return httpx.Response(status, json=payload)
        return httpx.Client(transport=httpx.MockTransport(handler),
                            base_url="https://api.x.ai/v1")

    def test_parses_models_envelope(self) -> None:
        specs = discover_remote_models(self._client(
            {"models": [{"id": "grok-imagine-image"},
                        {"id": "grok-imagine-image-quality", "image_price": "$0.05"}]}))
        self.assertEqual(set(specs), {"grok-imagine-image", "grok-imagine-image-quality"})
        s = specs["grok-imagine-image-quality"]
        self.assertEqual(s.family, "remote")
        self.assertEqual(s.remote.api_model, "grok-imagine-image-quality")
        self.assertEqual(s.remote.aspect_ratios, XAI_ASPECT_RATIOS)
        self.assertIn("$0.05", s.description)

    def test_parses_data_envelope_and_bare_list(self) -> None:
        for payload in ({"data": [{"id": "m1"}]}, [{"id": "m1"}]):
            specs = discover_remote_models(self._client(payload))
            self.assertEqual(set(specs), {"m1"})

    def test_skips_items_without_id(self) -> None:
        specs = discover_remote_models(self._client(
            {"models": [{"name": "from-name"}, {"nope": 1}]}))
        self.assertEqual(set(specs), {"from-name"})

    def test_non_200_raises(self) -> None:
        with self.assertRaises(RemoteBackendError):
            discover_remote_models(self._client({"error": "blocked"}, status=403))

    def test_no_key_returns_empty(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("XAI_API_KEY", None)
            self.assertEqual(discover_remote_models(), {})


class MergeRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._removed = {n: m for n, m in MODELS.items() if m.family == "remote"}
        for n in list(self._removed):
            del MODELS[n]

    def tearDown(self) -> None:
        for n in [k for k, v in MODELS.items() if v.family == "remote"]:
            del MODELS[n]
        MODELS.update(self._removed)

    def _client(self, payload) -> httpx.Client:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)
        return httpx.Client(transport=httpx.MockTransport(handler),
                            base_url="https://api.x.ai/v1")

    def test_off_when_no_key(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("XAI_API_KEY", None)
            report = merge_remote_into_registry(self._client({"models": [{"id": "m1"}]}))
        self.assertEqual(report["added"], [])
        self.assertNotIn("m1", MODELS)

    def test_adds_and_is_idempotent(self) -> None:
        with patch.dict(os.environ, {"XAI_API_KEY": "k"}):
            client = self._client({"models": [{"id": "m1"}, {"id": "m2"}]})
            r1 = merge_remote_into_registry(client)
            self.assertEqual(set(r1["added"]), {"m1", "m2"})
            self.assertIn("m1", MODELS)
            r2 = merge_remote_into_registry(self._client({"models": [{"id": "m1"}]}))
        self.assertEqual(r2["added"], ["m1"])
        self.assertIn("m1", MODELS)
        self.assertNotIn("m2", MODELS)  # stale remote entry was pruned

    def test_local_collision_is_skipped(self) -> None:
        local_name = "pony-v6-xl"  # a builtin local model
        with patch.dict(os.environ, {"XAI_API_KEY": "k"}):
            report = merge_remote_into_registry(self._client({"models": [{"id": local_name}]}))
        self.assertEqual(report["added"], [])
        self.assertTrue(any("collide" in e for e in report["errors"]))
        self.assertNotEqual(MODELS[local_name].family, "remote")

    def test_listing_failure_reported_not_raised(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"error": "no credits"})
        client = httpx.Client(transport=httpx.MockTransport(handler),
                              base_url="https://api.x.ai/v1")
        with patch.dict(os.environ, {"XAI_API_KEY": "k"}):
            report = merge_remote_into_registry(client)
        self.assertEqual(report["added"], [])
        self.assertTrue(report["errors"])
        self.assertIn("no credits", report["errors"][0])


class GenConfigFromSpecTests(unittest.TestCase):
    def test_remote_defaults(self) -> None:
        g = GenConfig.from_spec(_remote_spec())
        self.assertEqual(g.aspect_ratio, "1:1")
        self.assertEqual(g.resolution_tier, "1k")
        self.assertEqual((g.width, g.height, g.steps, g.cfg), (0, 0, 0, 0.0))

    def test_local_defaults(self) -> None:
        g = GenConfig.from_spec(MODELS["pony-v6-xl"])
        self.assertEqual(g.aspect_ratio, "")
        self.assertEqual(g.resolution_tier, "")
        self.assertGreater(g.steps, 0)


class GenerateEndToEndTests(unittest.TestCase):
    """generate() orchestration over a RemoteBackend: PNG saved with the
    remote provenance fields and without the local-only diffusers fields."""

    def test_generate_writes_remote_png_metadata(self) -> None:
        import tempfile
        from zimt.generate import generate

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "data": [{"b64_json": _png_b64()}],
                "revised_prompt": "revised!",
            })

        backend = _backend_with(handler)
        g = GenConfig.from_spec(backend.spec)
        g.aspect_ratio = "9:16"
        with tempfile.TemporaryDirectory() as d:
            out = generate(backend, g, "a dog", 123, raw=True, out_dir=d)
            meta = Image.open(out).info

        self.assertEqual(meta["model"], "grok-imagine-image-quality")
        self.assertEqual(meta["device"], "remote:xai")
        self.assertEqual(meta["api_model"], "grok-imagine-image-quality")
        self.assertEqual(meta["aspect_ratio"], "9:16")
        self.assertEqual(meta["resolution"], "1k")
        self.assertEqual(meta["revised_prompt"], "revised!")
        self.assertEqual(meta["seed"], "123")
        # Local-only diffusers fields must be absent for a remote image.
        for absent in ("steps", "cfg", "sampler", "clip_skip", "width", "height"):
            self.assertNotIn(absent, meta)


class ReplSettingTests(unittest.TestCase):
    """/aspect + /quality handlers and the remote-ignored-knob guard."""

    def _apply(self, cmd: str, args: list[str], g: GenConfig) -> str:
        import contextlib
        from zimt.repl.main import _apply_setting
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _apply_setting(cmd, args, g)
        return buf.getvalue()

    def test_aspect_sets_valid_value(self) -> None:
        g = GenConfig.from_spec(_remote_spec())
        out = self._apply("/aspect", ["16:9"], g)
        self.assertEqual(g.aspect_ratio, "16:9")
        self.assertIn("16:9", out)

    def test_aspect_rejects_unknown_value(self) -> None:
        g = GenConfig.from_spec(_remote_spec())
        out = self._apply("/aspect", ["banana"], g)
        self.assertEqual(g.aspect_ratio, "1:1")  # unchanged
        self.assertIn("unknown", out)

    def test_quality_sets_valid_tier(self) -> None:
        g = GenConfig.from_spec(_remote_spec())
        self._apply("/quality", ["2k"], g)
        self.assertEqual(g.resolution_tier, "2k")

    def test_aspect_rejected_on_local_model(self) -> None:
        g = GenConfig.from_spec(MODELS["pony-v6-xl"])
        out = self._apply("/aspect", ["16:9"], g)
        self.assertIn("only applies to remote", out)

    def test_local_knobs_noop_on_remote(self) -> None:
        g = GenConfig.from_spec(_remote_spec())
        out = self._apply("/steps", ["40"], g)
        self.assertEqual(g.steps, 0)  # unchanged
        self.assertIn("no effect for remote", out)


class LoadSpecTests(unittest.TestCase):
    def test_load_spec_builds_remote_backend(self) -> None:
        with patch.dict(os.environ, {"XAI_API_KEY": "k"}):
            backend = load_spec(_remote_spec(), DEFAULT_MEM)
        self.assertIsInstance(backend, RemoteBackend)
        backend.unload()

    def test_load_spec_remote_without_key_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("XAI_API_KEY", None)
            with self.assertRaises(RemoteBackendError):
                load_spec(_remote_spec(), DEFAULT_MEM)


if __name__ == "__main__":
    unittest.main()
