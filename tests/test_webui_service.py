from __future__ import annotations

import asyncio
import base64
import os
import threading
import unittest
from dataclasses import asdict
from unittest.mock import patch

from fastapi import HTTPException

import zimt.webui.app as web_app
from zimt.generate import GenConfig
from zimt.models.registry import MODELS
from zimt.webui import exec_api, loader, prefetch
from zimt.webui.exec_api import ExecBody
from zimt.webui.state import CANCEL_EVENTS, Job, STATE


def _config_for(name: str) -> GenConfig:
    spec = MODELS[name]
    return GenConfig(
        spec=spec,
        cfg=spec.default_cfg,
        negative_prompt=spec.default_negative,
        height=spec.default_h,
        width=spec.default_w,
        steps=spec.default_steps,
    )


class StateCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._pipe = STATE.pipe
        self._g = STATE.g
        self._loading_model = STATE.loading_model
        self._jobs = dict(STATE.jobs)
        self._cancel_events = dict(CANCEL_EVENTS)
        STATE.pipe = None
        STATE.g = None
        STATE.loading_model = None
        STATE.jobs.clear()
        CANCEL_EVENTS.clear()

    def tearDown(self) -> None:
        STATE.pipe = self._pipe
        STATE.g = self._g
        STATE.loading_model = self._loading_model
        STATE.jobs.clear()
        STATE.jobs.update(self._jobs)
        CANCEL_EVENTS.clear()
        CANCEL_EVENTS.update(self._cancel_events)


class LoaderTests(StateCase):
    async def test_failed_load_clears_stale_config_and_same_model_reloads(self) -> None:
        # regression: stale STATE.g previously made same-model reload a no-op
        # after the pipe had been unloaded by a failed model switch.
        STATE.pipe = object()
        STATE.g = _config_for("z-image-turbo")

        def fail_after_unload(_name: str) -> None:
            STATE.pipe = None
            raise RuntimeError("simulated load failure")

        with patch.object(loader, "_do_load_sync", fail_after_unload):
            with self.assertRaises(HTTPException):
                await loader.load_model("pony-v6-xl")

        self.assertIsNone(STATE.pipe)
        self.assertIsNone(STATE.g)

        calls: list[str] = []

        def succeed(name: str) -> None:
            calls.append(name)
            STATE.pipe = object()
            STATE.g = _config_for(name)

        with patch.object(loader, "_do_load_sync", succeed):
            await loader.load_model("z-image-turbo")

        self.assertEqual(calls, ["z-image-turbo"])
        self.assertIsNotNone(STATE.pipe)
        self.assertEqual(STATE.g.spec.name if STATE.g else None, "z-image-turbo")


class ExecValidationTests(StateCase):
    async def test_invalid_settings_do_not_mutate_loaded_config(self) -> None:
        STATE.pipe = object()
        STATE.g = _config_for("z-image-turbo")
        before = (STATE.g.cfg, STATE.g.steps, STATE.g.width, STATE.g.height)

        response = await exec_api.api_exec(ExecBody(
            line="/steps 9999 /cfg nan /size 99999 1024 /res 1024x99999"
        ))

        self.assertEqual(response["job_ids"], [])
        self.assertEqual((STATE.g.cfg, STATE.g.steps, STATE.g.width, STATE.g.height), before)
        log = "\n".join(response["log"])
        self.assertIn("/steps: expected int", log)
        self.assertIn("/cfg: expected finite float", log)
        self.assertIn("/size:", log)
        self.assertIn("/res:", log)

    async def test_generation_receives_config_snapshot_from_enqueue_time(self) -> None:
        STATE.pipe = object()
        STATE.g = _config_for("z-image-turbo")
        captured: list[GenConfig] = []

        async def fake_run_job(
            _job: Job, _raw_prompt: str, _seed: int, _raw: bool, g: GenConfig,
        ) -> None:
            captured.append(g)

        with patch.object(exec_api, "run_job", fake_run_job):
            response = await exec_api.api_exec(ExecBody(
                line="/steps 31 /cfg 4.5 /size 1024 1024 /seed 7 test prompt"
            ))
            assert STATE.g is not None
            STATE.g.steps = 9
            STATE.g.cfg = 1.0
            await asyncio.sleep(0)

        self.assertEqual(len(response["job_ids"]), 1)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].steps, 31)
        self.assertEqual(captured[0].cfg, 4.5)

    async def test_generation_rejected_while_model_load_pending(self) -> None:
        STATE.pipe = object()
        STATE.g = _config_for("z-image-turbo")
        STATE.loading_model = "pony-v6-xl"

        response = await exec_api.api_exec(ExecBody(line="test prompt"))

        self.assertEqual(response["job_ids"], [])
        self.assertIn("generate: model pony-v6-xl is loading", response["log"])

    async def test_model_load_waits_for_active_jobs_before_unload(self) -> None:
        STATE.pipe = object()
        STATE.g = _config_for("z-image-turbo")
        STATE.jobs["queued"] = Job(id="queued", status="queued")
        CANCEL_EVENTS["queued"] = threading.Event()
        load_calls: list[str] = []

        def succeed(name: str) -> None:
            load_calls.append(name)
            STATE.pipe = object()
            STATE.g = _config_for(name)

        async def finish_job() -> None:
            await asyncio.sleep(0.05)
            STATE.jobs["queued"].status = "done"

        finisher = asyncio.create_task(finish_job())
        with patch.object(loader, "_do_load_sync", succeed):
            await loader.load_model("pony-v6-xl")
        await finisher

        self.assertEqual(load_calls, ["pony-v6-xl"])
        self.assertEqual(STATE.g.spec.name if STATE.g else None, "pony-v6-xl")


class DownloadIdentityTests(StateCase):
    async def test_prefetch_job_exposes_download_target_kind(self) -> None:
        # regression: model-tab downloads previously exposed only kind="download"
        # plus model name, so same-named base and LoRA targets shared UI state.
        with patch.object(prefetch, "_snapshot_download_sync", lambda _repo_id: None):
            job_id = await prefetch.prefetch_model("pixel-art-xl", kind="lora")
            await asyncio.sleep(0)

        job = STATE.jobs[job_id]
        self.assertEqual(job.kind, "download")
        self.assertEqual(job.target_kind, "lora")
        self.assertEqual(job.model, "pixel-art-xl")

    async def test_model_load_download_job_exposes_base_target_kind(self) -> None:
        load_calls: list[str] = []

        def succeed(name: str) -> None:
            load_calls.append(name)
            STATE.pipe = object()
            STATE.g = _config_for(name)

        with patch.object(loader, "_do_load_sync", succeed):
            await loader.load_model("z-image-turbo")

        download_jobs = [
            job for job in STATE.jobs.values()
            if job.kind == "download" and job.model == "z-image-turbo"
        ]
        self.assertEqual(load_calls, ["z-image-turbo"])
        self.assertEqual(len(download_jobs), 1)
        self.assertEqual(download_jobs[0].target_kind, "base")

    def test_job_serialization_preserves_target_kind_separate_from_queue_kind(self) -> None:
        job = Job(id="job-1", kind="download", target_kind="base", model="shared")
        payload = asdict(job)

        self.assertEqual(payload["kind"], "download")
        self.assertEqual(payload["target_kind"], "base")
        self.assertEqual(payload["model"], "shared")


class RegistryTests(unittest.TestCase):
    def test_added_adult_models_use_distinct_diffusers_repositories(self) -> None:
        expected = {
            "wai-nsfw-illustrious-v110": "John6666/wai-nsfw-illustrious-v110-sdxl",
            "wai-mature-illustrious": "John6666/wai-mature-illustrious-v20-sdxl",
            "pony-realism-v23": "John6666/pony-realism-v23-sdxl",
            "spicy-realism-nsfw-mix": "John6666/spicy-realism-nsfw-mix-v30-sdxl",
        }

        repos = {
            name: getattr(MODELS[name].load, "repo_id", None)
            for name in expected
        }

        self.assertEqual(repos, expected)
        self.assertEqual(len(set(repos.values())), len(repos))
        community_repos = [
            getattr(spec.load, "repo_id", None)
            for spec in MODELS.values()
            if getattr(spec.load, "repo_id", None) is not None
        ]
        self.assertEqual(len(community_repos), len(set(community_repos)))


class AuthTests(unittest.TestCase):
    def test_auth_accepts_bearer_and_basic_token(self) -> None:
        basic = base64.b64encode(b"zimt:secret").decode("ascii")
        with patch.dict(os.environ, {"ZIMT_AUTH_TOKEN": "secret"}, clear=False):
            self.assertTrue(web_app._request_permitted({
                "host": "127.0.0.1:8000",
                "origin": "http://127.0.0.1:8000",
                "authorization": "Bearer secret",
            }))
            self.assertTrue(web_app._request_permitted({
                "host": "127.0.0.1:8000",
                "origin": "http://127.0.0.1:8000",
                "authorization": f"Basic {basic}",
            }))
            self.assertFalse(web_app._request_permitted({
                "host": "127.0.0.1:8000",
                "origin": "http://127.0.0.1:8000",
                "authorization": "Bearer wrong",
            }))

    def test_origin_must_match_host_unless_allow_list_is_set(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(web_app._request_permitted({
                "host": "127.0.0.1:8000",
                "origin": "http://127.0.0.1:8000",
                "authorization": "",
            }))
            self.assertFalse(web_app._request_permitted({
                "host": "127.0.0.1:8000",
                "origin": "https://example.test",
                "authorization": "",
            }))

        with patch.dict(os.environ, {"ZIMT_ALLOWED_ORIGINS": "https://example.test"}, clear=True):
            self.assertTrue(web_app._request_permitted({
                "host": "127.0.0.1:8000",
                "origin": "https://example.test",
                "authorization": "",
            }))
