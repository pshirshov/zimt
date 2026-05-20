from __future__ import annotations

import asyncio
import base64
import contextvars
import os
import threading
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

import zimt.webui.app as web_app
import zimt.webui.models_info as models_info
from zimt import generate as generate_mod
from zimt.generate import GenConfig, UninstalledLoraError, _apply_lora_stack
from zimt.lora_cmd import apply_lora_args
from zimt.models.registry import MODELS
from zimt.webui import downloads, exec_api, jobs as jobs_mod, loader, prefetch
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
        self._active_job_id = downloads._active_job_id
        STATE.pipe = None
        STATE.g = None
        STATE.loading_model = None
        STATE.jobs.clear()
        CANCEL_EVENTS.clear()
        downloads._active_job_id = None
        # CPython asyncio.Lock caches _loop on the contended path; resetting
        # here makes per-test loop isolation actually hold for tests that
        # contend _PREFETCH_LOCK (each IsolatedAsyncioTestCase gets a fresh
        # event loop, and a cached pointer from a prior test would cross-bind).
        setattr(prefetch._PREFETCH_LOCK, "_loop", None)

    def tearDown(self) -> None:
        STATE.pipe = self._pipe
        STATE.g = self._g
        STATE.loading_model = self._loading_model
        STATE.jobs.clear()
        STATE.jobs.update(self._jobs)
        CANCEL_EVENTS.clear()
        CANCEL_EVENTS.update(self._cancel_events)
        downloads._active_job_id = self._active_job_id


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


class DownloadOwnershipTests(StateCase):
    async def test_model_load_waits_for_active_prefetch_download_owner(self) -> None:
        # regression: a model load previously overwrote the active prefetch
        # progress owner, then cleared the slot while the prefetch still ran.
        loop = asyncio.get_running_loop()
        prefetch_started = asyncio.Event()
        release_prefetch = threading.Event()
        observed: dict[str, str | None] = {}

        def blocked_snapshot(_repo_id: str) -> None:
            loop.call_soon_threadsafe(prefetch_started.set)
            if not release_prefetch.wait(timeout=5):
                raise RuntimeError("timed out waiting to release blocked prefetch")

        def succeed(name: str) -> None:
            current = downloads.current_download()
            observed["during_load_owner"] = current["id"] if current is not None else None
            STATE.pipe = object()
            STATE.g = _config_for(name)

        with (
            patch.object(prefetch, "_snapshot_download_sync", blocked_snapshot),
            patch.object(loader, "_do_load_sync", succeed),
        ):
            prefetch_job_id = await prefetch.prefetch_model("pixel-art-xl", kind="lora")
            try:
                await asyncio.wait_for(prefetch_started.wait(), timeout=1)
                current = downloads.current_download()
                self.assertIsNotNone(current)
                self.assertEqual(current["id"], prefetch_job_id)

                await loader.load_model("z-image-turbo")
                current = downloads.current_download()
                observed["after_load_owner"] = current["id"] if current is not None else None

                self.assertEqual(
                    observed,
                    {
                        "during_load_owner": prefetch_job_id,
                        "after_load_owner": prefetch_job_id,
                    },
                )
            finally:
                release_prefetch.set()
                # Drain the released prefetch task before returning so
                # module-global state (_active_job_id) is restored by its
                # finally block before the next test begins.
                for _ in range(50):
                    job = STATE.jobs.get(prefetch_job_id)
                    if job is not None and job.status in {"done", "error"}:
                        break
                    await asyncio.sleep(0.02)
                else:
                    raise RuntimeError("prefetch task did not finish after release")


class DownloadCancelTests(StateCase):
    async def test_queued_prefetch_canceled_before_run(self) -> None:
        # Two prefetch jobs (distinct assets, since duplicate (kind, name)
        # requests collapse to the running job per PR-06) queue against the
        # _PREFETCH_LOCK. We cancel the second while the first is still
        # parked inside snapshot_download, then release the first and assert
        # the second never invoked the downloader.
        loop = asyncio.get_running_loop()
        first_started = asyncio.Event()
        release_first = threading.Event()
        call_log: list[str] = []

        def first_snapshot(_repo_id: str) -> None:
            call_log.append("first")
            loop.call_soon_threadsafe(first_started.set)
            if not release_first.wait(timeout=5):
                raise RuntimeError("timed out waiting to release first prefetch")

        def second_snapshot(_repo_id: str) -> None:
            call_log.append("second")

        # Patch dispatches based on call count: first invocation blocks, second
        # would record execution (and must not run).
        invocations = {"n": 0}

        def dispatching_snapshot(repo_id: str) -> None:
            invocations["n"] += 1
            if invocations["n"] == 1:
                first_snapshot(repo_id)
            else:
                second_snapshot(repo_id)

        with patch.object(prefetch, "_snapshot_download_sync", dispatching_snapshot):
            first_id = await prefetch.prefetch_model("pixel-art-xl", kind="lora")
            second_id = await prefetch.prefetch_model("ascii-art", kind="lora")
            try:
                await asyncio.wait_for(first_started.wait(), timeout=1)
                # Cancel the queued second job before it acquires the lock.
                ev = CANCEL_EVENTS.get(second_id)
                self.assertIsNotNone(ev)
                assert ev is not None
                ev.set()
            finally:
                release_first.set()
                # Drain both tasks.
                for _ in range(100):
                    first_job = STATE.jobs.get(first_id)
                    second_job = STATE.jobs.get(second_id)
                    if (first_job is not None
                            and first_job.status in {"done", "error", "canceled"}
                            and second_job is not None
                            and second_job.status in {"done", "error", "canceled"}):
                        break
                    await asyncio.sleep(0.02)
                else:
                    raise RuntimeError("prefetch tasks did not finish")

        self.assertEqual(call_log, ["first"])
        self.assertEqual(STATE.jobs[second_id].status, "canceled")

    async def test_running_prefetch_canceled_at_tqdm_boundary(self) -> None:
        # Drive a ProgressTqdm bar manually inside the patched snapshot fn.
        # First update is fine; then we set the cancel event and the next
        # update should raise DownloadCanceled.
        from zimt.webui.downloads import DownloadCanceled

        # Ensure the tqdm patch is installed against the running loop.
        import sys
        loop = asyncio.get_running_loop()
        downloads.install(loop)
        hf_tqdm_mod = sys.modules["huggingface_hub.utils.tqdm"]
        ProgressTqdm = hf_tqdm_mod.tqdm

        observed_exc: dict[str, BaseException | None] = {"e": None}

        def driving_snapshot(_repo_id: str) -> None:
            bar = ProgressTqdm(total=100, desc="weights.bin")
            try:
                bar.update(10)
                ev = CANCEL_EVENTS[job_id]
                ev.set()
                bar.update(10)  # should raise DownloadCanceled
            except DownloadCanceled as e:
                observed_exc["e"] = e
                raise
            finally:
                bar.close()

        with patch.object(prefetch, "_snapshot_download_sync", driving_snapshot):
            job_id = await prefetch.prefetch_model("pixel-art-xl", kind="lora")
            for _ in range(100):
                job = STATE.jobs.get(job_id)
                if job is not None and job.status in {"done", "error", "canceled"}:
                    break
                await asyncio.sleep(0.02)
            else:
                raise RuntimeError("prefetch did not finish")

        self.assertIsInstance(observed_exc["e"], DownloadCanceled)
        self.assertEqual(STATE.jobs[job_id].status, "canceled")

    async def test_cancel_all_covers_download_jobs(self) -> None:
        # Park a prefetch on the _PREFETCH_LOCK, queue a generation job,
        # then call jobs_cancel_all and check both cancel events fire.
        loop = asyncio.get_running_loop()
        prefetch_started = asyncio.Event()
        release_prefetch = threading.Event()

        def blocked_snapshot(_repo_id: str) -> None:
            loop.call_soon_threadsafe(prefetch_started.set)
            if not release_prefetch.wait(timeout=5):
                raise RuntimeError("timed out waiting to release blocked prefetch")

        gen_job = Job(id="gen-1", kind="generate", status="queued")
        STATE.jobs[gen_job.id] = gen_job
        CANCEL_EVENTS[gen_job.id] = threading.Event()

        with patch.object(prefetch, "_snapshot_download_sync", blocked_snapshot):
            dl_id = await prefetch.prefetch_model("pixel-art-xl", kind="lora")
            try:
                await asyncio.wait_for(prefetch_started.wait(), timeout=1)
                result = await web_app._rpc_jobs_cancel_all({})
                self.assertGreaterEqual(result["canceled"], 2)
                self.assertTrue(CANCEL_EVENTS[dl_id].is_set())
                self.assertTrue(CANCEL_EVENTS[gen_job.id].is_set())
            finally:
                release_prefetch.set()
                for _ in range(100):
                    job = STATE.jobs.get(dl_id)
                    if job is not None and job.status in {"done", "error", "canceled"}:
                        break
                    await asyncio.sleep(0.02)


class DownloadStateMachineTests(StateCase):
    async def _drain(self, *job_ids: str, timeout: float = 5.0) -> None:
        terminal = {"done", "error", "canceled"}
        deadline_iters = int(timeout / 0.02)
        for _ in range(deadline_iters):
            jobs = [STATE.jobs.get(jid) for jid in job_ids]
            if all(j is not None and j.status in terminal for j in jobs):
                return
            await asyncio.sleep(0.02)
        raise RuntimeError(f"jobs did not finish: {job_ids}")

    async def test_duplicate_prefetch_request_returns_existing_job_id_when_running(self) -> None:
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()

        def blocked_snapshot(_repo_id: str) -> None:
            loop.call_soon_threadsafe(started.set)
            if not release.wait(timeout=5):
                raise RuntimeError("timed out waiting to release blocked prefetch")

        with patch.object(prefetch, "_snapshot_download_sync", blocked_snapshot):
            job_id_1 = await prefetch.prefetch_model("pixel-art-xl", kind="lora")
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                self.assertEqual(STATE.jobs[job_id_1].status, "running")
                job_id_2 = await prefetch.prefetch_model("pixel-art-xl", kind="lora")
                self.assertEqual(job_id_2, job_id_1)
                matching = [
                    j for j in STATE.jobs.values()
                    if j.model == "pixel-art-xl" and j.target_kind == "lora"
                ]
                self.assertEqual(len(matching), 1)
            finally:
                release.set()
                await self._drain(job_id_1)
        self.assertEqual(STATE.jobs[job_id_1].status, "done")

    async def test_duplicate_prefetch_request_returns_existing_job_id_when_queued(self) -> None:
        loop = asyncio.get_running_loop()
        first_started = asyncio.Event()
        release_first = threading.Event()

        # First snapshot for model A blocks; any subsequent invocation returns.
        invocations = {"n": 0}

        def dispatching_snapshot(_repo_id: str) -> None:
            invocations["n"] += 1
            if invocations["n"] == 1:
                loop.call_soon_threadsafe(first_started.set)
                if not release_first.wait(timeout=5):
                    raise RuntimeError("timed out releasing first prefetch")

        with patch.object(prefetch, "_snapshot_download_sync", dispatching_snapshot):
            first_id = await prefetch.prefetch_model("pixel-art-xl", kind="lora")
            try:
                await asyncio.wait_for(first_started.wait(), timeout=1)
                # Model B prefetch queues behind _PREFETCH_LOCK (status "queued").
                queued_id = await prefetch.prefetch_model("ascii-art", kind="lora")
                self.assertEqual(STATE.jobs[queued_id].status, "queued")
                # Duplicate request for the queued one must return same id.
                duplicate_id = await prefetch.prefetch_model("ascii-art", kind="lora")
                self.assertEqual(duplicate_id, queued_id)
                matching = [
                    j for j in STATE.jobs.values()
                    if j.model == "ascii-art" and j.target_kind == "lora"
                ]
                self.assertEqual(len(matching), 1)
            finally:
                release_first.set()
                await self._drain(first_id, queued_id)
        self.assertEqual(STATE.jobs[first_id].status, "done")
        self.assertEqual(STATE.jobs[queued_id].status, "done")

    async def test_retry_after_completion_creates_new_job(self) -> None:
        with patch.object(prefetch, "_snapshot_download_sync", lambda _r: None):
            first_id = await prefetch.prefetch_model("pixel-art-xl", kind="lora")
            await self._drain(first_id)
            self.assertEqual(STATE.jobs[first_id].status, "done")
            second_id = await prefetch.prefetch_model("pixel-art-xl", kind="lora")
            await self._drain(second_id)
        self.assertNotEqual(second_id, first_id)
        self.assertEqual(STATE.jobs[second_id].status, "done")

    async def test_retry_after_failure_creates_new_job(self) -> None:
        def failing(_repo_id: str) -> None:
            raise RuntimeError("synthetic prefetch failure")

        with patch.object(prefetch, "_snapshot_download_sync", failing):
            first_id = await prefetch.prefetch_model("pixel-art-xl", kind="lora")
            await self._drain(first_id)
            self.assertEqual(STATE.jobs[first_id].status, "error")
            second_id = await prefetch.prefetch_model("pixel-art-xl", kind="lora")
            await self._drain(second_id)
        self.assertNotEqual(second_id, first_id)
        self.assertEqual(STATE.jobs[second_id].status, "error")

    async def test_retry_after_cancellation_creates_new_job(self) -> None:
        # Set cancel before the runner acquires the lock so it lands "canceled
        # before start". The snapshot fn must never be invoked for the first job.
        call_log: list[str] = []

        def should_not_run(_repo_id: str) -> None:
            call_log.append("ran")

        with patch.object(prefetch, "_snapshot_download_sync", should_not_run):
            first_id = await prefetch.prefetch_model("pixel-art-xl", kind="lora")
            # The runner schedules immediately on the executor; flip cancel
            # before the lock is acquired by setting the event synchronously.
            ev = CANCEL_EVENTS.get(first_id)
            self.assertIsNotNone(ev)
            assert ev is not None
            ev.set()
            await self._drain(first_id)
            self.assertEqual(STATE.jobs[first_id].status, "canceled")
            self.assertEqual(call_log, [])

            second_id = await prefetch.prefetch_model("pixel-art-xl", kind="lora")
            await self._drain(second_id)
        self.assertNotEqual(second_id, first_id)
        self.assertEqual(STATE.jobs[second_id].status, "done")

    async def test_duplicate_request_during_cancel_pending_window_creates_new_job(self) -> None:
        # Reproduces the cancel-pending race: the user clicks cancel (event
        # set) but the runner has not yet transitioned status to "canceled".
        # A duplicate prefetch request during this window must create a fresh
        # job rather than collapsing onto the doomed one.
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()

        def blocked_snapshot(_repo_id: str) -> None:
            loop.call_soon_threadsafe(started.set)
            if not release.wait(timeout=5):
                raise RuntimeError("timed out waiting to release blocked prefetch")

        with patch.object(prefetch, "_snapshot_download_sync", blocked_snapshot):
            first_id = await prefetch.prefetch_model("pixel-art-xl", kind="lora")
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                self.assertEqual(STATE.jobs[first_id].status, "running")

                # Simulate user clicking cancel: set the event directly.
                # Status is still "running" — the runner hasn't reached its
                # checkpoint yet.
                CANCEL_EVENTS[first_id].set()
                self.assertEqual(STATE.jobs[first_id].status, "running")

                # The user's retry must land on a fresh job, not the doomed one.
                second_id = await prefetch.prefetch_model("pixel-art-xl", kind="lora")
                self.assertNotEqual(second_id, first_id)
                self.assertIn(second_id, STATE.jobs)
            finally:
                release.set()
                # Drain both jobs to terminal state for clean teardown.
                await self._drain(first_id, second_id, timeout=5.0)

        # First job is in a terminal state (done or canceled depending on
        # whether the runner reached a cancel checkpoint before completing).
        self.assertIn(STATE.jobs[first_id].status, {"done", "canceled"})
        self.assertIn(STATE.jobs[second_id].status, {"done", "error", "canceled"})
        # Two distinct entries in STATE.jobs for the same asset.
        matching = [
            j for j in STATE.jobs.values()
            if j.model == "pixel-art-xl" and j.target_kind == "lora"
        ]
        self.assertEqual(len(matching), 2)


class DownloadOwnershipApiTests(StateCase):
    def test_set_active_download_is_idempotent_for_same_owner(self) -> None:
        jid = "job-a"
        other = "job-b"
        self.assertTrue(downloads.set_active_download(jid))
        self.assertTrue(downloads.set_active_download(jid))  # idempotent
        self.assertFalse(downloads.set_active_download(other))  # different owner refused
        downloads.clear_active_download(other)  # non-owner clear is a no-op
        self.assertEqual(downloads._active_job_id, jid)
        downloads.clear_active_download(jid)  # owner clears
        self.assertIsNone(downloads._active_job_id)


class DownloadContextPropagationTests(StateCase):
    async def test_owner_var_reaches_executor_worker_via_copy_context(self) -> None:
        from concurrent.futures import ThreadPoolExecutor
        captured: dict[str, str | None] = {}

        def worker() -> None:
            captured["owner"] = downloads._owner_var.get()

        ex = ThreadPoolExecutor(max_workers=1)
        try:
            loop = asyncio.get_running_loop()
            with downloads.download_context("job-xyz"):
                ctx = contextvars.copy_context()
                await loop.run_in_executor(ex, ctx.run, worker)
        finally:
            ex.shutdown(wait=True)
        self.assertEqual(captured, {"owner": "job-xyz"})

    async def test_progress_tqdm_captures_owner_inside_copy_context_worker(self) -> None:
        # Prove that a ProgressTqdm bar created inside a ctx.run-dispatched
        # worker receives the correct _zimt_owner (not None).
        import sys
        loop = asyncio.get_running_loop()
        downloads.install(loop)
        hf_tqdm_mod = sys.modules["huggingface_hub.utils.tqdm"]
        ProgressTqdm = hf_tqdm_mod.tqdm

        captured: dict[str, str | None] = {}

        def worker() -> None:
            bar = ProgressTqdm(total=10, desc="test.bin")
            captured["owner"] = bar._zimt_owner
            bar.close()

        from concurrent.futures import ThreadPoolExecutor
        ex = ThreadPoolExecutor(max_workers=1)
        try:
            with downloads.download_context("job-abc"):
                ctx = contextvars.copy_context()
                await loop.run_in_executor(ex, ctx.run, worker)
        finally:
            ex.shutdown(wait=True)
        self.assertEqual(captured, {"owner": "job-abc"})


class ProgressUnitTests(StateCase):
    def _progress_tqdm(self):
        import sys
        # Install with a fresh, immediately-closed loop so _schedule_emit
        # no-ops (closed-loop branch); we only want STATE mutation here.
        loop = asyncio.new_event_loop()
        loop.close()
        downloads.install(loop)
        return sys.modules["huggingface_hub.utils.tqdm"].tqdm

    def _setup_job(self, jid: str) -> Job:
        job = Job(id=jid, kind="download", target_kind="base", model="m")
        STATE.jobs[jid] = job
        CANCEL_EVENTS[jid] = threading.Event()
        downloads.set_active_download(jid)
        return job

    def test_byte_tqdm_event_sets_unit_bytes_and_does_not_change_files_done_on_close(self) -> None:
        ProgressTqdm = self._progress_tqdm()
        job = self._setup_job("byte-job")
        try:
            with downloads.download_context(job.id):
                bar = ProgressTqdm(total=12345, desc="model.safetensors", unit="B")
                bar.update(100)
                self.assertEqual(STATE.jobs[job.id].download_unit, "bytes")
                self.assertEqual(STATE.jobs[job.id].download_n, 100)
                self.assertEqual(STATE.jobs[job.id].download_total, 12345)
                self.assertEqual(STATE.jobs[job.id].download_files_done, 0)
                bar.close()
                # close() no longer mutates download_files_done; it stays 0.
                self.assertEqual(STATE.jobs[job.id].download_files_done, 0)
        finally:
            downloads.clear_active_download(job.id)

    def test_file_count_tqdm_event_classified_via_desc_and_drives_files_done(self) -> None:
        # HF emits unit='it' (tqdm default) for the outer "Fetching N files"
        # thread_map bar.  Classification must use desc, not unit.
        ProgressTqdm = self._progress_tqdm()
        job = self._setup_job("file-job")
        try:
            with downloads.download_context(job.id):
                bar = ProgressTqdm(total=7, desc="Fetching 7 files", unit="it")
                bar.update(3)
                self.assertEqual(STATE.jobs[job.id].download_unit, "files")
                self.assertEqual(STATE.jobs[job.id].download_n, 3)
                self.assertEqual(STATE.jobs[job.id].download_total, 7)
                self.assertEqual(STATE.jobs[job.id].download_files_done, 3)
                bar.close()
                # close() does not mutate download_files_done.
                self.assertEqual(STATE.jobs[job.id].download_files_done, 3)
        finally:
            downloads.clear_active_download(job.id)

    def test_combined_files_bar_and_bytes_bar_drive_job_state_correctly(self) -> None:
        # Models the real HF snapshot_download interleaving: outer
        # "Fetching N files" bar (unit='it', desc matches regex) plus
        # an aggregate bytes bar (unit='B').
        ProgressTqdm = self._progress_tqdm()
        job = self._setup_job("combined-job")
        try:
            with downloads.download_context(job.id):
                outer = ProgressTqdm(total=7, desc="Fetching 7 files", unit="it")
                outer.update(3)
                self.assertEqual(STATE.jobs[job.id].download_unit, "files")
                self.assertEqual(STATE.jobs[job.id].download_n, 3)
                self.assertEqual(STATE.jobs[job.id].download_total, 7)
                self.assertEqual(STATE.jobs[job.id].download_files_done, 3)

                inner = ProgressTqdm(total=100_000, desc="model.safetensors", unit="B")
                inner.update(1024)
                self.assertEqual(STATE.jobs[job.id].download_unit, "bytes")
                self.assertEqual(STATE.jobs[job.id].download_n, 1024)
                # bytes bar must NOT change download_files_done — still 3 from outer.
                self.assertEqual(STATE.jobs[job.id].download_files_done, 3)

                inner.close()
                self.assertEqual(STATE.jobs[job.id].download_files_done, 3)

                outer.update(4)
                self.assertEqual(STATE.jobs[job.id].download_files_done, 7)
                outer.close()
                # close() never mutates download_files_done.
                self.assertEqual(STATE.jobs[job.id].download_files_done, 7)
        finally:
            downloads.clear_active_download(job.id)

    def test_unknown_unit_tqdm_event_normalizes_to_items_and_does_not_bump_files_done(self) -> None:
        ProgressTqdm = self._progress_tqdm()
        job = self._setup_job("item-job")
        try:
            with downloads.download_context(job.id):
                bar = ProgressTqdm(total=5, desc="something", unit="it")
                bar.update(2)
                self.assertEqual(STATE.jobs[job.id].download_unit, "items")
                bar.close()
                self.assertEqual(STATE.jobs[job.id].download_files_done, 0)
        finally:
            downloads.clear_active_download(job.id)


class UninstalledLoraGuardTests(StateCase):
    def _sdxl_config(self) -> GenConfig:
        return _config_for("pony-v6-xl")

    def test_generation_with_uninstalled_lora_raises_before_load_lora_weights(self) -> None:
        pipe = MagicMock()
        pipe._zimt_loras_loaded = set()
        g = self._sdxl_config()
        g.lora_stack = [("pixel-art-xl", 0.8)]
        with patch.object(models_info, "is_installed", return_value=False):
            with self.assertRaises(UninstalledLoraError) as cm:
                _apply_lora_stack(pipe, g)
        self.assertIn("pixel-art-xl", cm.exception.names)
        pipe.load_lora_weights.assert_not_called()
        pipe.set_adapters.assert_not_called()

    def test_generation_with_installed_lora_proceeds_to_load_lora_weights(self) -> None:
        pipe = MagicMock()
        pipe._zimt_loras_loaded = set()
        g = self._sdxl_config()
        g.lora_stack = [("pixel-art-xl", 0.8)]
        with patch.object(models_info, "is_installed", return_value=True):
            _apply_lora_stack(pipe, g)
        pipe.load_lora_weights.assert_called_once()
        pipe.set_adapters.assert_called_once()

    def test_lora_cmd_add_rejects_uninstalled_lora_and_does_not_append_to_stack(self) -> None:
        base = MODELS["pony-v6-xl"]
        stack: list[tuple[str, float]] = []
        with patch.object(models_info, "is_installed", return_value=False):
            log = apply_lora_args(stack, ["pixel-art-xl"], base)
        self.assertEqual(stack, [])
        self.assertTrue(any("not installed" in line for line in log),
                        f"expected 'not installed' message in log, got {log!r}")

    async def test_run_job_reports_uninstalled_lora_error_with_clean_message(self) -> None:
        STATE.pipe = MagicMock()
        STATE.pipe._zimt_loras_loaded = set()
        STATE.g = self._sdxl_config()
        STATE.g.lora_stack = [("pixel-art-xl", 0.8)]
        # Short-circuit _ensure_sampler (would otherwise touch scheduler.config).
        STATE.pipe._zimt_sampler = STATE.g.spec.default_sampler

        job = Job(id="gen-uninstalled", kind="generate", status="queued")
        STATE.jobs[job.id] = job
        CANCEL_EVENTS[job.id] = threading.Event()

        with patch.object(models_info, "is_installed", return_value=False):
            await jobs_mod.run_job(job, "test prompt", 42, False, STATE.g)

        self.assertEqual(job.status, "error")
        self.assertIsNotNone(job.error)
        expected = str(UninstalledLoraError(["pixel-art-xl"]))
        self.assertEqual(job.error, expected)
        # Must be the plain message, not the repr-formatted one.
        self.assertFalse(job.error.startswith("UninstalledLoraError("),
                         f"error should be str(e), not repr(e): {job.error!r}")
        STATE.pipe.load_lora_weights.assert_not_called()
        STATE.pipe.assert_not_called()  # the pipeline itself was never invoked


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
