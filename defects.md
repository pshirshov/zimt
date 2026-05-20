# zimt — Defect Ledger

Persistent defect ledger for review-loop findings.

Status: `[ ]` open · `[~]` under fix · `[x]` resolved

---

## PR-02

## [PR-02-D01] tqdm progress events from a non-owning concurrent download are still broadcast under the current slot owner's job id
**Status:** resolved
**Severity:** major
**Location:** `src/zimt/webui/downloads.py:103-118` (`ProgressTqdm._emit`), `src/zimt/webui/downloads.py:91-101` (`ProgressTqdm.close`); reachable from `src/zimt/webui/loader.py:82-83` and `src/zimt/webui/prefetch.py:77`.
**Description:** PR-02 scopes the *acquire/release* of the download slot by job id, but the tqdm hook (`_emit` and `close`) still reads a single ambient `_active_job_id` under the lock and attributes every tqdm event to it — regardless of which job's thread is actually emitting. Concrete scenario: prefetch (job P) owns the slot, then `load_model` enters and calls `_do_load_sync`, which calls `load_spec → diffusers → huggingface_hub`. Those tqdm bars run on the load's executor thread but find `_active_job_id == P` and write `download_file`, `download_n`, `download_total` (in `_emit`) and bump `download_files_done` (in `close`) on prefetch's Job. The model tab's prefetch row then shows the load's bytes, the load's file name, and an inflated file counter. Symmetrically, if load owns first and prefetch races, the prefetch's bytes flow into the load's Job. This violates the PR-02 acceptance criterion "does not overwrite another job's progress" — progress is misattributed, which is worse than the pre-PR-02 overwrite behavior because it is silent.
**Root cause:** ownership of the slot does not imply ownership of the tqdm events firing in any thread. HF tqdm is process-wide and `ProgressTqdm` carries no per-bar identity.
**Fix:** added module-scope `_owner_var: contextvars.ContextVar[Optional[str]]` and `@contextlib.contextmanager` `download_context(job_id)` in `src/zimt/webui/downloads.py:58-78`. `ProgressTqdm.__init__` captures `self._zimt_owner = _owner_var.get()` at `downloads.py:142` before `super().__init__`. `_emit` (downloads.py:167) and `close` (downloads.py:157) both early-return when `jid is None or jid != self._zimt_owner`, gating *all* mutation of `STATE.jobs.*` and `download_files_done` increments on per-bar ownership. Callers wrap their executor invocations: `loader.py:103-104` wraps `loop.run_in_executor(EXECUTOR, _do_load_sync, name)` in `with download_context(job.id):`; `prefetch.py:82-85` does the same for `_snapshot_download_sync`. Python's `BaseEventLoop.run_in_executor` copies the current `contextvars.Context`, so the var propagates into the worker thread. Net effect: a non-owning concurrent download's bars are silent no-ops rather than corrupting the owner's progress fields.

## [PR-02-D02] module docstring claims "progress is not broadcast" for non-owners but the implementation does broadcast (under the owner's id)
**Status:** resolved
**Severity:** major
**Location:** `src/zimt/webui/downloads.py:15-19` (module docstring).
**Description:** The new "Ownership invariant" paragraph asserts that a concurrent download that cannot acquire the slot "proceeds normally but its tqdm progress is not broadcast." That is false: per D01 its progress *is* broadcast, but under the wrong job. This makes the docstring an outright lie about behavior, which is worse than no docstring.
**Fix:** `src/zimt/webui/downloads.py:11-28` — the module docstring now describes both the slot-level ownership invariant *and* the per-bar context-var mechanism, including the explicit statement that "a non-owning concurrent download's bars are no-ops rather than corrupting the owner's progress fields." Docstring matches implementation once the D01 fix landed.

## [PR-02-D03] download_files_done counter on the owning job is inflated by close() events from non-owning concurrent downloads
**Status:** resolved
**Severity:** major
**Location:** `src/zimt/webui/downloads.py:91-101` (`ProgressTqdm.close`).
**Description:** `close()` unconditionally bumps `STATE.jobs[_active_job_id].download_files_done` by 1 every time any tqdm bar closes, regardless of which job's thread created the bar. The pre-PR-02 code had the same shape but in practice the "second" download would steal the slot, so its bars at least counted toward its own Job (still incorrect attribution, but symmetric). With PR-02's slot protection in place, the non-owning download's `close()` calls deterministically pollute the owner's `download_files_done` for the entire run. The diff thereby strictly worsens the visible symptom of an underlying defect.
**Root cause:** same as D01 — `close()` has no per-bar identity.
**Fix:** `src/zimt/webui/downloads.py:157` — the `close()` body now guards the `download_files_done += 1` increment behind `if jid is None or jid != self._zimt_owner: return`, identical to the `_emit` gate. Same root-cause fix as D01; `_zimt_owner` is captured on the bar at construction time and only the owning job's `close()` events update its counter.

## [PR-02-D04] a load that fails to acquire the slot signals "busy" only via a single log line; the Job stays in status "running" with no UI affordance distinguishing it
**Status:** resolved (mitigated; row-level "busy" state deferred to PR-06)
**Severity:** minor
**Location:** `src/zimt/webui/loader.py:82-84`.
**Description:** The PR-02 acceptance criterion says a non-acquiring load should "wait, queue, or report a defined busy state." The diff chose "proceed and log one line." The Job is created at `loader.py:71-78` with `status="running"`; the user sees two "running" download rows, one of which silently has no progress (post-D01-fix). The log line is informational, not state. A user without the log view cannot tell which row is the "real" progress owner.
**Fix:** intentionally deferred to PR-06 (Consolidate model-tab download state transitions), which is already scoped to define the row-level state machine. PR-02 retains the log line as the passive busy signal; PR-06 must add a Job-level flag or distinct status indicating "running but not progress owner" and surface it in the queue/model tab rendering. Recorded here so the deferral is auditable; no functional change in PR-02 beyond what D01 already accomplishes (silent vs. misattributed progress for the loser).

## [PR-02-D05] prefetch.py does not mirror loader.py's busy-slot log line
**Status:** resolved
**Severity:** minor
**Location:** `src/zimt/webui/prefetch.py:77`.
**Description:** `set_active_download(job.id)`'s return value is discarded in prefetch. Although `_PREFETCH_LOCK` serializes prefetches relative to each other, the lock does not include `load_model` — so a prefetch starting while a load owns the slot will silently fail to acquire ownership without surfacing the condition. This is asymmetric with `loader.py`, which does emit a log line.
**Fix:** `src/zimt/webui/prefetch.py:78-80` — assigns `owns_progress = set_active_download(job.id)` and emits `f"download progress slot busy; prefetch of {name} progress will not be broadcast"` via `emit_log` when False. Symmetric to the loader's log at `loader.py:84`. Both share the substring `"download progress slot busy"` for grep-ability.

## [PR-02-D06] DownloadOwnershipTests does not deterministically drain the released prefetch task and the module-global _active_job_id is not reset between tests
**Status:** resolved
**Severity:** minor
**Location:** `tests/test_webui_service.py:207-250` (`DownloadOwnershipTests`), `tests/test_webui_service.py` (`StateCase` setUp/tearDown — see `:30` onward for the class).
**Description:** After `release_prefetch.set()` at the end of the test, the test method returns immediately. The released prefetch task then asynchronously runs its `finally`, which calls `clear_active_download(prefetch_job_id)`. The test relies on `IsolatedAsyncioTestCase` cancel-and-await semantics during loop shutdown to drain that pending task before the next test starts. In practice this works today, but the module-global `_active_job_id` is not part of `StateCase`'s reset, so an order/shutdown change could leak slot ownership across test instances.
**Fix:** `tests/test_webui_service.py` `StateCase.setUp/tearDown` now snapshot and restore `downloads._active_job_id` alongside other module-globals. `DownloadOwnershipTests.test_model_load_waits_for_active_prefetch_download_owner` adds a bounded drain in its `finally`: after `release_prefetch.set()`, it polls `STATE.jobs[prefetch_job_id].status in {"done", "error"}` up to 50 × 20 ms, raising `RuntimeError` if the drain times out. `release_prefetch.set()` runs first so the worker is unblocked before draining begins.

## [PR-02-D07] set_active_download's idempotent re-acquire branch (slot already owned by the same job id) has no caller and no test
**Status:** resolved
**Severity:** nit
**Location:** `src/zimt/webui/downloads.py:55-58`.
**Description:** The `_active_job_id == job_id` branch in `set_active_download` is unreachable from the current callers (`loader.load_model` and `prefetch.prefetch_model` each call acquire once per job and release in the same `finally`). The branch is defensive code with no test and no caller; if a future caller did re-acquire, the docstring claims idempotence but there is no test to lock in that contract.
**Fix:** `tests/test_webui_service.py` — added `DownloadOwnershipApiTests(StateCase).test_set_active_download_is_idempotent_for_same_owner`. It exercises same-owner re-acquire (returns True twice), different-owner refusal (returns False without mutating state), non-owner clear (no-op), and owner clear (slot becomes None). Locks in the documented API contract; the branch remains in `downloads.py` and is now covered.

## [PR-02-D08] set_active_download is invoked outside the try block that calls clear_active_download in its finally, so an exception between acquire and try: leaks slot ownership
**Status:** resolved
**Severity:** minor
**Location:** `src/zimt/webui/loader.py:82-85`, `src/zimt/webui/prefetch.py:77-80`.
**Description:** Both callers acquire the slot (`set_active_download(job.id)`) and then conditionally emit a busy-log line (`await emit_log(...)`) before entering the `try:` whose `finally:` releases the slot (`clear_active_download(job.id)`). If anything between the acquire and the `try:` raises — concretely, the `await emit_log(...)` — the slot ownership leaks for the lifetime of the process, and no subsequent download can broadcast progress. `emit_log` in practice swallows its own errors so the window is narrow, but the structural invariant "every successful acquire is paired with a release in finally" is violated. Round-1 had a similar shape; round-2 widened the gap by inserting `emit_log` between acquire and try.
**Fix:** moved `owns_progress = set_active_download(job.id)` and the conditional busy-log emit *inside* the existing outer `try:` block in both files — `src/zimt/webui/loader.py:82-85` (try opens at 82; acquire at 83; busy-log at 85; finally at 120) and `src/zimt/webui/prefetch.py:77-80` (try opens at 77; acquire at 78; busy-log at 80; finally at 99). The owner-scoped `clear_active_download(job.id)` in `finally` is now guaranteed to run after every successful acquire even if `emit_log` raises.

## [PR-02-D09] `# type: ignore[return]` on download_context masks a diagnostic that pyright does not actually emit
**Status:** resolved
**Severity:** nit
**Location:** `src/zimt/webui/downloads.py:64`.
**Description:** `def download_context(job_id: str):  # type: ignore[return]` carries a pragma directed at a diagnostic code that pyright does not produce for a `@contextlib.contextmanager`-decorated generator with a one-shot `yield`. Round-2 review confirmed (by removing the comment and re-running pyright) that no `report*` diagnostic is suppressed — the pragma is dead code masquerading as required tooling appeasement.
**Fix:** `src/zimt/webui/downloads.py:64` — `# type: ignore[return]` removed. Function signature is now plain `def download_context(job_id: str):`. Pyright verified clean (only the pre-existing tqdm monkey-patch finding at `downloads.py:181` remains, unchanged).

## [PR-02-D10] download_context's contextvars do not propagate through loop.run_in_executor, making the per-bar gating non-functional in production
**Status:** resolved
**Severity:** major
**Location:** `src/zimt/webui/loader.py:103-104` (`with download_context(job.id): await loop.run_in_executor(EXECUTOR, _do_load_sync, name)`), `src/zimt/webui/prefetch.py:82-85` (analogous), `src/zimt/webui/downloads.py:142,167,157` (the `_zimt_owner` capture and gates that depend on propagation).
**Description:** PR-02-D01's fix relies on the assumption that the `contextvars.Context` set by `download_context(job_id)` propagates into the worker thread that `loop.run_in_executor` dispatches to. **This assumption is false.** Verified by repro:
```python
import asyncio, contextvars
v = contextvars.ContextVar("v", default=None)
def worker(): return v.get()
async def main():
    v.set("hello")
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, worker)
print(asyncio.run(main()))  # prints: None
```
The CPython 3.13 `BaseEventLoop.run_in_executor` (`asyncio/base_events.py`) implementation is `executor.submit(func, *args)` with **no** `contextvars.copy_context()` wrapping. Only `asyncio.Task` creation propagates context; `run_in_executor` does not. Documented behavior, not a CPython bug. As a consequence, `ProgressTqdm.__init__`'s `self._zimt_owner = _owner_var.get()` (downloads.py:142) always sees the default `None` inside the worker, and the gates in `_emit` (downloads.py:167: `if jid is None or jid != self._zimt_owner: return`) and `close` (downloads.py:157) always evaluate True → all real tqdm events from HF are silently dropped. Net production behavior:
- The D01 "cross-attribution" fix is non-functional; what actually happens is that **no** progress is broadcast at all from any real HF download, because every bar's `_zimt_owner` is None.
- The model tab's download rows therefore show no live byte/file counters during real downloads (regression from pre-PR-02 behavior, where the slot owner at least saw its own bars).
- The cross-attribution problem D01 claimed to fix is "fixed" only in the trivial sense that nothing is broadcast.
The defect was missed by three rounds of adversarial review because every PR-02 test patched the executor payload (`_do_load_sync`, `_snapshot_download_sync`) to a function that does **not** create real tqdm bars, so the broken gate was never exercised. The DownloadOwnershipTests test reads `current_download()` which derives from `_active_job_id` (the slot, not the bar), so it passes regardless.
PR-03's first execution surfaced this when its `_zimt_check_cancel` (which uses the same `_zimt_owner` capture) refused to fire; the subagent worked around it with a fallback to `_active_job_id`, but that workaround re-introduces D01 (cross-attribution across concurrent downloads would misattribute both progress *and* cancellation).
**Root cause:** `loop.run_in_executor` in CPython does not copy contextvars into the worker thread. PR-02's design assumed it does. The docstring at `downloads.py:23-28` perpetuates the misconception ("Python copies the current context into ``loop.run_in_executor`` worker threads"); this is false.
**Fix:** `src/zimt/webui/loader.py:14,116-119` and `src/zimt/webui/prefetch.py:20,97-103` now explicitly `ctx = contextvars.copy_context()` inside `with download_context(job.id):` and dispatch via `loop.run_in_executor(executor, lambda: ctx.run(func, ...))` (the lambda satisfies `run_in_executor`'s `Callable[[], R]` signature). `src/zimt/webui/downloads.py:21-30,72-78` — the module docstring and `download_context` docstring now correctly state that callers must use explicit `copy_context()` and dispatch via `ctx.run`; without that wrapper `_owner_var` will be `None` in the worker and all progress events will be silently dropped. `src/zimt/webui/downloads.py:166-173` — `ProgressTqdm._zimt_check_cancel` no longer falls back to `_active_job_id`; it returns immediately when `self._zimt_owner is None`, eliminating the D01 re-introduction the workaround would have caused. `tests/test_webui_service.py` — new `DownloadContextPropagationTests` class with two tests verifying (a) `_owner_var` set in asyncio context reaches the executor worker when dispatched via `ctx.run`, and (b) a `ProgressTqdm` constructed in the worker captures the correct `_zimt_owner`.

---

## PR-04

## [PR-04-D01] PR-04's unit normalization and download_files_done semantics do not match the real huggingface_hub snapshot_download flow
**Status:** resolved
**Severity:** major
**Location:** `src/zimt/webui/downloads.py:50-61` (`_normalize_unit`), `src/zimt/webui/downloads.py:204-209` (`close()` increment gate), `tests/test_webui_service.py` (`ProgressUnitTests.test_file_count_tqdm_event_…` constructing `unit='file'`).
**Description:** PR-04 promised three things: (a) detect file-count progress via tqdm's `unit` attribute and render it as `"N / T files"`; (b) gate `download_files_done` so it only increments on byte-bar `close()`, on the comment-asserted invariant that "byte-unit bars represent a single completed file in snapshot_download"; (c) prove both with backend tests. Verified against the installed `huggingface_hub`'s `_snapshot_download.py`:
1. The outer "Fetching N files" thread_map bar (created via `thread_map(_inner_hf_hub_download, ..., desc=tqdm_desc, tqdm_class=tqdm_class)`) inherits tqdm's default `unit='it'` — HF does **not** pass `unit='file'`. `_normalize_unit('it')` returns `"items"`, so the UI will render `"3 / 7"` instead of `"3 / 7 files"`. Acceptance criterion "File-count progress does not render as bytes" is satisfied (items branch is plain numbers), but the explicit "Treat HF `Fetching N files` progress as file-count progress" plan promise is unmet.
2. The per-file byte bars are created via `_AggregatedTqdm`, a duck-type wrapper that does **not** subclass `ProgressTqdm`. Its `__init__`/`update`/`close` calls never enter the patched hook. The only byte-unit `ProgressTqdm` instance in the snapshot flow is `bytes_progress`, which exists for the lifetime of the whole snapshot and closes **once** at the end. Therefore `download_files_done` increments to exactly 1 per snapshot regardless of how many files were fetched — not "files done" but "snapshot done". The comment claim "Only byte-unit bars represent a single completed file in snapshot_download" is factually wrong.
3. The new `ProgressUnitTests.test_file_count_tqdm_event_sets_unit_files_and_does_not_bump_files_done` constructs a bar with `unit='file'`, a literal HF does not emit. The test passes but does not exercise the production path. The plan's PR-04 verification matrix called for "One integration-style in-process test where fake snapshot download emits both `fetching files` and per-file byte progress" — this was not added, so the misclassification slipped through.
Net production effects:
- For a real `snapshot_download` of 7 files, the model tab will display the file-count bar as `"3 / 7"` (items branch) and the byte bar as `"x B / y B"` (correct). `download_files_done` will display `"files done: 1"` after the entire snapshot finishes, which contradicts what the user reads.
- The plan's invariant "File completion counters track completed files, not closed tqdm instances" is **not** satisfied — it still tracks closed tqdm instances, just of one specific unit. The pre-PR-04 behavior (`download_files_done == 2` after a snapshot) was wrong; the post-PR-04 behavior (`== 1`) is also wrong.
**Root cause:** PR-04's design assumed HF would expose unit categories via the `unit=` kwarg per-bar, mirroring how it exposes byte semantics on `bytes_progress`. In practice HF only sets `unit="B"` explicitly; the outer file-count bar uses tqdm's default. The `_AggregatedTqdm` indirection further breaks the per-file-close intuition.
**Fix:** `src/zimt/webui/downloads.py` — added `import re` and module-scope `_HF_FETCHING_FILES_RE = re.compile(r"^(?:\[dry-run\]\s*)?Fetching\b")` (broadened to match HF's four `tqdm_desc` variants: `"Fetching N files"`, `"Fetching ... files"`, `"[dry-run] Fetching N files"`, `"[dry-run] Fetching ... files"`). Replaced `_normalize_unit(raw_unit)` with `_classify(raw_unit, desc)`: `unit=='B'` → `"bytes"`; `unit in {'file','files'}` → `"files"` (compat with hand-built bars); else if desc matches the regex → `"files"`; else → `"items"`. The unit check runs first so a byte bar with a `"Fetching…"` desc still classifies as `"bytes"`. In `_emit`, when the bar classifies as `"files"`, `job.download_files_done = int(self.n or 0)` (overwrite from the file-count bar's current `n`). In `close()`, removed the `download_files_done += 1` increment entirely; `close()` now only enforces ownership and schedules a final emit. `tests/test_webui_service.py` — renamed `test_byte_tqdm_event_…` to assert `download_files_done == 0` before and after close (the bytes bar must not mutate it); rewrote `test_file_count_tqdm_event_…` to use HF's real `unit='it'`+`desc='Fetching 7 files'` and assert `download_files_done` mirrors `n` (3 after `update(3)`, unchanged after `close()`); added new `test_combined_files_bar_and_bytes_bar_drive_job_state_correctly` covering the file/byte/file interleaving and asserting unit flips, files_done mirroring, and close non-mutation. Note: PR-06 is responsible for handling the user-visible flicker between the file-count and byte counters when both bars interleave (single `download_n/total` slot in the Job).

---

## PR-05

## [PR-05-D01] frontend `_loraAddBtnState` lacks an incompatible-LoRA test
**Status:** resolved
**Severity:** nit
**Location:** `tests/model_download_state.test.js` (PR-05's new `_loraAddBtnState` tests).
**Description:** Plan §PR-05 "Suggested verification" explicitly enumerates three frontend state cases: "compatible installed, compatible uninstalled, **and incompatible** LoRA entries." PR-05 added tests for the first two and for the active-uninstalled case, but no test exercises the `!compatible` branch of `_loraAddBtnState`. The incompatible-title path is therefore untested by PR-05, even though the plan listed it.
**Fix:** `tests/model_download_state.test.js` — added `_loraAddBtnState renders incompatible title and disabled when compatible=false` test asserting `disabled: true` and `title` matching `/^incompatible:/`. `baseTags` passed as a list `["zimage"]` to match how production callers spread the Set via `[...baseTags]` before passing in (the helper internally calls `.join()`).

## [PR-05-D02] `test_run_job_reports_uninstalled_lora_error_with_clean_message` lacks the plan-required "no HF network call" sentinel
**Status:** resolved
**Severity:** nit
**Location:** `tests/test_webui_service.py` — `UninstalledLoraGuardTests.test_run_job_reports_uninstalled_lora_error_with_clean_message`.
**Description:** Plan §PR-05 "Acceptance criteria" line 219-220 mandates "Generation with an uninstalled LoRA fails before pipeline execution and before any Hugging Face network call." Tests 1 and 2 check this at the `_apply_lora_stack` boundary. The end-to-end `run_job` test only asserts the cleaned-up error string and does not also assert that `STATE.pipe.load_lora_weights.assert_not_called()` nor that `STATE.pipe.assert_not_called()` (i.e. the pipeline as a whole was never invoked). Adding both assertions closes the gap.
**Fix:** `tests/test_webui_service.py` — appended `STATE.pipe.load_lora_weights.assert_not_called()` and `STATE.pipe.assert_not_called()` to `test_run_job_reports_uninstalled_lora_error_with_clean_message`. Both pass because the LoRA precondition raises before sampler/pipe invocation.

---

## PR-06

## [PR-06-D01] idempotency scan accepts jobs whose cancel event is set but whose status has not yet transitioned to "canceled", so retries during the cancel window collapse into a doomed job
**Status:** resolved
**Severity:** minor
**Location:** `src/zimt/webui/prefetch.py:67-74` (the new idempotency scan).
**Description:** Cancellation is two-phased in this codebase. `CANCEL_EVENTS[jid].set()` flips a `threading.Event` synchronously when the user clicks cancel, but `Job.status` only transitions to `"canceled"` when the runner reaches its cancel checkpoint (the pre-`_PREFETCH_LOCK` check at `prefetch.py:78-86`, or the `DownloadCanceled` raise inside `download_context`). The new idempotency scan accepts `existing.status in {"queued", "running"}` and returns the existing job id. Reproducible sequence: user clicks download → asset job is `running`, cancel event clear → user clicks cancel (event set; status still `"running"`) → user immediately clicks download again → scan sees `"running"` → returns the doomed job_id → frontend shortly observes the job land in `"canceled"`, and the user's retry intent was silently discarded (a fresh job was not created). The user must click download a third time. The plan's PR-06 acceptance criterion "Retrying a failed or canceled download creates a new job with a distinct job id" is satisfied for jobs that have *already* landed in `"canceled"` but is violated for the cancel-pending window.
**Fix:** `src/zimt/webui/prefetch.py:67-83` — the idempotency scan now calls `CANCEL_EVENTS.get(existing.id)` and `continue`s past any matched job whose cancel event is set, with an inline comment explaining the cancel-pending exclusion. New `DownloadStateMachineTests.test_duplicate_request_during_cancel_pending_window_creates_new_job` in `tests/test_webui_service.py` parks a prefetch in `running`, sets its `CANCEL_EVENTS` entry directly (mirrors what `job_cancel` does), calls `prefetch_model` again, and asserts the second id differs synchronously and that `STATE.jobs` contains two entries.

## [PR-06-D02] `_PREFETCH_LOCK._loop = None` workaround lives only in `DownloadStateMachineTests.setUp` and is order-fragile
**Status:** resolved
**Severity:** minor
**Location:** `tests/test_webui_service.py` — `DownloadStateMachineTests.setUp` setting `prefetch._PREFETCH_LOCK._loop = None`.
**Description:** Confirmed reproducible against Python 3.13: `asyncio.Lock` only sets `_loop` as an instance attribute on the *contended* path (`_get_loop()` is called from `_waiters` machinery). After the first contended use, the attribute caches the event loop pointer. A subsequent contended use on a new `IsolatedAsyncioTestCase`-spawned loop raises `RuntimeError: <Lock [locked]> is bound to a different event loop`. The runner task dies silently ("Task exception was never retrieved"), the job never reaches a terminal state, and the test's drain logic times out. PR-06 worked around this only inside `DownloadStateMachineTests.setUp`, so the fix is not inherited by `DownloadCancelTests` or future classes whose alphabetical ordering interleaves with the suite (today's order happens to work — but a future class named `DownloadE*` / `DownloadM*` that contends the lock would not be covered). The right home is `StateCase.setUp`.
**Fix:** `tests/test_webui_service.py:51-55` — the `setattr(prefetch._PREFETCH_LOCK, "_loop", None)` reset now lives in `StateCase.setUp` (alongside the other module-global resets) with an explanatory comment. `DownloadStateMachineTests.setUp` was removed entirely — the class now inherits `StateCase.setUp` directly, so the reset applies suite-wide.

---

## PR-12

## [PR-12-D01] ConnectionManager.destroy() leaks lifecycle listeners on window/document/navigator
**Status:** resolved
**Severity:** minor
**Location:** `static/connection.js` — `_wireLifecycle` registers `visibilitychange`/`pagehide`/`pageshow`/`online`/`navigator.connection.change` handlers; `destroy()` does not remove any of them.
**Description:** Each `new ConnectionManager(...)` adds five global listeners; `destroy()` clears timers and closes sockets but does not call `removeEventListener` on any of them. Handlers are guarded by `if (this.destroyed) return;` so they don't act after destroy, but the listener nodes themselves accumulate on `navigator.connection`, `window`, and `document` across repeated create/destroy cycles (hot-reload, test fixtures, future SPA flows). Symptom is memory growth, not incorrect behaviour. Pre-existing pattern (the prior single-socket manager had the same gap) but more visible after PR-12 added the NetInfo listener.
**Fix:** `static/connection.js` — `_wireLifecycle` (lines ~735-840) now creates each handler as `this._onVisibilityChange / _onPageHide / _onPageShow / _onOnline / _onNetInfoChange` before registering, and `destroy()` (lines ~369-385) calls `removeEventListener` for each with the same conditional guards used at registration (window/document fallback for pagehide/pageshow; `navigator.connection && typeof addEventListener === "function"` for NetInfo).

## [PR-12-D02] connection_manager.test.js "MAX_LIVE_CONNECTIONS cap" test does not exercise the cap-rejection branch
**Status:** resolved
**Severity:** nit
**Location:** `tests/connection_manager.test.js` — `test("MAX_LIVE_CONNECTIONS cap honoured ...")`.
**Description:** The test only asserts `liveCount <= 3` at the end of a STALE-chain drive. By design the chain only produces 2 live connections at any moment (each STALE goes DEAD before the next replacement reaches NEW), so the cap-rejection branch in `_spawnConnection` (`if (this._liveConnections().length >= MAX_LIVE_CONNECTIONS) return null;`) is never executed. The assertion would pass even with `MAX_LIVE_CONNECTIONS = 999`.
**Fix:** `tests/connection_manager.test.js` — the cap test now passes `extraOpts: { staleGraceMs: 30_000 }` so three concurrent STALE connections can coexist without the oldest's grace timer expiring. The test drives ws0 to STALE (spawning ws1), forces ws1 to STALE via direct `_enterStale()`, calls `manager._spawnConnection()` to add ws2 (2 live < 3 cap, succeeds), forces ws2 to STALE, then calls `manager._spawnConnection()` once more — this invocation hits the `if (this._liveConnections().length >= MAX_LIVE_CONNECTIONS) return null` branch. The assertion `wsRegistry.length === 3` (not 4) verifies the rejection.

---

## PR-07

## [PR-07-D01] LoaderTests.test_failed_load_clears_stale_config_and_same_model_reloads expects HTTPException but load_model raises ModelLoadError
**Status:** open
**Severity:** major
**Location:** `tests/test_webui_service.py:80` (`with self.assertRaises(HTTPException)`), `src/zimt/webui/loader.py:142` (`raise ModelLoadError(repr(e)) from e`).
**Description:** Pre-existing test failure carried through PR-01..PR-06 and PR-12. The test asserts `HTTPException` is raised when `_do_load_sync` fails, but `loader.load_model` raises `ModelLoadError` instead. `HTTPException` is only constructed at the RPC boundary (`app.py:238` — `raise rpc_error(str(e))` for `ModelLoadError`), so the test is exercising the wrong layer. `python -m unittest discover -s tests` consistently reports this as the single failing test out of the M1 suite. Either the test should be rewritten to call the RPC handler `web_app._rpc_model_switch` and assert that it raises `_RpcError`/the WS-layer equivalent, or the test should assert `ModelLoadError` directly. The current assertion has been failing for the entire review-loop programme.
**Root cause:** API surface evolved (HTTPException → ModelLoadError at the loader layer, with the HTTP/RPC translation happening at the boundary), but the regression test wasn't updated.
**Suggested fix:** `tests/test_webui_service.py:80` — replace `with self.assertRaises(HTTPException):` with `with self.assertRaises(loader.ModelLoadError):`. The rest of the test (stale-config-cleared and same-model-reloads checks) is still valid.

## [PR-07-D02] pyright reports an unsuppressed reportAttributeAccessIssue on the hf_tqdm_mod.tqdm monkey-patch
**Status:** open
**Severity:** major
**Location:** `src/zimt/webui/downloads.py:246` (`hf_tqdm_mod.tqdm = ProgressTqdm`).
**Description:** `nix develop --command pyright src/zimt` reports `Cannot assign to attribute "tqdm" for class "ModuleType" — Attribute "tqdm" is unknown (reportAttributeAccessIssue)`. The assignment is intentional (we monkey-patch `huggingface_hub.utils.tqdm.tqdm` so HF downloads route through `ProgressTqdm`). The previous similar assignment at L256 (`setattr(mod, "tqdm", ProgressTqdm)`) uses `setattr` which pyright accepts; L246 uses attribute assignment and pyright doesn't. The error has been carried through every PR-01..PR-06 + PR-12 verification with the comment "only the pre-existing tqdm monkey-patch finding remains". This is the single non-clean diagnostic that blocks `flake.nix#checks.pyright` from being trivially extended into CI gating.
**Root cause:** module attribute assignment lacks a pyright-friendly form when the target attribute isn't declared in the module's stubs.
**Suggested fix:** `src/zimt/webui/downloads.py:246` — rewrite to `setattr(hf_tqdm_mod, "tqdm", ProgressTqdm)` to match the pattern already used at L256. Pyright accepts the `setattr` form for arbitrary modules. Verify with `nix develop --command pyright src/zimt` and expect 0 errors.

## [PR-07-D03] STATE.loading_model check-then-set across an await in loader.load_model permits two concurrent loads to race past the gate
**Status:** open
**Severity:** major
**Location:** `src/zimt/webui/loader.py:69-84` (`while STATE.loading_model is not None: await asyncio.sleep(0.25)` followed by `STATE.jobs[job.id] = job; STATE.loading_model = name`).
**Description:** Two concurrent `load_model("A")` and `load_model("B")` calls observe `STATE.loading_model is None` at the top, enter the while-loop, find it None again, and both proceed to create download Jobs and both set `STATE.loading_model`. The second assignment overwrites the first. Both subsequently `async with PIPE_LOCK:` serializes the actual `_do_load_sync` calls, so the load operations themselves don't corrupt each other — but `STATE.loading_model` is now "B" while "A" is also running; the model-state pill displays only "loading B", and the user's "A" job is hidden from the UI's loading affordance. The race is also reachable by an RPC pattern where a `/model A` exec is in flight and the user issues a second `/model B` before `load_model("A")` reaches `STATE.loading_model = name`. The check at L65 (`already loaded`) protects against same-model double-load but does not serialize different-model attempts.
**Root cause:** the busy-wait pattern without a lock is not an atomic check-and-set. The `await asyncio.sleep(0.25)` yields the loop, allowing another coroutine to pass the same check.
**Suggested fix:** `src/zimt/webui/loader.py:69-84` — introduce a dedicated `LOADER_LOCK = asyncio.Lock()` (or reuse `PIPE_LOCK` for the entire load_model body, accepting the queue effect) and acquire it around the check-and-set sequence:
```python
async with LOADER_LOCK:
    while STATE.loading_model is not None:
        await asyncio.sleep(0.25)
    STATE.loading_model = name
```
Then create the Job after the assignment so it can't be created and discarded. Backend tests: add a regression that fires two concurrent `load_model("A")` and `load_model("B")` and asserts both jobs are observed but `STATE.loading_model` was held by each in sequence (no overwrite).

## [PR-07-D04] outputs_cleanup follows symlinks via os.path.isfile and deletes their targets
**Status:** open
**Severity:** major
**Location:** `src/zimt/webui/app.py:407-423` (`_rpc_outputs_cleanup`).
**Description:** `outputs_cleanup` iterates `os.listdir(OUT_DIR)`, filters by `*.png` extension, and calls `os.remove(path)` for any entry where `os.path.isfile(path)` is True. `os.path.isfile` follows symlinks. A symlink in `OUT_DIR` named `foo.png` pointing to `/etc/important.png` (or any path the server user can write/delete) is treated as a regular file and unlinked by `os.remove`. The reverse-engineering attack: a process that can write into `OUT_DIR` (or a user who runs `ln -s /home/user/secret.png out/secret.png`) can cause `outputs_cleanup` to delete arbitrary files belonging to the server user. The output directory is normally writable only by the server process, but `OUT_DIR` defaults to `$PROJECT_ROOT/out` and is documented as user-managed (the README invites the user to put their own files in `out/fav`). Severity is **major** because the RPC requires only an authenticated session — there is no additional confirmation step beyond the frontend's `confirm()` dialog, and the operation is irreversible.
**Root cause:** `os.path.isfile` follows symlinks; the cleanup code does not check `os.path.islink` before deleting.
**Suggested fix:** `src/zimt/webui/app.py:413-419` — replace
```python
if not os.path.isfile(path):
    continue
try:
    os.remove(path)
```
with
```python
if os.path.islink(path) or not os.path.isfile(path):
    continue
try:
    # Use os.unlink which never follows symlinks; combined with the
    # islink guard, we refuse to delete anything but real regular files.
    os.unlink(path)
```
Backend test: create a symlink in a temp OUT_DIR pointing to a sentinel file outside OUT_DIR, invoke `_rpc_outputs_cleanup`, assert sentinel still exists and the symlink entry was skipped.

## [PR-07-D05] Background tasks created with asyncio.create_task are not retained and are vulnerable to GC mid-run
**Status:** open
**Severity:** major
**Location:** `src/zimt/webui/app.py:189` (`_gpu_stats_loop`), `src/zimt/webui/app.py:482` (per-ping `_watchdog`), `src/zimt/webui/app.py:526` (per-request `dispatch`), `src/zimt/webui/prefetch.py:146` (`_run()` task).
**Description:** Python's `asyncio.create_task` returns a `Task` whose only strong reference is the asyncio event loop's internal `_running_tasks` set — but CPython's implementation has historically warned that "If the application does not keep a reference to the task, it may be garbage collected at any time, even before it's done." (See `asyncio.create_task` docs, CPython 3.13.) Concrete consequences observed in the codebase: (a) `_gpu_stats_loop` is spawned once at startup; under memory pressure or during a GC cycle, the loop could be cancelled, silently terminating GPU stats updates. (b) `prefetch._run()` is spawned per download; if the asyncio loop's scheduler queue is congested, the task could be GC'd between `create_task` and its first `await async with _PREFETCH_LOCK:` execution — the download would never start, the job would stay queued forever, and the user would see "starting…" indefinitely. (c) Per-request `dispatch` tasks (L526) outlive the WS connection if the RPC handler is slow; the task continues sending to a closed socket (rpc.py's `_send` swallows the error). The watchdog (L482) is bounded by `PONG_TIMEOUT_S` so its GC risk is small.
**Root cause:** `asyncio.create_task` documents that the caller must retain the returned task to guarantee survival. None of these call sites do.
**Suggested fix:** Introduce a module-level `_BACKGROUND_TASKS: set[asyncio.Task] = set()` in `src/zimt/webui/state.py` and a helper:
```python
def spawn_background(coro: Coroutine) -> asyncio.Task:
    t = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(t)
    t.add_done_callback(_BACKGROUND_TASKS.discard)
    return t
```
Replace every `asyncio.create_task(...)` in `app.py`, `prefetch.py`, and `exec_api.py:330` (`asyncio.create_task(run_job(...))`) with `spawn_background(...)`. For per-request `dispatch` tasks the same pattern applies; the `done_callback` ensures the set doesn't grow unbounded. Add a backend regression test that spawns 1000 short-lived tasks under simulated GC pressure (call `gc.collect()` between spawns) and asserts all reach their `done` state.

## [PR-07-D06] empty-Origin fallback in _origin_allowed permits any non-browser client to bypass the CSRF check
**Status:** open
**Severity:** minor
**Location:** `src/zimt/webui/app.py:85-93` (`_origin_allowed`), `src/zimt/webui/app.py:96-101` (`_request_permitted`).
**Description:** `_origin_allowed("", host)` returns `True` because of the `if not origin: return True` guard at L86-87. Combined with `_auth_matches` returning `True` for empty `ZIMT_AUTH_TOKEN`, a non-browser client (curl, an MCP tool, a malicious local process) that omits both the `Origin` and `Authorization` headers will be permitted to call every RPC method. Browsers always send `Origin` for cross-origin requests so the actual CSRF surface for a browser-driven attack is intact — the issue is that the documentation in the auth chain implies an Origin check is enforced, but it is opt-in. If `ZIMT_AUTH_TOKEN` is set, the empty-origin path still bypasses origin validation as long as auth matches; an attacker who steals the token via any means (env leak, log capture) can replay it from anywhere without an Origin header.
**Root cause:** `_origin_allowed` was written to handle the "browser navigates directly to the page" case (no Origin), but applies the same rule to RPC requests where Origin should be required.
**Suggested fix:** `src/zimt/webui/app.py:85-93` — split the check by request type. For the `/ws` handshake and any RPC-bearing HTTP, require an Origin header. For the initial `GET /` and `/static/*` paths, allow missing Origin (a browser doesn't send Origin for top-level navigations). Concretely: extend `_request_permitted` to take a `require_origin: bool` flag and pass `True` from `ws_endpoint` and any RPC handler. Update tests `AuthTests.test_origin_must_match_host_unless_allow_list_is_set` to cover the missing-Origin → reject case for RPC.

## [PR-07-D07] _graceful_shutdown drains EXECUTOR but not _PREFETCH_EXECUTOR
**Status:** open
**Severity:** minor
**Location:** `src/zimt/webui/app.py:192-208` (`_graceful_shutdown`), `src/zimt/webui/prefetch.py:35` (`_PREFETCH_EXECUTOR`).
**Description:** SIGTERM handler sets every cancel event and calls `EXECUTOR.shutdown(wait=True, cancel_futures=True)`. `EXECUTOR` is the inference executor (state.py:29). The prefetch executor (`_PREFETCH_EXECUTOR` in prefetch.py:35) is a separate `ThreadPoolExecutor` and is **not** drained. A prefetch in flight during a systemd `TimeoutStopSec` will be SIGKILLed mid-`snapshot_download`, leaving partial files in the HF cache. HF's cache layout uses `.incomplete` suffixes for in-progress downloads which it normally cleans up on the next download attempt, so the post-mortem state is recoverable — but the in-flight Job's cancel event is set with no opportunity for the worker to honor it.
**Root cause:** prefetch was added in PR-06 era with its own executor for isolation from inference; the shutdown hook was not updated.
**Suggested fix:** `src/zimt/webui/app.py:207` — add `from .prefetch import _PREFETCH_EXECUTOR` and call `_PREFETCH_EXECUTOR.shutdown(wait=True, cancel_futures=True)` immediately after the existing `EXECUTOR.shutdown(...)`. Document in `prefetch.py` that callers must not retain executor references that survive the shutdown.

## [PR-07-D08] dataclasses.replace(STATE.g) shares the lora_stack list reference with the live config
**Status:** open
**Severity:** minor
**Location:** `src/zimt/webui/exec_api.py:325` (`g_snapshot = replace(STATE.g)`), `src/zimt/generate.py:64` (`lora_stack: list[tuple[str, float]] = field(default_factory=list)`).
**Description:** `dataclasses.replace(obj)` is a shallow copy — it constructs a new instance with the same field values. For mutable fields like `lora_stack: list[tuple[str, float]]`, the new instance holds a reference to the **same list**. The intent of the snapshot pattern (PR-01 / cross-cutting note) is that a `/lora` issued between enqueue and run does not affect the in-flight job. Today: enqueue captures `g_snapshot` with `g_snapshot.lora_stack is STATE.g.lora_stack`. A subsequent `/lora foo` mutates the list in place via `apply_lora_args`, and the in-flight job sees the mutation. The job then runs with the wrong adapter set. The test `ExecValidationTests.test_generation_receives_config_snapshot_from_enqueue_time` checks `steps` and `cfg` (scalar fields) but does not exercise `lora_stack` — the defect is uncovered.
**Root cause:** `replace` semantics are documented but easy to miss; tests covered only scalar fields.
**Suggested fix:** `src/zimt/webui/exec_api.py:325` — replace `g_snapshot = replace(STATE.g)` with `g_snapshot = replace(STATE.g, lora_stack=list(STATE.g.lora_stack))` (explicit list copy). Add a regression test:
```python
async def test_generation_snapshot_isolates_lora_stack_from_post_enqueue_mutation(self):
    STATE.g.lora_stack = [("pixel-art-xl", 1.0)]
    ...  # enqueue, then STATE.g.lora_stack.append(("ascii-art", 0.8))
    self.assertEqual(captured[0].lora_stack, [("pixel-art-xl", 1.0)])
```

## [PR-07-D09] jobs.run_job reports UninstalledLoraError as str(e) but other exceptions as repr(e), so a useful pipeline error message is hidden behind RuntimeError(…)
**Status:** open
**Severity:** minor
**Location:** `src/zimt/webui/jobs.py:93-99` (`except UninstalledLoraError as e: job.error = str(e)`), `src/zimt/webui/jobs.py:100-106` (`except Exception as e: job.error = repr(e)`).
**Description:** PR-05 added a dedicated branch for `UninstalledLoraError` that formats the error as the plain message (`str(e)`) so the UI shows "LoRA(s) not installed: …" cleanly. Every other pipeline error falls through to the generic catch which formats as `repr(e)`. A diffusers error like `RuntimeError("CUDA out of memory")` becomes `"RuntimeError('CUDA out of memory')"` in `job.error`. The user reads the noise rather than the message. The same pattern exists in `loader.py:137` (`job.error = repr(e)`). The frontend error popup at `app.js:345-349` simply prints `job.error` verbatim, so the user-facing text inherits the repr-vs-str inconsistency.
**Root cause:** `repr(e)` was the original convention; PR-05 introduced `str(e)` for one specific exception type without revisiting the general case.
**Suggested fix:** `src/zimt/webui/jobs.py:100-106` — change `job.error = repr(e)` to `job.error = f"{type(e).__name__}: {e}"` (class name + message, no quote-doubling). Same change at `loader.py:137`. The class-name prefix preserves the diagnostic info the original `repr(e)` carried, but renders the message readably. Update any test that pattern-matches on `repr(e)` formatting.

## [PR-07-D10] ProgressTqdm worker thread writes job.download_n / download_total / download_unit independently from the asyncio reader, allowing torn snapshots
**Status:** open
**Severity:** minor
**Location:** `src/zimt/webui/downloads.py:230-243` (`_emit` body), `src/zimt/webui/ws.py:35-36` (`emit_job` reads via `asdict(job)`).
**Description:** The tqdm worker thread mutates `job.download_file`, `job.download_unit`, `job.download_n`, `job.download_total`, `job.download_files_done` as separate Python attribute assignments. Between any two assignments, the asyncio loop can run an `emit_job(job)` (scheduled via `_schedule_emit` for a *previous* tqdm event) and read a partially-updated Job — e.g., `download_unit = "files"` (newly set) with `download_n = 95_232` and `download_total = 100_000` (stale byte-bar values from the previous emit). The frontend then formats `"95232 / 100000 files"`, which is gibberish. The GIL guarantees individual `setattr` is atomic but provides no multi-field consistency. Empirically the window is microsecond-scale, so frequency of observation depends on tqdm event rate. During a heavy snapshot with both file and byte bars firing, the bar-flicker PR-04 documented likely interacts with this — the visible "flicker" may not be only between unit categories but also between consistent and torn snapshots.
**Root cause:** no synchronization between the tqdm worker's mutation of `Job` fields and the asyncio loop's `asdict(job)` snapshot.
**Suggested fix:** introduce a per-Job `threading.Lock` (or a single module-level `_progress_lock = threading.Lock()` since at most one tqdm bar emits at a time per slot owner) and hold it across the field updates in `_emit` and around `asdict(job)` calls in `emit_job` when the job is `kind == "download"`. Alternative: replace the four scalar fields with a single `download_progress: ProgressSnapshot` dataclass that is atomically swapped under the lock, then `asdict` only sees consistent snapshots. Backend regression: spin up a tight loop that drives `ProgressTqdm.update` from a worker thread while another thread runs `asdict(job)` 10⁶ times; assert no snapshot has `unit == "files"` with `total > 99` (the byte-bar's total).

## [PR-07-D11] ws.broadcast crashes the whole broadcast loop on a non-JSON-serializable event payload rather than skipping the bad event
**Status:** open
**Severity:** minor
**Location:** `src/zimt/webui/ws.py:19-28` (`broadcast`).
**Description:** `broadcast` calls `payload = json.dumps(event)` once at the top, then loops over `STATE.clients`. The `except Exception` (L25) catches `ws.send_text` errors per-client. But `json.dumps(event)` is outside the try and outside the loop — if a caller passes an event whose `event["state"]` includes a non-serializable value (e.g., a `datetime` object, a `Path`, a numpy scalar), `json.dumps` raises `TypeError`, the call returns to the awaiting coroutine with an unhandled exception, and the broadcast fails for every client at once. There's no per-event isolation: a single bad `emit_state()` call would tear down the message stream for the entire session. The likelihood of hitting this in production is low (today's `state_dict()` only contains JSON-safe primitives), but every new field added to `Job` or `AppState` is one more chance to introduce a non-serializable value. The defect is also a regression-detection concern: adding a `datetime` field to `Job` would crash `emit_job` and the test suite would have to catch the exception via the broadcast path.
**Root cause:** `json.dumps` is outside the try/except.
**Suggested fix:** `src/zimt/webui/ws.py:19-28` — wrap `json.dumps(event)` in a try/except `TypeError, ValueError` that logs the bad event to stderr and returns early:
```python
async def broadcast(event: dict[str, Any]) -> None:
    try:
        payload = json.dumps(event, default=str)  # default=str as last-resort coercion
    except (TypeError, ValueError) as e:
        print(f"zimt: dropped non-serializable broadcast: {e!r} event={event!r}", file=sys.stderr)
        return
    ...
```
The `default=str` argument turns non-serializable values into their `str()` form rather than crashing — defensive enough that we don't need to chase every type that ends up in an event. Backend test: pass a `dataclass` field of type `set[int]` and assert the broadcast returns without raising.

## [PR-07-D12] PIPE_LOCK (asyncio.Lock) has the same cross-loop binding hazard as _PREFETCH_LOCK
**Status:** open
**Severity:** minor
**Location:** `src/zimt/webui/state.py:30` (`PIPE_LOCK = asyncio.Lock()`), `src/zimt/webui/jobs.py:48` (`async with PIPE_LOCK:`), `src/zimt/webui/loader.py:105` (`async with PIPE_LOCK:`).
**Description:** PR-06-D02 closed the cross-loop binding bug for `_PREFETCH_LOCK` by adding `setattr(prefetch._PREFETCH_LOCK, "_loop", None)` to `StateCase.setUp`. `PIPE_LOCK` has the same shape (module-global `asyncio.Lock`) but is not reset between tests. Today no `IsolatedAsyncioTestCase` test contends `PIPE_LOCK` from inside its body (the `LoaderTests.test_failed_load_…` test patches `_do_load_sync` to a sync function so the lock is only acquired briefly), so the bug is latent. The first test that genuinely contends `PIPE_LOCK` across two test methods will fail with `RuntimeError: <Lock [locked]> is bound to a different event loop`. The fix is mechanical and cheap; the only reason it isn't done is that no test has needed it yet.
**Root cause:** CPython `asyncio.Lock` caches `_loop` on the contended path.
**Suggested fix:** `tests/test_webui_service.py:51-55` — add `setattr(jobs_mod.PIPE_LOCK, "_loop", None)` next to the existing `_PREFETCH_LOCK._loop` reset in `StateCase.setUp`. Use `jobs_mod` (already imported) to reach the lock object. No production code change required — this is purely test-infrastructure hardening.

## [PR-07-D13] is_installed runs a full scan_cache_dir on every generation via _apply_lora_stack
**Status:** open
**Severity:** minor
**Location:** `src/zimt/webui/models_info.py:23-46` (`_scan_cache`), `src/zimt/webui/models_info.py:49-59` (`is_installed`), `src/zimt/generate.py:127-139` (`_apply_lora_stack` calls `is_installed` per LoRA in stack).
**Description:** `_apply_lora_stack` calls `is_installed(spec.repo_id)` once per LoRA in `g.lora_stack` (L136), and `is_installed` calls `_scan_cache()` which invokes `huggingface_hub.scan_cache_dir()` — a full walk of every cached repo's blob directory tree, stat()'ing every file. PR-05's notes acknowledge "few ms per repo" — for an HF cache with a dozen large models, this is in the 100-500 ms range. Every image generation pays this cost before the pipeline starts, multiplied by the number of LoRAs in the stack. For a `/many 16` invocation with a 3-LoRA stack, that's 16 × 3 = 48 cache scans for what is fundamentally the same answer (the cache state can't change between two generations in the same call). The PR-05 notes explicitly call out "No in-process caching was added because cache state can change between calls (a parallel prefetch can install a LoRA mid-session)" — true, but the *per-generation* call could safely cache for the duration of a single `run_job` invocation.
**Root cause:** correctness was prioritized over performance; the safer default (no cache) is paid by every generation.
**Suggested fix:** `src/zimt/generate.py:127-139` — hoist the cache scan out of the LoRA loop:
```python
if g.lora_stack:
    from .webui.models_info import _scan_cache
    cache = _scan_cache()
    missing: list[str] = []
    for name, _w in g.lora_stack:
        spec = LORAS.get(name)
        if spec is None:
            continue
        if spec.repo_id and spec.repo_id not in cache:
            missing.append(name)
    if missing:
        raise UninstalledLoraError(missing)
```
This calls `_scan_cache` once per `_apply_lora_stack` invocation rather than once per LoRA. The `_scan_cache` helper is module-private but used here as the same source of truth; promote it (or expose a `cache_snapshot()` accessor) so this isn't reaching into a private. No semantic change — both `is_installed(repo_id)` and `repo_id in cache` evaluate identically.

## [PR-07-D14] _apply_lora_stack silently ignores g.lora_stack for non-SDXL families, so a user can /lora foo on Z-Image and the LoRA is never applied or warned about
**Status:** open
**Severity:** minor
**Location:** `src/zimt/generate.py:123-126` (`if g.spec.family != "sdxl": return`).
**Description:** The early return at L123-126 says ZImagePipeline doesn't currently expose `set_adapters`, so we skip the entire LoRA application. But: (a) `lora_cmd.apply_lora_args` does **not** check the family before appending to the stack — it only checks `compatible_with` overlap. (b) `LoraSpec.family = "zimage"` is a valid value, and a user can legitimately add a Z-Image LoRA via the custom-add modal. (c) Once the LoRA is in `g.lora_stack`, the generation silently runs without it; the user sees the LoRA listed in `/state` output, the badge "active @ weight" rendered in the model tab, and the trigger tags surfaced, but the model never receives the adapter. (d) The PNG metadata at `generate._pnginfo:106` records `loras` = the stack contents, so the saved image claims to have used a LoRA it didn't. This is the kind of silent semantic divergence the user discovers only when they wonder why their LoRA had no effect.
**Root cause:** the early-skip was added defensively when Z-Image support landed; it predates LoRA-stack-tracking changes.
**Suggested fix:** `src/zimt/generate.py:123-126` — either (a) raise an explicit error like `raise ValueError(f"LoRA stacking is not supported for family={g.spec.family}; remove with /lora -")` so the user is forced to clear the stack, or (b) emit a one-time-per-config warning via the existing `print` channel ("note: family=zimage ignores LoRA stack") and clear `g.lora_stack` in place so the metadata doesn't lie. Option (a) is the fail-fast path consistent with the project's style; option (b) is the graceful-degradation path. Either way, the silent skip must end. Also update `lora_cmd.apply_lora_args` to reject Z-Image LoRAs against a non-SDXL base (or vice versa) at add time rather than letting the spec-compatibility check be the only gate.

## [PR-07-D15] MAX_QUEUE_SHOWN evicts the oldest entry which may be the running long-lived job rather than a completed one
**Status:** open
**Severity:** minor
**Location:** `static/app.js:11-12` (`let jobs = new Map(); ... const MAX_QUEUE_SHOWN = 30;`), `static/app.js:218-230` (`onJob`).
**Description:** `onJob` updates the `jobs` Map by id; when `jobs.size > 30`, it deletes `jobs.keys().next().value` — the **insertion-order oldest** entry. If the user has been running the session a while and the oldest entry happens to be the still-running download (e.g., a multi-gigabyte base model still pulling), that entry is evicted from the UI even though the job is alive. The user sees the download disappear from the queue, has no way to cancel it, and only the log line "downloading X" persists. The backend `STATE.jobs` still has it. The eviction policy should be FIFO-among-completed, not strict-FIFO.
**Root cause:** `Map.keys().next().value` always returns the first-inserted key; insertion is preserved for active jobs.
**Suggested fix:** `static/app.js:218-230` — change the cap-enforcement loop to scan for completed-status entries first:
```javascript
if (jobs.size > MAX_QUEUE_SHOWN) {
  // Evict the oldest completed entry first; if all are active, evict
  // the oldest active as a last resort (today's behaviour).
  let toEvict = null;
  for (const [id, job] of jobs) {
    if (job.status === "done" || job.status === "error" || job.status === "canceled") {
      toEvict = id;
      break;
    }
  }
  if (toEvict == null) toEvict = jobs.keys().next().value;
  jobs.delete(toEvict);
}
```
Frontend test: enqueue 31 download jobs with the first 5 in status `running` and the rest `done`; assert the running ones are all still present after the cap is hit.

## [PR-07-D16] _modelDlBtnState computes pct from download_n/total regardless of download_unit, so the model-tab button text flickers between file-percent and byte-percent during a snapshot
**Status:** open
**Severity:** minor
**Location:** `static/app.js:1163-1185` (`_modelDlBtnState`).
**Description:** The PR-04 unit-aware formatting was applied to `fmtDownloadCounter` (which renders the queue row's counter text) but **not** to `_modelDlBtnState` (which renders the model-tab download button). The button text computation at L1164-1167 is `pct = dlJob.download_total > 0 ? Math.round(100 * dlJob.download_n / dlJob.download_total) : null`. During a real `snapshot_download`, the file-count bar fires `n=3 total=7 unit="files"` → button reads "downloading 43%", then the byte bar fires `n=1024 total=100000000 unit="bytes"` → button reads "downloading 0%", then the file-count fires again with `n=4 total=7` → "downloading 57%", etc. The percentage oscillates between two incomparable scales. The task notes for PR-04 explicitly document this as a deferred follow-up that requires "a Job model refactor (new field for `progress_owner: bool`, splitting `download_n/total` into per-unit slots)". PR-07-D16 reflags it at the button-state site so it isn't lost.
**Root cause:** single `download_n/total` slot in `Job` shared by file-count and byte bars (PR-04 follow-up).
**Suggested fix:** add `download_n_bytes`, `download_total_bytes`, `download_n_files`, `download_total_files` separate slots to `Job`; update `_emit` to write only the slot matching the bar's unit. The button reads `download_n_files / download_total_files` when unit=="files" was last seen and bytes otherwise. Until that refactor lands, mitigate in `_modelDlBtnState` by suppressing the percentage when `dlJob.download_unit === "items"` or `""`, and adding a unit suffix when known: `text: "downloading " + (unit === "bytes" ? pct + "% bytes" : pct + "% files")`.

## [PR-07-D17] dlBtn.onclick sets textContent to "starting…" but doesn't restore it on the idempotent-collapse path
**Status:** open
**Severity:** minor
**Location:** `static/app.js:1104-1113` (`dlBtn.onclick`).
**Description:** Click handler sets `dlBtn.disabled = true; dlBtn.textContent = "starting…";` then awaits `wsRequest("model_download", { name, kind })`. If the backend collapses to an existing job (PR-06 idempotency), the response is `{ job_id: <existing> }` and no fresh `job` event fires for that id (the job was already running). The next time `renderBases()` runs (triggered by something else — a `state` event, a periodic refresh, or another job's progress), `_addCommonMeta` re-creates the button with the correct state. But until then, the button stays at "starting…". The user sees a button that lies about the actual state. The catch branch restores `disabled` (L1111) but not textContent.
**Root cause:** optimistic UI assumption that a fresh job event will arrive imminently; not true on the idempotent path.
**Suggested fix:** `static/app.js:1104-1113` — on the success path, explicitly trigger a re-render of just the bases/loras list after the await:
```javascript
dlBtn.onclick = async () => {
  dlBtn.disabled = true; dlBtn.textContent = "starting…";
  try {
    await wsRequest("model_download", { name: m.name, kind });
    appendLog(`download started: ${m.name}`);
    // The button will be re-rendered by the next 'job' event, but if
    // the request collapsed to an existing job there may be no fresh
    // event. Force a re-render to clear the optimistic "starting…".
    if (kind === "base") renderBases(); else renderLoras();
  } catch (e) {
    appendLog(`download: ${e.message}`, "error");
    dlBtn.disabled = false; dlBtn.textContent = m.installed ? "redownload" : "download";
  }
};
```
Frontend test: stub `wsRequest` to resolve without firing a job event, click the button, assert `dlBtn.textContent` is "downloading…" (per `_modelDlBtnState`) after the await resolves, not "starting…".

## [PR-07-D18] buildRestorePromptLine emits an invalid /lora line for multi-LoRA stacks (arity-1 parser drops all but the first)
**Status:** open
**Severity:** minor
**Location:** `static/restore_prompt.js:18` (`if (meta.loras) parts.push(\`/lora ${meta.loras.replace(/,/g, " ")}\`);`).
**Description:** The PNG metadata records the LoRA stack as `loras="pixel-art-xl:0.8,ascii-art:0.7"` (comma-separated, per `generate.py:106`). `restore_prompt.js` replaces commas with spaces and emits a single `/lora pixel-art-xl:0.8 ascii-art:0.7` token. The current `/lora` parser is arity-1 (per the task ledger's "/lora: arity-1 parser" note from commit 53e4f25) — it accepts exactly one token after `/lora`. The second LoRA `ascii-art:0.7` becomes a tail token that the parser interprets as part of the prompt (it doesn't start with `/`, so it's appended to `prompt_acc`). Result: the restored generation has only one LoRA in the stack and "ascii-art:0.7" smuggled into the prompt text. Reproducible by saving an image with two LoRAs, clicking the restore (↻) button on its thumbnail, and inspecting the textarea content.
**Root cause:** `restore_prompt.js` predates the arity-1 `/lora` parser refactor and assumes the old greedy/multi-arg form.
**Suggested fix:** `static/restore_prompt.js:18` — emit one `/lora` command per LoRA:
```javascript
if (meta.loras) {
  for (const entry of meta.loras.split(",").map(s => s.trim()).filter(Boolean)) {
    parts.push(`/lora ${entry}`);
  }
}
```
Frontend test `tests/restore_prompt.test.js` — add a case with `metadata.loras = "pixel-art-xl:0.8,ascii-art:0.7"` and assert the output contains two `/lora` commands.

## [PR-07-D19] run.sh hardcodes a Nix store hash in LD_LIBRARY_PATH that goes stale on any nixpkgs bump
**Status:** open
**Severity:** nit
**Location:** `run.sh:17` (`export LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib:/run/opengl-driver/lib`).
**Description:** The path `/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib` references a specific Nix store derivation hash. After any `nixpkgs` flake input update, the gcc store path changes and this path no longer exists. `LD_LIBRARY_PATH` to a non-existent prefix is harmless (the dynamic linker silently skips missing directories), but the project's intent — making libstdc++ available to the manylinux torch wheel — is silently defeated, and the user gets the same `ZE_RESULT_ERROR_UNINITIALIZED` failure mode the file's comment was written to prevent. The same path is also literal-grep'd in the M1 task notes (every PR's `LD_LIBRARY_PATH=...HF_HOME=...` verification command), so a bump cascades into stale test commands too.
**Root cause:** Nix store paths are content-addressed; hardcoding one couples the dev script to a specific nixpkgs revision.
**Suggested fix:** `run.sh:17` — derive the path dynamically:
```bash
GCC_LIB=$(nix eval --raw nixpkgs#gcc.cc.lib.outPath 2>/dev/null)/lib
export LD_LIBRARY_PATH="${GCC_LIB:+$GCC_LIB:}/run/opengl-driver/lib"
```
Or — preferred for the dev shell — move the export into the flake's `devShells.default.shellHook` so it tracks the same nixpkgs revision as the rest of the dev environment. Either approach removes the hash literal from the source tree.

