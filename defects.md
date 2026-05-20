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
