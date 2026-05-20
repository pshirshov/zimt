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
