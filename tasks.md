# zimt — Task Ledger

Authoritative ledger for the requested model-tab download fixes and whole-codebase review. Detailed plan: `./docs/drafts/20260519-2333-model-download-review-loop-plan.md`.

Status: `[ ]` planned · `[~]` in progress · `[x]` done · `[!]` blocked

---

## Milestones (high-level)

- [~] **M1** — Resolve known model-tab/download correctness defects with regression tests.
- [ ] **M2** — Perform whole-codebase adversarial review and execute follow-up fixes for confirmed defects.

---

## Milestone 1 — PR breakdown

Detail in `./docs/drafts/20260519-2333-model-download-review-loop-plan.md`. One line per PR here; sub-task detail stays in the plan doc.

- [x] **PR-01** — Introduce base/LoRA download target identity.
- [ ] **PR-02** — Make download progress ownership job-scoped.
- [ ] **PR-03** — Add cancellation semantics for download jobs.
- [ ] **PR-04** — Normalize Hugging Face progress units.
- [ ] **PR-05** — Prevent untracked LoRA downloads during generation.
- [ ] **PR-06** — Consolidate model-tab download state transitions.

---

## Milestone 2 — PR breakdown

Detail in `./docs/drafts/20260519-2333-model-download-review-loop-plan.md`.

- [ ] **PR-07** — Whole-codebase review inventory and defect triage.
- [ ] **PR-08** — Backend concurrency and lifecycle follow-up fixes.
- [ ] **PR-09** — Frontend state and API-contract follow-up fixes.
- [ ] **PR-10** — Filesystem, configuration, and external-boundary follow-up fixes.
- [ ] **PR-11** — Final adversarial review and release verification.

---

## Cross-cutting architectural notes (locked)

- [x] Download jobs need asset identity separate from queue job type: `kind="download"` remains the queue discriminator; `target_kind` distinguishes base-model assets from LoRA assets. Lands in PR-01.
- [ ] Download progress ownership invariant — decide and encode in PR-02: one active Hugging Face download progress owner at a time unless the bridge gains independent per-job channels.
- [ ] Download cancellation invariant — decide and encode in PR-03: queued downloads must be cancelable before start; running downloads must enter a terminal canceled state at a controlled boundary if immediate upstream abort is unavailable.
- [ ] Progress unit invariant — decide and encode in PR-04: counters render only with units proven by the progress event.
- [ ] LoRA dependency invariant — decide and encode in PR-05: generation must not initiate implicit Hugging Face downloads.

---

## Completed

- **PR-01** (2026-05-19) — Download queue jobs now keep queue type and
  downloadable asset identity separate: download jobs still use
  `kind="download"`, and base/LoRA identity travels as `target_kind`. Base
  model load jobs set `target_kind="base"`; model-tab prefetch jobs set
  `target_kind` from the requested target kind; model-tab active download
  lookup filters by both `target_kind` and model name so same-named base and
  LoRA entries do not share progress state.
  Reproduction before fix:
  - `node --test tests/model_download_state.test.js` failed because
    `_downloadingByName("lora")` returned the active base download for the
    shared model name.
  - `LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib:/run/opengl-driver/lib OCL_ICD_VENDORS=/run/opengl-driver/etc/OpenCL/vendors HF_HOME=/home/pavel/work/safe/zimt/hf_cache HF_XET_HIGH_PERFORMANCE=1 .venv/bin/python -m unittest tests.test_webui_service.DownloadIdentityTests`
    failed with missing `Job.target_kind` / unexpected `target_kind`
    constructor argument.
  Verification:
  - `node --test tests/model_download_state.test.js` → pass, 1 test.
  - `LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib:/run/opengl-driver/lib OCL_ICD_VENDORS=/run/opengl-driver/etc/OpenCL/vendors HF_HOME=/home/pavel/work/safe/zimt/hf_cache HF_XET_HIGH_PERFORMANCE=1 .venv/bin/python -m unittest tests.test_webui_service.DownloadIdentityTests`
    → pass, 3 tests.
  - `node --test tests/*.test.js` → pass, 10 tests.
  Review:
  - Adversarial review reported no PR-01 defects. Residual low-risk gaps:
    serialization coverage uses `asdict` directly rather than `emit_job`, and
    frontend coverage checks lookup separation rather than full rendered
    same-name base/LoRA row state.
  Notes / constraints:
  - `pytest` is not installed in the virtualenv; backend tests were run via
    `unittest`.
  - `tests.test_webui_service` still has an unrelated pre-existing failure:
    `LoaderTests.test_failed_load_clears_stale_config_and_same_model_reloads`
    expects `HTTPException` while `loader.load_model()` raises
    `ModelLoadError`.
  - `nix develop --command pyright src/zimt` still reports an unrelated
    pre-existing `ModuleType.tqdm` monkey-patch error in
    `src/zimt/webui/downloads.py:120`.
