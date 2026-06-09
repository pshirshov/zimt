const $ = (id) => document.getElementById(id);

// ---------- state ----------
let state = null;
let outputs = [];
let currentTab = "all";
let currentPage = 1;
let hasMore = false;
let totalCount = 0;
let modalEntry = null;
let jobs = new Map();        // jobId -> job (preserve insertion order via Map)
const MAX_QUEUE_SHOWN = 30;

// Prompt history + favourites + globals are server-side, scoped to the
// active profile (see "profiles" section below). `promptCache` mirrors the
// active profile's prompts as `[{text, fav}]`, newest-first (server order).
// `LS.recents()` returns a copy of that cache; `LS.history()` returns the
// text array in display order (favs first) for arrow-key nav. Mutations
// route through RPC with optimistic local updates — see pushRecent /
// toggleRecentFav / deleteRecent and the profiles section.
let promptCache = [];          // [{text, fav}] — active profile's prompts
const LS = {
  recents: () => promptCache.map(e => ({ text: e.text, fav: e.fav })),
  history: () => {
    const favs = promptCache.filter(e => e.fav).map(e => e.text);
    const rest = promptCache.filter(e => !e.fav).map(e => e.text);
    return [...favs, ...rest];
  },
};

// Shown in place of the empty history so a first-time user can see the
// command shape at a glance. Each entry exercises a different feature:
// model swap, /res + /sampler, /many, /lora stacking, A1111 weighting,
// /negprompt + /cfg + /steps composition.
const SHOWCASE_PROMPTS = [
  "/model z-image-turbo cute anime girl riding a unicorn through a sunflower field",
  "/model pony-v6-xl serene mountain lake at dawn",
  "/res landscape /sampler dpmpp-2m-karras a dragon flying over volcanic plains",
  "/many 4 cozy cabin in a snowy forest at night",
  "/model illustrious-xl-v1 /lora pixel-art-xl pixel art castle on a cliff",
  "(detailed:1.3) portrait of a fox in a wizard hat",
  "/negprompt blurry, low quality /cfg 7 /steps 30 a galaxy in a glass jar",
];

// ---------- websocket + RPC ----------
// Connection lifecycle (state machine, heartbeat, backoff, time-jump,
// defer-while-hidden, terminal state) lives in connection.js. We just
// pipe messages out and surface state to the topbar pill + title.
let conn = null;

function wsRequest(method, params) {
  if (!conn) return Promise.reject(new Error("not initialized"));
  return conn.request(method, params || {});
}

function dispatchEvent(m) {
  if (!m || typeof m !== "object") return;
  if (m.type === "state") onStateUpdate(m.state);
  else if (m.type === "job") onJob(m.job);
  else if (m.type === "output_added") onOutputAdded(m.entry, m.profile);
  else if (m.type === "favorite_changed") onFavoriteChanged(m.entry, m.profile);
  else if (m.type === "outputs_cleared") { if (m.profile === activeProfileName()) refreshOutputs(); }
  else if (m.type === "jobs_cleared") {
    for (const id of m.ids) jobs.delete(id);
    renderQueue();
  }
  else if (m.type === "gpu_stats") onGpuStats(m.stats);
  else if (m.type === "log") appendLog(m.message, m.level);
  else if (m.type === "model_loading") onModelLoading(m.model);
  else if (m.type === "model_loaded")  onModelLoaded(m.model);
  else if (m.type === "model_error")   onModelError(m.error);
  else if (m.type === "model_prefetched") onModelPrefetched();
  else if (m.type === "registries_changed") onModelPrefetched();
}

function onModelPrefetched() {
  if (leftTab === "models") refreshModels();
}

function deriveWidget(s) {
  // V2: state is *derived* from manager stats so it can't drift.
  // V10: never lie — terminal/deferred/stale each get their own label.
  if (s.isTerminal) return { label: "stopped", cls: "err",
    title: `connection stopped after ${s.attempt} attempts (last close: ${s.lastCloseCode} ${s.lastCloseReason})` };
  if (s.deferredOnVisible) return { label: "paused (hidden)", cls: "loading",
    title: "reconnect deferred until tab is visible" };
  switch (s.state) {
    case "ALIVE":
      return { label: "connected", cls: "ok",
        title: `connected · pending pings ${s.pendingPings}` };
    case "STALE":
      return { label: "stalled", cls: "loading",
        title: "no pong from server — verifying" };
    case "NEW":
      return { label: "connecting…", cls: "loading", title: "opening websocket" };
    case "DEAD":
    default: {
      const next = s.nextReconnectInMs;
      const inS = next == null ? null : Math.max(0, Math.round(next / 1000));
      const label = inS == null
        ? "disconnected"
        : `reconnecting in ${inS}s (${s.attempt}/${s.maxAttempts})`;
      return { label, cls: "err",
        title: `last close: ${s.lastCloseCode || "?"} ${s.lastCloseReason || ""}` };
    }
  }
}

function renderConnectionPill(s) {
  const w = deriveWidget(s);
  const el = $("ws-state");
  el.textContent = w.label;
  el.className = "pill " + w.cls;
  el.title = w.title;
  el.setAttribute("aria-label", `connection: ${w.label}`);
  // V7: mirror state in the document title so hidden tabs surface it.
  const base = "zimt";
  if (s.state === "ALIVE" && !s.isTerminal) {
    document.title = base;
  } else if (s.isTerminal) {
    document.title = `[stopped] ${base}`;
  } else if (s.state === "STALE") {
    document.title = `[stalled] ${base}`;
  } else {
    document.title = `[offline] ${base}`;
  }
}

// rAF-throttled re-render so the countdown text ticks smoothly without
// burning the loop. We re-derive from the last stats snapshot each
// frame; the manager pushes a fresh snapshot whenever state changes.
let _lastStats = null;
let _renderScheduled = false;
function scheduleRender() {
  if (_renderScheduled) return;
  _renderScheduled = true;
  requestAnimationFrame(() => {
    _renderScheduled = false;
    if (_lastStats) renderConnectionPill(_lastStats);
  });
}
// Also drive a 1Hz tick so the countdown moves while the state itself
// is steady (DEAD with reconnect pending).
setInterval(() => { if (_lastStats) scheduleRender(); }, 1000);

function connectWs() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  conn = new ZimtConnectionManager({
    url: `${proto}//${location.host}/ws`,
    onMessage: dispatchEvent,
    onStateChange: (s) => { _lastStats = s; scheduleRender(); },
  });
}

// ---------- log line (rolling) ----------
function appendLog(msg, level = "info") {
  const root = $("log");
  // keep just the last few lines
  while (root.children.length >= 4) root.removeChild(root.firstChild);
  const d = document.createElement("div");
  d.className = "log-line" + (level === "error" ? " error" : "");
  d.textContent = msg;
  root.appendChild(d);
}

// ---------- state text + queue ----------
function fmtState(s) {
  if (!s.loaded) return "no model loaded";
  const st = s.settings;
  const model = s.models.find(m => m.name === s.model);
  // Remote (hosted-API) models expose aspect_ratio + resolution tier rather
  // than the diffusers knobs — show those instead of dormant 0/0 fields.
  if (st.is_remote) {
    const rc = model?.remote;
    const lines = [
      `model: ${s.model}  (remote)`,
      `aspect:  ${st.aspect_ratio ?? rc?.default_aspect_ratio ?? "?"}`,
      `quality: ${st.resolution_tier ?? rc?.default_resolution ?? "?"}`,
    ];
    return lines.join("\n");
  }
  const presets = model?.resolutions ?? [];
  const preset = presets.find(p => p.w === st.width && p.h === st.height);
  const sz = `${st.width}x${st.height}` + (preset ? `  [${preset.label}]` : "  [custom]");
  const lines = [
    `model: ${s.model}`,
    `cfg:   ${st.cfg}`,
    `steps: ${st.steps}`,
    `size:  ${sz}`,
    `sampler: ${st.sampler ?? "?"}    clip_skip: ${st.clip_skip ?? 0}`,
    `negprompt: ${JSON.stringify(st.negative_prompt ?? "")}`,
  ];
  if (st.score_tags) lines.push(`auto-prefix: ${JSON.stringify(st.score_tags)}`);
  const stack = st.lora_stack || [];
  const loraTxt = stack.length
    ? stack.map(e => `${e.name}:${e.weight}`).join(", ")
    : "(none)";
  lines.push(`loras: ${loraTxt}`);
  return lines.join("\n");
}

// Model-loading state for the topbar pill. While loading we override the
// pill text + add a pulsing class until model_loaded / model_error fires.
let modelLoading = "";

function onStateUpdate(s) {
  state = s;
  if (!modelLoading) {
    $("model-state").textContent = s.loaded ? s.model : "no model";
    $("model-state").className = "pill" + (s.loaded ? " ok" : "");
  }
  const pre = $("state-text");
  pre.textContent = fmtState(s);
  pre.classList.toggle("empty", !s.loaded);
  if (leftTab === "models" && (modelsBases.length || modelsLoras.length)) {
    renderBases(); renderLoras();
  }
}

function onModelLoading(name) {
  modelLoading = name;
  $("model-state").textContent = `loading ${name}…`;
  $("model-state").className = "pill loading";
  appendLog(`loading ${name}…`);
}
function onModelLoaded(name) {
  modelLoading = "";
  // onStateUpdate will run next (the server emits state right after) — the
  // pill will catch up. Update immediately as well so users see the change.
  $("model-state").textContent = name;
  $("model-state").className = "pill ok";
  appendLog(`loaded ${name}`);
}
function onModelError(err) {
  modelLoading = "";
  $("model-state").textContent = "load error";
  $("model-state").className = "pill err";
  appendLog(`load error: ${err}`, "error");
}

function onJob(j) {
  jobs.set(j.id, j);
  // bound size — evict the oldest completed entry first to avoid displacing
  // a still-running long-lived job (PR-07-D15).
  if (jobs.size > MAX_QUEUE_SHOWN) {
    let toEvict = null;
    for (const [id, job] of jobs) {
      if (job.status === "done" || job.status === "error" || job.status === "canceled") {
        toEvict = id;
        break;
      }
    }
    if (toEvict == null) {
      // All entries are active; fall back to oldest insertion (last resort).
      toEvict = jobs.keys().next().value;
      console.warn("onJob: all queue entries are active — evicting oldest (queue > MAX_QUEUE_SHOWN)");
    }
    jobs.delete(toEvict);
  }
  renderQueue();
  // Mirror download-job progress into the models tab if it's visible.
  if (leftTab === "models" && j.kind === "download") {
    renderBases(); renderLoras();
  }
}

function fmtBytes(n) {
  if (!n || n <= 0) return "?";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

// Distinguish "downloading bytes" from "loading from cache". A model load
// is modelled as kind=="download" because that's where the HF byte-bar
// bridge attaches; but when the snapshot is already cached, no tqdm bar
// ever fires and the user is just waiting for the in-memory pipeline to
// build. Saying "downloading … (starting)" in that case is misleading.
function _dlPhase(j) {
  const active = (j.download_file && j.download_file !== "")
              || (j.download_bytes_total || 0) > 0
              || (j.download_files_total || 0) > 0
              || (j.download_bytes_n || 0) > 0
              || (j.download_files_n || 0) > 0;
  return active ? "downloading" : "loading";
}

function fmtDuration(seconds) {
  if (!isFinite(seconds) || seconds < 0) return "";
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

function fmtDownloadCounter(j) {
  // Per-unit slots: files and bytes can both be populated during a real
  // HF snapshot_download (outer thread_map bar + per-file byte bar).
  const fn = j.download_files_n || 0;
  const ft = j.download_files_total || 0;
  const bn = j.download_bytes_n || 0;
  const bt = j.download_bytes_total || 0;
  const hasFiles = fn > 0 || ft > 0;
  const hasBytes = bn > 0 || bt > 0;
  const filesStr = ft > 0 ? `${fn} / ${ft} files` : (fn > 0 ? `${fn} files` : "");
  const bytesStr = bt > 0 ? `${fmtBytes(bn)} / ${fmtBytes(bt)}` : (bn > 0 ? fmtBytes(bn) : "");
  if (hasFiles && hasBytes) return `${filesStr}  ·  ${bytesStr}`;
  if (hasFiles) return filesStr;
  if (hasBytes) return bytesStr;
  return "";
}

function renderQueue() {
  renderQueueSummary();
  const root = $("queue-list");
  root.innerHTML = "";
  const arr = Array.from(jobs.values()).reverse();
  for (const j of arr) {
    const li = document.createElement("li");
    li.className = "queue-item status-" + j.status + " kind-" + (j.kind || "generate");
    const status = document.createElement("span"); status.className = "qstatus";
    const dlPhase = j.kind === "download" ? _dlPhase(j) : null;
    // For finished/error/canceled download jobs, fall through to j.status so
    // the row reads "done" / "error" / "canceled" rather than "loading".
    if (j.kind === "download" && (j.status === "running" || j.status === "queued")) {
      status.textContent = dlPhase;
    } else {
      status.textContent = j.status;
    }
    li.appendChild(status);
    if (j.kind === "download") {
      const label = document.createElement("span"); label.className = "qprompt";
      // PR-02-D04: when this job doesn't own the progress slot, surface
      // a "waiting" affordance instead of progress text and suppress the bar.
      const waiting = j.progress_owner === false;
      const counter = waiting ? "waiting for slot…" : fmtDownloadCounter(j);
      // Only show the per-file segment when we're actually transferring
      // bytes — cache-only loads never set download_file, so "(starting)"
      // would just be filler that looks like a stuck download.
      const showFile = dlPhase === "downloading";
      const file = showFile ? (j.download_file || "(starting)") : "";
      const parts = [j.model];
      if (file) parts.push(file);
      if (counter) parts.push(counter);
      label.textContent = parts.join("  ·  ");
      label.title = label.textContent;
      li.appendChild(label);
      if (!waiting && j.download_files_done > 0) {
        const meta = document.createElement("span"); meta.className = "qseed";
        meta.textContent = `files done: ${j.download_files_done}`;
        li.appendChild(meta);
      }
      const bytesTotal = j.download_bytes_total || 0;
      const filesTotal = j.download_files_total || 0;
      const barTotal = bytesTotal > 0 ? bytesTotal : filesTotal;
      const barN = bytesTotal > 0 ? (j.download_bytes_n || 0) : (j.download_files_n || 0);
      if (!waiting && j.status === "running" && barTotal > 0) {
        const bar = document.createElement("div");
        bar.className = "qprogress";
        const pct = Math.min(100, Math.round(100 * barN / barTotal));
        bar.title = counter || `${pct}%`;
        const fill = document.createElement("div");
        fill.style.width = pct + "%";
        bar.appendChild(fill);
        li.appendChild(bar);
      }
      if (j.status === "queued" || j.status === "running") {
        const cancel = document.createElement("button");
        cancel.className = "cancel-btn"; cancel.textContent = "✕";
        cancel.title = "cancel download";
        cancel.onclick = async () => {
          cancel.disabled = true;
          try { await wsRequest("job_cancel", { id: j.id }); }
          catch (e) { appendLog(`cancel: ${e.message}`, "error"); cancel.disabled = false; }
        };
        li.appendChild(cancel);
      }
    } else {
      const prompt = document.createElement("span"); prompt.className = "qprompt";
      prompt.textContent = (j.full_prompt || j.raw_prompt || "").slice(0, 200);
      prompt.title = j.full_prompt || j.raw_prompt || "";
      const seed = document.createElement("span"); seed.className = "qseed";
      seed.textContent = `seed=${j.seed}`;
      li.append(prompt, seed);
      if (j.status === "running" && j.total_steps > 0) {
        const bar = document.createElement("div");
        bar.className = "qprogress";
        const pct = Math.min(100, Math.round(100 * j.step / j.total_steps));
        bar.title = `${j.step}/${j.total_steps}`;
        const fill = document.createElement("div");
        fill.style.width = pct + "%";
        bar.appendChild(fill);
        li.appendChild(bar);
      }
      if (j.status === "queued" || j.status === "running") {
        const cancel = document.createElement("button");
        cancel.className = "cancel-btn"; cancel.textContent = "✕";
        cancel.title = "cancel job";
        cancel.onclick = async () => {
          cancel.disabled = true;
          try { await wsRequest("job_cancel", { id: j.id }); }
          catch (e) { appendLog(`cancel: ${e.message}`, "error"); cancel.disabled = false; }
        };
        li.appendChild(cancel);
      }
    }
    if (j.status === "error" && j.error) {
      const details = document.createElement("button");
      details.className = "details-btn"; details.textContent = "details";
      details.title = "show error message";
      details.onclick = () => showJobError(j);
      li.appendChild(details);
    }
    // Wall-clock duration for finished jobs. Uses ts_queued as start since
    // download jobs skip the queued state and generate jobs spend at most a
    // tick in it before the executor picks them up.
    const finished = j.status === "done" || j.status === "error" || j.status === "canceled";
    if (finished && j.ts_done && j.ts_queued) {
      const dur = document.createElement("span");
      dur.className = "qduration";
      dur.textContent = fmtDuration(j.ts_done - j.ts_queued);
      dur.title = "elapsed time";
      li.appendChild(dur);
    }
    root.appendChild(li);
  }
}

// ---------- queue bottom panel ----------
// Collapsed-state summary: counts of done / pending / errors plus a
// single "current task" row with prompt + progress bar + elapsed time.
// The summary is recomputed by renderQueueSummary() — called from
// renderQueue() (event-driven WS updates) AND from a 1s tick while a
// job is running so the elapsed-time counter advances even without
// new step events. The tick auto-clears when nothing is running.
let queuePinned = JSON.parse(localStorage.getItem("zimt.queue-pinned") || "false");
let queueTickHandle = null;

function _qPickRunning() {
  for (const j of jobs.values()) if (j.status === "running") return j;
  return null;
}

function _qCounts() {
  let done = 0, queued = 0, running = 0, error = 0, canceled = 0;
  for (const j of jobs.values()) {
    if (j.status === "done") done++;
    else if (j.status === "queued") queued++;
    else if (j.status === "running") running++;
    else if (j.status === "error") error++;
    else if (j.status === "canceled") canceled++;
  }
  return { done, queued, running, error, canceled };
}

function _qSummaryProgress(j) {
  // Returns {pct, label} or null if no progress info available.
  if (j.kind === "download") {
    const bytesTotal = j.download_bytes_total || 0;
    const filesTotal = j.download_files_total || 0;
    const total = bytesTotal > 0 ? bytesTotal : filesTotal;
    const n = bytesTotal > 0 ? (j.download_bytes_n || 0) : (j.download_files_n || 0);
    if (total <= 0) return null;
    return { pct: Math.min(100, Math.round(100 * n / total)),
             label: `${n}/${total}` };
  }
  if ((j.total_steps || 0) > 0) {
    return { pct: Math.min(100, Math.round(100 * j.step / j.total_steps)),
             label: `${j.step}/${j.total_steps}` };
  }
  return null;
}

function renderQueueSummary() {
  const counts = _qCounts();
  const pending = counts.queued + counts.running;
  const finished = counts.done + counts.error + counts.canceled;
  const total = finished + pending;

  $("qsumm-done").textContent = `done: ${counts.done}`;
  $("qsumm-pending").textContent = `pending: ${pending}`;

  // Progress bars and cancel-all only make sense while there's work in
  // flight. With nothing pending the batch is over, so hide them.
  const bars = $("qsumm-bars");
  const cancelAll = $("btn-cancel-all-summary");
  if (pending === 0) {
    bars.hidden = true;
    cancelAll.hidden = true;
    _stopQueueTick();
    return;
  }
  bars.hidden = false;
  cancelAll.hidden = false;

  // Total progress: finished jobs out of all jobs (e.g. 18/20).
  $("qsumm-total-label").textContent = `${finished}/${total}`;
  const totalPct = total > 0 ? Math.round(100 * finished / total) : 0;
  $("qsumm-total-bar-fill").style.width = totalPct + "%";

  // Current-task progress: step/byte progress of the running job.
  const running = _qPickRunning();
  const prog = running ? _qSummaryProgress(running) : null;
  const curGroup = $("qsumm-current-group");
  if (prog) {
    curGroup.hidden = false;
    $("qsumm-current-label").textContent = prog.label;
    $("qsumm-current-bar-fill").style.width = prog.pct + "%";
  } else {
    curGroup.hidden = true;
  }

  if (running) _startQueueTick();
  else _stopQueueTick();
}

function _startQueueTick() {
  if (queueTickHandle !== null) return;
  queueTickHandle = setInterval(renderQueueSummary, 1000);
}
function _stopQueueTick() {
  if (queueTickHandle === null) return;
  clearInterval(queueTickHandle);
  queueTickHandle = null;
}

function setQueuePinned(pinned) {
  queuePinned = pinned;
  localStorage.setItem("zimt.queue-pinned", JSON.stringify(pinned));
  applyQueuePinned();
}
function applyQueuePinned() {
  const panel = $("queue-panel");
  panel.classList.toggle("expanded", queuePinned);
  panel.classList.toggle("collapsed", !queuePinned);
  $("queue-full").hidden = !queuePinned;
  $("queue-summary-row").hidden = queuePinned;
}
function setupQueuePin() {
  $("btn-queue-pin").onclick = () => setQueuePinned(true);
  $("btn-queue-unpin").onclick = () => setQueuePinned(false);
  applyQueuePinned();
}

// ---------- collapsible state panel ----------
let stateCollapsed = JSON.parse(localStorage.getItem("zimt.state-collapsed") || "false");
function applyStateCollapsed() {
  $("state-panel").classList.toggle("collapsed", stateCollapsed);
  $("state-toggle").setAttribute("aria-expanded", String(!stateCollapsed));
}
function setupStatePanel() {
  $("state-toggle").onclick = () => {
    stateCollapsed = !stateCollapsed;
    localStorage.setItem("zimt.state-collapsed", JSON.stringify(stateCollapsed));
    applyStateCollapsed();
  };
  applyStateCollapsed();
}

// ---------- error popup ----------
function showJobError(j) {
  $("error-popup-text").textContent = j.error || "(no error message)";
  $("error-popup").classList.add("open");
  $("error-popup")._currentError = j.error || "";
}
$("error-popup-close").onclick = () => $("error-popup").classList.remove("open");
$("error-popup").onclick = (e) => {
  if (e.target === $("error-popup")) $("error-popup").classList.remove("open");
};
$("error-popup-copy").onclick = () => {
  navigator.clipboard.writeText($("error-popup")._currentError || "");
};

// ---------- thumbnails ----------
// Build a profile-scoped image URL. Outputs live under
// /api/outputs/<profile>/<file>; the active profile supplies the segment.
function _outUrl(name, thumb) {
  const prof = activeProfileName() || "Default";
  const base = `/api/outputs/${encodeURIComponent(prof)}/${encodeURIComponent(name)}`;
  return thumb ? `${base}?thumb=${thumb}` : base;
}
function onOutputAdded(entry, profile) {
  // Per-tab scoping: ignore additions for a profile this tab isn't viewing.
  if (profile !== activeProfileName()) return;
  if (currentTab === "all") {
    outputs.unshift(entry); totalCount += 1; renderThumbs();
  }
}
function onFavoriteChanged(entry, profile) {
  if (profile !== activeProfileName()) return;
  const idx = outputs.findIndex(x => x.name === entry.name);
  if (currentTab === "favs") {
    if (entry.fav && idx < 0) { outputs.unshift(entry); totalCount += 1; }
    else if (!entry.fav && idx >= 0) { outputs.splice(idx, 1); totalCount -= 1; }
  } else { if (idx >= 0) outputs[idx] = entry; }
  if (modalEntry && modalEntry.name === entry.name) {
    modalEntry = entry; updateModalFavButton();
  }
  renderThumbs();
}
function renderThumbs() {
  $("outputs-count").textContent = `${outputs.length}${hasMore ? "+" : ""} / ${totalCount}`;
  // Keep modal prev/next in sync — if the active entry was removed
  // (e.g. unfavorited while on the fav tab), the boundaries shift.
  if ($("modal").classList.contains("open")) updateModalNav();
  const root = $("thumbs"); root.innerHTML = "";
  for (const e of outputs) {
    const d = document.createElement("div"); d.className = "thumb";
    const img = document.createElement("img"); img.loading = "lazy";
    img.src = _outUrl(e.name, 192);
    d.appendChild(img);
    // Three stacked reload buttons (top to bottom):
    //  1. "1:1"  — fill the prompt with the exact original command line
    //              (uses the command_line PNG metadata field; disabled
    //              on older images that predate the field).
    //  2. "↺"   — restore prompt + settings, fresh random seed.
    //  3. "="   — restore prompt + settings + the original seed (pinned).
    const fillCmd = document.createElement("button");
    fillCmd.className = "reload-btn reload-cmd-btn";
    fillCmd.textContent = "1:1";
    if (e.metadata && e.metadata.command_line) {
      fillCmd.title = "fill prompt with the exact original input line (1:1)";
      fillCmd.onclick = (ev) => { ev.stopPropagation(); fillCommandLine(e); };
    } else {
      fillCmd.title = "no command_line metadata on this image (predates the field)";
      fillCmd.disabled = true;
    }
    d.appendChild(fillCmd);
    const reloadFreshSeed = document.createElement("button");
    reloadFreshSeed.className = "reload-btn reload-no-seed-btn";
    reloadFreshSeed.textContent = "↺";
    reloadFreshSeed.title = "restore prompt + settings, but use a new random seed";
    reloadFreshSeed.onclick = (ev) => {
      ev.stopPropagation();
      restoreToPrompt(e, { includeSeed: false });
    };
    d.appendChild(reloadFreshSeed);
    const reloadWithSeed = document.createElement("button");
    reloadWithSeed.className = "reload-btn reload-with-seed-btn";
    reloadWithSeed.textContent = "=";
    reloadWithSeed.title = "restore prompt + settings + original seed";
    reloadWithSeed.onclick = (ev) => { ev.stopPropagation(); restoreToPrompt(e); };
    d.appendChild(reloadWithSeed);
    // Top-right: favorite toggle.
    const star = document.createElement("button");
    star.className = "star-btn" + (e.fav ? " on" : "");
    star.textContent = e.fav ? "★" : "☆";
    star.title = e.fav ? "remove favorite" : "favorite";
    star.onclick = (ev) => { ev.stopPropagation(); toggleFavorite(e); };
    d.appendChild(star);
    // Right column, below the star: copy full image to clipboard.
    const copy = document.createElement("button");
    copy.className = "copy-btn";
    copy.textContent = "⧉";
    copy.title = "copy full image to clipboard";
    copy.onclick = (ev) => { ev.stopPropagation(); copyImageToClipboard(e, copy); };
    d.appendChild(copy);
    d.onclick = () => openModal(e);
    root.appendChild(d);
  }
  $("load-more").style.display = hasMore ? "" : "none";
}
async function toggleFavorite(entry) {
  try {
    await wsRequest("output_favorite",
      { profile: activeProfileName(), name: entry.name, favorite: !entry.fav });
  } catch (e) { appendLog(`favorite: ${e.message}`, "error"); }
}
// Copy the full (non-thumbnail) image to the OS clipboard as image/png.
// The Clipboard API needs a Promise<Blob> handed to ClipboardItem so the
// fetch can run inside the user-gesture context — passing an already-
// resolved blob works in modern browsers but the Promise form is the
// portable spelling that Safari requires.
async function copyImageToClipboard(entry, btn) {
  const prev = btn.textContent;
  btn.disabled = true;
  try {
    const url = _outUrl(entry.name);
    const blob = await fetch(url).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.blob();
    });
    await navigator.clipboard.write([
      new ClipboardItem({ [blob.type || "image/png"]: blob }),
    ]);
    btn.textContent = "✓";
    appendLog(`copied ${entry.name} to clipboard`);
  } catch (err) {
    btn.textContent = "✕";
    appendLog(`copy: ${err.message}`, "error");
  } finally {
    setTimeout(() => { btn.textContent = prev; btn.disabled = false; }, 1200);
  }
}
async function loadOutputs(tab, page) {
  return wsRequest("outputs_list", { profile: activeProfileName(), tab, page, per_page: 60 });
}
async function refreshOutputs() {
  // No active profile yet (pre-boot) → nothing to show.
  if (activeProfileId == null) {
    outputs = []; hasMore = false; totalCount = 0; currentPage = 1;
    renderThumbs();
    return;
  }
  try {
    const r = await loadOutputs(currentTab, 1);
    outputs = r.entries; hasMore = r.has_more; totalCount = r.total; currentPage = 1;
    renderThumbs();
  } catch (e) { appendLog(`outputs: ${e.message}`, "error"); }
}
async function loadMore() {
  try {
    const r = await loadOutputs(currentTab, currentPage + 1);
    outputs = outputs.concat(r.entries); hasMore = r.has_more; totalCount = r.total;
    currentPage += 1; renderThumbs();
  } catch (e) { appendLog(`outputs: ${e.message}`, "error"); }
}
function setTab(tab) {
  if (tab === currentTab) return;
  currentTab = tab;
  for (const b of document.querySelectorAll(".tab")) {
    b.classList.toggle("active", b.dataset.tab === tab);
  }
  refreshOutputs();
}
for (const b of document.querySelectorAll(".tab")) {
  b.addEventListener("click", () => setTab(b.dataset.tab));
}
$("load-more").onclick = loadMore;

// ---------- utility buttons under the prompt ----------
$("btn-cleanup-outputs").onclick = async () => {
  if (!confirm("Delete every non-favorite output? Favorites are kept.")) return;
  try {
    const r = await wsRequest("outputs_cleanup", { profile: activeProfileName() });
    appendLog(`cleaned ${r.deleted} files`);
    await refreshOutputs();
  } catch (e) { appendLog(`cleanup: ${e.message}`, "error"); }
};
async function _cancelAllJobs() {
  try {
    const r = await wsRequest("jobs_cancel_all");
    appendLog(`canceled ${r.canceled} jobs`);
  } catch (e) { appendLog(`cancel-all: ${e.message}`, "error"); }
}
$("btn-cancel-all").onclick = _cancelAllJobs;
$("btn-cancel-all-summary").onclick = _cancelAllJobs;
$("btn-clear-completed").onclick = async () => {
  try {
    const r = await wsRequest("jobs_clear_completed");
    appendLog(`cleared ${r.cleared} completed jobs`);
  } catch (e) { appendLog(`clear-completed: ${e.message}`, "error"); }
};
$("btn-clear-log").onclick = () => { $("log").innerHTML = ""; };
$("btn-refresh-models").onclick = () => refreshModels();
$("btn-generate").onclick = () => submit();
// Recent-prompts UI state — declared here so the button wirings below
// (and the initial `_syncWrapBtn()` call) can see them without TDZ.
let recentSearch = "";
let recentWrap = JSON.parse(localStorage.getItem("zimt.recent-wrap") || "true");
$("recent-search").addEventListener("input", (ev) => {
  recentSearch = ev.target.value;
  renderRecent();
});
$("btn-recent-wrap").onclick = () => {
  recentWrap = !recentWrap;
  localStorage.setItem("zimt.recent-wrap", JSON.stringify(recentWrap));
  _syncWrapBtn();
  renderRecent();
};
function _syncWrapBtn() {
  const btn = $("btn-recent-wrap");
  btn.textContent = recentWrap ? "wrap" : "ellipse";
  btn.setAttribute("aria-pressed", recentWrap ? "true" : "false");
  btn.title = recentWrap
    ? "currently wrapping long prompts — click to ellipse"
    : "currently ellipsing long prompts — click to wrap";
}
_syncWrapBtn();
$("btn-clear-recent").onclick = async () => {
  if (activeProfileId == null) return;
  const favs = promptCache.filter(e => e.fav);
  const removed = promptCache.length - favs.length;
  if (!removed) {
    appendLog("nothing to clear (no non-favourite prompts)");
    return;
  }
  if (!confirm(`Clear ${removed} non-favourite recent prompt(s)? Favourites are kept.`)) return;
  promptCache = favs;          // optimistic
  renderRecent();
  try {
    const r = await wsRequest("prompts_clear", { profile_id: activeProfileId });
    appendLog(`cleared ${r.removed} non-favourite prompt(s)`);
  } catch (e) { appendLog(`clear: ${e.message}`, "error"); await hydrate(activeProfileId); }
};

// ---------- recent prompts ----------
// Each <li> is a row: ★ fav-toggle · prompt text · ✕ delete. Favourites
// render first; within each group, insertion order (newest on top).
// `recentSearch` (substring, case-insensitive) filters both groups in
// place — favs that match still group above non-favs. Showcase prompts
// only render when the user has no real history AND no active filter —
// they're click-to-load examples, not stored entries.
function renderRecent() {
  const root = $("recent-list"); root.innerHTML = "";
  root.classList.toggle("ellipsed", !recentWrap);
  const xs = LS.recents();
  const q = recentSearch.trim().toLowerCase();
  const matches = q ? xs.filter(e => e.text.toLowerCase().includes(q)) : xs;
  if (!matches.length) {
    const hint = document.createElement("li");
    hint.className = "recent-hint";
    if (q) {
      hint.textContent = `no prompts match "${recentSearch}"`;
      root.appendChild(hint);
      return;
    }
    hint.textContent = "no recent prompts — click one to load it:";
    root.appendChild(hint);
    for (const text of SHOWCASE_PROMPTS) {
      const li = document.createElement("li");
      li.className = "recent-item showcase";
      li.title = `example: ${text}`;
      li.textContent = text;
      li.onclick = () => loadPromptIntoInput(text);
      root.appendChild(li);
    }
    return;
  }
  const favs = matches.filter(e => e.fav);
  const rest = matches.filter(e => !e.fav);
  for (const e of [...favs, ...rest]) root.appendChild(_recentItem(e));
}
function _recentItem(entry) {
  const li = document.createElement("li");
  li.className = "recent-item" + (entry.fav ? " fav" : "");
  const starBtn = document.createElement("button");
  starBtn.className = "recent-fav" + (entry.fav ? " on" : "");
  starBtn.type = "button";
  starBtn.title = entry.fav ? "unfavourite" : "favourite";
  starBtn.textContent = entry.fav ? "★" : "☆";
  starBtn.onclick = (ev) => { ev.stopPropagation(); toggleRecentFav(entry.text); };
  const textSpan = document.createElement("span");
  textSpan.className = "recent-text";
  textSpan.textContent = entry.text;
  textSpan.title = "click to load into prompt";
  textSpan.onclick = () => loadPromptIntoInput(entry.text);
  const delBtn = document.createElement("button");
  delBtn.className = "recent-del";
  delBtn.type = "button";
  delBtn.title = "delete this prompt";
  delBtn.textContent = "✕";
  delBtn.onclick = (ev) => { ev.stopPropagation(); deleteRecent(entry.text); };
  li.appendChild(starBtn);
  li.appendChild(textSpan);
  li.appendChild(delBtn);
  return li;
}
function loadPromptIntoInput(text) {
  $("prompt-input").value = text;
  setLeftTab("inference");
  $("prompt-input").focus();
  autoResize();
}
function toggleRecentFav(text) {
  if (activeProfileId == null) return;
  const e = promptCache.find(x => x.text === text);
  if (!e) return;
  const fav = !e.fav;
  e.fav = fav;                 // optimistic
  renderRecent();
  wsRequest("prompt_fav", { profile_id: activeProfileId, text, fav })
    .catch(err => { appendLog(`favourite: ${err.message}`, "error"); hydrate(activeProfileId); });
}
function deleteRecent(text) {
  if (activeProfileId == null) return;
  promptCache = promptCache.filter(e => e.text !== text);   // optimistic
  renderRecent();
  wsRequest("prompt_delete", { profile_id: activeProfileId, text })
    .catch(err => { appendLog(`delete: ${err.message}`, "error"); hydrate(activeProfileId); });
}
function pushRecent(p) {
  if (activeProfileId == null) return;
  // Optimistic: move/insert at front keeping any existing fav flag. The
  // server applies the same upsert + non-fav cap and returns the canonical
  // list, which we reconcile against on response.
  const existing = promptCache.find(e => e.text === p);
  const fav = existing ? existing.fav : false;
  promptCache = [{ text: p, fav }, ...promptCache.filter(e => e.text !== p)];
  renderRecent();
  wsRequest("prompt_push", { profile_id: activeProfileId, text: p })
    .then(r => { if (r && Array.isArray(r.prompts)) { promptCache = r.prompts; renderRecent(); } })
    .catch(err => { appendLog(`history: ${err.message}`, "error"); });
}

// ---------- modal ----------
function updateModalFavButton() {
  if (!modalEntry) return;
  $("modal-fav").textContent = modalEntry.fav ? "★ unfavorite" : "☆ favorite";
}
function openModal(entry) {
  modalEntry = entry;
  $("modal-img").src = _outUrl(entry.name);
  // Add the metadata expanded_prompt right after raw_prompt so a
  // user looking at a template-generated image sees both the typed
  // template and the resolved text at a glance.
  const grid = $("modal-meta"); grid.innerHTML = "";
  // command_line is the original user input (with all /cmd parts and
  // template syntax intact) — shown first so a glance at the modal
  // reveals exactly what was typed. The decomposed fields below stay
  // for backward compat and quick parameter inspection.
  const keys = ["model", "repo_id", "repo_url",
                "command_line",
                "raw_prompt", "expanded_prompt", "prompt",
                "negative_prompt", "seed",
                "steps", "cfg", "sampler", "clip_skip", "loras",
                "width", "height", "dtype", "device"];
  const meta = entry.metadata || {};
  for (const k of keys) {
    if (!(k in meta)) continue;
    const keyDiv = document.createElement("div"); keyDiv.className = "meta-key"; keyDiv.textContent = k;
    const valDiv = document.createElement("div"); valDiv.className = "meta-val"; valDiv.textContent = meta[k];
    const btn = document.createElement("button");
    btn.className = "copy-btn"; btn.textContent = "copy";
    btn.onclick = () => navigator.clipboard.writeText(meta[k]);
    grid.appendChild(keyDiv); grid.appendChild(valDiv); grid.appendChild(btn);
  }
  updateModalFavButton();
  updateModalNav();
  $("modal").classList.add("open");
}
// Hide prev/next at boundaries. Uses the `outputs` array which is
// already filtered by the active tab (all/fav) — so navigation
// implicitly respects the current filter without any extra wiring.
function updateModalNav() {
  if (!modalEntry) return;
  const prev = neighborInOutputs(outputs, modalEntry.name, -1);
  const next = neighborInOutputs(outputs, modalEntry.name, +1);
  $("modal-prev").hidden = prev === null;
  $("modal-next").hidden = next === null;
}
function navigateModal(direction) {
  if (!modalEntry) return;
  const target = neighborInOutputs(outputs, modalEntry.name, direction);
  if (target !== null) openModal(target);
}
$("modal-prev").onclick = (e) => { e.stopPropagation(); navigateModal(-1); };
$("modal-next").onclick = (e) => { e.stopPropagation(); navigateModal(+1); };
$("modal-close").onclick = () => $("modal").classList.remove("open");
$("modal").onclick = (e) => { if (e.target === $("modal")) $("modal").classList.remove("open"); };
$("modal-fav").onclick = () => { if (modalEntry) toggleFavorite(modalEntry); };

// Help modal — static reference popup. Topbar `?` opens; click outside,
// the close button, or Esc dismisses. The content lives in index.html
// (search for `id="help-modal"`) so non-engineers can edit the wording
// without touching JS.
$("help-btn").onclick = () => {
  $("help-modal").classList.add("open");
  $("help-modal").setAttribute("aria-hidden", "false");
};
function closeHelp() {
  $("help-modal").classList.remove("open");
  $("help-modal").setAttribute("aria-hidden", "true");
}
$("help-close").onclick = closeHelp;
$("help-modal").onclick = (e) => { if (e.target === $("help-modal")) closeHelp(); };

// Keyboard: ←/→ navigate the image-preview modal, Esc closes whichever
// modal happens to be open (preview or help). Gated on the user NOT
// being in an editable element so the prompt editor's keys keep their
// own behaviour.
document.addEventListener("keydown", (e) => {
  const tgt = e.target;
  const editing = tgt && (tgt.tagName === "INPUT" || tgt.tagName === "TEXTAREA"
                          || tgt.isContentEditable);
  if (editing) return;
  const previewOpen = $("modal").classList.contains("open");
  const helpOpen = $("help-modal").classList.contains("open");
  if (!previewOpen && !helpOpen) return;
  if (e.key === "Escape") {
    if (helpOpen) closeHelp();
    if (previewOpen) $("modal").classList.remove("open");
    return;
  }
  // Arrow-key navigation only applies to the image preview modal.
  if (previewOpen && e.key === "ArrowLeft") { e.preventDefault(); navigateModal(-1); }
  else if (previewOpen && e.key === "ArrowRight") { e.preventDefault(); navigateModal(+1); }
});

// Build a single /api/exec line from previous-run metadata, and load it into
// the prompt textarea. Shared by the modal restore button and thumbnail reload
// buttons.
function restoreToPrompt(entry, options = {}) {
  if (!entry) return;
  $("prompt-input").value = buildRestorePromptLine(entry.metadata || {}, options);
  autoResize();
  $("modal").classList.remove("open");
  $("prompt-input").focus();
}
// Fill the prompt input verbatim from the PNG's `command_line` field —
// a 1:1 round-trip of what the user originally typed (with /cmd parts
// and template syntax intact). Distinct from restoreToPrompt which
// reconstructs the line from decomposed metadata fields.
function fillCommandLine(entry) {
  const line = entry?.metadata?.command_line;
  if (!line) return;
  $("prompt-input").value = stripInjectedGlobals(line);
  autoResize();
  $("modal").classList.remove("open");
  $("prompt-input").focus();
}
// applyGlobalsToLine() splices the active globals into the sent line wrapped
// in `<!-- globals:begin -->`/`<!-- globals:end -->` markers, and that sent
// line is what gets recorded as command_line. A 1:1 restore should show the
// user's original input, so strip the marker-delimited region (and the
// markers) back out. The marker constants are declared further down, so build
// the regex lazily to avoid the temporal-dead-zone reference at load time.
// The markers contain no regex metacharacters.
function stripInjectedGlobals(line) {
  const re = new RegExp(
    `\\s*${GLOBALS_BEGIN_MARKER}[\\s\\S]*?${GLOBALS_END_MARKER}\\s*`, "g");
  return line.replace(re, " ").trim();
}
$("modal-restore").onclick = () => restoreToPrompt(modalEntry);

// ---------- prompt input: readline-style ----------
const input = $("prompt-input");
let historyIdx = -1;          // -1 = editing fresh; otherwise index into LS.history()
let editingDraft = "";        // saved when entering history mode

// Auto-grow the textarea wrap (which sizes the absolute-positioned overlay)
// to fit the content. Also re-renders the syntax-highlighted mirror.
// `clampToMaxHeight` is provided by static/prompt_resize.js.
const inputWrap = document.querySelector(".textarea-wrap");
const highlight = $("prompt-highlight");

function autoResize() {
  // Skip while the inference pane is hidden — the textarea has no layout
  // box, so scrollHeight is 0 and we'd otherwise collapse it permanently.
  if (input.offsetParent === null) return;
  // The textarea has `position: absolute; inset: 0` in the stylesheet,
  // so it's pinned to all four edges of the wrap. Setting
  // `input.style.height = "auto"` alone does NOT release that constraint
  // — the textarea still fills whatever height the wrap currently has,
  // which means scrollHeight is floored by the previous box size. That
  // breaks two ways:
  //   * after `submit()` clears the value the wrap stays large, and
  //     each keystroke trims one pixel of sub-pixel rounding noise
  //     until the box finally matches the now-tiny content;
  //   * while a multi-line prompt grows, scrollHeight oscillates ±1
  //     line because the rounded line-height (14 * 1.4 = 19.6px) keeps
  //     crossing the currently-painted height boundary.
  // To get a clean measurement we drop the wrap's inline height so the
  // wrap collapses to its CSS min-height (2.5em). The textarea then
  // fills that minimal box, and scrollHeight returns the true content
  // height regardless of whatever height we had set last time.
  inputWrap.style.height = "";
  input.style.height = "auto";
  const contentH = input.scrollHeight;
  // Cap at the textarea's own CSS max-height so the wrap (and therefore
  // the highlight underlay, which is absolute-positioned within the
  // wrap) cannot grow past the visible editor. Beyond the cap, the
  // textarea handles overflow internally via its own scrollbar, and
  // the input.scroll → highlight.scrollTop sync below keeps the
  // highlight aligned with what's visible.
  const maxH = parseFloat(getComputedStyle(input).maxHeight);
  const h = clampToMaxHeight(contentH, maxH);
  input.style.height = h + "px";
  inputWrap.style.height = h + "px";
  highlight.style.height = h + "px";
  renderHighlight();
}
input.addEventListener("input", autoResize);
input.addEventListener("scroll", () => { highlight.scrollTop = input.scrollTop; });
window.addEventListener("resize", autoResize);

// ---------- prompt syntax highlighting ----------
// The tokenizer + renderer lives in static/prompt_highlight.js so it can
// be unit-tested without a DOM. It also owns the HL_CMDS table since
// command-to-class mapping is part of the same concern.
function esc(s) {
  return s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
}
function renderHighlight() {
  highlight.innerHTML = renderPromptHTML(input.value);
}

// ---------- intellisense-style suggestion popup ----------
// Updated on every input event; positioned at the textarea caret via a
// short-lived measurement mirror.
//
// Key handling — the popup is driven by Tab (sublime-style cycling). Each
// Tab rewrites the token at the cursor with the next candidate; Shift+Tab
// rewinds. The original typed prefix is preserved in `tokenText` so cycling
// stays consistent across replacements.
//
//   Tab        → cycle selection forward, rewrite token
//   Shift+Tab  → cycle selection backward, rewrite token
//   Esc        → dismiss popup (without undoing the current replacement)
//   click item → jump to that item + dismiss
//   typing     → rebuild popup from the new token, selected=0
//   Enter      → submit (unchanged; popup never captures it)
const popup = $("suggest-popup");
let suggest = {
  visible: false,
  items: [],
  selected: 0,
  tokenStart: 0,
  tokenEnd: 0,
  tokenText: "",
  cycling: false,   // true once Tab has been pressed; further Tabs cycle
};

// Measure the pixel position of a character offset inside the textarea.
// Used both for the popup anchor (token start) and, indirectly, for caret
// positioning. Standard "mirror div" technique — ~1ms per call.
function offsetCoords(offset) {
  const cs = getComputedStyle(input);
  const mirror = document.createElement("div");
  for (const p of ["boxSizing", "width", "padding", "border",
                   "font", "lineHeight", "whiteSpace",
                   "wordBreak", "overflowWrap"]) {
    mirror.style.setProperty(p, cs.getPropertyValue(p));
  }
  mirror.style.position = "absolute";
  mirror.style.visibility = "hidden";
  mirror.style.top = "0";
  mirror.style.left = "0";
  mirror.style.pointerEvents = "none";
  inputWrap.appendChild(mirror);

  mirror.textContent = input.value.slice(0, offset);
  const marker = document.createElement("span");
  marker.textContent = "​";          // zero-width but participates in layout
  mirror.appendChild(marker);

  const wrapRect = inputWrap.getBoundingClientRect();
  const mr = marker.getBoundingClientRect();
  const lineHeight = parseFloat(cs.lineHeight) || mr.height || 18;
  inputWrap.removeChild(mirror);
  return {
    left: mr.left - wrapRect.left - input.scrollLeft,
    top:  mr.top  - wrapRect.top  - input.scrollTop,
    lineHeight,
  };
}

function renderSuggest() {
  const items = suggest.items;
  if (!items.length) { popup.classList.remove("open"); popup.setAttribute("aria-hidden", "true"); return; }
  popup.innerHTML = items.map((it, i) => {
    const lbl = esc(it.label);
    const matchLen = suggest.tokenText.length;
    const labelHtml = matchLen
      ? `<span class="suggest-match">${esc(it.label.slice(0, matchLen))}</span>${esc(it.label.slice(matchLen))}`
      : lbl;
    const desc = it.desc ? `<span class="suggest-desc">${esc(it.desc)}</span>` : "";
    return `<div class="suggest-item${i === suggest.selected ? " selected" : ""}" data-i="${i}">`
      + `<span class="suggest-label">${labelHtml}</span>${desc}</div>`;
  }).join("");
  popup.classList.add("open");
  popup.setAttribute("aria-hidden", "false");
  // Anchor at the START of the token being completed, not the caret —
  // otherwise the popup jumps right as the user types more characters.
  const c = offsetCoords(suggest.tokenStart);
  popup.style.left = Math.max(0, c.left) + "px";
  popup.style.top  = (c.top + c.lineHeight + 2) + "px";
  // Make sure the selected item is in view.
  const sel = popup.querySelector(".suggest-item.selected");
  if (sel) sel.scrollIntoView({ block: "nearest" });
}

function updateSuggest() {
  const value = input.value;
  const cursor = input.selectionStart;
  let tok = tokenAtCursor(value, cursor);
  // If the cursor sits inside an unclosed `${...`, narrow the "current
  // token" to the variable head — start at the `$`, end past the
  // matching `}` (if any). The replacement range covers the WHOLE
  // existing reference so clicking a suggestion on `${clothesA}` with
  // the cursor mid-name swaps the entire reference for `${clothesB}`,
  // not just from the `$` to the cursor (which would leave a `thesA}`
  // suffix). `tok.text` still ends at the cursor — that's the typed
  // prefix used to filter the completion list.
  const varOpen = unclosedVarOpenIndex(value.slice(0, cursor));
  if (varOpen >= 0) {
    tok = {
      start: varOpen,
      end: naturalVarEnd(value, cursor),
      text: value.slice(varOpen, cursor),
    };
  }
  const items = completionItems(value, tok.start, tok.text, state);
  if (!items.length) {
    suggest = { visible: false, items: [], selected: 0,
                tokenStart: 0, tokenEnd: 0, tokenText: "", cycling: false };
    popup.classList.remove("open");
    popup.setAttribute("aria-hidden", "true");
    return;
  }
  suggest = {
    visible: true, items, selected: 0,
    tokenStart: tok.start, tokenEnd: tok.end, tokenText: tok.text,
    cycling: false,
  };
  renderSuggest();
}

function hideSuggest() {
  suggest.visible = false;
  suggest.cycling = false;
  popup.classList.remove("open");
  popup.setAttribute("aria-hidden", "true");
}

// /commands that DON'T need a trailing space when accepted (zero arity, or
// /raw which is itself a flag that precedes the prompt text).
const NO_TRAILING_SPACE_CMDS = new Set([
  "/help", "/?", "/quit", "/exit", "/q", "/raw",
]);

// Tab/Shift-Tab driver: advance selection, rewrite the token in the textarea.
// First Tab "accepts" the already-highlighted top match (selected stays at 0
// the very first time, advances on each subsequent Tab). No trailing space —
// this is preview-style cycling; commit happens via Enter or click.
function cycleSuggest(direction) {
  if (!suggest.visible || !suggest.items.length) return false;
  const len = suggest.items.length;
  if (suggest.cycling) {
    suggest.selected = (suggest.selected + direction + len) % len;
  }
  suggest.cycling = true;
  const choice = suggest.items[suggest.selected].label;
  const before = input.value.slice(0, suggest.tokenStart);
  const after  = input.value.slice(suggest.tokenEnd);
  input.value = before + choice + after;
  const newCursor = suggest.tokenStart + choice.length;
  input.setSelectionRange(newCursor, newCursor);
  suggest.tokenEnd = newCursor;
  // Manually grow + redraw the highlight layer — we don't go through
  // `updateSuggest` because that would rebuild the items list with the
  // just-replaced text and we'd lose the cycle state.
  autoResize();
  renderSuggest();
  return true;
}

function moveSuggestSelection(direction) {
  if (!suggest.visible || !suggest.items.length) return false;
  suggest.selected = popupArrowSelection(
    suggest.selected, suggest.items.length, direction < 0 ? "ArrowUp" : "ArrowDown",
  );
  suggest.cycling = false;
  renderSuggest();
  return true;
}

// "Commit" — used by Enter and click. Inserts the highlighted item AND a
// trailing space for /commands that take arguments, then dismisses the
// popup. For /cmd accepts the popup re-opens immediately at the new
// cursor position so the user can chain through model/sampler/preset
// without retyping anything.
function commitSuggest() {
  if (!suggest.visible || !suggest.items.length) return false;
  suggest.cycling = true;
  const choice = suggest.items[suggest.selected].label;
  const isCmdWithArgs = choice.startsWith("/") && !NO_TRAILING_SPACE_CMDS.has(choice);
  const inject = choice + (isCmdWithArgs ? " " : "");
  const before = input.value.slice(0, suggest.tokenStart);
  const after  = input.value.slice(suggest.tokenEnd);
  input.value = before + inject + after;
  const newCursor = suggest.tokenStart + inject.length;
  input.setSelectionRange(newCursor, newCursor);
  suggest.tokenEnd = newCursor;
  autoResize();
  hideSuggest();
  // After a /cmd commit, immediately surface the next-arg suggestions so
  // /model<Enter> flows straight into picking the model name.
  if (isCmdWithArgs) updateSuggest();
  return true;
}

// Click on an item: select it, then commit (with trailing space + popup
// re-opens for the next context, just like Enter).
popup.addEventListener("mousedown", (e) => {
  const target = e.target.closest(".suggest-item");
  if (!target) return;
  e.preventDefault();  // keep focus in the textarea
  suggest.selected = Number(target.dataset.i) || 0;
  commitSuggest();
});

input.addEventListener("blur", () => {
  // Defer so a click on the popup can still register before we hide.
  setTimeout(hideSuggest, 100);
});

input.addEventListener("keydown", (e) => {
  // Ctrl/Cmd+/ — toggle line comments over the selection (or cursor line).
  if (e.key === "/" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    const r = toggleComment(input.value, input.selectionStart, input.selectionEnd);
    input.value = r.value;
    input.setSelectionRange(r.selectionStart, r.selectionEnd);
    autoResize();
    updateSuggest();
    return;
  }
  // Tab / Shift-Tab — popup cycling. Open the popup if it isn't already.
  if (e.key === "Tab") {
    e.preventDefault();
    if (!suggest.visible) updateSuggest();
    cycleSuggest(e.shiftKey ? -1 : +1);
    return;
  }
  // Escape always dismisses the popup if it's open; otherwise harmless.
  if (e.key === "Escape" && suggest.visible) {
    e.preventDefault();
    hideSuggest();
    return;
  }
  // Enter-key key bindings:
  //   * Shift+Enter   — queue (submit) but don't clear the textarea.
  //                     Lets the user iterate on the same prompt.
  //   * Ctrl+Enter    — queue (submit) and clear the textarea.
  //                     The "I'm done with this prompt" gesture.
  //   * Enter (popup) — accept the highlighted completion item.
  //   * Enter (plain) — insert a newline, like a normal multi-line editor.
  // Submit gestures take precedence over the popup-open accept so the
  // user can always submit even with the popup showing.
  if (e.key === "Enter" && e.shiftKey && !e.ctrlKey) {
    e.preventDefault();
    hideSuggest();
    submit({ keepValue: true });
    return;
  }
  if (e.key === "Enter" && e.ctrlKey) {
    e.preventDefault();
    hideSuggest();
    submit();
    return;
  }
  if (e.key === "Enter" && suggest.visible) {
    e.preventDefault();
    commitSuggest();
    return;
  }
  if (e.key === "Enter") {
    // Plain Enter inserts a newline. Default textarea behaviour does
    // exactly that, so don't preventDefault — let the browser handle
    // it and then update the highlight/resize via the input event.
    return;
  }
  // Arrows behave like a normal multiline editor except at absolute text
  // boundaries: start + Up enters older prompt history, end + Down returns
  // toward the current draft.
  if (e.key === "ArrowUp" || e.key === "ArrowDown") {
    if (suggest.visible) {
      e.preventDefault();
      moveSuggestSelection(e.key === "ArrowUp" ? -1 : +1);
      return;
    }
    const action = promptArrowAction(
      input.value, input.selectionStart, input.selectionEnd, e.key,
    );
    if (action === "native") return;
    e.preventDefault();
    if (action === "move-start") {
      input.setSelectionRange(0, 0);
      updateSuggest();
    } else if (action === "move-end") {
      moveCursorEnd();
      updateSuggest();
    } else if (action === "history-prev") {
      const h = LS.history();
      if (!h.length) return;
      if (historyIdx === -1) { editingDraft = input.value; historyIdx = 0; }
      else if (historyIdx < h.length - 1) historyIdx += 1;
      input.value = h[historyIdx];
      moveCursorEnd();
      autoResize();
      updateSuggest();
    } else if (action === "history-next") {
      if (historyIdx === -1) return;
      historyIdx -= 1;
      input.value = historyIdx < 0 ? editingDraft : LS.history()[historyIdx];
      moveCursorEnd();
      autoResize();
      updateSuggest();
    }
    return;
  }
  if (e.key.length === 1 || e.key === "Backspace" || e.key === "Delete") historyIdx = -1;
});

// Recompute suggestions on every value or cursor change.
input.addEventListener("input", updateSuggest);
input.addEventListener("click", updateSuggest);
input.addEventListener("keyup", (e) => {
  // Arrow keys don't fire "input" but they do move the cursor — keep popup in sync.
  if (e.key.startsWith("Arrow") && !suggest.visible) updateSuggest();
});

function moveCursorEnd() {
  const n = input.value.length; input.setSelectionRange(n, n);
}

async function submit({ keepValue = false } = {}) {
  const line = input.value.trim();
  if (!line) return;
  // Record everything except bare commands in the recent list.
  if (!line.startsWith("/") || line.includes(" ")) pushRecent(line);
  // optimistic clear feels nicer; restore on error
  const saved = input.value;
  if (!keepValue) {
    input.value = ""; historyIdx = -1;
    hideSuggest();
    autoResize();
  }
  const sentLine = applyGlobalsToLine(line);
  try { await wsRequest("exec", { line: sentLine, profile: activeProfileName() }); }
  catch (e) {
    appendLog(`exec: ${e.message}`, "error");
    if (!keepValue) { input.value = saved; autoResize(); }
  }
}

// ---------- outputs pane show/hide ----------
// Hides both `.right` and the `.splitter` so the inference column gets
// the full width. State persists in localStorage; the topbar pill is
// the only affordance to bring it back.
let outputsShown = JSON.parse(localStorage.getItem("zimt.outputs-shown") || "true");
function applyOutputsShown() {
  const right = $("right-pane");
  const splitter = $("splitter");
  const btn = $("btn-toggle-outputs");
  right.hidden = !outputsShown;
  splitter.hidden = !outputsShown;
  btn.textContent = outputsShown ? "outputs ◀" : "outputs ▶";
  btn.setAttribute("aria-pressed", outputsShown ? "true" : "false");
  btn.title = outputsShown ? "hide outputs panel" : "show outputs panel";
}
function setupOutputsToggle() {
  $("btn-toggle-outputs").onclick = () => {
    outputsShown = !outputsShown;
    localStorage.setItem("zimt.outputs-shown", JSON.stringify(outputsShown));
    applyOutputsShown();
  };
  applyOutputsShown();
}

// ---------- splitter (drag-resizable right pane) ----------
function setupSplitter() {
  const splitter = $("splitter");
  const right = $("right-pane");
  // restore previous width
  const saved = parseInt(localStorage.getItem("zimt.right-w") || "", 10);
  if (Number.isFinite(saved) && saved > 200 && saved < 2000) right.style.width = saved + "px";

  let dragging = false, startX = 0, startW = 0;
  splitter.addEventListener("mousedown", (e) => {
    dragging = true; startX = e.clientX; startW = right.offsetWidth;
    document.body.style.cursor = "col-resize"; document.body.style.userSelect = "none";
    e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const dx = startX - e.clientX;
    const w = Math.max(260, Math.min(window.innerWidth - 300, startW + dx));
    right.style.width = w + "px";
  });
  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false; document.body.style.cursor = ""; document.body.style.userSelect = "";
    localStorage.setItem("zimt.right-w", String(right.offsetWidth));
  });
}

// ---------- init ----------
// ---------- GPU stats pill ----------
function fmtGiB(bytes) { return (bytes / (1024**3)).toFixed(1); }
function onGpuStats(s) {
  const el = $("gpu-stats");
  if (!s) { el.textContent = ""; el.title = ""; return; }
  const dev = (s.device || "").replace(/^Intel\(R\) /, "").replace(/ Graphics$/, "");
  if (s.total_bytes > 0) {
    const pct = Math.round(100 * s.allocated_bytes / s.total_bytes);
    el.textContent = `${dev}  ${fmtGiB(s.allocated_bytes)}/${fmtGiB(s.total_bytes)} GB`;
    el.title = `allocated ${fmtGiB(s.allocated_bytes)} GB  ·  `
             + `reserved ${fmtGiB(s.reserved_bytes)} GB  ·  `
             + `total ${fmtGiB(s.total_bytes)} GB  (${pct}%)`;
  } else {
    el.textContent = `${dev}  ${fmtGiB(s.allocated_bytes)} GB`;
    el.title = `RSS ${fmtGiB(s.allocated_bytes)} GB`;
  }
}

// ---------- GPU pill dropdown menu (unload model, …) ----------
function _positionGpuMenu() {
  // Align the menu's right edge with the pill's right edge, just below
  // the pill. position:fixed coords from the pill's bounding rect; this
  // is robust to scroll and responsive layout shifts.
  const pill = $("gpu-stats");
  const menu = $("gpu-menu");
  if (!pill || !menu) return;
  const r = pill.getBoundingClientRect();
  const menuW = menu.offsetWidth || 140;
  // Right-align under the pill, but never run off the left edge.
  const rightEdge = Math.max(menuW + 4, r.right);
  menu.style.left = Math.round(rightEdge - menuW) + "px";
  menu.style.top  = Math.round(r.bottom + 4) + "px";
}

function _setGpuMenu(open) {
  const pill = $("gpu-stats");
  const menu = $("gpu-menu");
  if (!pill || !menu) return;
  pill.setAttribute("aria-expanded", open ? "true" : "false");
  menu.hidden = !open;
  if (open) {
    // Refresh the unload item's enabled-state on each open. A model
    // must be loaded; the backend also refuses while generations
    // are in flight, but the user-visible signal is "no model" only.
    const unload = $("gpu-menu-unload");
    if (unload) {
      unload.disabled = !(state && state.loaded);
      unload.title = unload.disabled ? "no model is currently loaded" : "release the pipeline and free device memory";
    }
    _positionGpuMenu();
  }
}

function _initGpuMenu() {
  if (typeof document === "undefined" ||
      typeof document.addEventListener !== "function") return;
  const pill = $("gpu-stats");
  const menu = $("gpu-menu");
  const unload = $("gpu-menu-unload");
  if (!pill || !menu || !unload) return;
  pill.addEventListener("click", (ev) => {
    ev.stopPropagation();
    _setGpuMenu(menu.hidden);
  });
  pill.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      _setGpuMenu(menu.hidden);
    } else if (ev.key === "Escape") {
      _setGpuMenu(false);
    }
  });
  unload.addEventListener("click", async () => {
    _setGpuMenu(false);
    try {
      const r = await wsRequest("model_unload");
      if (r && r.reason) appendLog(r.reason);
      else appendLog("model unloaded");
    } catch (e) {
      appendLog(`unload: ${e.message}`, "error");
    }
  });
  // Click anywhere outside the menu (or press Escape globally) closes it.
  document.addEventListener("click", (ev) => {
    if (menu.hidden) return;
    if (menu.contains(ev.target) || pill.contains(ev.target)) return;
    _setGpuMenu(false);
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && !menu.hidden) _setGpuMenu(false);
  });
  // Keep the menu anchored to the pill when the viewport shifts.
  window.addEventListener("resize", () => { if (!menu.hidden) _positionGpuMenu(); });
  window.addEventListener("scroll", () => { if (!menu.hidden) _positionGpuMenu(); },
                         { passive: true });
}
_initGpuMenu();

// ---------- profiles ----------
// A profile is a named, isolated set of {prompt history, favourites,
// globals}, persisted server-side in SQLite. Selection is per-browser-tab:
// the active id lives in localStorage and travels as `profile_id` on every
// scoped RPC. The server keeps no active-profile state, so nothing here
// goes through the `state` broadcast.
const PROFILE_LS_KEY = "zimt.profile-id";
const MIGRATED_LS_KEY = "zimt.db-migrated";
let activeProfileId = null;
let profilesList = [];          // [{id, name, prompt_count}]

function _activeProfile() {
  return profilesList.find(p => p.id === activeProfileId) || null;
}
function activeProfileName() {
  const p = _activeProfile();
  return p ? p.name : null;
}
function renderProfileButton() {
  const p = _activeProfile();
  $("profile-name").textContent = p ? p.name : "profile";
}

// Parse legacy localStorage history into [{text, fav}] (display order),
// tolerating the bare-string entries the old store also accepted.
function _legacyHistory() {
  let raw;
  try { raw = JSON.parse(localStorage.getItem("zimt.history") || "[]"); }
  catch { return []; }
  if (!Array.isArray(raw)) return [];
  const out = [], seen = new Set();
  for (const e of raw) {
    const o = typeof e === "string" ? { text: e, fav: false } : e;
    if (!o || typeof o.text !== "string" || !o.text || seen.has(o.text)) continue;
    seen.add(o.text);
    out.push({ text: o.text, fav: Boolean(o.fav) });
  }
  return out;
}

// One-time migration of this browser's localStorage prompts/globals into
// the server DB. Runs only on a genuinely fresh DB (a single empty Default
// profile), so a browser connecting to an already-populated server adopts
// the server data instead of merging/duplicating. Idempotent via
// MIGRATED_LS_KEY.
async function _maybeMigrate(defaultId) {
  if (localStorage.getItem(MIGRATED_LS_KEY)) return false;
  const prompts = _legacyHistory();
  const globals = localStorage.getItem("zimt.globals") || "";
  if (!prompts.length && !globals) {
    localStorage.setItem(MIGRATED_LS_KEY, "1");
    return false;
  }
  try {
    if (globals) {
      await wsRequest("profile_globals_set", { profile_id: defaultId, text: globals });
    }
    // Push oldest-first so the newest ends up at the front; re-apply fav
    // flags after insert (prompt_push always stores fav=0 for new rows).
    for (let i = prompts.length - 1; i >= 0; i--) {
      await wsRequest("prompt_push", { profile_id: defaultId, text: prompts[i].text });
      if (prompts[i].fav) {
        await wsRequest("prompt_fav",
          { profile_id: defaultId, text: prompts[i].text, fav: true });
      }
    }
    localStorage.setItem(MIGRATED_LS_KEY, "1");
    appendLog(`migrated ${prompts.length} prompt(s) into "Default"`);
    return true;
  } catch (e) {
    appendLog(`migrate: ${e.message}`, "error");
    return false;
  }
}

async function _refreshProfiles() {
  const r = await wsRequest("profiles_list");
  profilesList = Array.isArray(r.profiles) ? r.profiles : [];
  return profilesList;
}

// Load a profile's prompts + globals into the in-memory caches and repaint.
async function hydrate(id) {
  try {
    const p = await wsRequest("profile_get", { profile_id: id });
    activeProfileId = p.id;
    localStorage.setItem(PROFILE_LS_KEY, String(p.id));
    promptCache = Array.isArray(p.prompts) ? p.prompts : [];
    _setGlobalsTextareaValue(typeof p.globals === "string" ? p.globals : "");
    renderRecent();
    renderProfileButton();
    // The gallery is profile-scoped — reload it for the newly active profile.
    await refreshOutputs();
  } catch (e) {
    appendLog(`profile load: ${e.message}`, "error");
  }
}

async function bootProfiles() {
  let profiles;
  try { profiles = await _refreshProfiles(); }
  catch (e) { appendLog(`profiles: ${e.message}`, "error"); return; }
  if (!profiles.length) return;   // server always auto-creates a Default

  // Migrate into the lone empty Default the very first time only.
  const lone = profiles.length === 1 ? profiles[0] : null;
  if (lone && lone.prompt_count === 0) {
    const migrated = await _maybeMigrate(lone.id);
    if (migrated) await _refreshProfiles();
  } else {
    localStorage.setItem(MIGRATED_LS_KEY, "1");
  }

  const storedId = parseInt(localStorage.getItem(PROFILE_LS_KEY) || "", 10);
  const exists = profilesList.find(p => p.id === storedId);
  await hydrate(exists ? storedId : profilesList[0].id);
  renderProfileMenu();
}

// ----- dropdown menu + settings dialog -----
function _positionProfileMenu() {
  const btn = $("profile-btn"), menu = $("profile-menu");
  if (!btn || !menu) return;
  const r = btn.getBoundingClientRect();
  const menuW = menu.offsetWidth || 200;
  const left = Math.max(4, Math.min(r.left, window.innerWidth - menuW - 4));
  menu.style.left = Math.round(left) + "px";
  menu.style.top = Math.round(r.bottom + 4) + "px";
}
function _setProfileMenu(open) {
  const btn = $("profile-btn"), menu = $("profile-menu");
  if (!btn || !menu) return;
  btn.setAttribute("aria-expanded", open ? "true" : "false");
  menu.hidden = !open;
  if (open) { renderProfileMenu(); _positionProfileMenu(); }
}
function renderProfileMenu() {
  const menu = $("profile-menu");
  if (!menu) return;
  menu.innerHTML = "";
  for (const p of profilesList) {
    const row = document.createElement("div");
    row.className = "profile-row" + (p.id === activeProfileId ? " active" : "");
    const pick = document.createElement("button");
    pick.type = "button"; pick.className = "profile-pick"; pick.role = "menuitem";
    const check = document.createElement("span");
    check.className = "profile-check";
    check.textContent = p.id === activeProfileId ? "✓" : "";
    pick.appendChild(check);
    pick.appendChild(document.createTextNode(p.name));
    pick.onclick = () => { _setProfileMenu(false); switchProfile(p.id); };
    const gear = document.createElement("button");
    gear.type = "button"; gear.className = "profile-gear";
    gear.title = "profile settings"; gear.textContent = "⚙";
    gear.onclick = (ev) => { ev.stopPropagation(); _setProfileMenu(false); openProfileSettings(p.id); };
    row.appendChild(pick); row.appendChild(gear);
    menu.appendChild(row);
  }
  const sep = document.createElement("div");
  sep.className = "profile-menu-sep";
  menu.appendChild(sep);
  const add = document.createElement("button");
  add.type = "button"; add.className = "profile-add"; add.role = "menuitem";
  add.textContent = "+ add profile";
  add.onclick = () => { _setProfileMenu(false); addProfile(); };
  menu.appendChild(add);
}
async function switchProfile(id) {
  if (id === activeProfileId) return;
  await hydrate(id);
}
async function addProfile() {
  const name = (window.prompt("New profile name:") || "").trim();
  if (!name) return;
  try {
    const r = await wsRequest("profile_create", { name });
    await _refreshProfiles();
    await hydrate(r.profile.id);
  } catch (e) { appendLog(`add profile: ${e.message}`, "error"); }
}
let _settingsProfileId = null;
function openProfileSettings(id) {
  const p = profilesList.find(x => x.id === id);
  if (!p) return;
  _settingsProfileId = id;
  $("profile-modal-name").textContent = p.name;
  $("profile-modal-count").textContent =
    `${p.prompt_count} prompt${p.prompt_count === 1 ? "" : "s"} stored`;
  const del = $("profile-modal-delete");
  del.disabled = profilesList.length <= 1;
  del.title = del.disabled ? "cannot delete the last remaining profile" : "";
  $("profile-modal").classList.add("open");
}
function _closeProfileSettings() {
  $("profile-modal").classList.remove("open");
  _settingsProfileId = null;
}
function _initProfileUi() {
  const btn = $("profile-btn"), menu = $("profile-menu");
  if (!btn || !menu) return;
  btn.addEventListener("click", (ev) => { ev.stopPropagation(); _setProfileMenu(menu.hidden); });
  btn.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); _setProfileMenu(menu.hidden); }
    else if (ev.key === "Escape") _setProfileMenu(false);
  });
  document.addEventListener("click", (ev) => {
    if (menu.hidden) return;
    if (menu.contains(ev.target) || btn.contains(ev.target)) return;
    _setProfileMenu(false);
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && !menu.hidden) _setProfileMenu(false);
  });
  window.addEventListener("resize", () => { if (!menu.hidden) _positionProfileMenu(); });
  window.addEventListener("scroll", () => { if (!menu.hidden) _positionProfileMenu(); },
                         { passive: true });

  $("profile-modal-close").onclick = _closeProfileSettings;
  $("profile-modal").onclick = (e) => { if (e.target === $("profile-modal")) _closeProfileSettings(); };
  $("profile-modal-delete").onclick = async () => {
    const id = _settingsProfileId;
    if (id == null) return;
    const p = profilesList.find(x => x.id === id);
    if (!confirm(`Delete profile "${p ? p.name : id}" and all its prompts? This cannot be undone.`)) return;
    try {
      await wsRequest("profile_delete", { profile_id: id });
      await _refreshProfiles();
      _closeProfileSettings();
      // If we deleted the active profile, fall back to the first remaining.
      if (id === activeProfileId) await hydrate(profilesList[0].id);
      else renderProfileMenu();
    } catch (e) { appendLog(`delete profile: ${e.message}`, "error"); }
  };
}
_initProfileUi();

// ---------- globals editor ----------
// A persistent block of template variable definitions (e.g.
//   ${shoes=[runners|sneakers|high heels|platform boots]}
// ) that's spliced into every generation line at submit time. Persisted
// server-side per active profile (see the profiles section); held in the
// `globalsText` in-memory cache, hydrated on profile switch and saved
// debounced on edit. The textarea/highlight pair mirrors the prompt
// editor's overlay technique so the same syntax highlighter
// (renderPromptHTML) paints both.
//
// Injection rules — see CLAUDE.md `src/zimt/repl/commands.py` for the
// command-line grammar that drives placement:
//   * If the line carries a non-empty prompt portion (anything that
//     isn't a known /cmd or its args), splice globals in. Otherwise
//     skip — globals belong with generation, not with bare settings
//     commands like `/cfg 7`.
//   * Placement: after `/model <name>` if present, otherwise prepended.
//     The parser treats any non-command token anywhere on the line as
//     prompt text, so positioning is cosmetic — but "after /model X"
//     reads naturally and keeps the user's typed prompt intact.
const GLOBALS_BEGIN_MARKER = "<!-- globals:begin -->";
const GLOBALS_END_MARKER = "<!-- globals:end -->";
// Arity table mirrors COMMAND_ARITY in src/zimt/repl/commands.py. Kept
// inline so we don't have to thread it through every static file; the
// completion-contract test catches command-list drift.
const _GLOBALS_ARITY = {
  "/help": 0, "/?": 0, "/quit": 0, "/exit": 0, "/q": 0,
  "/model": 1, "/cfg": 1, "/steps": 1, "/seed": 1, "/res": 1,
  "/clip_skip": 1, "/sampler": 1, "/size": 2,
  "/raw": 0,
  "/many": "MANY",
  "/negprompt": "GREEDY", "/tokenize": "GREEDY",
  "/lora": 1,
  "/mem": "GREEDY",
};
function _linePromptAcc(line) {
  const toks = line.split(/\s+/).filter(Boolean);
  const acc = [];
  let i = 0;
  while (i < toks.length) {
    const t = toks[i];
    if (!(t in _GLOBALS_ARITY)) { acc.push(t); i += 1; continue; }
    const a = _GLOBALS_ARITY[t];
    if (a === 0) { i += 1; continue; }
    if (a === "GREEDY") {
      i += 1;
      while (i < toks.length && !(toks[i] in _GLOBALS_ARITY)) i += 1;
      continue;
    }
    if (a === "MANY") {
      i += 1;
      if (i < toks.length) i += 1; // N
      // /many's greedy arg is the prompt (see exec_api.py /many branch).
      // Treat those tokens as prompt content so a line like
      // `/many 4 a cute girl` still triggers globals injection.
      while (i < toks.length && !(toks[i] in _GLOBALS_ARITY)) {
        acc.push(toks[i]);
        i += 1;
      }
      continue;
    }
    i += 1 + a; // fixed arity
  }
  return acc.join(" ");
}
// globalsText holds the active profile's globals block in memory; it is
// hydrated from the server on profile switch and persisted (debounced) on
// edit. Injection (applyGlobalsToLine) and markers are unchanged.
let globalsText = "";
let _globalsTextarea = null;
let _globalsSaveTimer = null;
function getGlobalsText() {
  return globalsText;
}
function _setGlobalsTextareaValue(text) {
  globalsText = text;
  if (_globalsTextarea) {
    _globalsTextarea.value = text;
    const hl = $("globals-highlight");
    if (hl) hl.innerHTML = renderPromptHTML(text);
  }
}
function _saveGlobalsDebounced() {
  if (activeProfileId == null) return;
  if (_globalsSaveTimer) clearTimeout(_globalsSaveTimer);
  const pid = activeProfileId;
  _globalsSaveTimer = setTimeout(() => {
    wsRequest("profile_globals_set", { profile_id: pid, text: globalsText })
      .catch(err => appendLog(`globals: ${err.message}`, "error"));
  }, 400);
}
function applyGlobalsToLine(line) {
  const g = getGlobalsText().trim();
  if (!g) return line;
  // Skip injection when the line has no prompt portion — no generation
  // will be triggered, so globals would be noise.
  if (!_linePromptAcc(line)) return line;
  // Normalise globals: collapse any internal newlines/tabs to single
  // spaces. The line itself stays single-line because the textarea is
  // submitted as-is.
  const flat = g.replace(/\s+/g, " ").trim();
  // Wrap the injected globals in dynamics comment markers so the spliced
  // region is identifiable in the sent prompt. `<!-- ... -->` is stripped
  // by the dynamics parser, so the markers never reach the model.
  const wrapped = `${GLOBALS_BEGIN_MARKER} ${flat} ${GLOBALS_END_MARKER}`;
  // Insert after `/model <name>` if present. Match the first occurrence
  // at a word boundary; trailing token is the model name.
  const re = /(?:^|\s)\/model\s+\S+/;
  const m = re.exec(line);
  if (m) {
    const at = m.index + m[0].length;
    return line.slice(0, at) + " " + wrapped + line.slice(at);
  }
  return wrapped + " " + line;
}
function _initGlobalsEditor() {
  const ta = $("globals-input");
  const hl = $("globals-highlight");
  if (!ta || !hl) return;
  _globalsTextarea = ta;
  ta.value = globalsText;
  const render = () => { hl.innerHTML = renderPromptHTML(ta.value); };
  render();
  ta.addEventListener("input", () => {
    globalsText = ta.value;
    render();
    _saveGlobalsDebounced();
  });
  ta.addEventListener("keydown", (e) => {
    if (e.key === "/" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      const r = toggleComment(ta.value, ta.selectionStart, ta.selectionEnd);
      ta.value = r.value;
      ta.setSelectionRange(r.selectionStart, r.selectionEnd);
      globalsText = ta.value;
      render();
      _saveGlobalsDebounced();
    }
  });
  ta.addEventListener("scroll", () => { hl.scrollTop = ta.scrollTop; });
}
_initGlobalsEditor();

// ---------- left-column tab switcher (inference / models) ----------
let leftTab = "inference";
let modelsBases = [];
let modelsLoras = [];

function setLeftTab(tab) {
  if (tab === leftTab) return;
  leftTab = tab;
  for (const b of document.querySelectorAll(".ltab")) {
    const on = b.dataset.ltab === tab;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
  }
  for (const p of document.querySelectorAll(".lpane")) {
    p.hidden = p.dataset.ltab !== tab;
  }
  if (tab === "models") refreshModels();
  if (tab === "inference") autoResize();
}

for (const b of document.querySelectorAll(".ltab")) {
  b.addEventListener("click", () => setLeftTab(b.dataset.ltab));
}

function fmtGB(bytes) {
  if (!bytes || bytes <= 0) return "0";
  return (bytes / (1024**3)).toFixed(2);
}

async function refreshModels() {
  try {
    const r = await wsRequest("models_info");
    modelsBases = r.bases || [];
    modelsLoras = r.loras || [];
    renderBases();
    renderLoras();
  } catch (e) { appendLog(`models_info: ${e.message}`, "error"); }
}

function _downloadingByName(kind) {
  // Map: name -> Job for the most-recent active download job of that kind.
  const out = new Map();
  for (const j of jobs.values()) {
    if (j.kind !== "download") continue;
    if (j.target_kind !== kind) continue;
    if (j.status !== "queued" && j.status !== "running") continue;
    out.set(j.model, j);
  }
  return out;
}

function _activeLoraMap() {
  const m = new Map();
  for (const e of state?.settings?.lora_stack ?? []) m.set(e.name, e.weight);
  return m;
}

function _addCommonMeta(li, m, kind, downloadingByName, opts = {}) {
  const meta = document.createElement("div");
  meta.className = "model-meta";
  if (m.repo_url) {
    const a = document.createElement("a");
    a.href = m.repo_url; a.target = "_blank"; a.rel = "noopener noreferrer";
    a.textContent = m.repo_id; a.title = m.repo_url;
    meta.appendChild(a);
  }
  const status = document.createElement("span");
  status.className = "model-status " + (m.installed ? "ok" : "missing");
  status.textContent = m.installed
    ? `installed · ${fmtGB(m.size_bytes)} GB`
    : "not installed";
  meta.appendChild(status);
  if (!m.is_builtin) {
    const b = document.createElement("span");
    b.className = "custom-badge"; b.textContent = "custom";
    meta.appendChild(b);
  }
  if (opts.tags && opts.tags.length) {
    const t = document.createElement("span");
    t.className = "model-tags";
    t.textContent = opts.tags.join(" · ");
    meta.appendChild(t);
  }
  li.appendChild(meta);

  const actions = document.createElement("div");
  actions.className = "model-actions";
  const dlJob = downloadingByName.get(m.name);

  const dlBtn = document.createElement("button");
  dlBtn.className = "section-btn";
  const dlState = _modelDlBtnState({ dlJob, installed: !!m.installed });
  dlBtn.textContent = dlState.text;
  dlBtn.disabled = dlState.disabled;
  dlBtn.title = dlState.title;
  dlBtn.onclick = async () => {
    dlBtn.disabled = true; dlBtn.textContent = "starting…";
    try {
      await wsRequest("model_download", { name: m.name, kind });
      appendLog(`download started: ${m.name}`);
      // Force a re-render so the button reverts from the optimistic "starting…"
      // text on the idempotent-collapse path (no fresh job event will arrive
      // for a pre-existing running job) — PR-07-D17.
      if (kind === "base") renderBases(); else renderLoras();
    } catch (e) {
      appendLog(`download: ${e.message}`, "error");
      dlBtn.disabled = false;
      dlBtn.textContent = dlState.text;
    }
  };
  actions.appendChild(dlBtn);

  return { meta, actions, dlJob };
}

function renderBases() {
  const root = $("models-list");
  root.innerHTML = "";
  const loadedName = state?.model;
  const downloading = _downloadingByName("base");
  for (const m of modelsBases) {
    const li = document.createElement("li");
    li.className = "model-item " + (m.installed ? "installed" : "missing")
                 + (m.name === loadedName ? " loaded" : "");

    const nameWrap = document.createElement("div");
    nameWrap.className = "model-name";
    nameWrap.textContent = m.name;
    const fam = document.createElement("span");
    fam.className = "model-family"; fam.textContent = m.family;
    nameWrap.appendChild(fam);
    li.appendChild(nameWrap);

    const desc = document.createElement("div");
    desc.className = "model-desc"; desc.textContent = m.description;
    li.appendChild(desc);

    const { actions, dlJob } = _addCommonMeta(li, m, "base", downloading,
      { tags: m.compatibility_tags || [] });

    const loadBtn = document.createElement("button");
    loadBtn.className = "section-btn";
    loadBtn.textContent = m.name === loadedName ? "loaded" : "load";
    loadBtn.disabled = m.name === loadedName || !!dlJob;
    loadBtn.title = "swap this in as the active model";
    loadBtn.onclick = async () => {
      loadBtn.disabled = true;
      try { await wsRequest("model_switch", { name: m.name }); }
      catch (e) { appendLog(`load: ${e.message}`, "error"); loadBtn.disabled = false; }
    };
    actions.appendChild(loadBtn);

    if (!m.is_builtin) actions.appendChild(_removeBtn("base", m.name));

    li.appendChild(actions);
    root.appendChild(li);
  }
}

function _modelDlBtnState({ dlJob, installed }) {
  if (dlJob) {
    // PR-02-D04: a job that didn't acquire the progress slot has no
    // tqdm events to report — show a waiting affordance instead.
    if (dlJob.progress_owner === false) {
      return {
        text: "downloading (waiting)…",
        disabled: true,
        title: "another download owns the progress slot",
      };
    }
    // PR-04 follow-up: per-unit slots. Prefer bytes percent (more granular)
    // and fall back to files when only files are populated.
    const bytesTotal = dlJob.download_bytes_total || 0;
    const filesTotal = dlJob.download_files_total || 0;
    let pct = null;
    let unit = "";
    if (bytesTotal > 0) {
      pct = Math.round(100 * (dlJob.download_bytes_n || 0) / bytesTotal);
      unit = "bytes";
    } else if (filesTotal > 0) {
      pct = Math.round(100 * (dlJob.download_files_n || 0) / filesTotal);
      unit = "files";
    }
    let text;
    if (pct == null) {
      text = "downloading…";
    } else if (unit === "bytes") {
      text = `downloading ${pct}% bytes`;
    } else if (unit === "files") {
      text = `downloading ${pct}% files`;
    } else {
      text = `downloading ${pct}%`;
    }
    return {
      text,
      disabled: true,
      title: "download in progress",
    };
  }
  if (installed) {
    return {
      text: "redownload",
      disabled: false,
      title: "re-fetch from HF (will refresh any updated weights)",
    };
  }
  return {
    text: "download",
    disabled: false,
    title: "fetch from HF without loading",
  };
}

function _loraAddBtnState({ compatible, loaded, dlJob, isActive, installed, compat, baseTags }) {
  // Active stack entries are always removable, regardless of install state —
  // the user must be able to clear the stack even if a LoRA gets uninstalled.
  const text = isActive ? "remove" : "add";
  const disabled = !compatible || !loaded || !!dlJob || (!isActive && !installed);
  let title;
  if (!compatible) {
    title = `incompatible: lora needs ${compat.join("/") || "?"}; base is ${baseTags.join("/") || "—"}`;
  } else if (!installed && !isActive) {
    title = "not installed — use the download button below first";
  } else if (isActive) {
    title = "remove from active LoRA stack";
  } else {
    title = "add to active LoRA stack";
  }
  return { text, disabled, title };
}

function renderLoras() {
  const root = $("loras-list");
  root.innerHTML = "";
  const downloading = _downloadingByName("lora");
  const active = _activeLoraMap();
  const base = modelsBases.find((b) => b.name === state?.model);
  const baseTags = new Set(base?.compatibility_tags || []);

  for (const m of modelsLoras) {
    const compat = (m.compatible_with || []);
    const compatible = !baseTags.size || compat.some((t) => baseTags.has(t));
    const isActive = active.has(m.name);

    const li = document.createElement("li");
    li.className = "model-item " + (m.installed ? "installed" : "missing")
                 + (compatible ? "" : " incompatible");

    const nameWrap = document.createElement("div");
    nameWrap.className = "model-name";
    nameWrap.textContent = m.name;
    const fam = document.createElement("span");
    fam.className = "model-family"; fam.textContent = "lora · " + m.family;
    nameWrap.appendChild(fam);
    if (isActive) {
      const badge = document.createElement("span");
      badge.className = "model-active-badge";
      badge.style.marginLeft = "6px";
      badge.textContent = `active @ ${active.get(m.name)}`;
      nameWrap.appendChild(badge);
    }
    li.appendChild(nameWrap);

    const desc = document.createElement("div");
    desc.className = "model-desc"; desc.textContent = m.description;
    li.appendChild(desc);

    if (m.trigger_tags) {
      const trig = document.createElement("div");
      trig.className = "model-trigger";
      trig.textContent = `trigger: ${m.trigger_tags}`;
      li.appendChild(trig);
    }

    const { actions, dlJob } = _addCommonMeta(li, m, "lora", downloading,
      { tags: compat });

    const addBtn = document.createElement("button");
    addBtn.className = "section-btn";
    const btnState = _loraAddBtnState({
      compatible, loaded: !!state?.loaded, dlJob,
      isActive, installed: !!m.installed,
      compat, baseTags: [...baseTags],
    });
    addBtn.textContent = btnState.text;
    addBtn.title = btnState.title;
    addBtn.disabled = btnState.disabled;
    addBtn.onclick = async () => {
      addBtn.disabled = true;
      const arg = isActive ? `-${m.name}` : m.name;
      try { await wsRequest("exec", { line: `/lora ${arg}` }); }
      catch (e) { appendLog(`/lora: ${e.message}`, "error"); addBtn.disabled = false; }
    };
    actions.appendChild(addBtn);

    if (!m.is_builtin) actions.appendChild(_removeBtn("lora", m.name));

    li.appendChild(actions);
    root.appendChild(li);
  }
}

function _removeBtn(kind, name) {
  const btn = document.createElement("button");
  btn.className = "section-btn";
  btn.textContent = "✕";
  btn.title = `remove custom ${kind} (deletes the JSON descriptor)`;
  btn.onclick = async () => {
    if (!confirm(`Delete custom ${kind} "${name}"?`)) return;
    btn.disabled = true;
    try {
      await wsRequest("custom_remove", { kind, name });
      appendLog(`removed custom ${kind}: ${name}`);
      await refreshModels();
    } catch (e) { appendLog(`remove: ${e.message}`, "error"); btn.disabled = false; }
  };
  return btn;
}

// ---------- custom-add modal ----------
let customKind = "lora";

function openCustomForm(kind) {
  customKind = kind;
  $("custom-modal-title").textContent = `add custom ${kind}`;
  $("custom-form").reset();
  $("custom-form-err").textContent = "";
  for (const el of document.querySelectorAll(".custom-lora-only")) {
    el.hidden = kind !== "lora";
  }
  for (const el of document.querySelectorAll(".custom-base-only")) {
    el.hidden = kind !== "base";
  }
  $("custom-modal").classList.add("open");
}

function closeCustomForm() {
  $("custom-modal").classList.remove("open");
}

$("btn-add-base").onclick = () => openCustomForm("base");
$("btn-add-lora").onclick = () => openCustomForm("lora");
$("custom-cancel").onclick = closeCustomForm;
$("custom-modal").onclick = (e) => {
  if (e.target === $("custom-modal")) closeCustomForm();
};
$("custom-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = new FormData(e.target);
  const tagsRaw = (form.get("tags") || "").trim();
  const tags = tagsRaw ? tagsRaw.split(",").map(s => s.trim()).filter(Boolean) : [];
  const desc = {
    name: form.get("name"),
    repo_id: form.get("repo_id"),
    description: form.get("description"),
    family: form.get("family"),
  };
  if (customKind === "lora") {
    desc.compatible_with = tags.length ? tags : [desc.family];
    if (form.get("weight_name")) desc.weight_name = form.get("weight_name");
    desc.default_weight = parseFloat(form.get("default_weight") || "1.0");
    if (form.get("trigger_tags")) desc.trigger_tags = form.get("trigger_tags");
  } else {
    desc.compatibility_tags = tags.length ? tags : [desc.family];
    desc.default_steps = parseInt(form.get("default_steps") || "28", 10);
    desc.default_cfg = parseFloat(form.get("default_cfg") || "6.0");
    desc.default_sampler = form.get("default_sampler") || "euler-a";
    if (form.get("score_tags")) desc.score_tags = form.get("score_tags");
    if (form.get("default_negative")) desc.default_negative = form.get("default_negative");
  }
  $("custom-form-err").textContent = "";
  try {
    await wsRequest("custom_add", { kind: customKind, descriptor: desc });
    appendLog(`added custom ${customKind}: ${desc.name}`);
    closeCustomForm();
    await refreshModels();
  } catch (err) {
    $("custom-form-err").textContent = err.message;
  }
});

// ---------- mobile tab toggle ----------
function setupMobileTabs() {
  function applyDefault() {
    // Default to the prompt view on first load when we're below the
    // CSS breakpoint. Doesn't reapply on resize — once the user has
    // picked a tab their choice sticks.
    if (window.innerWidth <= 800 &&
        !document.body.classList.contains("mobile-view-browser")) {
      document.body.classList.add("mobile-view-prompt");
    }
  }
  applyDefault();
  window.addEventListener("resize", applyDefault, { passive: true });

  for (const btn of document.querySelectorAll(".mtab")) {
    btn.addEventListener("click", () => {
      const view = btn.dataset.view;  // "prompt" | "browser"
      document.body.classList.remove("mobile-view-prompt", "mobile-view-browser");
      document.body.classList.add(`mobile-view-${view}`);
      for (const x of document.querySelectorAll(".mtab")) {
        x.classList.toggle("active", x === btn);
      }
    });
  }
}

async function init() {
  renderRecent();
  setupOutputsToggle();
  setupQueuePin();
  setupStatePanel();
  renderQueueSummary();
  setupSplitter();
  setupMobileTabs();
  autoResize();
  renderHighlight();
  connectWs();
  // State arrives as a "state" event on WS connect (server hello), so
  // no explicit fetch is needed. Outputs + profiles still need a one-shot
  // pull (the connection manager queues these until the socket is open).
  try { await refreshOutputs(); }
  catch (e) { appendLog(`init: ${e.message}`, "error"); }
  try { await bootProfiles(); }
  catch (e) { appendLog(`profiles init: ${e.message}`, "error"); }
}
init();
