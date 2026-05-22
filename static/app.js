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

const LS = {
  history: () => JSON.parse(localStorage.getItem("zimt.history") || "[]"),
  setHistory: (xs) => localStorage.setItem("zimt.history", JSON.stringify(xs.slice(0, 50))),
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
  else if (m.type === "output_added") onOutputAdded(m.entry);
  else if (m.type === "favorite_changed") onFavoriteChanged(m.entry);
  else if (m.type === "outputs_cleared") refreshOutputs();
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
function onOutputAdded(entry) {
  if (currentTab === "all") {
    outputs.unshift(entry); totalCount += 1; renderThumbs();
  }
}
function onFavoriteChanged(entry) {
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
  const root = $("thumbs"); root.innerHTML = "";
  for (const e of outputs) {
    const d = document.createElement("div"); d.className = "thumb";
    const img = document.createElement("img"); img.loading = "lazy";
    img.src = `/api/outputs/${encodeURIComponent(e.name)}?thumb=192`;
    d.appendChild(img);
    // Top-left: reload-to-prompt. Mirrors the modal's Restore button.
    const reload = document.createElement("button");
    reload.className = "reload-btn";
    reload.textContent = "↻";
    reload.title = "restore prompt + settings from this image";
    reload.onclick = (ev) => { ev.stopPropagation(); restoreToPrompt(e); };
    d.appendChild(reload);
    const reloadFreshSeed = document.createElement("button");
    reloadFreshSeed.className = "reload-btn reload-no-seed-btn";
    reloadFreshSeed.textContent = "↺";
    reloadFreshSeed.title = "restore prompt + settings, but use a new random seed";
    reloadFreshSeed.onclick = (ev) => {
      ev.stopPropagation();
      restoreToPrompt(e, { includeSeed: false });
    };
    d.appendChild(reloadFreshSeed);
    // Top-right: favorite toggle.
    const star = document.createElement("button");
    star.className = "star-btn" + (e.fav ? " on" : "");
    star.textContent = e.fav ? "★" : "☆";
    star.title = e.fav ? "remove favorite" : "favorite";
    star.onclick = (ev) => { ev.stopPropagation(); toggleFavorite(e); };
    d.appendChild(star);
    d.onclick = () => openModal(e);
    root.appendChild(d);
  }
  $("load-more").style.display = hasMore ? "" : "none";
}
async function toggleFavorite(entry) {
  try { await wsRequest("output_favorite", { name: entry.name, favorite: !entry.fav }); }
  catch (e) { appendLog(`favorite: ${e.message}`, "error"); }
}
async function loadOutputs(tab, page) {
  return wsRequest("outputs_list", { tab, page, per_page: 60 });
}
async function refreshOutputs() {
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
    const r = await wsRequest("outputs_cleanup");
    appendLog(`cleaned ${r.deleted} files`);
    await refreshOutputs();
  } catch (e) { appendLog(`cleanup: ${e.message}`, "error"); }
};
$("btn-cancel-all").onclick = async () => {
  try {
    const r = await wsRequest("jobs_cancel_all");
    appendLog(`canceled ${r.canceled} jobs`);
  } catch (e) { appendLog(`cancel-all: ${e.message}`, "error"); }
};
$("btn-clear-completed").onclick = async () => {
  try {
    const r = await wsRequest("jobs_clear_completed");
    appendLog(`cleared ${r.cleared} completed jobs`);
  } catch (e) { appendLog(`clear-completed: ${e.message}`, "error"); }
};
$("btn-clear-log").onclick = () => { $("log").innerHTML = ""; };
$("btn-refresh-models").onclick = () => refreshModels();
$("btn-generate").onclick = () => submit();
$("btn-clear-recent").onclick = () => {
  if (!confirm("Clear all recent prompts? This only affects this browser.")) return;
  LS.setHistory([]);
  renderRecent();
  appendLog("recent prompts cleared");
};

// ---------- recent prompts ----------
function renderRecent() {
  const root = $("recent-list"); root.innerHTML = "";
  const history = LS.history();
  const entries = history.length
    ? history.map((p) => ({ text: p, showcase: false }))
    : SHOWCASE_PROMPTS.map((p) => ({ text: p, showcase: true }));
  if (!history.length) {
    const hint = document.createElement("li");
    hint.className = "recent-hint";
    hint.textContent = "no recent prompts — click one to load it:";
    root.appendChild(hint);
  }
  for (const e of entries) {
    const li = document.createElement("li");
    li.className = "recent-item" + (e.showcase ? " showcase" : "");
    li.title = e.showcase ? `example: ${e.text}` : e.text;
    li.textContent = e.text;
    li.onclick = () => {
      $("prompt-input").value = e.text;
      $("prompt-input").focus();
      autoResize();
    };
    root.appendChild(li);
  }
}
function pushRecent(p) {
  const xs = LS.history().filter(x => x !== p);
  xs.unshift(p); LS.setHistory(xs); renderRecent();
}

// ---------- modal ----------
function updateModalFavButton() {
  if (!modalEntry) return;
  $("modal-fav").textContent = modalEntry.fav ? "★ unfavorite" : "☆ favorite";
}
function openModal(entry) {
  modalEntry = entry;
  $("modal-img").src = `/api/outputs/${encodeURIComponent(entry.name)}`;
  const grid = $("modal-meta"); grid.innerHTML = "";
  const keys = ["model", "repo_id", "repo_url",
                "raw_prompt", "prompt", "negative_prompt", "seed",
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
  $("modal").classList.add("open");
}
$("modal-close").onclick = () => $("modal").classList.remove("open");
$("modal").onclick = (e) => { if (e.target === $("modal")) $("modal").classList.remove("open"); };
$("modal-fav").onclick = () => { if (modalEntry) toggleFavorite(modalEntry); };

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
$("modal-restore").onclick = () => restoreToPrompt(modalEntry);

// ---------- prompt input: readline-style ----------
const input = $("prompt-input");
let historyIdx = -1;          // -1 = editing fresh; otherwise index into LS.history()
let editingDraft = "";        // saved when entering history mode

// Auto-grow the textarea wrap (which sizes the absolute-positioned overlay)
// to fit the content. Also re-renders the syntax-highlighted mirror.
const inputWrap = document.querySelector(".textarea-wrap");
const highlight = $("prompt-highlight");

function autoResize() {
  // Skip while the inference pane is hidden — the textarea has no layout
  // box, so scrollHeight is 0 and we'd otherwise collapse it permanently.
  if (input.offsetParent === null) return;
  input.style.height = "auto";
  const h = input.scrollHeight;
  input.style.height = h + "px";
  inputWrap.style.height = h + "px";
  highlight.style.height = h + "px";
  renderHighlight();
}
input.addEventListener("input", autoResize);
input.addEventListener("scroll", () => { highlight.scrollTop = input.scrollTop; });
window.addEventListener("resize", autoResize);

// ---------- prompt syntax highlighting ----------
// Command → arg-spec: {n: fixedN, greedy: true|false}
const HL_CMDS = {
  "/help": {n: 0}, "/?": {n: 0}, "/quit": {n: 0}, "/exit": {n: 0}, "/q": {n: 0},
  "/raw": {n: 0},
  "/model": {n: 1, cls: "hl-model"},
  "/lora": {n: 1, cls: "hl-lora"},
  "/sampler": {n: 1, cls: "hl-sampler"},
  "/cfg": {n: 1, cls: "hl-num"},
  "/steps": {n: 1, cls: "hl-num"},
  "/seed": {n: 1, cls: "hl-num"},
  "/clip_skip": {n: 1, cls: "hl-num"},
  "/res": {n: 1, cls: "hl-num"},
  "/size": {n: 2, cls: "hl-num"},
  "/many": {n: 1, cls: "hl-num", thenGreedy: true},
  "/negprompt": {greedy: true, cls: "hl-neg"},
  "/tokenize": {greedy: true},
};

function esc(s) {
  return s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
}

function renderHighlight() {
  const text = input.value;
  if (text === "") { highlight.innerHTML = "&nbsp;"; return; }
  // Tokenize keeping whitespace so the rendered string preserves the user's
  // exact formatting and the caret aligns with what they typed.
  const parts = text.split(/(\s+)/);
  let out = "";
  let argsLeft = 0;
  let argClass = "";
  let greedyClass = null;
  for (const p of parts) {
    if (/^\s+$/.test(p)) { out += p; continue; }
    if (p === "") continue;
    if (p in HL_CMDS) {
      const spec = HL_CMDS[p];
      out += `<span class="hl-cmd">${esc(p)}</span>`;
      if (spec.greedy) {
        greedyClass = spec.cls || "";
        argsLeft = 0;
      } else {
        argsLeft = spec.n;
        argClass = spec.cls || "";
        if (spec.thenGreedy) {
          // /many: N is one arg, then prompt is free-form (no highlight).
          // We render N as num, then drop out of arg mode.
        }
      }
      continue;
    }
    // Token that *looks* like a command but isn't recognised.
    if (p.startsWith("/")) {
      out += `<span class="hl-unknown-cmd">${esc(p)}</span>`;
      continue;
    }
    if (greedyClass !== null) {
      out += greedyClass ? `<span class="${greedyClass}">${esc(p)}</span>` : esc(p);
      continue;
    }
    if (argsLeft > 0) {
      out += argClass ? `<span class="${argClass}">${esc(p)}</span>` : esc(p);
      argsLeft -= 1;
      continue;
    }
    // Free-form prompt text — default colour.
    out += esc(p);
  }
  // Trailing newline guards against the browser collapsing the final line.
  if (text.endsWith("\n")) out += "\n";
  highlight.innerHTML = out;
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
  const tok = tokenAtCursor(value, cursor);
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
  // Shift+Enter: submit but DON'T clear the textarea (iterate-friendly).
  // Dismisses the popup as a side effect. Takes precedence over the
  // other Enter handlers regardless of popup state.
  if (e.key === "Enter" && e.shiftKey && !e.ctrlKey) {
    e.preventDefault();
    hideSuggest();
    submit({ keepValue: true });
    return;
  }
  // Ctrl+Enter: insert a newline (since Shift+Enter no longer does that).
  if (e.key === "Enter" && e.ctrlKey) {
    // Let the textarea handle Enter normally — but Ctrl+Enter doesn't
    // insert a newline by default either, so do it ourselves.
    e.preventDefault();
    const c = input.selectionStart;
    input.value = input.value.slice(0, c) + "\n" + input.value.slice(c);
    input.setSelectionRange(c + 1, c + 1);
    autoResize();
    return;
  }
  // Enter while popup is open: accept the highlighted item (+ trailing space
  // for /commands that take args) and dismiss the popup. The next Enter
  // submits, unless the popup re-opened for the next-arg context.
  if (e.key === "Enter" && suggest.visible) {
    e.preventDefault();
    commitSuggest();
    return;
  }
  // Enter (popup closed): submit + clear.
  if (e.key === "Enter") {
    e.preventDefault();
    submit();
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
  try { await wsRequest("exec", { line }); }
  catch (e) {
    appendLog(`exec: ${e.message}`, "error");
    if (!keepValue) { input.value = saved; autoResize(); }
  }
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
  setupSplitter();
  setupMobileTabs();
  autoResize();
  renderHighlight();
  connectWs();
  // State arrives as a "state" event on WS connect (server hello), so
  // no explicit fetch is needed. Outputs do still need a one-shot pull.
  try { await refreshOutputs(); }
  catch (e) { appendLog(`init: ${e.message}`, "error"); }
}
init();
