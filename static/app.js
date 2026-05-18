const $ = (id) => document.getElementById(id);

// ---------- constants ----------
const COMMANDS = [
  "/help","/?","/raw","/many","/seed","/negprompt","/cfg","/steps","/size",
  "/res","/model","/tokenize","/quit","/exit","/q",
].sort();

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

// ---------- websocket ----------
let ws = null, reconnectTimer = null;
function connectWs() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen = () => { $("ws-state").textContent = "connected"; $("ws-state").className = "pill ok"; };
  ws.onclose = () => {
    $("ws-state").textContent = "disconnected"; $("ws-state").className = "pill err";
    reconnectTimer = setTimeout(connectWs, 1500);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
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
  };
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
  // bound size
  if (jobs.size > MAX_QUEUE_SHOWN) {
    const oldest = jobs.keys().next().value;
    jobs.delete(oldest);
  }
  renderQueue();
}

function renderQueue() {
  const root = $("queue-list");
  root.innerHTML = "";
  const arr = Array.from(jobs.values()).reverse();
  for (const j of arr) {
    const li = document.createElement("li");
    li.className = "queue-item status-" + j.status;
    const status = document.createElement("span"); status.className = "qstatus";
    status.textContent = j.status;
    const prompt = document.createElement("span"); prompt.className = "qprompt";
    prompt.textContent = (j.full_prompt || j.raw_prompt || "").slice(0, 200);
    prompt.title = j.full_prompt || j.raw_prompt || "";
    const seed = document.createElement("span"); seed.className = "qseed";
    seed.textContent = `seed=${j.seed}`;
    li.append(status, prompt, seed);
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
        try { await apiPost(`/api/jobs/${encodeURIComponent(j.id)}/cancel`, {}); }
        catch (e) { appendLog(`cancel: ${e.message}`, "error"); cancel.disabled = false; }
      };
      li.appendChild(cancel);
    }
    if (j.status === "error" && j.error) {
      const details = document.createElement("button");
      details.className = "details-btn"; details.textContent = "details";
      details.title = "show error message";
      details.onclick = () => showJobError(j);
      li.appendChild(details);
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
  try { await apiPost(`/api/outputs/${encodeURIComponent(entry.name)}/favorite`, { favorite: !entry.fav }); }
  catch (e) { appendLog(`favorite: ${e.message}`, "error"); }
}
async function loadOutputs(tab, page) {
  const r = await fetch(`/api/outputs?tab=${tab}&page=${page}&per_page=60`);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
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
    const r = await apiPost("/api/outputs/cleanup", {});
    appendLog(`cleaned ${r.deleted} files`);
    await refreshOutputs();
  } catch (e) { appendLog(`cleanup: ${e.message}`, "error"); }
};
$("btn-cancel-all").onclick = async () => {
  try {
    const r = await apiPost("/api/jobs/cancel-all", {});
    appendLog(`canceled ${r.canceled} jobs`);
  } catch (e) { appendLog(`cancel-all: ${e.message}`, "error"); }
};
$("btn-clear-completed").onclick = async () => {
  try {
    const r = await apiPost("/api/jobs/clear-completed", {});
    appendLog(`cleared ${r.cleared} completed jobs`);
  } catch (e) { appendLog(`clear-completed: ${e.message}`, "error"); }
};
$("btn-clear-log").onclick = () => { $("log").innerHTML = ""; };
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
  for (const p of LS.history()) {
    const li = document.createElement("li");
    li.className = "recent-item"; li.title = p; li.textContent = p;
    li.onclick = () => {
      $("prompt-input").value = p;
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
  const keys = ["model", "raw_prompt", "prompt", "negative_prompt", "seed",
                "steps", "cfg", "sampler", "clip_skip",
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

// Build a single /api/exec line that recreates a previous run, and load it
// into the prompt textarea. Shared between the modal's Restore button and
// the per-thumbnail ↻ reload button.
function restoreToPrompt(entry) {
  if (!entry) return;
  const meta = entry.metadata || {};
  const parts = [];
  if (meta.model) parts.push(`/model ${meta.model}`);
  if (meta.cfg)   parts.push(`/cfg ${meta.cfg}`);
  if (meta.steps) parts.push(`/steps ${meta.steps}`);
  if (meta.sampler) parts.push(`/sampler ${meta.sampler}`);
  if (meta.clip_skip && Number(meta.clip_skip) > 0) parts.push(`/clip_skip ${meta.clip_skip}`);
  if (meta.width && meta.height) parts.push(`/size ${meta.width} ${meta.height}`);
  if (meta.negative_prompt != null) parts.push(`/negprompt ${meta.negative_prompt}`);
  if (meta.seed)  parts.push(`/seed ${meta.seed}`);
  // Use the FULL composed prompt (`meta.prompt`) plus `/raw` to avoid
  // re-applying the model's auto-prefix. `raw_prompt` is what the user
  // typed; replaying it under /raw would silently drop the score-tags
  // and yield a different image for the same seed.
  const text = meta.prompt ?? meta.raw_prompt ?? "";
  if (text) parts.push("/raw", text);
  $("prompt-input").value = parts.join(" ");
  autoResize();
  $("modal").classList.remove("open");
  $("prompt-input").focus();
}
$("modal-restore").onclick = () => restoreToPrompt(modalEntry);

// ---------- prompt input: readline-style ----------
const input = $("prompt-input");
let historyIdx = -1;          // -1 = editing fresh; otherwise index into LS.history()
let editingDraft = "";        // saved when entering history mode

function tokenAtCursor(value, cursor) {
  // bounded by whitespace
  let s = cursor; while (s > 0 && !/\s/.test(value[s - 1])) s--;
  let e = cursor; while (e < value.length && !/\s/.test(value[e])) e++;
  return { start: s, end: e, text: value.slice(s, e) };
}

function tokensBefore(value, end) {
  return value.slice(0, end).split(/\s+/).filter(Boolean);
}

// Short one-line help shown next to /commands in the suggest popup.
const COMMAND_HELP = {
  "/help": "show command list",
  "/?": "show command list",
  "/quit": "exit (REPL only)",
  "/exit": "exit (REPL only)",
  "/q": "exit (REPL only)",
  "/raw": "skip the model's auto-prefix",
  "/many": "generate N images with sequential seeds",
  "/seed": "pin seed for next gen",
  "/cfg": "guidance scale",
  "/steps": "num inference steps",
  "/size": "set W H (free-form)",
  "/res": "pick a model-preset resolution",
  "/sampler": "switch scheduler",
  "/clip_skip": "SDXL only — skip top N CLIP layers",
  "/model": "load a model",
  "/tokenize": "per-encoder token analysis",
  "/negprompt": "set/clear negative prompt",
};

function completionItems(value, tokStart, tokText) {
  // Returns a list of {label, desc} suggestions for the token at the cursor,
  // filtered by what the user has typed so far.
  const before = tokensBefore(value, tokStart);
  const prev = before.length ? before[before.length - 1] : "";
  const lower = tokText.toLowerCase();

  if (prev === "/model") {
    const models = state?.models ?? [];
    return models
      .filter(m => m.name.toLowerCase().startsWith(lower))
      .map(m => ({ label: m.name, desc: m.description }));
  }
  if (prev === "/sampler") {
    if (!state?.loaded) return [];
    const model = (state.models ?? []).find(m => m.name === state.model);
    return (model?.samplers ?? [])
      .filter(s => s.toLowerCase().startsWith(lower))
      .map(s => ({ label: s, desc: "" }));
  }
  if (prev === "/res") {
    if (!state?.loaded) return [];
    const model = (state.models ?? []).find(m => m.name === state.model);
    const presets = model?.resolutions ?? [];
    // Orientation keywords: largest preset by area in each category.
    const byArea = (a, b) => (b.w * b.h) - (a.w * a.h);
    const sq = [...presets].sort(byArea).find(r => r.w === r.h);
    const ls = [...presets].sort(byArea).find(r => r.w > r.h);
    const pt = [...presets].sort(byArea).find(r => r.w < r.h);
    const orientations = [
      sq && { label: "square",    desc: `largest 1:1 (${sq.w}x${sq.h})` },
      ls && { label: "landscape", desc: `largest W>H (${ls.w}x${ls.h})` },
      pt && { label: "portrait",  desc: `largest W<H (${pt.w}x${pt.h})` },
    ].filter(Boolean);
    const explicit = presets.map(r => ({ label: `${r.w}x${r.h}`, desc: r.label }));
    return [...orientations, ...explicit]
      .filter(it => it.label.toLowerCase().startsWith(lower));
  }
  if (tokText.startsWith("/")) {
    return COMMANDS
      .filter(c => c.toLowerCase().startsWith(lower))
      .map(c => ({ label: c, desc: COMMAND_HELP[c] ?? "" }));
  }
  return [];
}

// Auto-grow the textarea wrap (which sizes the absolute-positioned overlay)
// to fit the content. Also re-renders the syntax-highlighted mirror.
const inputWrap = document.querySelector(".textarea-wrap");
const highlight = $("prompt-highlight");

function autoResize() {
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

function cursorOnFirstLine() {
  return input.value.lastIndexOf("\n", input.selectionStart - 1) === -1;
}
function cursorOnLastLine() {
  return input.value.indexOf("\n", input.selectionStart) === -1;
}

// ---------- intellisense-style suggestion popup ----------
// Updated on every input event; positioned at the textarea caret via a
// short-lived measurement mirror.
//
// Key handling — arrows are reserved for history navigation, so the popup
// is driven by Tab (sublime-style cycling). Each Tab rewrites the token
// at the cursor with the next candidate; Shift+Tab rewinds. The original
// typed prefix is preserved in `tokenText` so cycling stays consistent
// across replacements.
//
//   Tab        → cycle selection forward, rewrite token
//   Shift+Tab  → cycle selection backward, rewrite token
//   Esc        → dismiss popup (without undoing the current replacement)
//   click item → jump to that item + dismiss
//   typing     → rebuild popup from the new token, selected=0
//   ↑ / ↓      → history navigation (unchanged; popup never captures them)
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

function caretCoords() {
  // Stand up a hidden mirror with the same content+style as the textarea,
  // place a marker at the caret, measure, tear it down. ~1ms.
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

  const before = input.value.slice(0, input.selectionStart);
  mirror.textContent = before;
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
  // Position at caret (one line below).
  const c = caretCoords();
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
  const items = completionItems(value, tok.start, tok.text);
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
  // Enter while popup is open: accept the highlighted item (+ trailing space
  // for /commands that take args) and dismiss the popup. The next Enter
  // submits, unless the popup re-opened for the next-arg context.
  if (e.key === "Enter" && !e.shiftKey && suggest.visible) {
    e.preventDefault();
    commitSuggest();
    return;
  }
  // Enter (popup closed): submit. Shift+Enter inserts a newline.
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submit();
    return;
  }
  // Arrows always do history navigation when on the edge line of the
  // textarea — they NEVER navigate the popup. (Popup is Tab-driven so the
  // muscle memory for readline-style up/down history stays intact.)
  if (e.key === "ArrowUp" && cursorOnFirstLine()) {
    const h = LS.history();
    if (!h.length) return;
    if (historyIdx === -1) { editingDraft = input.value; historyIdx = 0; }
    else if (historyIdx < h.length - 1) historyIdx += 1;
    input.value = h[historyIdx];
    moveCursorEnd();
    autoResize();
    e.preventDefault();
    return;
  }
  if (e.key === "ArrowDown" && cursorOnLastLine()) {
    if (historyIdx === -1) return;
    historyIdx -= 1;
    input.value = historyIdx < 0 ? editingDraft : LS.history()[historyIdx];
    moveCursorEnd();
    autoResize();
    e.preventDefault();
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

async function submit() {
  const line = input.value.trim();
  if (!line) return;
  // Record everything except bare commands in the recent list.
  if (!line.startsWith("/") || line.includes(" ")) pushRecent(line);
  // optimistic clear feels nicer; restore on error
  const saved = input.value;
  input.value = ""; historyIdx = -1;
  hideSuggest();
  autoResize();
  try { await apiPost("/api/exec", { line }); }
  catch (e) {
    appendLog(`exec: ${e.message}`, "error");
    input.value = saved; autoResize();
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

// ---------- HTTP helper ----------
async function apiPost(path, body) {
  const r = await fetch(path, { method: "POST",
    headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
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

async function init() {
  renderRecent();
  setupSplitter();
  autoResize();
  renderHighlight();
  connectWs();
  try {
    const s = await (await fetch("/api/state")).json();
    onStateUpdate(s);
    await refreshOutputs();
  } catch (e) { appendLog(`init: ${e.message}`, "error"); }
}
init();
