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
    else if (m.type === "log") appendLog(m.message, m.level);
    else if (m.type === "model_loading") appendLog(`loading ${m.model}…`);
    else if (m.type === "model_loaded")  appendLog(`loaded ${m.model}`);
    else if (m.type === "model_error")   appendLog(`load error: ${m.error}`, "error");
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
    `negprompt: ${JSON.stringify(st.negative_prompt ?? "")}`,
  ];
  if (st.score_tags) lines.push(`auto-prefix: ${JSON.stringify(st.score_tags)}`);
  return lines.join("\n");
}

function onStateUpdate(s) {
  state = s;
  $("model-state").textContent = s.loaded ? s.model : "no model";
  $("model-state").className = "pill" + (s.loaded ? " ok" : "");
  const pre = $("state-text");
  pre.textContent = fmtState(s);
  pre.classList.toggle("empty", !s.loaded);
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
                "steps", "cfg", "width", "height", "dtype", "device"];
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

// "Restore to prompt": build a single /api/exec line that recreates the run.
$("modal-restore").onclick = () => {
  if (!modalEntry) return;
  const meta = modalEntry.metadata || {};
  const parts = [];
  if (meta.model) parts.push(`/model ${meta.model}`);
  if (meta.cfg)   parts.push(`/cfg ${meta.cfg}`);
  if (meta.steps) parts.push(`/steps ${meta.steps}`);
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
};

// ---------- prompt input: readline-style ----------
const input = $("prompt-input");
let historyIdx = -1;          // -1 = editing fresh; otherwise index into LS.history()
let editingDraft = "";        // saved when entering history mode

// Tab completion cycling state
let tabCtx = null;            // {origValue, origCursor, tokenStart, tokenEnd, matches, idx}

function tokenAtCursor(value, cursor) {
  // bounded by whitespace
  let s = cursor; while (s > 0 && !/\s/.test(value[s - 1])) s--;
  let e = cursor; while (e < value.length && !/\s/.test(value[e])) e++;
  return { start: s, end: e, text: value.slice(s, e) };
}

function tokensBefore(value, end) {
  return value.slice(0, end).split(/\s+/).filter(Boolean);
}

function completionPool(value, tokStart, tokText) {
  // Pool depends on whether we're completing a /command or an argument of one.
  // Heuristic: look at the previous non-empty token; if it's /model, complete model names.
  const before = tokensBefore(value, tokStart);
  const prev = before.length ? before[before.length - 1] : "";
  if (prev === "/model") {
    return (state?.models ?? []).map(m => m.name);
  }
  // Default: only complete tokens that look like a /command.
  if (tokText.startsWith("/")) return COMMANDS;
  return [];
}

// Auto-grow the textarea to fit its contents, up to the CSS max-height.
function autoResize() {
  input.style.height = "auto";
  input.style.height = input.scrollHeight + "px";
}
input.addEventListener("input", autoResize);
window.addEventListener("resize", autoResize);

function cursorOnFirstLine() {
  return input.value.lastIndexOf("\n", input.selectionStart - 1) === -1;
}
function cursorOnLastLine() {
  return input.value.indexOf("\n", input.selectionStart) === -1;
}

input.addEventListener("keydown", (e) => {
  if (e.key === "Tab") {
    e.preventDefault();
    handleTab(e.shiftKey);
    return;
  }
  // Enter submits; Shift+Enter inserts a newline (textarea default).
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submit();
    return;
  }
  // Arrow up/down navigate history only when on the edge line of the textarea —
  // otherwise let the native caret motion happen (within the multi-line value).
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
  // Any other typing breaks tab/history cycling.
  tabCtx = null;
  if (e.key.length === 1 || e.key === "Backspace" || e.key === "Delete") historyIdx = -1;
});

function moveCursorEnd() {
  const n = input.value.length; input.setSelectionRange(n, n);
}

function handleTab(shift) {
  const value = input.value, cursor = input.selectionStart;
  // (Re)build context if cursor/value diverged from saved state.
  if (!tabCtx ||
      tabCtx.origValue !== value ||
      tabCtx.origCursor !== cursor) {
    const tok = tokenAtCursor(value, cursor);
    const pool = completionPool(value, tok.start, tok.text);
    const matches = pool.filter(p => p.startsWith(tok.text));
    if (matches.length === 0) return;
    tabCtx = {
      origValue: value, origCursor: cursor,
      tokenStart: tok.start, tokenEnd: tok.end,
      matches, idx: -1,
    };
  }
  tabCtx.idx = (tabCtx.idx + (shift ? -1 : 1) + tabCtx.matches.length) % tabCtx.matches.length;
  const choice = tabCtx.matches[tabCtx.idx];
  const newValue = tabCtx.origValue.slice(0, tabCtx.tokenStart) + choice
                 + tabCtx.origValue.slice(tabCtx.tokenEnd);
  const newCursor = tabCtx.tokenStart + choice.length;
  input.value = newValue;
  input.setSelectionRange(newCursor, newCursor);
  autoResize();
  // pin the saved state to the new value/cursor so the next Tab keeps cycling
  tabCtx.origValue = newValue; tabCtx.origCursor = newCursor;
  tabCtx.tokenEnd = newCursor;
}

async function submit() {
  const line = input.value.trim();
  if (!line) return;
  // Record everything except bare commands in the recent list.
  if (!line.startsWith("/") || line.includes(" ")) pushRecent(line);
  // optimistic clear feels nicer; restore on error
  const saved = input.value;
  input.value = ""; historyIdx = -1; tabCtx = null;
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
async function init() {
  renderRecent();
  setupSplitter();
  autoResize();
  connectWs();
  try {
    const s = await (await fetch("/api/state")).json();
    onStateUpdate(s);
    await refreshOutputs();
  } catch (e) { appendLog(`init: ${e.message}`, "error"); }
}
init();
