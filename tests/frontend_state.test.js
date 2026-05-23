// Tests for PR-09 defects: D15 (queue eviction), D16 (download unit percentage),
// D17 (dlBtn text after click), D18 (multi-LoRA restore line).
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");

// ---------- minimal DOM stubs (shared with model_download_state.test.js) ----------

class FakeClassList {
  constructor() { this.names = new Set(); }
  add(...names) { for (const n of names) this.names.add(n); }
  remove(...names) { for (const n of names) this.names.delete(n); }
  toggle(name, force) {
    const next = force === undefined ? !this.names.has(name) : Boolean(force);
    if (next) this.names.add(name); else this.names.delete(name);
    return next;
  }
  contains(name) { return this.names.has(name); }
}

class FakeElement {
  constructor(tagName = "div") {
    this.tagName = tagName;
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.classList = new FakeClassList();
    this.attributes = new Map();
    this.value = "";
    this.selectionStart = 0;
    this.selectionEnd = 0;
    this.scrollHeight = 24;
    this.scrollTop = 0;
    this.scrollLeft = 0;
    this.offsetParent = null;
  }
  appendChild(child) { this.children.push(child); child.parentNode = this; return child; }
  append(...nodes) { for (const n of nodes) this.appendChild(n); }
  removeChild(child) {
    const i = this.children.indexOf(child);
    if (i >= 0) this.children.splice(i, 1);
    return child;
  }
  get firstChild() { return this.children[0] ?? null; }
  set innerHTML(_html) { this.children = []; }
  get innerHTML() { return ""; }
  addEventListener() {}
  setAttribute(name, value) { this.attributes.set(name, value); }
  querySelector() { return null; }
  focus() {}
  setSelectionRange(s, e) { this.selectionStart = s; this.selectionEnd = e; }
  getBoundingClientRect() { return { left: 0, top: 0, height: 18 }; }
}

function loadApp(overrides = {}) {
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new FakeElement());
    return elements.get(id);
  };
  const storage = new Map();
  const context = {
    console,
    location: { protocol: "http:", host: "localhost" },
    navigator: { clipboard: { writeText() {} } },
    localStorage: {
      getItem(key) { return storage.get(key) ?? null; },
      setItem(key, value) { storage.set(key, String(value)); },
    },
    document: {
      body: new FakeElement("body"),
      title: "zimt",
      getElementById: element,
      createElement(tagName) { return new FakeElement(tagName); },
      querySelector(selector) {
        if (selector === ".textarea-wrap") return element("textarea-wrap");
        return null;
      },
      querySelectorAll() { return []; },
      // Stubbed for the modal-nav keydown listener; see comment in
      // tests/model_download_state.test.js.
      addEventListener() {},
    },
    window: {
      innerWidth: 1200,
      addEventListener() {},
    },
    FormData: class {
      constructor() { this.fields = new Map(); }
      get(name) { return this.fields.get(name) ?? ""; }
    },
    ZimtConnectionManager: class {
      constructor() {
        // Allow overrides.wsRequest to intercept conn.request calls.
        this._override = overrides.wsRequest;
      }
      request(method, params) {
        if (this._override) return this._override(method, params);
        if (method === "outputs_list") {
          return Promise.resolve({ entries: [], has_more: false, total: 0 });
        }
        return Promise.resolve({});
      }
    },
    buildRestorePromptLine() { return ""; },
    neighborInOutputs() { return null; },
    renderPromptHTML() { return ""; },
    unclosedVarOpenIndex() { return -1; },
    naturalVarEnd(text, cursor) { return cursor; },
    tokenAtCursor() { return { start: 0, end: 0, text: "" }; },
    completionItems() { return []; },
    promptArrowAction() { return "native"; },
    popupArrowSelection(selected) { return selected; },
    confirm() { return true; },
    getComputedStyle() {
      return { getPropertyValue() { return ""; }, lineHeight: "18px" };
    },
    requestAnimationFrame(callback) { callback(); },
    setInterval() { return 0; },
    setTimeout(callback) { callback(); return 0; },
    clearTimeout() {},
  };
  context.globalThis = context;
  context.window.window = context.window;
  context.window.document = context.document;

  const appPath = path.join(__dirname, "..", "static", "app.js");
  vm.runInNewContext(fs.readFileSync(appPath, "utf8"), context, { filename: appPath });
  return context;
}

// Collect model names from the rendered queue-list children.
// Each queue item has a .qprompt span with text "model · file · counter" or
// just "model · file" for downloads; the model name appears before the first "  ·  ".
function queueModelNames(app) {
  const root = app.document.getElementById("queue-list");
  const names = new Set();
  for (const li of root.children) {
    for (const child of li.children) {
      if (typeof child.className === "string" && child.className.includes("qprompt")) {
        const text = child.textContent || "";
        const name = text.split("  ·  ")[0].trim();
        if (name) names.add(name);
      }
    }
  }
  return names;
}

// ---------- D15: queue eviction skips running/queued entries ----------

test("D15: onJob evicts a completed entry, not a running one, when queue is full", () => {
  const app = loadApp();

  // Fill queue to MAX_QUEUE_SHOWN (30) with one running job first, rest done.
  app.onJob({
    id: "running-0",
    kind: "download",
    target_kind: "base",
    status: "running",
    model: "big-model",
    download_bytes_n: 0, download_bytes_total: 0, download_files_done: 0,
  });
  for (let i = 1; i < 30; i++) {
    app.onJob({
      id: `done-${i}`,
      kind: "download",
      target_kind: "base",
      status: "done",
      model: `model-${i}`,
      download_bytes_n: 100, download_bytes_total: 100, download_files_done: 1,
    });
  }
  // Queue is now exactly 30 (MAX_QUEUE_SHOWN). Adding one more triggers eviction.
  app.onJob({
    id: "done-trigger",
    kind: "download",
    target_kind: "base",
    status: "done",
    model: "trigger-model",
    download_bytes_n: 100, download_bytes_total: 100, download_files_done: 1,
  });

  const queueList = app.document.getElementById("queue-list");
  const names = queueModelNames(app);
  assert.equal(queueList.children.length, 30,
    "queue should remain at MAX_QUEUE_SHOWN after eviction");
  assert.ok(names.has("big-model"),
    "running job (big-model) must not be evicted when completed entries are available");
});

test("D15: onJob evicts the oldest active entry only when all entries are active", () => {
  const app = loadApp();

  // Fill queue with 30 running jobs — no completed entries available.
  for (let i = 0; i < 30; i++) {
    app.onJob({
      id: `running-${i}`,
      kind: "download",
      target_kind: "base",
      status: "running",
      model: `model-${i}`,
      download_bytes_n: 0, download_bytes_total: 0, download_files_done: 0,
    });
  }
  // Trigger eviction — all entries are running so oldest active must be evicted.
  app.onJob({
    id: "running-new",
    kind: "download",
    target_kind: "base",
    status: "running",
    model: "new-model",
    download_bytes_n: 0, download_bytes_total: 0, download_files_done: 0,
  });

  const queueList = app.document.getElementById("queue-list");
  const names = queueModelNames(app);
  assert.equal(queueList.children.length, 30,
    "queue must remain at MAX_QUEUE_SHOWN after last-resort eviction");
  assert.ok(!names.has("model-0"),
    "oldest running entry (model-0) should be evicted as last resort");
  assert.ok(names.has("new-model"),
    "newly added job must be present");
});

// ---------- D16: _modelDlBtnState uses unit-aware percentage ----------

test("D16: _modelDlBtnState shows bytes percentage with unit suffix when bytes slot is populated", () => {
  const app = loadApp();
  const r = app._modelDlBtnState({
    dlJob: { download_bytes_n: 512, download_bytes_total: 1024 },
    installed: false,
  });
  // Percent should be 50; text should indicate bytes unit.
  assert.match(r.text, /50/,
    `expected 50 in button text, got: ${r.text}`);
  assert.match(r.text, /bytes/,
    `expected 'bytes' unit label in button text, got: ${r.text}`);
  assert.equal(r.disabled, true);
});

test("D16: _modelDlBtnState shows files percentage with unit suffix when only files slot is populated", () => {
  const app = loadApp();
  const r = app._modelDlBtnState({
    dlJob: { download_files_n: 3, download_files_total: 7 },
    installed: false,
  });
  // Math.round(100 * 3 / 7) = 43
  assert.match(r.text, /43/,
    `expected 43 in button text, got: ${r.text}`);
  assert.match(r.text, /files/,
    `expected 'files' unit label in button text, got: ${r.text}`);
  assert.equal(r.disabled, true);
});

test("PR-04: _modelDlBtnState prefers bytes percent when both slots are populated", () => {
  const app = loadApp();
  // Real HF snapshot_download interleaves both bars; the button must pick
  // ONE percentage rather than oscillate. Bytes is more granular.
  const r = app._modelDlBtnState({
    dlJob: {
      download_files_n: 3, download_files_total: 7,       // 43% files
      download_bytes_n: 512, download_bytes_total: 1024,  // 50% bytes
    },
    installed: false,
  });
  assert.match(r.text, /50/,
    `expected bytes percent (50) in button text, got: ${r.text}`);
  assert.match(r.text, /bytes/,
    `expected 'bytes' unit label in button text, got: ${r.text}`);
  assert.equal(r.disabled, true);
});

test("PR-02-D04: _modelDlBtnState shows waiting affordance when progress_owner=false", () => {
  const app = loadApp();
  const r = app._modelDlBtnState({
    dlJob: {
      progress_owner: false,
      download_bytes_n: 0, download_bytes_total: 0,
    },
    installed: false,
  });
  assert.match(r.text, /waiting/,
    `expected 'waiting' in button text, got: ${r.text}`);
  assert.equal(r.disabled, true);
});

test("PR-02-D04: queue row for a non-owning download shows 'waiting' label and no progress bar", () => {
  const app = loadApp();
  app.onJob({
    id: "dl-waiting",
    kind: "download",
    target_kind: "base",
    status: "running",
    model: "model-x",
    download_file: "weights.bin",
    progress_owner: false,
    download_bytes_n: 0, download_bytes_total: 0,
    download_files_n: 0, download_files_total: 0,
    download_files_done: 0,
  });
  const queueList = app.document.getElementById("queue-list");
  const row = queueList.children[0];
  // Find the qprompt span.
  let prompt = null;
  let bar = null;
  for (const child of row.children) {
    const cn = typeof child.className === "string" ? child.className : "";
    if (cn.includes("qprompt")) prompt = child;
    if (cn.includes("qprogress")) bar = child;
  }
  assert.ok(prompt, "expected a .qprompt span on the row");
  assert.match(prompt.textContent, /waiting/,
    `expected 'waiting' in row label, got: ${prompt.textContent}`);
  assert.equal(bar, null, "expected no .qprogress bar when progress_owner=false");
});

// ---------- cached-load phase: model load with no actual bytes downloaded ----------

function findChild(row, className) {
  for (const child of row.children) {
    const cn = typeof child.className === "string" ? child.className : "";
    if (cn.includes(className)) return child;
  }
  return null;
}

test("cached load: queue row says 'loading' (not 'downloading') when no progress fields are populated", () => {
  const app = loadApp();
  // load_model() creates a kind=="download" job before the tqdm bridge has
  // fired. For a fully cached snapshot, the bridge never fires at all and
  // download_file / counters stay empty.
  app.onJob({
    id: "load-cached",
    kind: "download",
    target_kind: "base",
    status: "running",
    model: "z-image-turbo",
    download_file: "",
    download_bytes_n: 0, download_bytes_total: 0,
    download_files_n: 0, download_files_total: 0,
    download_files_done: 0,
    progress_owner: true,
  });
  const queueList = app.document.getElementById("queue-list");
  const row = queueList.children[0];
  const status = findChild(row, "qstatus");
  const prompt = findChild(row, "qprompt");
  assert.ok(status, "expected qstatus");
  assert.ok(prompt, "expected qprompt");
  assert.equal(status.textContent, "loading",
    `expected status 'loading', got: ${status.textContent}`);
  // The "(starting)" filler must not be shown when nothing is being
  // transferred — the row reads "z-image-turbo" alone.
  assert.equal(prompt.textContent, "z-image-turbo",
    `expected bare model name, got: ${prompt.textContent}`);
});

test("active download: queue row says 'downloading' and shows the file + counter", () => {
  const app = loadApp();
  app.onJob({
    id: "dl-active",
    kind: "download",
    target_kind: "base",
    status: "running",
    model: "z-image-turbo",
    download_file: "weights.safetensors",
    download_bytes_n: 512, download_bytes_total: 1024,
    download_files_n: 0, download_files_total: 0,
    download_files_done: 0,
    progress_owner: true,
  });
  const queueList = app.document.getElementById("queue-list");
  const row = queueList.children[0];
  const status = findChild(row, "qstatus");
  const prompt = findChild(row, "qprompt");
  assert.equal(status.textContent, "downloading");
  assert.match(prompt.textContent, /weights\.safetensors/);
  assert.match(prompt.textContent, /512/);
});

test("finished download: status reads the terminal state, not 'loading'", () => {
  const app = loadApp();
  app.onJob({
    id: "dl-done",
    kind: "download",
    target_kind: "base",
    status: "done",
    model: "z-image-turbo",
    download_file: "",
    download_bytes_n: 0, download_bytes_total: 0,
    download_files_n: 0, download_files_total: 0,
    download_files_done: 0,
    ts_queued: 1000.0,
    ts_done: 1012.5,
  });
  const row = app.document.getElementById("queue-list").children[0];
  const status = findChild(row, "qstatus");
  assert.equal(status.textContent, "done",
    "finished cached-load row must read 'done' rather than the in-progress phase");
});

// ---------- duration display on finished jobs ----------

test("finished generate job shows duration in seconds", () => {
  const app = loadApp();
  app.onJob({
    id: "gen-1",
    kind: "generate",
    status: "done",
    model: "z-image-turbo",
    raw_prompt: "a cat",
    seed: 42,
    step: 28, total_steps: 28,
    ts_queued: 1000.0,
    ts_done: 1012.5,
  });
  const row = app.document.getElementById("queue-list").children[0];
  const dur = findChild(row, "qduration");
  assert.ok(dur, "expected a .qduration span on a finished generate row");
  assert.equal(dur.textContent, "12.5s");
});

test("finished download job shows duration", () => {
  const app = loadApp();
  app.onJob({
    id: "dl-done-2",
    kind: "download",
    target_kind: "base",
    status: "done",
    model: "pony-v6-xl",
    download_bytes_n: 0, download_bytes_total: 0,
    download_files_n: 0, download_files_total: 0,
    ts_queued: 2000.0,
    ts_done: 2090.0,
  });
  const row = app.document.getElementById("queue-list").children[0];
  const dur = findChild(row, "qduration");
  assert.ok(dur, "expected a .qduration span on a finished download row");
  assert.equal(dur.textContent, "1m 30s",
    `expected '1m 30s', got: ${dur.textContent}`);
});

test("running job has no duration span yet", () => {
  const app = loadApp();
  app.onJob({
    id: "gen-running",
    kind: "generate",
    status: "running",
    model: "z-image-turbo",
    raw_prompt: "a dog",
    seed: 7,
    step: 5, total_steps: 28,
    ts_queued: 1000.0,
    ts_done: null,
  });
  const row = app.document.getElementById("queue-list").children[0];
  const dur = findChild(row, "qduration");
  assert.equal(dur, null, "running rows must not carry a duration");
});

test("fmtDuration covers sub-second, second, and minute ranges", () => {
  const app = loadApp();
  assert.equal(app.fmtDuration(0.25), "250ms");
  assert.equal(app.fmtDuration(0.999), "999ms");
  assert.equal(app.fmtDuration(1.0), "1.0s");
  assert.equal(app.fmtDuration(45.3), "45.3s");
  assert.equal(app.fmtDuration(60), "1m 0s");
  assert.equal(app.fmtDuration(125), "2m 5s");
});

// ---------- D17: dlBtn restores text after click resolves ----------

test("D17: download button text is not 'starting…' after wsRequest resolves", async () => {
  // Stub wsRequest: return models on models_info; resolve model_download immediately
  // without firing a job event (idempotent-collapse path).
  const app = loadApp({
    wsRequest(method) {
      if (method === "outputs_list") {
        return Promise.resolve({ entries: [], has_more: false, total: 0 });
      }
      if (method === "models_info") {
        return Promise.resolve({
          bases: [{ name: "test-base", installed: false, is_builtin: true,
                    family: "test", description: "", compatibility_tags: [] }],
          loras: [],
        });
      }
      // model_download — resolve as if request collapsed to existing job.
      return Promise.resolve({ job_id: "existing-123" });
    },
  });

  // Populate the bases list via refreshModels() (async — it calls wsRequest).
  await app.refreshModels();

  const modelsList = app.document.getElementById("models-list");
  // Walk the rendered tree to find the first button with class "section-btn"
  // (that is the download button — it is the first action button).
  function findBtn(node) {
    if (!node) return null;
    if (node.tagName === "button" && node.className === "section-btn") return node;
    for (const child of node.children || []) {
      const hit = findBtn(child);
      if (hit) return hit;
    }
    return null;
  }
  const btn = findBtn(modelsList);
  assert.ok(btn, "expected to find a section-btn in the rendered bases list");
  assert.ok(btn.onclick, "expected dlBtn.onclick to be set");

  // Capture the initial text before the click.
  const initialText = btn.textContent;

  // Click the button and wait for the async handler to finish.
  await btn.onclick();

  // After the click resolves, the fix triggers renderBases() which re-creates the
  // DOM subtree. Re-find the button in the updated list.
  const btnAfter = findBtn(modelsList);
  assert.ok(btnAfter, "expected a section-btn to be present after click resolves");
  // The freshly-rendered button must reflect the derived state, not "starting…".
  assert.notEqual(btnAfter.textContent, "starting…",
    "newly rendered dlBtn must not show 'starting…' after wsRequest resolves");
});

// ---------- D18: buildRestorePromptLine emits one /lora per entry ----------

const { buildRestorePromptLine } = require("../static/restore_prompt.js");

test("D18: buildRestorePromptLine emits separate /lora commands for each LoRA in the stack", () => {
  const line = buildRestorePromptLine({
    model: "pony-v6-xl",
    loras: "pixel-art-xl:0.8,ascii-art:0.7,neon-glow:0.5",
    prompt: "a castle",
  });

  // Must contain three separate /lora tokens — not one combined token.
  const loraCmds = line.match(/\/lora\s+\S+/g) || [];
  assert.equal(loraCmds.length, 3,
    `expected 3 /lora commands, got ${loraCmds.length}: ${line}`);
  assert.ok(loraCmds.some(c => c.includes("pixel-art-xl:0.8")),
    `expected /lora pixel-art-xl:0.8 in line: ${line}`);
  assert.ok(loraCmds.some(c => c.includes("ascii-art:0.7")),
    `expected /lora ascii-art:0.7 in line: ${line}`);
  assert.ok(loraCmds.some(c => c.includes("neon-glow:0.5")),
    `expected /lora neon-glow:0.5 in line: ${line}`);
});

test("D18: each /lora token in the restored line is parseable by the arity-1 rule (single token per command)", () => {
  const line = buildRestorePromptLine({
    loras: "foo:0.7,bar:0.3",
    prompt: "test",
  });

  // Verify that no /lora command in the line is followed by more than one
  // whitespace-separated non-command token (i.e. arity-1 is respected).
  const tokens = line.split(/\s+/);
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i] === "/lora") {
      const next = tokens[i + 1];
      const afterNext = tokens[i + 2];
      assert.ok(next && !next.startsWith("/"),
        `/lora at position ${i} must be followed by exactly one argument token`);
      // The token two positions ahead must be either another command or end-of-tokens.
      if (afterNext !== undefined) {
        assert.ok(afterNext.startsWith("/"),
          `/lora argument '${next}' must not be followed by a second non-command token '${afterNext}' — arity-1 violated`);
      }
    }
  }
});
