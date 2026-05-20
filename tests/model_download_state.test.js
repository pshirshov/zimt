const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");

class FakeClassList {
  constructor() {
    this.names = new Set();
  }

  add(...names) {
    for (const name of names) this.names.add(name);
  }

  remove(...names) {
    for (const name of names) this.names.delete(name);
  }

  toggle(name, force) {
    const next = force === undefined ? !this.names.has(name) : Boolean(force);
    if (next) this.names.add(name);
    else this.names.delete(name);
    return next;
  }

  contains(name) {
    return this.names.has(name);
  }
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

  appendChild(child) {
    this.children.push(child);
    child.parentNode = this;
    return child;
  }

  removeChild(child) {
    const index = this.children.indexOf(child);
    if (index >= 0) this.children.splice(index, 1);
    return child;
  }

  get firstChild() {
    return this.children[0] ?? null;
  }

  set innerHTML(_html) {
    this.children = [];
  }

  get innerHTML() {
    return "";
  }

  addEventListener() {}

  setAttribute(name, value) {
    this.attributes.set(name, value);
  }

  querySelector() {
    return null;
  }

  focus() {}

  setSelectionRange(start, end) {
    this.selectionStart = start;
    this.selectionEnd = end;
  }

  getBoundingClientRect() {
    return { left: 0, top: 0, height: 18 };
  }
}

function loadApp() {
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new FakeElement());
    return elements.get(id);
  };
  const body = new FakeElement("body");
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
      body,
      title: "zimt",
      getElementById: element,
      createElement(tagName) { return new FakeElement(tagName); },
      querySelector(selector) {
        if (selector === ".textarea-wrap") return element("textarea-wrap");
        return null;
      },
      querySelectorAll() { return []; },
    },
    window: {
      innerWidth: 1200,
      addEventListener() {},
    },
    FormData: class {
      constructor() {
        this.fields = new Map();
      }
      get(name) {
        return this.fields.get(name) ?? "";
      }
    },
    ZimtConnectionManager: class {
      request(method) {
        if (method === "outputs_list") {
          return Promise.resolve({ entries: [], has_more: false, total: 0 });
        }
        return Promise.resolve({});
      }
    },
    buildRestorePromptLine() { return ""; },
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
  vm.runInNewContext(fs.readFileSync(appPath, "utf8"), context, {
    filename: appPath,
  });
  return context;
}

function findByClass(node, cls) {
  if (!node) return null;
  // The fake harness sets `className` as a string; `classList` is only
  // populated by explicit classList.add() calls. Match either.
  const cn = typeof node.className === "string" ? node.className : "";
  const hitByName = cn.split(/\s+/).includes(cls);
  const hitByList = node.classList && node.classList.contains
    && node.classList.contains(cls);
  if (hitByName || hitByList) return node;
  if (node.children) {
    for (const child of node.children) {
      const hit = findByClass(child, cls);
      if (hit) return hit;
    }
  }
  return null;
}

test("download queue row renders a cancel button when running and omits it when done", () => {
  const app = loadApp();

  app.onJob({
    id: "dl-running",
    kind: "download",
    target_kind: "base",
    status: "running",
    model: "some-model",
    download_file: "weights.bin",
    download_bytes_n: 10,
    download_bytes_total: 100,
    download_files_done: 0,
  });
  const queueList = app.document.getElementById("queue-list");
  const runningRow = queueList.children[0];
  const cancelBtn = findByClass(runningRow, "cancel-btn");
  assert.ok(cancelBtn, "expected cancel-btn on a running download row");

  app.onJob({
    id: "dl-done",
    kind: "download",
    target_kind: "base",
    status: "done",
    model: "some-model-2",
    download_file: "weights.bin",
    download_bytes_n: 100,
    download_bytes_total: 100,
    download_files_done: 1,
  });
  const doneRow = queueList.children[0]; // reverse order: newest first
  assert.equal(findByClass(doneRow, "cancel-btn"), null,
    "expected no cancel-btn on a completed download row");
});

test("model-tab active download lookup separates base and LoRA targets with the same name", () => {
  const app = loadApp();

  app.onJob({
    id: "base-download",
    kind: "download",
    target_kind: "base",
    status: "running",
    model: "shared-name",
    download_file: "model.safetensors",
    download_bytes_n: 25,
    download_bytes_total: 100,
    download_files_done: 0,
  });
  const queueList = app.document.getElementById("queue-list");

  const baseDownloads = app._downloadingByName("base");
  const loraDownloads = app._downloadingByName("lora");

  assert.equal(queueList.children[0].children[0].textContent, "downloading");
  assert.equal(baseDownloads.get("shared-name").id, "base-download");
  assert.equal(loraDownloads.has("shared-name"), false);
});

test("download counter formats bytes slot as bytes", () => {
  const app = loadApp();
  const out = app.fmtDownloadCounter({
    download_bytes_n: 1024,
    download_bytes_total: 4096,
  });
  assert.equal(out, "1.0 KB / 4.0 KB");
});

test("download counter formats files slot with files label", () => {
  const app = loadApp();
  const out = app.fmtDownloadCounter({
    download_files_n: 3,
    download_files_total: 7,
  });
  assert.equal(out, "3 / 7 files");
});

test("download counter renders both files and bytes when both slots populated", () => {
  const app = loadApp();
  const out = app.fmtDownloadCounter({
    download_files_n: 3,
    download_files_total: 7,
    download_bytes_n: 512,
    download_bytes_total: 4096,
  });
  assert.equal(out, "3 / 7 files  ·  512 B / 4.0 KB");
});

test("download counter with all-zero counters returns empty", () => {
  const app = loadApp();
  const out = app.fmtDownloadCounter({
    download_files_n: 0,
    download_files_total: 0,
    download_bytes_n: 0,
    download_bytes_total: 0,
  });
  assert.equal(out, "");
});

test("lora add button disabled when not installed", () => {
  const app = loadApp();
  const r = app._loraAddBtnState({
    compatible: true, loaded: true, dlJob: null,
    isActive: false, installed: false,
    compat: ["sdxl"], baseTags: ["sdxl"],
  });
  assert.equal(r.text, "add");
  assert.equal(r.disabled, true);
  assert.ok(r.title.includes("not installed"),
            `expected title to mention 'not installed', got: ${r.title}`);
});

test("lora add button enabled when installed", () => {
  const app = loadApp();
  const r = app._loraAddBtnState({
    compatible: true, loaded: true, dlJob: null,
    isActive: false, installed: true,
    compat: ["sdxl"], baseTags: ["sdxl"],
  });
  assert.equal(r.text, "add");
  assert.equal(r.disabled, false);
});

test("lora active stack can be removed even if not installed", () => {
  const app = loadApp();
  const r = app._loraAddBtnState({
    compatible: true, loaded: true, dlJob: null,
    isActive: true, installed: false,
    compat: ["sdxl"], baseTags: ["sdxl"],
  });
  assert.equal(r.text, "remove");
  assert.equal(r.disabled, false);
});

test("download button shows 'download' for uninstalled model with no active job", () => {
  const app = loadApp();
  const r = app._modelDlBtnState({ dlJob: null, installed: false });
  assert.equal(r.text, "download");
  assert.equal(r.disabled, false);
  assert.ok(typeof r.title === "string" && r.title.length > 0);
});

test("download button shows 'redownload' for installed model with no active job", () => {
  const app = loadApp();
  const r = app._modelDlBtnState({ dlJob: null, installed: true });
  assert.equal(r.text, "redownload");
  assert.equal(r.disabled, false);
  assert.ok(typeof r.title === "string" && r.title.length > 0);
});

test("download button shows 'downloading X%' and is disabled when bytes slot has progress", () => {
  const app = loadApp();
  const r = app._modelDlBtnState({
    dlJob: { download_bytes_n: 50, download_bytes_total: 100 },
    installed: false,
  });
  assert.equal(r.text, "downloading 50% bytes");
  assert.equal(r.disabled, true);
});

test("download button shows 'downloading…' and is disabled when an active job has no total", () => {
  const app = loadApp();
  const r = app._modelDlBtnState({
    dlJob: {},
    installed: false,
  });
  assert.equal(r.text, "downloading…");
  assert.equal(r.disabled, true);
});

test("download button stays disabled while active even for already-installed model", () => {
  const app = loadApp();
  const r = app._modelDlBtnState({
    dlJob: { download_bytes_n: 30, download_bytes_total: 60 },
    installed: true,
  });
  assert.equal(r.text, "downloading 50% bytes");
  assert.equal(r.disabled, true);
});

test("_loraAddBtnState renders incompatible title and disabled when compatible=false", () => {
  const app = loadApp();
  const out = app._loraAddBtnState({
    compatible: false,
    loaded: true,
    dlJob: null,
    isActive: false,
    installed: true,
    compat: ["sdxl"],
    baseTags: ["zimage"],
  });
  assert.equal(out.disabled, true);
  assert.match(out.title, /^incompatible:/);
});
