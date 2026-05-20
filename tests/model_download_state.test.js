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
    download_n: 10,
    download_total: 100,
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
    download_n: 100,
    download_total: 100,
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
    download_n: 25,
    download_total: 100,
    download_files_done: 0,
  });
  const queueList = app.document.getElementById("queue-list");

  const baseDownloads = app._downloadingByName("base");
  const loraDownloads = app._downloadingByName("lora");

  assert.equal(queueList.children[0].children[0].textContent, "downloading");
  assert.equal(baseDownloads.get("shared-name").id, "base-download");
  assert.equal(loraDownloads.has("shared-name"), false);
});
