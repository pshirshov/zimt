const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");

// ----- Fake WebSocket (no real I/O) -----------------------------------
const CONNECTING = 0;
const OPEN = 1;
const CLOSING = 2;
const CLOSED = 3;

function makeFakeWSClass(registry) {
  class FakeWS {
    constructor(url) {
      this.url = url;
      this.readyState = CONNECTING;
      this.sent = [];
      this.closes = [];
      this.onopen = null;
      this.onmessage = null;
      this.onerror = null;
      this.onclose = null;
      registry.push(this);
    }
    send(msg) { this.sent.push(msg); }
    close(code, reason) {
      if (this.readyState === CLOSED) return;
      this.readyState = CLOSED;
      this.closes.push({ code: code || 1006, reason: reason || "" });
      if (this.onclose) this.onclose({ code: code || 1006, reason: reason || "" });
    }
    // Test helpers
    fireOpen() {
      this.readyState = OPEN;
      if (this.onopen) this.onopen();
    }
    fireMessage(obj) {
      if (this.onmessage) this.onmessage({ data: JSON.stringify(obj) });
    }
    // Find last ping nonce sent on this socket.
    lastPingNonce() {
      for (let i = this.sent.length - 1; i >= 0; i--) {
        try {
          const m = JSON.parse(this.sent[i]);
          if (m.type === "ping") return m.nonce;
        } catch (_) {}
      }
      return null;
    }
    respondToPingWithPong() {
      const n = this.lastPingNonce();
      if (n) this.fireMessage({ type: "pong", nonce: n, ts: 0 });
      return n;
    }
  }
  FakeWS.CONNECTING = CONNECTING;
  FakeWS.OPEN = OPEN;
  FakeWS.CLOSING = CLOSING;
  FakeWS.CLOSED = CLOSED;
  return FakeWS;
}

// ----- Controllable timers --------------------------------------------
function makeTimerHarness() {
  const timers = new Map();
  const intervals = new Map();
  let nextId = 1;
  let nowMs = 1_000_000;
  return {
    setTimeout(fn, delayMs) {
      const id = nextId++;
      timers.set(id, { fn, dueAt: nowMs + (delayMs || 0), type: "timeout" });
      return id;
    },
    clearTimeout(id) { timers.delete(id); },
    setInterval(fn, periodMs) {
      const id = nextId++;
      intervals.set(id, { fn, period: periodMs || 0, lastRanAt: nowMs });
      return id;
    },
    clearInterval(id) { intervals.delete(id); },
    now() { return nowMs; },
    // Advance virtual time and fire any due timeouts / intervals.
    advance(deltaMs) {
      const target = nowMs + deltaMs;
      // We process events in time order. Intervals can re-arm, so cap
      // iterations.
      let safety = 10000;
      while (safety-- > 0) {
        let next = null;
        for (const [, t] of timers) {
          if (next == null || t.dueAt < next) next = t.dueAt;
        }
        for (const [, iv] of intervals) {
          const due = iv.lastRanAt + iv.period;
          if (next == null || due < next) next = due;
        }
        if (next == null || next > target) break;
        nowMs = next;
        // Fire all timers with dueAt <= nowMs.
        for (const [id, t] of [...timers]) {
          if (t.dueAt <= nowMs) {
            timers.delete(id);
            try { t.fn(); } catch (e) { /* swallow */ }
          }
        }
        // Fire all intervals due at or before nowMs.
        for (const [, iv] of [...intervals]) {
          if (iv.lastRanAt + iv.period <= nowMs) {
            iv.lastRanAt = nowMs;
            try { iv.fn(); } catch (e) { /* swallow */ }
          }
        }
      }
      nowMs = target;
    },
    // Pending timer count (for assertions).
    pendingCount() { return timers.size; },
  };
}

// ----- Load connection.js into a sandbox ------------------------------
function loadConnection(opts) {
  opts = opts || {};
  const wsRegistry = [];
  const FakeWS = makeFakeWSClass(wsRegistry);
  const harness = makeTimerHarness();

  const listeners = { document: {}, window: {} };
  const addListener = (target) => (event, fn) => {
    (listeners[target][event] = listeners[target][event] || []).push(fn);
  };
  const fire = (target, event, ev) => {
    for (const fn of (listeners[target][event] || [])) fn(ev);
  };

  const document = {
    visibilityState: "visible",
    addEventListener: addListener("document"),
  };

  // The IIFE in connection.js binds globals onto `window` if it's
  // defined. We make the sandbox's globalThis === window so the
  // exported `ZimtConnectionManager` is reachable directly.
  const context = {
    console,
    Date: { now: () => harness.now() },
    setTimeout: harness.setTimeout.bind(harness),
    clearTimeout: harness.clearTimeout.bind(harness),
    setInterval: harness.setInterval.bind(harness),
    clearInterval: harness.clearInterval.bind(harness),
    document,
    navigator: opts.navigator || {},
    crypto: undefined,
    Math,
    JSON,
    Promise,
    Error,
    Object,
    Map,
    Set,
    Array,
    Uint8Array,
    Symbol,
    addEventListener: addListener("window"),
  };
  context.window = context;
  context.globalThis = context;

  const connPath = path.join(__dirname, "..", "static", "connection.js");
  vm.runInNewContext(fs.readFileSync(connPath, "utf8"), context, {
    filename: connPath,
  });

  const messages = [];
  const stateChanges = [];
  const manager = new context.ZimtConnectionManager(Object.assign({
    url: "ws://localhost/ws",
    WebSocket: FakeWS,
    onMessage: (m) => messages.push(m),
    onStateChange: (s) => stateChanges.push(s),
    // Tighter timings make the tests independent of DEFAULTS.
    pingIntervalMs: 1000,
    pongTimeoutMs: 500,
    staleGraceMs: 1000,
    connectTimeoutMs: 2000,
  }, opts.extraOpts || {}));

  return {
    context,
    manager,
    wsRegistry,
    FakeWS,
    harness,
    messages,
    stateChanges,
    fire,
    document,
  };
}

// ----- Tests ----------------------------------------------------------

test("single-connection happy path: NEW -> ALIVE via open + pong", () => {
  const t = loadConnection();
  assert.equal(t.wsRegistry.length, 1, "exactly one socket spawned at start");
  const ws = t.wsRegistry[0];
  assert.equal(ws.readyState, CONNECTING);
  assert.equal(t.manager.stats().state, "NEW");

  ws.fireOpen();
  assert.equal(t.manager.stats().state, "ALIVE");
  const stats = t.manager.stats();
  assert.equal(stats.connections.length, 1);
  assert.equal(stats.connections[0].active, true);
});

test("STALE triggers replacement immediately", () => {
  const t = loadConnection();
  const ws1 = t.wsRegistry[0];
  ws1.fireOpen();
  assert.equal(t.manager.stats().state, "ALIVE");

  // Advance past pingInterval so a ping fires, then past pongTimeout
  // without a pong -> STALE.
  t.harness.advance(1000); // ping sent
  // ping should now be in sent[]
  assert.ok(ws1.lastPingNonce(), "ping was sent");
  t.harness.advance(500); // pong timeout -> STALE
  assert.equal(ws1.readyState, OPEN, "stale conn not yet closed (grace period)");
  // Manager should have spawned a replacement.
  assert.equal(t.wsRegistry.length, 2,
    `expected replacement on STALE; got ${t.wsRegistry.length} sockets`);
  assert.equal(t.wsRegistry[1].readyState, CONNECTING);
  // Active is still the stale connection at this point.
  const stats = t.manager.stats();
  assert.equal(stats.state, "STALE");
  assert.equal(stats.connections.length, 2);
});

test("late pong on stale promotes it back; replacement is closed as superseded", () => {
  const t = loadConnection();
  const ws1 = t.wsRegistry[0];
  ws1.fireOpen();
  t.harness.advance(1000); // ping
  t.harness.advance(500);  // pong timeout -> STALE + replacement spawned
  assert.equal(t.wsRegistry.length, 2);
  const ws2 = t.wsRegistry[1];

  // Late pong arrives on ws1 — should promote back to ALIVE.
  ws1.respondToPingWithPong();
  assert.equal(t.manager.stats().state, "ALIVE");

  // Now the replacement (ws2) opens and goes ALIVE. Because the active
  // is already ALIVE, ws2 is the "extra" and should be closed superseded.
  ws2.fireOpen();
  assert.equal(ws2.readyState, CLOSED,
    "expected replacement to be closed when original was already ALIVE");
  const last = ws2.closes[ws2.closes.length - 1];
  assert.equal(last.code, 4002, "superseded code");
});

test("replacement reaches ALIVE first; stale is closed as superseded", () => {
  const t = loadConnection();
  const ws1 = t.wsRegistry[0];
  ws1.fireOpen();
  t.harness.advance(1000); // ping
  t.harness.advance(500);  // STALE + replacement
  assert.equal(t.wsRegistry.length, 2);
  const ws2 = t.wsRegistry[1];

  // ws2 opens before ws1 is rescued.
  ws2.fireOpen();
  // The stale ws1 should be closed with 4002 superseded.
  assert.equal(ws1.readyState, CLOSED,
    "stale original should be closed when replacement promotes");
  const lastClose = ws1.closes[ws1.closes.length - 1];
  assert.equal(lastClose.code, 4002);

  // ws2 is now the active connection.
  const stats = t.manager.stats();
  assert.equal(stats.state, "ALIVE");
  assert.equal(stats.activeConnectionId, ws2 === t.wsRegistry[1] ?
    stats.connections.find(c => c.active).id : null);
});

test("MAX_LIVE_CONNECTIONS cap honoured across successive STALEs", () => {
  // Drive three connections to STALE simultaneously and confirm that a fourth
  // _spawnConnection call returns null — exercising the cap-rejection branch.
  //
  // manager._pool holds Connection instances (not FakeWS).  We use
  // _enterStale() directly on pool members to park them in STALE without
  // opening them (Connection allows STALE from NEW state).
  // staleGraceMs=30_000 ensures no grace timer fires during the test.
  const t = loadConnection({
    extraOpts: { staleGraceMs: 30_000 },
  });

  // conn0 was spawned by the constructor. Open it so it becomes ALIVE and
  // active, then let the pong timeout fire to make it STALE.
  const conn0 = t.manager._pool[0];
  t.wsRegistry[0].fireOpen();
  t.harness.advance(1000); // ping fires on conn0
  t.harness.advance(500);  // pong timeout -> conn0 STALE; _ensureReplacement spawns conn1
  assert.equal(t.wsRegistry.length, 2, "conn1 spawned when conn0 goes STALE");
  assert.equal(t.manager._liveConnections().length, 2); // conn0 STALE, conn1 NEW

  // conn1 is NEW. Transition it directly to STALE — the manager's
  // _handleStale callback does nothing here because conn0 (not conn1) is
  // the active connection. The pool now has two STALE connections.
  const conn1 = t.manager._pool.find(c => c !== conn0);
  conn1._enterStale();
  assert.equal(conn1.state, "STALE", "conn1 is now STALE");
  assert.equal(t.manager._liveConnections().length, 2); // conn0 STALE, conn1 STALE

  // Spawn conn2 manually — two live connections < cap of 3, so this succeeds.
  const conn2 = t.manager._spawnConnection();
  assert.ok(conn2 !== null, "conn2 spawned: only 2 live connections before spawn");
  assert.equal(t.wsRegistry.length, 3, "conn2 registered");
  assert.equal(t.manager._liveConnections().length, 3); // conn0 STALE, conn1 STALE, conn2 NEW

  // Transition conn2 to STALE so all three are live-STALE simultaneously.
  conn2._enterStale();
  assert.equal(t.manager._liveConnections().length, 3,
    "all three still live after conn2 goes STALE (staleGraceMs=30s)");

  // Now _spawnConnection must return null: 3 live >= MAX_LIVE_CONNECTIONS (3).
  const refused = t.manager._spawnConnection();
  assert.equal(refused, null, "cap blocks a fourth connection");
  assert.equal(t.wsRegistry.length, 3,
    `wsRegistry must not grow past 3; got ${t.wsRegistry.length}`);
});

test("BFCache: pagehide(persisted) closes all sockets; no reconnect scheduled", () => {
  const t = loadConnection();
  const ws1 = t.wsRegistry[0];
  ws1.fireOpen();
  t.harness.advance(1000); t.harness.advance(500); // STALE + replacement
  assert.equal(t.wsRegistry.length, 2);

  // Fire pagehide(persisted: true)
  t.fire("window", "pagehide", { persisted: true });
  for (const ws of t.wsRegistry) {
    assert.equal(ws.readyState, CLOSED,
      `expected ws ${ws.url} closed after pagehide persisted`);
  }
  // No reconnect timer should be armed.
  // Advance a long while; no new sockets must appear.
  const before = t.wsRegistry.length;
  t.harness.advance(60_000);
  assert.equal(t.wsRegistry.length, before,
    "no reconnect should occur while page is in BFCache");
});

test("BFCache: pageshow(persisted) reconnects immediately", () => {
  const t = loadConnection();
  const ws1 = t.wsRegistry[0];
  ws1.fireOpen();
  t.fire("window", "pagehide", { persisted: true });
  const before = t.wsRegistry.length;

  t.fire("window", "pageshow", { persisted: true });
  // Schedule fires on next tick; advance just past 0.
  t.harness.advance(1);
  assert.ok(t.wsRegistry.length > before,
    "expected a fresh socket after pageshow persisted");
  const fresh = t.wsRegistry[t.wsRegistry.length - 1];
  assert.equal(fresh.readyState, CONNECTING);
  // Attempt counter should be reset (it advanced by 1 due to schedule
  // bookkeeping, but the cap should not be inherited).
  const stats = t.manager.stats();
  assert.ok(stats.attempt <= 1, "attempt counter should be near zero post-BFCache return");
});

test("outgoing request() routes through the promoted active connection", () => {
  const t = loadConnection();
  const ws1 = t.wsRegistry[0];
  ws1.fireOpen();
  t.harness.advance(1000); t.harness.advance(500); // STALE + replacement
  const ws2 = t.wsRegistry[1];
  ws2.fireOpen(); // promotion: ws1 closed superseded, ws2 active

  // Reset sent buffers so we can attribute the next send.
  ws1.sent.length = 0;
  ws2.sent.length = 0;

  t.manager.request("ping_me", { foo: 1 });
  // The request should have been sent on ws2, not ws1.
  assert.equal(ws1.sent.length, 0, "no request on superseded socket");
  assert.equal(ws2.sent.length, 1, "request routed to active (replacement) socket");
  const sent = JSON.parse(ws2.sent[0]);
  assert.equal(sent.type, "req");
  assert.equal(sent.method, "ping_me");
});
