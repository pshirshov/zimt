// Resilient WebSocket connection manager.
//
// Implements the high-value subset of the resilient-ws-ui pattern for
// the zimt UI:
//   * NEW / ALIVE / STALE / DEAD state machine — a single boolean
//     `connected` is too coarse to drive correct UI or recovery logic.
//   * App-level heartbeat with per-ping nonces and per-ping timeouts.
//     Transport `readyState` lies on mobile freeze, IP change, idle NAT
//     drop, and Firefox bug 920074.
//   * Exponential backoff with full jitter, cap, max attempts, and a
//     terminal "stopped" state — silent infinite retry trains users to
//     ignore the indicator.
//   * Connect timeout (separate from pong timeout) so a hanging TLS /
//     captive-portal session doesn't sit in CONNECTING forever.
//   * Time-jump detector: a 1s interval comparing wall-clock deltas
//     catches OS sleep, mobile tab freeze, debugger pauses — all
//     scenarios where the platform won't fire `close` for us.
//   * Defer-while-hidden reconnect (Phoenix pattern) — don't burn the
//     retry budget on a user who isn't there.
//   * Overlapping-connection failover (R6): when the active socket goes
//     STALE the manager spins up a replacement in parallel. Whichever
//     reaches ALIVE first wins; the other is closed with code 4002.
//     Counters the silent-NAT-drop class of bugs (Firefox 920074).
//   * BFCache wiring (R9): pagehide(persisted) closes sockets cleanly
//     so the page is eligible for BFCache; pageshow(persisted)
//     immediately reconnects with a fresh attempt counter.
//   * Network Information API: a `change` event on `navigator.connection`
//     triggers an immediate ping on the active socket so we don't wait
//     for the next heartbeat to discover a path-MTU / NAT-rebind change.
//   * online/offline + visibilitychange wiring.
//
// Intentional gaps (documented for honesty per the skill's R14):
//   * Heartbeat timer runs on the main thread (R14). Chrome throttles
//     main-thread timers in heavily-backgrounded tabs; the time-jump
//     detector recovers on resume, but heartbeats themselves stop
//     until visible. A Web Worker heartbeat would fix this but isn't
//     worth the complexity for a single-tab tool.
//   * No session resumption on reconnect — a reconnect is a fresh
//     session. The server re-sends a "state" hello + in-flight jobs,
//     which is sufficient for this app.

(function (global) {
  "use strict";

  const STATE = Object.freeze({
    NEW: "NEW",
    ALIVE: "ALIVE",
    STALE: "STALE",
    DEAD: "DEAD",
  });

  // 1002..1015 minus the codes we treat as transient. The retriable
  // set is informational; we use NON_RETRIABLE_CODES to make terminal
  // decisions.
  const NON_RETRIABLE_CODES = new Set([1002, 1003, 1007, 1008, 1009, 1010, 1015]);

  const MAX_LIVE_CONNECTIONS = 3;
  // Close code used when promotion supersedes an extra parallel socket.
  // 4000-4999 is the application-private range per RFC 6455 §7.4.2.
  const SUPERSEDED_CODE = 4002;

  const DEFAULTS = Object.freeze({
    connectTimeoutMs: 10_000,
    pingIntervalMs: 15_000,
    pongTimeoutMs: 10_000,
    staleGraceMs:  5_000,
    backoffBaseMs: 1_000,
    backoffCapMs: 30_000,
    maxAttempts: 15,
    timeJumpThresholdMs: 5_000,
  });

  // -----------------------------------------------------------------
  // Connection: per-socket lifecycle.
  //
  // Owns exactly one WebSocket and drives one NEW/ALIVE/STALE/DEAD
  // machine. Reports lifecycle transitions to its parent manager via
  // callbacks; never schedules a reconnect itself.
  // -----------------------------------------------------------------
  let _nextConnId = 1;

  class Connection {
    constructor(parent, opts) {
      this.id = _nextConnId++;
      this.parent = parent;
      this.cfg = parent.cfg;
      this.WS = opts.WebSocket || global.WebSocket;
      this.url = opts.url;

      this.ws = null;
      this.state = STATE.NEW;
      this.lastCloseCode = null;
      this.lastCloseReason = "";
      this.openedAt = null;
      this._pendingPings = new Map();  // nonce -> sent_at_ms
      this._connectTimer = null;
      this._pingTimer = null;
      this._staleTimer = null;
      this._dead = false;
      this._supersededClose = false;
    }

    open() {
      try {
        this.ws = new this.WS(this.url);
      } catch (_) {
        this._transitionTo(STATE.DEAD, { code: 1006, reason: "ctor failed" });
        return;
      }
      this._connectTimer = setTimeout(() => {
        if (this.state === STATE.NEW &&
            this.ws && this.ws.readyState === this.WS.CONNECTING) {
          try { this.ws.close(); } catch (_) {}
        }
      }, this.cfg.connectTimeoutMs);

      this.ws.onopen = () => this._onOpen();
      this.ws.onmessage = (ev) => this._onMessage(ev);
      this.ws.onerror = () => { /* `close` follows; drive state there */ };
      this.ws.onclose = (ev) => this._onClose(ev);
    }

    // Closes the socket. The eventual `onclose` will still fire and
    // route through _onClose. If `superseded` is true, we mark that so
    // the parent knows not to count this against retries.
    close(code, reason, superseded) {
      if (superseded) this._supersededClose = true;
      try { if (this.ws) this.ws.close(code, reason); } catch (_) {}
    }

    send(text) {
      if (this.ws && this.ws.readyState === this.WS.OPEN) {
        this.ws.send(text);
        return true;
      }
      return false;
    }

    isOpen() {
      return this.ws && this.ws.readyState === this.WS.OPEN;
    }

    snapshot() {
      return {
        id: this.id,
        state: this.state,
        lastCloseCode: this.lastCloseCode,
        lastCloseReason: this.lastCloseReason,
        pendingPings: this._pendingPings.size,
        uptimeMs: this.openedAt ? Date.now() - this.openedAt : 0,
        superseded: this._supersededClose,
      };
    }

    // ---- transitions ----

    _transitionTo(s, closeInfo) {
      if (this.state === s) return;
      this.state = s;
      if (s === STATE.DEAD) {
        this._dead = true;
        if (closeInfo) {
          this.lastCloseCode = closeInfo.code;
          this.lastCloseReason = closeInfo.reason || "";
        }
      }
      this.parent._onConnectionStateChange(this);
    }

    _onOpen() {
      this._clearTimer("_connectTimer");
      this.openedAt = Date.now();
      this._transitionTo(STATE.ALIVE);
      this._schedulePing();
    }

    _onMessage(ev) {
      let m;
      try { m = JSON.parse(ev.data); } catch (_) { return; }
      if (m && m.type === "ping") {
        try {
          this.ws.send(JSON.stringify({ type: "pong",
            nonce: m.nonce, ts: m.ts }));
        } catch (_) {}
        return;
      }
      if (m && m.type === "pong") {
        if (typeof m.nonce === "string") this._pendingPings.delete(m.nonce);
        // Late pong while STALE — peer is still talking. Promote back.
        if (this.state === STATE.STALE) {
          this._clearTimer("_staleTimer");
          this._transitionTo(STATE.ALIVE);
        }
        return;
      }
      this.parent._onConnectionMessage(this, m);
    }

    _onClose(ev) {
      this._clearTimer("_connectTimer");
      this._clearTimer("_pingTimer");
      this._clearTimer("_staleTimer");
      this._pendingPings.clear();
      const code = (ev && ev.code) || 0;
      const reason = (ev && ev.reason) || "";
      this._transitionTo(STATE.DEAD, { code, reason });
    }

    _schedulePing() {
      this._clearTimer("_pingTimer");
      if (this._dead) return;
      this._pingTimer = setTimeout(() => this._sendPing(),
                                   this.cfg.pingIntervalMs);
    }

    _sendPing() {
      if (this._dead) return;
      if (!this.ws || this.ws.readyState !== this.WS.OPEN) return;
      const nonce = _randomNonce();
      this._pendingPings.set(nonce, Date.now());
      try {
        this.ws.send(JSON.stringify({ type: "ping", nonce, ts: Date.now() }));
      } catch (_) {
        return;
      }
      setTimeout(() => {
        if (!this._pendingPings.has(nonce)) return;
        this._enterStale();
      }, this.cfg.pongTimeoutMs);
      this._schedulePing();
    }

    _enterStale() {
      if (this.state !== STATE.ALIVE && this.state !== STATE.NEW) return;
      this._transitionTo(STATE.STALE);
      this._clearTimer("_staleTimer");
      this._staleTimer = setTimeout(() => {
        try { if (this.ws) this.ws.close(4000, "stale"); } catch (_) {}
      }, this.cfg.staleGraceMs);
    }

    _clearTimer(field) {
      if (this[field]) {
        clearTimeout(this[field]);
        this[field] = null;
      }
    }

    _clearAllTimers() {
      this._clearTimer("_connectTimer");
      this._clearTimer("_pingTimer");
      this._clearTimer("_staleTimer");
    }
  }

  // -----------------------------------------------------------------
  // ConnectionManager: pool orchestrator + reconnect/backoff + lifecycle.
  //
  // Public API (back-compat with the previous single-socket manager):
  //   request(method, params) → Promise
  //   stats() → flat status object (top-level `state`, `attempt`, etc.
  //             stay where they were; `connections`, `activeConnectionId`,
  //             and `pool` are added)
  //   destroy()
  // -----------------------------------------------------------------
  class ConnectionManager {
    constructor(opts) {
      this.url = opts.url;
      this.cfg = Object.assign({}, DEFAULTS, opts || {});
      this.onMessage = opts.onMessage || (() => {});
      this.onStateChange = opts.onStateChange || (() => {});
      this.WS = opts.WebSocket || global.WebSocket;

      this._pool = [];                  // Connection instances (live + recently dead awaiting drain)
      this._activeConnectionId = null;
      this.attempt = 0;
      this.isTerminal = false;
      this.deferredOnVisible = false;
      this.destroyed = false;
      // Set while the page is parked in BFCache. Suppresses
      // reconnect-on-close so the page stays eligible for the cache.
      this._bfcacheParked = false;
      this._lastCloseCode = null;
      this._lastCloseReason = "";

      this._pending = new Map();        // RPC id -> {resolve, reject}
      this._nextReqId = 1;
      this._outbox = [];

      this._reconnectTimer = null;
      this._reconnectAt = null;
      this._lastTickAt = Date.now();
      this._tickTimer = setInterval(() => this._tick(), 1000);

      this._wireLifecycle();
      this._spawnConnection();
    }

    // ---- public API ----

    request(method, params) {
      params = params || {};
      return new Promise((resolve, reject) => {
        if (this.isTerminal) {
          reject(new Error("connection stopped — refresh the page"));
          return;
        }
        const id = String(this._nextReqId++);
        this._pending.set(id, { resolve, reject });
        const msg = JSON.stringify({ type: "req", id, method, params });
        const active = this._activeConnection();
        if (active && active.send(msg)) return;
        this._outbox.push(msg);
      });
    }

    stats() {
      const active = this._activeConnection();
      // Top-level `state` is the active connection's state. With no
      // active, fall back to the "best" pool member so the UI shows
      // NEW during the first connect, not DEAD.
      let state;
      if (active) {
        state = active.state;
      } else {
        const bestRank = { ALIVE: 0, NEW: 1, STALE: 2, DEAD: 3 };
        let best = null;
        for (const c of this._pool) {
          if (best == null || bestRank[c.state] < bestRank[best.state]) best = c;
        }
        state = best ? best.state : STATE.DEAD;
      }
      const conns = this._pool.map(c => Object.assign(c.snapshot(),
                                                       { active: c.id === this._activeConnectionId }));
      const aliveCount = this._pool.filter(c => c.state === STATE.NEW ||
                                                 c.state === STATE.ALIVE ||
                                                 c.state === STATE.STALE).length;
      const lastCloseCode = active ? active.lastCloseCode : this._lastCloseCode;
      const lastCloseReason = active ? active.lastCloseReason : this._lastCloseReason;
      return {
        state,
        attempt: this.attempt,
        maxAttempts: this.cfg.maxAttempts,
        isTerminal: this.isTerminal,
        deferredOnVisible: this.deferredOnVisible,
        pendingPings: active ? active._pendingPings.size : 0,
        nextReconnectInMs: this._nextReconnectInMs(),
        lastCloseCode,
        lastCloseReason,
        activeConnectionId: this._activeConnectionId,
        connections: conns,
        pool: { alive: aliveCount, total: this._pool.length },
        frozen: false,
      };
    }

    destroy() {
      this.destroyed = true;
      this._clearReconnectTimer();
      if (this._tickTimer) {
        clearInterval(this._tickTimer);
        this._tickTimer = null;
      }
      this._rejectPending("manager destroyed");
      for (const c of this._pool) {
        c._clearAllTimers();
        try { if (c.ws) c.ws.close(1000, "client shutdown"); } catch (_) {}
      }
      if (typeof document !== "undefined") {
        document.removeEventListener("visibilitychange", this._onVisibilityChange);
        if (typeof window !== "undefined") {
          window.removeEventListener("pagehide", this._onPageHide);
          window.removeEventListener("pageshow", this._onPageShow);
        } else if (typeof document.addEventListener === "function") {
          document.removeEventListener("pagehide", this._onPageHide);
          document.removeEventListener("pageshow", this._onPageShow);
        }
      }
      if (typeof window !== "undefined") {
        window.removeEventListener("online", this._onOnline);
      }
      if (typeof navigator !== "undefined" && navigator.connection &&
          typeof navigator.connection.removeEventListener === "function") {
        navigator.connection.removeEventListener("change", this._onNetInfoChange);
      }
    }

    // ---- pool management ----

    _activeConnection() {
      if (this._activeConnectionId == null) return null;
      return this._pool.find(c => c.id === this._activeConnectionId) || null;
    }

    _liveConnections() {
      return this._pool.filter(c => c.state === STATE.NEW ||
                                    c.state === STATE.ALIVE ||
                                    c.state === STATE.STALE);
    }

    _spawnConnection() {
      if (this.destroyed || this.isTerminal) return null;
      if (this._liveConnections().length >= MAX_LIVE_CONNECTIONS) return null;
      const c = new Connection(this, { url: this.url, WebSocket: this.WS });
      this._pool.push(c);
      c.open();
      this.onStateChange(this.stats());
      return c;
    }

    // Called when an active connection enters STALE: start a parallel
    // replacement so we don't have to wait for the original to die.
    _ensureReplacement() {
      const others = this._pool.filter(c => c.state === STATE.NEW ||
                                            c.state === STATE.ALIVE);
      if (others.length > 0) return;
      this._spawnConnection();
    }

    // Hooks called by Connection instances.

    _onConnectionStateChange(conn) {
      if (this.destroyed) return;
      if (conn.state === STATE.ALIVE) this._handleAlive(conn);
      else if (conn.state === STATE.STALE) this._handleStale(conn);
      else if (conn.state === STATE.DEAD) this._handleDead(conn);
      this.onStateChange(this.stats());
    }

    _onConnectionMessage(conn, m) {
      // RPC responses can come from any connection; the request was
      // dispatched on whichever was active at the time, but a slow pong
      // or replay shouldn't get dropped.
      if (m && m.type === "resp") {
        const p = this._pending.get(m.id);
        if (!p) return;
        this._pending.delete(m.id);
        if (m.ok) p.resolve(m.result);
        else p.reject(new Error(m.error || "request failed"));
        return;
      }
      this.onMessage(m);
    }

    _handleAlive(conn) {
      // Successful open of *some* connection — reset the manager-level
      // backoff. The replacement-on-STALE path shouldn't penalise the
      // happy case.
      this.attempt = 0;
      const active = this._activeConnection();
      if (!active || active === conn) {
        this._activeConnectionId = conn.id;
        this._flushOutbox(conn);
        return;
      }
      if (active.state === STATE.STALE) {
        // Promotion: replacement wins, supersede the stale active.
        active.close(SUPERSEDED_CODE, "superseded", true);
        this._activeConnectionId = conn.id;
        this._flushOutbox(conn);
        return;
      }
      if (active.state === STATE.ALIVE) {
        // Both healthy — keep the original, close the extra.
        conn.close(SUPERSEDED_CODE, "superseded", true);
        return;
      }
      // Active is NEW (still opening). Take the one that just won the race.
      this._activeConnectionId = conn.id;
      this._flushOutbox(conn);
    }

    _handleStale(conn) {
      if (conn.id === this._activeConnectionId) this._ensureReplacement();
    }

    _handleDead(conn) {
      // Drain the pool: connection slot is gone.
      this._lastCloseCode = conn.lastCloseCode;
      this._lastCloseReason = conn.lastCloseReason;
      this._pool = this._pool.filter(c => c !== conn);

      if (conn._supersededClose) {
        // Not a real failure — don't count it, don't reconnect.
        if (conn.id === this._activeConnectionId) {
          // Active was the superseded one. Pick a live successor if any.
          this._pickActiveFromPool();
        }
        return;
      }

      // Non-retriable close codes: stop forever.
      if (NON_RETRIABLE_CODES.has(conn.lastCloseCode)) {
        this.isTerminal = true;
        return;
      }

      if (conn.id === this._activeConnectionId) {
        // Lost the active socket. If a successor is already live, promote it.
        const succ = this._pool.find(c => c.state === STATE.ALIVE) ||
                     this._pool.find(c => c.state === STATE.NEW);
        if (succ) {
          this._activeConnectionId = succ.id;
          if (succ.state === STATE.ALIVE) this._flushOutbox(succ);
          return;
        }
        this._activeConnectionId = null;
      }

      // No live successor and we have no active. If anything in the pool
      // is still trying (e.g. a parallel replacement still NEW), wait
      // for it. Otherwise schedule a fresh reconnect — unless we are
      // parked in BFCache, in which case stay quiet until pageshow.
      if (this._liveConnections().length === 0 && !this._bfcacheParked) {
        this._rejectPending("connection lost");
        this._scheduleReconnect();
      }
    }

    _pickActiveFromPool() {
      const alive = this._pool.find(c => c.state === STATE.ALIVE);
      if (alive) {
        this._activeConnectionId = alive.id;
        this._flushOutbox(alive);
        return;
      }
      const newc = this._pool.find(c => c.state === STATE.NEW);
      if (newc) {
        this._activeConnectionId = newc.id;
        return;
      }
      const stale = this._pool.find(c => c.state === STATE.STALE);
      if (stale) {
        this._activeConnectionId = stale.id;
        return;
      }
      this._activeConnectionId = null;
    }

    _flushOutbox(conn) {
      while (this._outbox.length) {
        const msg = this._outbox.shift();
        if (!conn.send(msg)) {
          // Couldn't actually send; push back and stop.
          this._outbox.unshift(msg);
          return;
        }
      }
    }

    // ---- reconnect ----

    _scheduleReconnect(opts) {
      if (this.destroyed || this.isTerminal) return;
      opts = opts || {};
      if (opts.resetAttempts) this.attempt = 0;
      this.attempt += 1;
      if (this.attempt > this.cfg.maxAttempts) {
        this.isTerminal = true;
        this.onStateChange(this.stats());
        return;
      }
      if (!opts.ignoreVisibility &&
          typeof document !== "undefined" &&
          document.visibilityState === "hidden") {
        this.deferredOnVisible = true;
        this.onStateChange(this.stats());
        return;
      }
      this.deferredOnVisible = false;
      const delay = opts.immediate ? 0 : _backoff(this.cfg, this.attempt);
      this._reconnectAt = Date.now() + delay;
      this.onStateChange(this.stats());
      this._reconnectTimer = setTimeout(() => {
        this._clearReconnectTimer();
        this._spawnConnection();
      }, delay);
    }

    _clearReconnectTimer() {
      if (this._reconnectTimer) {
        clearTimeout(this._reconnectTimer);
        this._reconnectTimer = null;
      }
      this._reconnectAt = null;
    }

    _nextReconnectInMs() {
      if (!this._reconnectTimer || !this._reconnectAt) return null;
      return Math.max(0, this._reconnectAt - Date.now());
    }

    // ---- time-jump detector ----

    _tick() {
      const now = Date.now();
      const elapsed = now - this._lastTickAt;
      this._lastTickAt = now;
      if (elapsed > 1000 + this.cfg.timeJumpThresholdMs) {
        this._handleResume(elapsed);
      }
    }

    _handleResume(elapsedMs) {
      const active = this._activeConnection();
      if (!active) return;
      if (elapsedMs >= this.cfg.pongTimeoutMs &&
          active.ws && active.ws.readyState !== this.WS.CLOSED) {
        try { active.ws.close(4001, "time-jump"); } catch (_) {}
      } else if (active.state === STATE.ALIVE) {
        active._sendPing();
      }
    }

    // ---- lifecycle wiring ----

    _wireLifecycle() {
      this._onVisibilityChange = () => {
        if (this.destroyed) return;
        if (document.visibilityState === "visible") {
          if (this.deferredOnVisible) {
            this.deferredOnVisible = false;
            this._scheduleReconnect();
          } else {
            const active = this._activeConnection();
            if (active && active.isOpen()) active._sendPing();
          }
        }
      };
      // BFCache (R9). pagehide(persisted=true) means the page is going
      // into BFCache; close cleanly and do NOT schedule a reconnect.
      // pageshow(persisted=true) is the return; reconnect immediately
      // with a fresh attempt counter.
      this._onPageHide = (ev) => {
        if (this.destroyed) return;
        if (!ev || !ev.persisted) return;
        this._bfcacheParked = true;
        this._clearReconnectTimer();
        this.deferredOnVisible = false;
        for (const c of this._pool) {
          c._clearAllTimers();
          try { if (c.ws) c.ws.close(1001, "bfcache"); } catch (_) {}
        }
        this.onStateChange(this.stats());
      };
      this._onPageShow = (ev) => {
        if (this.destroyed) return;
        if (!ev || !ev.persisted) return;
        this._bfcacheParked = false;
        this.deferredOnVisible = false;
        this.attempt = 0;
        this._clearReconnectTimer();
        this._scheduleReconnect({ immediate: true,
                                  resetAttempts: true,
                                  ignoreVisibility: true });
      };
      this._onOnline = () => {
        if (this.destroyed) return;
        const active = this._activeConnection();
        if (active && active.isOpen()) {
          active._sendPing();
        } else if (!this.isTerminal && this._liveConnections().length === 0) {
          this._clearReconnectTimer();
          this._scheduleReconnect({ immediate: true });
        }
      };
      // Network Information API: type/effective-type/downlink changed.
      // Verify the path with an immediate ping; STALE pipeline takes
      // over from there if the path is actually gone.
      this._onNetInfoChange = () => {
        if (this.destroyed) return;
        const active = this._activeConnection();
        if (active && active.state === STATE.ALIVE) active._sendPing();
      };

      if (typeof document !== "undefined") {
        document.addEventListener("visibilitychange", this._onVisibilityChange);
        if (typeof window !== "undefined") {
          window.addEventListener("pagehide", this._onPageHide);
          window.addEventListener("pageshow", this._onPageShow);
        } else if (typeof document.addEventListener === "function") {
          document.addEventListener("pagehide", this._onPageHide);
          document.addEventListener("pageshow", this._onPageShow);
        }
      }
      if (typeof window !== "undefined") {
        window.addEventListener("online", this._onOnline);
      }
      if (typeof navigator !== "undefined" && navigator.connection &&
          typeof navigator.connection.addEventListener === "function") {
        navigator.connection.addEventListener("change", this._onNetInfoChange);
      }
    }

    _rejectPending(reason) {
      for (const [, p] of this._pending) {
        try { p.reject(new Error(reason)); } catch (_) {}
      }
      this._pending.clear();
    }
  }

  function _randomNonce() {
    if (global.crypto && global.crypto.getRandomValues) {
      const a = new Uint8Array(8);
      global.crypto.getRandomValues(a);
      return Array.from(a, (b) => b.toString(16).padStart(2, "0")).join("");
    }
    return Math.random().toString(36).slice(2);
  }

  // Full jitter, capped exponential backoff (R5).
  function _backoff(cfg, attempt) {
    const computed = Math.min(cfg.backoffCapMs, cfg.backoffBaseMs * (2 ** (attempt - 1)));
    return computed * (0.5 + Math.random() * 0.5);
  }

  global.ZimtConnectionManager = ConnectionManager;
  global.ZIMT_CONN_STATE = STATE;
})(typeof window !== "undefined" ? window : globalThis);
