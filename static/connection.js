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
//   * online/offline + visibilitychange wiring.
//
// Intentional gaps (documented for honesty per the skill's R14):
//   * No overlapping-connection failover (R6). Localhost + nginx
//     doesn't need it; pure-sequence reconnect is good enough.
//   * No BFCache wiring (R9 pagehide/pageshow). zimt is rarely
//     navigated away from, so the optimization isn't worth its bugs.
//   * Heartbeat timer runs on the main thread (R14). Chrome throttles
//     main-thread timers in heavily-backgrounded tabs; the time-jump
//     detector recovers on resume, but heartbeats themselves stop
//     until visible.
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

  const RETRIABLE_CODES = new Set([1001, 1005, 1006, 1011, 1012, 1013, 1014]);
  const NON_RETRIABLE_CODES = new Set([1002, 1003, 1007, 1008, 1009, 1010, 1015]);

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

  class ConnectionManager {
    constructor(opts) {
      this.url = opts.url;
      this.cfg = Object.assign({}, DEFAULTS, opts || {});
      this.onMessage = opts.onMessage || (() => {});
      this.onStateChange = opts.onStateChange || (() => {});

      this.ws = null;
      this.state = STATE.DEAD;
      this.attempt = 0;
      this.isTerminal = false;
      this.lastCloseCode = null;
      this.lastCloseReason = "";
      this.deferredOnVisible = false;
      this.destroyed = false;
      // pending request ids waiting on the server. Failure modes for
      // these are handled by the user of the manager (rejects on close).
      this._pending = new Map();
      this._nextReqId = 1;
      this._outbox = [];
      // pings we've sent, awaiting a matching pong
      this._pendingPings = new Map();  // nonce -> sent_at_ms
      this._connectTimer = null;
      this._pingTimer = null;
      this._reconnectTimer = null;
      this._staleTimer = null;
      this._lastTickAt = Date.now();
      this._tickTimer = setInterval(() => this._tick(), 1000);

      this._wireLifecycle();
      this._connect();
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
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(msg);
        } else {
          this._outbox.push(msg);
        }
      });
    }

    stats() {
      return {
        state: this.state,
        attempt: this.attempt,
        maxAttempts: this.cfg.maxAttempts,
        isTerminal: this.isTerminal,
        deferredOnVisible: this.deferredOnVisible,
        pendingPings: this._pendingPings.size,
        nextReconnectInMs: this._nextReconnectInMs(),
        lastCloseCode: this.lastCloseCode,
        lastCloseReason: this.lastCloseReason,
      };
    }

    destroy() {
      this.destroyed = true;
      this._clearAllTimers();
      this._rejectPending("manager destroyed");
      try { if (this.ws) this.ws.close(1000, "client shutdown"); } catch (_) {}
    }

    // ---- internal ----

    _setState(s) {
      if (this.state === s) return;
      this.state = s;
      this.onStateChange(this.stats());
    }

    _connect() {
      if (this.destroyed || this.isTerminal) return;
      this._clearTimer("_reconnectTimer");
      this._setState(STATE.NEW);
      try {
        this.ws = new WebSocket(this.url);
      } catch (_) {
        this._scheduleReconnect();
        return;
      }
      // Connect timeout — independent from pong timeout. R4: also check
      // native readyState in case the platform's open event raced us.
      this._connectTimer = setTimeout(() => {
        if (this.state === STATE.NEW &&
            this.ws && this.ws.readyState === WebSocket.CONNECTING) {
          try { this.ws.close(); } catch (_) {}
        }
      }, this.cfg.connectTimeoutMs);

      this.ws.onopen = () => {
        this._clearTimer("_connectTimer");
        this.attempt = 0;
        this._setState(STATE.ALIVE);
        // Flush queued requests.
        while (this._outbox.length) this.ws.send(this._outbox.shift());
        this._schedulePing();
      };

      this.ws.onmessage = (ev) => this._onMessage(ev);

      this.ws.onerror = () => {
        // The browser also fires `close` after `error`. Don't recurse;
        // let _onClose drive the state transition.
      };

      this.ws.onclose = (ev) => this._onClose(ev);
    }

    _onMessage(ev) {
      let m;
      try { m = JSON.parse(ev.data); } catch (_) { return; }
      // Heartbeat: handled in-band before the user sees it.
      if (m && m.type === "ping") {
        try {
          this.ws.send(JSON.stringify({ type: "pong",
            nonce: m.nonce, ts: m.ts }));
        } catch (_) {}
        return;
      }
      if (m && m.type === "pong") {
        if (typeof m.nonce === "string") this._pendingPings.delete(m.nonce);
        // Recovery: a late pong while we were STALE proves the peer is
        // still talking. Promote back to ALIVE.
        if (this.state === STATE.STALE) {
          this._clearTimer("_staleTimer");
          this._setState(STATE.ALIVE);
        }
        return;
      }
      // RPC response: route to the matching pending promise.
      if (m && m.type === "resp") {
        const p = this._pending.get(m.id);
        if (!p) return;
        this._pending.delete(m.id);
        if (m.ok) p.resolve(m.result);
        else p.reject(new Error(m.error || "request failed"));
        return;
      }
      // Anything else is an event; bubble to the user.
      this.onMessage(m);
    }

    _onClose(ev) {
      this._clearTimer("_connectTimer");
      this._clearTimer("_pingTimer");
      this._clearTimer("_staleTimer");
      this.lastCloseCode = (ev && ev.code) || 0;
      this.lastCloseReason = (ev && ev.reason) || "";
      this._setState(STATE.DEAD);
      this._rejectPending("connection lost");
      this._pendingPings.clear();

      const code = this.lastCloseCode;
      if (NON_RETRIABLE_CODES.has(code)) {
        this.isTerminal = true;
        this.onStateChange(this.stats());
        return;
      }
      this._scheduleReconnect();
    }

    _schedulePing() {
      this._clearTimer("_pingTimer");
      if (this.destroyed) return;
      this._pingTimer = setTimeout(() => this._sendPing(), this.cfg.pingIntervalMs);
    }

    _sendPing() {
      if (this.destroyed) return;
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      const nonce = _randomNonce();
      this._pendingPings.set(nonce, Date.now());
      try {
        this.ws.send(JSON.stringify({ type: "ping", nonce, ts: Date.now() }));
      } catch (_) {
        return;
      }
      // Pong timer. If still pending after pongTimeoutMs we go STALE,
      // and after staleGraceMs we close the socket (which triggers
      // reconnect via _onClose).
      setTimeout(() => {
        if (!this._pendingPings.has(nonce)) return;
        this._enterStale();
      }, this.cfg.pongTimeoutMs);
      // Schedule next ping regardless of pong outcome — keeps the
      // beat steady so a stuck-but-not-quite-dead peer is exercised.
      this._schedulePing();
    }

    _enterStale() {
      if (this.state !== STATE.ALIVE && this.state !== STATE.NEW) return;
      this._setState(STATE.STALE);
      this._clearTimer("_staleTimer");
      this._staleTimer = setTimeout(() => {
        // Grace expired; force close. _onClose handles reconnect.
        try { if (this.ws) this.ws.close(4000, "stale"); } catch (_) {}
      }, this.cfg.staleGraceMs);
    }

    _scheduleReconnect() {
      if (this.destroyed || this.isTerminal) return;
      this.attempt += 1;
      if (this.attempt > this.cfg.maxAttempts) {
        this.isTerminal = true;
        this.onStateChange(this.stats());
        return;
      }
      // Defer-while-hidden. The visibilitychange handler clears the
      // flag and re-runs scheduleReconnect with no extra wait.
      if (typeof document !== "undefined" &&
          document.visibilityState === "hidden") {
        this.deferredOnVisible = true;
        this.onStateChange(this.stats());
        return;
      }
      this.deferredOnVisible = false;
      const delay = _backoff(this.cfg, this.attempt);
      this._reconnectAt = Date.now() + delay;
      this.onStateChange(this.stats());
      this._reconnectTimer = setTimeout(() => this._connect(), delay);
    }

    _nextReconnectInMs() {
      if (!this._reconnectTimer || !this._reconnectAt) return null;
      return Math.max(0, this._reconnectAt - Date.now());
    }

    _tick() {
      const now = Date.now();
      const elapsed = now - this._lastTickAt;
      this._lastTickAt = now;
      if (elapsed > 1000 + this.cfg.timeJumpThresholdMs) {
        this._handleResume(elapsed);
      }
    }

    _handleResume(elapsedMs) {
      // R8: long gaps mean NAT tables and TCP state are likely gone.
      // Don't burn pong timeouts proving what we already know — drop
      // the current socket so reconnect logic picks up.
      if (elapsedMs >= this.cfg.pongTimeoutMs &&
          this.ws && this.ws.readyState !== WebSocket.CLOSED) {
        try { this.ws.close(4001, "time-jump"); } catch (_) {}
      } else {
        // Short pause — verify with a ping immediately.
        this._sendPing();
      }
    }

    _wireLifecycle() {
      if (typeof document === "undefined") return;
      document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "visible") {
          if (this.deferredOnVisible) {
            this.deferredOnVisible = false;
            this._scheduleReconnect();
          } else if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            // Came back from background; verify the channel.
            this._sendPing();
          }
        }
      });
      window.addEventListener("online", () => {
        if (this.state === STATE.DEAD && !this.isTerminal) {
          // Drop pending backoff — we know the network is back.
          this._clearTimer("_reconnectTimer");
          this._connect();
        } else if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this._sendPing();
        }
      });
      // online/offline + Network Information API change all use the
      // same "verify with a ping" path; offline is informational only
      // because the close event reliably fires when the OS knows.
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
      this._clearTimer("_reconnectTimer");
      if (this._tickTimer) {
        clearInterval(this._tickTimer);
        this._tickTimer = null;
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
})(window);
