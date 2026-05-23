(function (root) {
  "use strict";

  const COMMANDS = [
    "/help", "/?", "/raw", "/many", "/seed", "/negprompt", "/cfg", "/steps",
    "/size", "/res", "/sampler", "/clip_skip", "/model", "/lora", "/tokenize",
    "/mem",
    "/quit", "/exit", "/q",
  ].sort();

  // Static suggestions for `/mem max <size>`. Free-form — accelerate
  // accepts any size string ("4GiB", "6GB", ...); this list is just UX
  // convenience.
  const MEM_MODES = ["off", "max", "cpuoffload", "cpuoffload-seq"];
  const MEM_SIZE_HINTS = ["2GiB", "4GiB", "6GiB", "8GiB", "10GiB", "12GiB", "16GiB"];

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
    "/clip_skip": "SDXL only; skip top N CLIP layers",
    "/model": "load a model",
    "/lora": "add/remove LoRAs (name | name:0.8 | -name | -)",
    "/tokenize": "per-encoder token analysis",
    "/negprompt": "set/clear negative prompt",
    "/mem": "memory strategy (off | max <size> | cpuoffload | cpuoffload-seq)",
  };

  function tokenAtCursor(value, cursor) {
    let start = cursor;
    while (start > 0 && !/\s/.test(value[start - 1])) start -= 1;
    let end = cursor;
    while (end < value.length && !/\s/.test(value[end])) end += 1;
    return { start, end, text: value.slice(start, end) };
  }

  // Returns the offset of the most recent unclosed `${` in `textBefore`,
  // or -1 if the cursor isn't in a variable-name position. "Unclosed"
  // means no matching `}` AND no `=` between the `${` and the cursor —
  // i.e. we're still in the name part of a ${name=...} or ${name}.
  function unclosedVarOpenIndex(textBefore) {
    for (let i = textBefore.length - 1; i >= 0; i -= 1) {
      const c = textBefore[i];
      if (c === "}" || c === "=") return -1;  // closed or past the name
      if (c === "{" && i > 0 && textBefore[i - 1] === "$") return i - 1;
    }
    return -1;
  }

  // Returns the natural end of a `${name...` reference scanning forward
  // from `cursor`. Stops at the first non-name character — including the
  // closing `}` (which is INCLUDED in the return, i.e. the position just
  // after it). Used by the completion harness to set the replacement
  // range so that clicking a suggestion while the cursor sits IN the
  // middle of an existing `${clothesA}` reference replaces the entire
  // reference rather than leaving a `thesA}` suffix behind.
  function naturalVarEnd(text, cursor) {
    for (let i = cursor; i < text.length; i += 1) {
      const c = text[i];
      if (c === "}") return i + 1;
      // Any char that can't be part of a name terminates the var,
      // exclusive — whitespace, `=`, opening braces, etc. stay where
      // they are; we don't swallow them into the replacement range.
      if (!/[A-Za-z0-9_.]/.test(c)) return i;
    }
    return text.length;
  }

  // All variable names defined earlier in `text`, including:
  //   * scalar defs:        ${name=...}
  //   * flat-dotted defs:   ${a.b.c=...}
  //   * composite literals: ${person={hair=...}, {clothes=...}} → emits
  //     "person.hair" and "person.clothes"
  // For composite definitions only the leaf paths are returned (matching
  // what's actually bound in env at expansion time); the parent name
  // itself isn't included because `${person}` would raise.
  function definedVarNames(text) {
    const seen = new Set();
    const out = [];
    const add = (n) => {
      if (!seen.has(n)) { seen.add(n); out.push(n); }
    };
    const re = /\$\{([A-Za-z_][A-Za-z0-9_.]*)\s*=/g;
    let m;
    while ((m = re.exec(text)) !== null) {
      const baseName = m[1];
      const valueStart = m.index + m[0].length;
      const fields = _collectObjectFields(text, valueStart, baseName);
      if (fields.length === 0) {
        add(baseName);
      } else {
        fields.forEach(add);
      }
    }
    return out;
  }

  function _skipWs(text, i) {
    while (i < text.length && /\s/.test(text[i])) i += 1;
    return i;
  }

  // Scan past a balanced `{...}` sequence starting from `start` (which
  // points at the FIRST character of the value, i.e. just after `=`).
  // Returns the position of the closing `}` at the original depth, or
  // -1 if the input is unbalanced. Respects backslash escapes.
  function _skipBalancedClose(text, start) {
    let depth = 0;
    for (let i = start; i < text.length; i += 1) {
      const c = text[i];
      if (c === "\\" && i + 1 < text.length) { i += 1; continue; }
      if (c === "{") depth += 1;
      else if (c === "}") {
        if (depth === 0) return i;
        depth -= 1;
      }
    }
    return -1;
  }

  // Returns flat dotted field paths if text[start:] is an object literal
  // rooted at `prefix`. Returns [] when the value isn't a composite.
  // Mirrors the Python parser's _parse_object_literal_body so the same
  // shape is recognised on both sides.
  function _collectObjectFields(text, start, prefix) {
    const out = [];
    // Peek: must start with `{<ident>=` (after optional whitespace) for
    // this to be an object literal.
    let pos = _skipWs(text, start);
    if (text[pos] !== "{") return out;
    const probe = _skipWs(text, pos + 1);
    const peek = /[A-Za-z_][A-Za-z0-9_]*/y;
    peek.lastIndex = probe;
    const pm = peek.exec(text);
    if (!pm || pm.index !== probe) return out;
    const after = _skipWs(text, pm.index + pm[0].length);
    if (text[after] !== "=") return out;

    // Walk all comma-separated field blocks.
    while (true) {
      pos = _skipWs(text, pos);
      if (text[pos] !== "{") return out;
      pos += 1;
      pos = _skipWs(text, pos);
      const fnRe = /[A-Za-z_][A-Za-z0-9_]*/y;
      fnRe.lastIndex = pos;
      const fm = fnRe.exec(text);
      if (!fm || fm.index !== pos) return out;
      const fieldName = fm[0];
      pos = _skipWs(text, fm.index + fm[0].length);
      if (text[pos] !== "=") return out;
      pos += 1;
      const fullName = `${prefix}.${fieldName}`;
      const nested = _collectObjectFields(text, pos, fullName);
      if (nested.length > 0) {
        out.push(...nested);
      } else {
        out.push(fullName);
      }
      // Skip past the field block's closing `}` regardless of nesting.
      pos = _skipBalancedClose(text, pos);
      if (pos < 0) return out;
      pos += 1;
      pos = _skipWs(text, pos);
      if (text[pos] !== ",") return out;
      pos += 1;
    }
  }

  function tokensBefore(value, end) {
    return value.slice(0, end).split(/\s+/).filter(Boolean);
  }

  function modelForContext(tokens, appState) {
    const models = appState?.models ?? [];
    for (let i = tokens.length - 2; i >= 0; i -= 1) {
      if (tokens[i] !== "/model") continue;
      const name = tokens[i + 1];
      if (!name || name.startsWith("/")) continue;
      const explicit = models.find((m) => m.name === name);
      if (explicit) return explicit;
    }
    if (!appState?.model) return null;
    return models.find((m) => m.name === appState.model) ?? null;
  }

  function completionItems(value, tokStart, tokText, appState) {
    const before = tokensBefore(value, tokStart);
    const prev = before.length ? before[before.length - 1] : "";
    const lower = tokText.toLowerCase();

    // Variable-name completion. We rely on the caller (updateSuggest in
    // app.js) to have already shifted tokStart to point at the `$` of an
    // unclosed `${...` — so when tokText starts with `${`, the user is
    // typing inside a variable head. Accepted labels include the closing
    // brace, so picking one doesn't leave the editor with an unclosed
    // var. Only definitions BEFORE this `${` are offered; forward refs
    // would raise at expansion time, so suggesting them would mislead.
    if (tokText.startsWith("${")) {
      const namePart = tokText.slice(2).toLowerCase();
      const defs = definedVarNames(value.slice(0, tokStart));
      return defs
        .filter((n) => n.toLowerCase().startsWith(namePart))
        .map((n) => ({ label: `\${${n}}`, desc: "variable" }));
    }

    if (prev === "/model") {
      const models = appState?.models ?? [];
      return models
        .filter((m) => m.name.toLowerCase().startsWith(lower))
        .map((m) => ({ label: m.name, desc: m.description }));
    }

    // /lora takes exactly one LoRA-name argument. Complete only the token
    // immediately following /lora, filtered by compatibility with the
    // active base model (or the most recent /model arg in this line).
    if (prev === "/lora") {
      const loras = appState?.loras ?? [];
      const base = modelForContext(before, appState);
      const baseTags = new Set(base?.compatibility_tags ?? []);
      let core = lower;
      if (core.startsWith("-")) core = core.slice(1);
      if (core.includes(":")) return [];   // user typing a weight
      return loras
        .filter((l) => l.name.toLowerCase().startsWith(core))
        .filter((l) => {
          if (!baseTags.size) return true;
          return (l.compatible_with || []).some((t) => baseTags.has(t));
        })
        .map((l) => ({
          label: l.name,
          desc: l.trigger_tags ? `trigger: ${l.trigger_tags}` : l.description,
        }));
    }

    if (prev === "/sampler") {
      const model = modelForContext(before, appState);
      return (model?.samplers ?? [])
        .filter((s) => s.toLowerCase().startsWith(lower))
        .map((s) => ({ label: s, desc: "" }));
    }

    if (prev === "/res") {
      const model = modelForContext(before, appState);
      const presets = model?.resolutions ?? [];
      const byArea = (a, b) => (b.w * b.h) - (a.w * a.h);
      const square = [...presets].sort(byArea).find((r) => r.w === r.h);
      const landscape = [...presets].sort(byArea).find((r) => r.w > r.h);
      const portrait = [...presets].sort(byArea).find((r) => r.w < r.h);
      const orientations = [
        square && { label: "square", desc: `largest 1:1 (${square.w}x${square.h})` },
        landscape && { label: "landscape", desc: `largest W>H (${landscape.w}x${landscape.h})` },
        portrait && { label: "portrait", desc: `largest W<H (${portrait.w}x${portrait.h})` },
      ].filter(Boolean);
      const explicit = presets.map((r) => ({ label: `${r.w}x${r.h}`, desc: r.label }));
      return [...orientations, ...explicit]
        .filter((it) => it.label.toLowerCase().startsWith(lower));
    }

    if (prev === "/mem") {
      return MEM_MODES
        .filter((m) => m.toLowerCase().startsWith(lower))
        .map((m) => ({ label: m, desc: "" }));
    }

    // `/mem max <size>`: suggest common caps when the user is on the
    // size token. `/mem` is GREEDY in the Python parser, so we can't
    // rely on `prev` alone — look back for `/mem max`.
    if (prev === "max") {
      const memIdx = before.lastIndexOf("/mem");
      if (memIdx >= 0 && before[memIdx + 1] === "max") {
        return MEM_SIZE_HINTS
          .filter((s) => s.toLowerCase().startsWith(lower))
          .map((s) => ({ label: s, desc: "" }));
      }
    }

    if (tokText.startsWith("/")) {
      return COMMANDS
        .filter((c) => c.toLowerCase().startsWith(lower))
        .map((c) => ({ label: c, desc: COMMAND_HELP[c] ?? "" }));
    }

    return [];
  }

  root.tokenAtCursor = tokenAtCursor;
  root.completionItems = completionItems;
  root.unclosedVarOpenIndex = unclosedVarOpenIndex;
  root.naturalVarEnd = naturalVarEnd;
  root.definedVarNames = definedVarNames;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      COMMANDS, COMMAND_HELP, tokenAtCursor, completionItems,
      unclosedVarOpenIndex, naturalVarEnd, definedVarNames,
    };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
