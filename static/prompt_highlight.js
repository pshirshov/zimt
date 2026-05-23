// Char-by-char prompt tokenizer + HTML renderer.
//
// Recognises the NEW template syntax (see src/zimt/dynamics.py for the
// authoritative spec):
//   [a|b|c]                  alternation (only when `|` appears between
//                            matched brackets; bare `[word]` is literal
//                            so compel's `[word]-` weighting passes
//                            through unchanged)
//   ${name=...}              scalar variable definition
//   ${name}                  reference (dots allowed for composite paths)
//   ${obj={k=v, k=v}}        composite — field name/eq painted distinctly
//   `verbatim text`          stored as raw, rendered as the inner text
//   \[ \] \{ \} \| \$ \` \\  escapes (highlighted dim)
//   <!-- foo -->             HTML-style comment
//
// Slash-command tokens are recognised at word boundaries with the same
// `HL_CMDS` table the old highlighter used; their args carry the
// per-command colour while template special chars override at their
// positions so the user can SEE the template structure even inside a
// greedy arg run (e.g. `/negprompt [foo|bar]`).

(function (root) {
  "use strict";

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
    "/many": {n: 1, cls: "hl-num"},
    "/negprompt": {greedy: true, cls: "hl-neg"},
    "/tokenize": {greedy: true},
    "/mem": {greedy: true, cls: "hl-flag"},
  };

  function esc(s) {
    return s.replace(/[&<>"]/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;",
    }[c]));
  }
  function span(cls, text) {
    if (!cls) return esc(text);
    return `<span class="${cls}">${esc(text)}</span>`;
  }

  // Find the next un-escaped backtick at or after `start`.
  function findVerbatimEnd(text, start) {
    let i = start;
    while (i < text.length) {
      const c = text[i];
      if (c === "\\" && i + 1 < text.length) { i += 2; continue; }
      if (c === "`") return i;
      i += 1;
    }
    return -1;
  }

  // Lookahead for `[`: find matching `]` at the same depth and report
  // whether a top-level `|` separator was seen. Mirrors the Python
  // parser's _check_alternation so the highlighter and the runtime
  // agree on what's an alternation.
  function checkAlternation(text, start) {
    if (text[start] !== "[") return [-1, false];
    let i = start + 1;
    let depth = 1;
    let hasPipe = false;
    while (i < text.length) {
      const c = text[i];
      if (c === "\\" && i + 1 < text.length) { i += 2; continue; }
      if (c === "`") {
        const end = findVerbatimEnd(text, i + 1);
        if (end < 0) return [-1, hasPipe];
        i = end + 1;
        continue;
      }
      if (c === "$" && text[i + 1] === "{") {
        // Skip past the ${...} block by counting braces.
        let j = i + 2;
        let d = 0;
        while (j < text.length) {
          const cc = text[j];
          if (cc === "\\" && j + 1 < text.length) { j += 2; continue; }
          if (cc === "`") {
            const ve = findVerbatimEnd(text, j + 1);
            if (ve < 0) return [-1, hasPipe];
            j = ve + 1;
            continue;
          }
          if (cc === "{") { d += 1; j += 1; }
          else if (cc === "}") {
            if (d === 0) break;
            d -= 1;
            j += 1;
          } else { j += 1; }
        }
        if (j >= text.length) return [-1, hasPipe];
        i = j + 1;
        continue;
      }
      if (c === "[") { depth += 1; i += 1; }
      else if (c === "]") {
        depth -= 1;
        if (depth === 0) return [i, hasPipe];
        i += 1;
      } else if (c === "|" && depth === 1) {
        hasPipe = true;
        i += 1;
      } else { i += 1; }
    }
    return [-1, hasPipe];
  }

  function renderPromptHTML(text) {
    if (text === "") return "&nbsp;";
    const N = text.length;
    let out = "";
    let i = 0;

    // Slash-command state — argsLeft decrements once per whitespace-
    // delimited non-template token; greedyClass persists until the next
    // `/cmd` is encountered.
    let argsLeft = 0;
    let argClass = "";
    let greedyClass = null;
    let plainEmitted = false;
    let atWordStart = true;

    // Stack entries: {cls, kind} where kind is "alt" / "obj" / "var".
    // `cls` is the colour used for the matching close; `kind` lets us
    // detect "we're inside an object literal" so a `,` followed by an
    // `<ident>=` token paints the next field name + `=` distinctly.
    const closeStack = [];
    const inObject = () => closeStack.length > 0
      && closeStack[closeStack.length - 1].kind === "obj";

    const plainClass = () => {
      if (greedyClass !== null) return greedyClass;
      if (argsLeft > 0) return argClass;
      return "";
    };

    while (i < N) {
      const c = text[i];

      if (/\s/.test(c)) {
        if (plainEmitted && argsLeft > 0) argsLeft -= 1;
        plainEmitted = false;
        out += c;
        i += 1;
        atWordStart = true;
        continue;
      }

      // Comment
      if (c === "<" && text.substring(i, i + 4) === "<!--") {
        const end = text.indexOf("-->", i + 4);
        if (end >= 0) {
          out += span("hl-comment", text.substring(i, end + 3));
          i = end + 3;
        } else {
          out += span("hl-comment", text.substring(i));
          i = N;
        }
        plainEmitted = true;
        atWordStart = false;
        continue;
      }

      // Escape — emit both chars dim.
      if (c === "\\" && i + 1 < N) {
        out += span("hl-escape", text.substring(i, i + 2));
        i += 2;
        plainEmitted = true;
        atWordStart = false;
        continue;
      }

      // Verbatim string — paint the whole `...` run including delimiters
      // in the verbatim colour so the user can see it's a literal.
      if (c === "`") {
        const end = findVerbatimEnd(text, i + 1);
        const cls = "hl-verbatim";
        if (end < 0) {
          out += span(cls, text.substring(i));
          i = N;
        } else {
          out += span(cls, text.substring(i, end + 1));
          i = end + 1;
        }
        plainEmitted = true;
        atWordStart = false;
        continue;
      }

      // Helper: peek `<ident>=` at position j (skipping leading ws). If
      // matched, emit the whitespace + name + `=` and return the new
      // position. Used both for the opening of an object literal (after
      // `{`) and for subsequent fields (after `,`).
      const tryEmitField = (j) => {
        let k = j;
        while (k < N && /\s/.test(text[k])) k += 1;
        const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(text.substring(k));
        if (!m) return -1;
        let kEnd = k + m[0].length;
        let kEq = kEnd;
        while (kEq < N && /\s/.test(text[kEq])) kEq += 1;
        if (text[kEq] !== "=") return -1;
        out += text.substring(j, k);
        out += span("hl-var-name", m[0]);
        out += text.substring(kEnd, kEq);
        out += span("hl-var-eq", "=");
        return kEq + 1;
      };

      // Variable: ${name}, ${name=...}
      if (c === "$" && text[i + 1] === "{") {
        out += span("hl-var", "${");
        i += 2;
        let nameEnd = i;
        while (nameEnd < N && /[A-Za-z0-9_.]/.test(text[nameEnd])) nameEnd += 1;
        if (nameEnd > i) {
          out += span("hl-var-name", text.substring(i, nameEnd));
        }
        i = nameEnd;
        if (i < N && text[i] === "=") {
          out += span("hl-var-eq", "=");
          i += 1;
          closeStack.push({cls: "hl-var", kind: "var"});
        } else if (i < N && text[i] === "}") {
          out += span("hl-var", "}");
          i += 1;
        }
        plainEmitted = true;
        atWordStart = false;
        continue;
      }

      // Object literal opener — peek for `{<ident>=` to detect; literal
      // `{` (no `<ident>=` inside) stays plain. We peek WITHOUT
      // emitting first (lookahead only) so the `{` itself can be
      // emitted as hl-brace before the field-name span.
      if (c === "{") {
        let probe = i + 1;
        while (probe < N && /\s/.test(text[probe])) probe += 1;
        const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(text.substring(probe));
        let isField = false;
        if (m) {
          let kEq = probe + m[0].length;
          while (kEq < N && /\s/.test(text[kEq])) kEq += 1;
          isField = text[kEq] === "=";
        }
        if (isField) {
          out += span("hl-brace", "{");
          closeStack.push({cls: "hl-brace", kind: "obj"});
          i += 1;
          const next = tryEmitField(i);
          if (next >= 0) i = next;
          plainEmitted = true;
          atWordStart = false;
          continue;
        }
        out += span(plainClass(), "{");
        i += 1;
        plainEmitted = true;
        atWordStart = false;
        continue;
      }
      if (c === "}") {
        if (closeStack.length > 0) {
          out += span(closeStack.pop().cls, "}");
        } else {
          out += span(plainClass(), "}");
        }
        i += 1;
        plainEmitted = true;
        atWordStart = false;
        continue;
      }

      // Alternation `[...|...]` — only when the matched `]` contains a
      // top-level `|`. Bare `[word]` stays literal (compel-friendly).
      if (c === "[") {
        const [closePos, hasPipe] = checkAlternation(text, i);
        if (closePos > 0 && hasPipe) {
          out += span("hl-brace", "[");
          closeStack.push({cls: "hl-brace", kind: "alt"});
          i += 1;
          continue;
        }
        const klass = (hasPipe && closePos < 0) ? "hl-unknown-cmd" : plainClass();
        out += span(klass, "[");
        i += 1;
        plainEmitted = true;
        atWordStart = false;
        continue;
      }
      if (c === "]") {
        if (closeStack.length > 0 && closeStack[closeStack.length - 1].kind === "alt") {
          out += span(closeStack.pop().cls, "]");
        } else {
          out += span(plainClass(), "]");
        }
        i += 1;
        plainEmitted = true;
        atWordStart = false;
        continue;
      }

      // Pipe — separator inside any open bracket, literal elsewhere.
      if (c === "|") {
        if (closeStack.length > 0) {
          out += span("hl-brace", "|");
        } else {
          out += span(plainClass(), "|");
        }
        i += 1;
        plainEmitted = true;
        atWordStart = false;
        continue;
      }

      // Comma — field separator inside an object literal. After the
      // comma we peek for the next `<ident>=` and paint it as a field
      // (mirrors what we do after `{`). Outside an object literal `,`
      // is just literal text.
      if (c === ",") {
        if (inObject()) {
          out += span("hl-var-eq", ",");
          i += 1;
          const next = tryEmitField(i);
          if (next >= 0) i = next;
          plainEmitted = true;
          atWordStart = false;
          continue;
        }
        out += span(plainClass(), ",");
        i += 1;
        plainEmitted = true;
        atWordStart = false;
        continue;
      }

      // Slash command at word boundary.
      if (c === "/" && atWordStart) {
        let cmdEnd = i + 1;
        while (cmdEnd < N && /[A-Za-z0-9_?]/.test(text[cmdEnd])) cmdEnd += 1;
        const cmd = text.substring(i, cmdEnd);
        if (cmd in HL_CMDS) {
          out += span("hl-cmd", cmd);
          const spec = HL_CMDS[cmd];
          if (spec.greedy) {
            greedyClass = spec.cls || "";
            argsLeft = 0;
          } else {
            argsLeft = spec.n;
            argClass = spec.cls || "";
            greedyClass = null;
          }
          plainEmitted = false;
        } else {
          out += span("hl-unknown-cmd", cmd);
          plainEmitted = true;
        }
        i = cmdEnd;
        atWordStart = false;
        continue;
      }

      // Plain run — read until the next special char or whitespace.
      let runEnd = i + 1;
      while (runEnd < N) {
        const cc = text[runEnd];
        if (/\s/.test(cc)) break;
        if (cc === "[" || cc === "]" || cc === "|" || cc === "\\"
            || cc === "{" || cc === "}" || cc === "," || cc === "`") break;
        if (cc === "$" && text[runEnd + 1] === "{") break;
        if (cc === "<" && text.substring(runEnd, runEnd + 4) === "<!--") break;
        runEnd += 1;
      }
      out += span(plainClass(), text.substring(i, runEnd));
      i = runEnd;
      plainEmitted = true;
      atWordStart = false;
    }

    if (text.endsWith("\n")) out += "\n";
    return out;
  }

  root.renderPromptHTML = renderPromptHTML;
  root.HL_CMDS = HL_CMDS;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { renderPromptHTML, HL_CMDS };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
