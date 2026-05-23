// Char-by-char prompt tokenizer + HTML renderer.
//
// Replaces the original whitespace-split renderer in app.js. The motivation
// is that template syntax (`{a|b}` alternation and `${var=...}` references)
// can span whitespace boundaries — e.g. `{red | blue}` puts the matching
// `}` three tokens away from the `{` — so a per-word approach cannot
// colour matching delimiters consistently.
//
// The scanner threads two pieces of state through one pass:
//   * slash-command context (argsLeft, argClass, greedyClass) — same
//     semantics as the previous renderer: argsLeft decrements once per
//     non-template whitespace-delimited word.
//   * braceStack — the colour class for each pending close. `{` pushes
//     "hl-brace", `${` pushes "hl-var", so the matching `}` always paints
//     in the colour of its opener even when they're nested or far apart.
//
// Where slash-command coloring and template coloring would overlap (e.g.
// `/negprompt ${c=red|blue}${c}`) the template colours win for the special
// chars and the slash-cmd colour applies to the surrounding plain text.
// That keeps the template structure visible no matter what command it's
// nested inside.

(function (root) {
  "use strict";

  // Command → arg-spec for highlighting. Kept here (rather than in app.js)
  // so the tokenizer is self-contained and unit-testable. App.js imports
  // this and only adds the surface glue (input/scroll listeners).
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

  // Chars that are NOT part of a plain text run — when scanning a literal
  // word we stop at any of these (plus whitespace).
  function isSpecial(text, i) {
    const c = text[i];
    if (c === "{" || c === "}" || c === "|" || c === "\\") return true;
    if (c === "$" && text[i + 1] === "{") return true;
    if (c === "<" && text.substring(i, i + 4) === "<!--") return true;
    return false;
  }

  function esc(s) {
    return s.replace(/[&<>"]/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;",
    }[c]));
  }

  function span(cls, text) {
    if (!cls) return esc(text);
    return `<span class="${cls}">${esc(text)}</span>`;
  }

  // Main entry — takes the prompt text, returns HTML for the highlight mirror.
  function renderPromptHTML(text) {
    if (text === "") return "&nbsp;";
    const N = text.length;
    let out = "";
    let i = 0;

    // Slash-command state. argsLeft decrements once per whitespace-delimited
    // non-template token; greedyClass persists until the next /cmd.
    let argsLeft = 0;
    let argClass = "";
    let greedyClass = null;

    // Pending non-WS-token plain-content flag — set when we emit any
    // plain content since the last whitespace. Used to decide whether to
    // decrement argsLeft at the next whitespace boundary.
    let plainEmitted = false;

    // Stack of close-brace colour classes. Each `{` or `${` pushes; each
    // `}` pops. Template syntax can span whitespace so this state must
    // be threaded through the whole scan, not reset per token.
    const braceStack = [];

    // Whether the very next non-WS char would be the start of a new
    // word boundary. Slash-cmds are only recognised at word boundaries.
    let atWordStart = true;

    const plainClass = () => {
      if (greedyClass !== null) return greedyClass;
      if (argsLeft > 0) return argClass;
      return "";
    };

    while (i < N) {
      const c = text[i];

      // Whitespace: emit verbatim, decrement argsLeft if we just finished
      // a plain-content token.
      if (/\s/.test(c)) {
        if (plainEmitted && argsLeft > 0) argsLeft -= 1;
        plainEmitted = false;
        out += c;
        i += 1;
        atWordStart = true;
        continue;
      }

      // Comment: `<!-- ... -->`. Treat the whole run as one comment span.
      // If unclosed, paint to EOF — same lenient policy the Python parser
      // uses (it just strips the comment and continues).
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

      // Escape: `\X`. Paint both chars dim so the user sees the escape
      // is recognised. Trailing lone `\` is just literal.
      if (c === "\\" && i + 1 < N) {
        out += span("hl-escape", text.substring(i, i + 2));
        i += 2;
        plainEmitted = true;
        atWordStart = false;
        continue;
      }

      // Variable: `${name}` or `${name=...}`. Parse the name and the `=`
      // (if any) inline so the colouring within the `${...}` head is
      // contiguous regardless of what follows.
      if (c === "$" && text[i + 1] === "{") {
        out += span("hl-var", "${");
        i += 2;
        let nameEnd = i;
        while (nameEnd < N && /[A-Za-z0-9_]/.test(text[nameEnd])) nameEnd += 1;
        if (nameEnd > i) {
          out += span("hl-var-name", text.substring(i, nameEnd));
        }
        i = nameEnd;
        if (i < N && text[i] === "=") {
          // Definition: `${name=...}` — the matching `}` closes the var,
          // so push hl-var for that close. The RHS is parsed by the main
          // loop, which means braces / pipes inside the RHS get
          // hl-brace as usual.
          out += span("hl-var-eq", "=");
          i += 1;
          braceStack.push("hl-var");
        } else if (i < N && text[i] === "}") {
          // Reference: `${name}` — emit the close immediately.
          out += span("hl-var", "}");
          i += 1;
        }
        // If neither `=` nor `}` follows (e.g. unfinished `${` at EOF or
        // a stray space), leave the stack alone and let the main loop
        // continue. The user will see the partial highlighting and know
        // their var head isn't complete.
        plainEmitted = true;
        atWordStart = false;
        continue;
      }

      // Choice open
      if (c === "{") {
        out += span("hl-brace", "{");
        braceStack.push("hl-brace");
        i += 1;
        plainEmitted = true;
        atWordStart = false;
        continue;
      }

      // Close brace — paints in the colour of its opener; stray `}`
      // outside any open brace gets the surrounding plain class.
      if (c === "}") {
        if (braceStack.length > 0) {
          out += span(braceStack.pop(), "}");
        } else {
          out += span(plainClass(), "}");
        }
        i += 1;
        plainEmitted = true;
        atWordStart = false;
        continue;
      }

      // Pipe — separator inside any open brace, literal elsewhere.
      // Always paint as hl-brace inside a brace context (even inside a
      // `${...}` definition's RHS) so the alternation reads as one
      // syntax-family regardless of which container holds it.
      if (c === "|") {
        if (braceStack.length > 0) {
          out += span("hl-brace", "|");
        } else {
          out += span(plainClass(), "|");
        }
        i += 1;
        plainEmitted = true;
        atWordStart = false;
        continue;
      }

      // Slash command — only recognised at the start of a word.
      if (c === "/" && atWordStart) {
        let cmdEnd = i + 1;
        // Allow `?` for `/?` and the usual identifier chars.
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
          // The command itself is not a plain-content token.
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
      while (runEnd < N && !/\s/.test(text[runEnd]) && !isSpecial(text, runEnd)) {
        runEnd += 1;
      }
      out += span(plainClass(), text.substring(i, runEnd));
      i = runEnd;
      plainEmitted = true;
      atWordStart = false;
    }

    // Trailing newline guard — browsers collapse a final \n in <pre>.
    if (text.endsWith("\n")) out += "\n";
    return out;
  }

  root.renderPromptHTML = renderPromptHTML;
  root.HL_CMDS = HL_CMDS;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { renderPromptHTML, HL_CMDS };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
