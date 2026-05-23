"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { renderPromptHTML } = require("../static/prompt_highlight.js");

// Helper: collect (class, text) pairs from the rendered HTML. Decodes
// the few HTML entities `esc()` emits so test assertions can use the
// original characters. Empty trim() means whitespace-only tokens are
// dropped; the special chars (`{`, `}`, `|`, etc.) survive as their own
// tokens because the renderer wraps each in a span.
function tokens(html) {
  const decode = (s) => s
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"').replace(/&amp;/g, "&");
  const re = /<span class="([^"]+)">([^<]*)<\/span>|([^<]+)/g;
  const out = [];
  let m;
  while ((m = re.exec(html)) !== null) {
    if (m[1] !== undefined) out.push({ cls: m[1], text: decode(m[2]) });
    else {
      const t = decode(m[3]);
      if (t.trim()) out.push({ cls: "", text: t });
    }
  }
  return out;
}

test("empty input renders a non-breaking space placeholder", () => {
  assert.equal(renderPromptHTML(""), "&nbsp;");
});

test("plain text passes through with HTML escaping but no spans", () => {
  const html = renderPromptHTML("a serene mountain lake");
  assert.equal(tokens(html).length, 1);
  assert.deepEqual(tokens(html)[0], { cls: "", text: "a serene mountain lake" });
});

test("HTML special chars in plain text are escaped", () => {
  const html = renderPromptHTML("a < b > c");
  // We don't care about the exact span breakdown — just that raw `<` and
  // `>` don't leak into the HTML output. `<span` is OK; standalone `<` is
  // what we're guarding against.
  assert.ok(!/(^|[^"a-z])</.test(html.replace(/<span[^>]*>/g, "").replace(/<\/span>/g, "")));
});

test("simple alternation gets brace colouring", () => {
  const html = renderPromptHTML("{red|blue}");
  const toks = tokens(html);
  assert.deepEqual(toks, [
    { cls: "hl-brace", text: "{" },
    { cls: "", text: "red" },
    { cls: "hl-brace", text: "|" },
    { cls: "", text: "blue" },
    { cls: "hl-brace", text: "}" },
  ]);
});

test("braces match across whitespace", () => {
  // The regression motivating the new char-by-char scanner: the old
  // per-word renderer couldn't pair `{` and `}` separated by spaces.
  const html = renderPromptHTML("{red | blue}");
  const braceTokens = tokens(html).filter((t) => t.cls === "hl-brace");
  assert.deepEqual(braceTokens.map((t) => t.text), ["{", "|", "}"]);
});

test("variable reference gets variable colouring", () => {
  const toks = tokens(renderPromptHTML("${color}"));
  assert.deepEqual(toks, [
    { cls: "hl-var", text: "${" },
    { cls: "hl-var-name", text: "color" },
    { cls: "hl-var", text: "}" },
  ]);
});

test("variable definition splits into bracket / name / equals / body / close", () => {
  const toks = tokens(renderPromptHTML("${color=red|blue}"));
  assert.deepEqual(toks, [
    { cls: "hl-var", text: "${" },
    { cls: "hl-var-name", text: "color" },
    { cls: "hl-var-eq", text: "=" },
    { cls: "", text: "red" },
    { cls: "hl-brace", text: "|" },
    { cls: "", text: "blue" },
    { cls: "hl-var", text: "}" },
  ]);
});

test("nested braces close in the right colour each", () => {
  // Outer `{` is plain alternation, inner `${var}` is a variable —
  // the two `}` should paint hl-brace and hl-var respectively.
  const toks = tokens(renderPromptHTML("${color={a|b}|c}"));
  const closers = toks.filter((t) => t.text === "}" || t.text === "{");
  // Sequence: ${ ... ={ ... } ... }
  // Opener stack (LIFO): hl-var (from ${), then hl-brace (from inner {)
  // First `}` (inner) → hl-brace, second `}` (outer) → hl-var
  assert.deepEqual(closers, [
    { cls: "hl-brace", text: "{" },
    { cls: "hl-brace", text: "}" },
    { cls: "hl-var", text: "}" },
  ]);
});

test("comments are rendered as one comment span", () => {
  const toks = tokens(renderPromptHTML("a <!-- note --> b"));
  const comment = toks.find((t) => t.cls === "hl-comment");
  assert.ok(comment);
  assert.equal(comment.text, "<!-- note -->");
});

test("unclosed comment is painted to EOF", () => {
  const toks = tokens(renderPromptHTML("hello <!-- forgot to close"));
  const comment = toks.find((t) => t.cls === "hl-comment");
  assert.ok(comment);
  assert.equal(comment.text, "<!-- forgot to close");
});

test("escaped chars get dimmed in escape colour", () => {
  const toks = tokens(renderPromptHTML(String.raw`\{a\|b\}`));
  const escs = toks.filter((t) => t.cls === "hl-escape");
  assert.deepEqual(escs.map((t) => t.text), ["\\{", "\\|", "\\}"]);
});

test("slash commands still colour their args", () => {
  const html = renderPromptHTML("/cfg 5");
  const toks = tokens(html);
  assert.ok(toks.some((t) => t.cls === "hl-cmd" && t.text === "/cfg"));
  assert.ok(toks.some((t) => t.cls === "hl-num" && t.text === "5"));
});

test("template syntax inside a greedy /negprompt still highlights braces", () => {
  // Plain `bar` should be coloured hl-neg (greedy continues), but the
  // `{` `|` `}` get hl-brace because template colouring overrides arg
  // colouring at the special chars.
  const toks = tokens(renderPromptHTML("/negprompt {foo|bar}"));
  const negs = toks.filter((t) => t.cls === "hl-neg").map((t) => t.text);
  assert.deepEqual(negs, ["foo", "bar"]);
  const braces = toks.filter((t) => t.cls === "hl-brace").map((t) => t.text);
  assert.deepEqual(braces, ["{", "|", "}"]);
});

test("unclosed brace leaves stack but doesn't crash", () => {
  // The user is mid-typing; render must still produce valid HTML.
  const html = renderPromptHTML("{red|blue");
  assert.ok(html.includes("hl-brace"));
});

test("stray closing brace gets plain colour, not brace colour", () => {
  // `}` outside any `{` is literal — matches the lenient Python parser.
  // We check by ensuring NO hl-brace span appears in the output at all.
  const html = renderPromptHTML("hello } world");
  assert.ok(!/class="hl-brace"/.test(html),
            `expected no hl-brace spans, got: ${html}`);
});

test("plain dollar without brace is literal", () => {
  // `$5.99` and `$cost` must not trigger the variable parser.
  const toks = tokens(renderPromptHTML("price: $5.99"));
  assert.ok(!toks.some((t) => t.cls === "hl-var"));
});

test("known slash command is recognised after a template close", () => {
  // After `}`, the next word boundary is at the whitespace; `/cfg`
  // should still be detected.
  const html = renderPromptHTML("{red|blue} /cfg 5");
  assert.ok(html.includes('class="hl-cmd">/cfg'));
});
