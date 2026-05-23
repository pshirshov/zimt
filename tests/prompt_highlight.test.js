"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { renderPromptHTML } = require("../static/prompt_highlight.js");

// Helper: collect (class, text) pairs from the rendered HTML. Decodes
// the few HTML entities `esc()` emits so test assertions can use the
// original characters. Whitespace-only untagged tokens are dropped.
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

test("empty input renders a non-breaking space", () => {
  assert.equal(renderPromptHTML(""), "&nbsp;");
});

test("plain text passes through as one untagged token", () => {
  const html = renderPromptHTML("a serene mountain lake");
  assert.equal(tokens(html).length, 1);
  assert.deepEqual(tokens(html)[0], { cls: "", text: "a serene mountain lake" });
});

test("alternation `[a|b]` highlights brackets and pipe in hl-brace", () => {
  const toks = tokens(renderPromptHTML("[red|blue]"));
  const braces = toks.filter((t) => t.cls === "hl-brace").map((t) => t.text);
  assert.deepEqual(braces, ["[", "|", "]"]);
});

test("alternation matches across whitespace", () => {
  const html = renderPromptHTML("[red | blue]");
  const braces = tokens(html).filter((t) => t.cls === "hl-brace").map((t) => t.text);
  assert.deepEqual(braces, ["[", "|", "]"]);
});

test("bare `[word]` (no pipe) stays literal — compel-friendly", () => {
  // Critical: compel's `[word]-` negative weighting must not be eaten
  // by the highlighter. No hl-brace spans, brackets render as plain.
  const html = renderPromptHTML("[bad anatomy]-");
  assert.ok(!/class="hl-brace"/.test(html));
});

test("variable reference: ${ name }", () => {
  const toks = tokens(renderPromptHTML("${color}"));
  assert.deepEqual(toks, [
    { cls: "hl-var", text: "${" },
    { cls: "hl-var-name", text: "color" },
    { cls: "hl-var", text: "}" },
  ]);
});

test("dotted reference highlights the whole path as one var-name", () => {
  const toks = tokens(renderPromptHTML("${person.hair}"));
  const names = toks.filter((t) => t.cls === "hl-var-name").map((t) => t.text);
  assert.deepEqual(names, ["person.hair"]);
});

test("scalar definition: ${name=value}", () => {
  const toks = tokens(renderPromptHTML("${color=red}"));
  const seq = toks.map((t) => `${t.cls}:${t.text}`);
  // Outer brackets paint hl-var; name paints hl-var-name; '=' paints
  // hl-var-eq; the value `red` is plain.
  assert.deepEqual(seq, [
    "hl-var:${",
    "hl-var-name:color",
    "hl-var-eq:=",
    ":red",
    "hl-var:}",
  ]);
});

test("composite definition: ${obj={k=v, k=v}}", () => {
  // The new JSON-like syntax — field names + `=` use the variable
  // colour family; the comma uses hl-var-eq for visual consistency.
  const toks = tokens(renderPromptHTML(
    "${person={hair=blond, clothes=red}}"
  ));
  const names = toks.filter((t) => t.cls === "hl-var-name").map((t) => t.text);
  assert.deepEqual(names, ["person", "hair", "clothes"]);
  const eqs = toks.filter((t) => t.cls === "hl-var-eq").map((t) => t.text);
  assert.deepEqual(eqs, ["=", "=", ",", "="]);
});

test("nested composite: ${a={b={c=v}}}", () => {
  const toks = tokens(renderPromptHTML("${a={b={c=v}}}"));
  const names = toks.filter((t) => t.cls === "hl-var-name").map((t) => t.text);
  assert.deepEqual(names, ["a", "b", "c"]);
});

test("verbatim string is rendered as one verbatim span", () => {
  const toks = tokens(renderPromptHTML("a `verbatim text` b"));
  const verb = toks.find((t) => t.cls === "hl-verbatim");
  assert.ok(verb);
  assert.equal(verb.text, "`verbatim text`");
});

test("verbatim variable definition: ${name=`raw`}", () => {
  const html = renderPromptHTML("${tt=`[red|blue]`}");
  // The backticked content paints as hl-verbatim; bracket chars inside
  // are NOT separately coloured because they're inside the verbatim
  // run (rendered literally).
  assert.ok(html.includes('class="hl-verbatim">`[red|blue]`'));
});

test("comments render as one comment span", () => {
  const toks = tokens(renderPromptHTML("a <!-- note --> b"));
  const c = toks.find((t) => t.cls === "hl-comment");
  assert.ok(c);
  assert.equal(c.text, "<!-- note -->");
});

test("escaped chars dim in hl-escape", () => {
  const toks = tokens(renderPromptHTML(String.raw`\[a\|b\]`));
  const escs = toks.filter((t) => t.cls === "hl-escape");
  assert.deepEqual(escs.map((t) => t.text), ["\\[", "\\|", "\\]"]);
});

test("slash commands keep their existing colours", () => {
  const html = renderPromptHTML("/cfg 5");
  assert.ok(html.includes('class="hl-cmd">/cfg'));
  assert.ok(html.includes('class="hl-num">5'));
});

test("template syntax inside greedy /negprompt highlights brackets and plain text together", () => {
  const toks = tokens(renderPromptHTML("/negprompt [foo|bar]"));
  const negs = toks.filter((t) => t.cls === "hl-neg").map((t) => t.text);
  assert.deepEqual(negs, ["foo", "bar"]);
  const braces = toks.filter((t) => t.cls === "hl-brace").map((t) => t.text);
  assert.deepEqual(braces, ["[", "|", "]"]);
});

test("unclosed alternation (with pipe) is flagged in red", () => {
  // The parser will raise on this at expansion time. The highlighter
  // surfaces it via hl-unknown-cmd so the user sees something's off
  // before submitting.
  const html = renderPromptHTML("[red|blue");
  assert.ok(/class="hl-unknown-cmd">\[/.test(html));
});

test("bare `{` outside object-literal context renders as plain", () => {
  // Not a field block (no `<ident>=` after), and `{` isn't a Choice
  // opener anymore — should NOT be coloured hl-brace.
  const html = renderPromptHTML("a { stuff } b");
  assert.ok(!/class="hl-brace"/.test(html));
});
