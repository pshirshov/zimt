"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const {
  tokenAtCursor, completionItems,
  unclosedVarOpenIndex, definedVarNames,
} = require("../static/completion.js");

test("unclosedVarOpenIndex finds the most recent unclosed ${", () => {
  // Standard case: ${ open, no = yet.
  assert.equal(unclosedVarOpenIndex("hello ${c"), 6);
});

test("unclosedVarOpenIndex returns -1 once = is typed", () => {
  // Past the name — we're in the RHS now, completions should be plain.
  assert.equal(unclosedVarOpenIndex("hello ${color="), -1);
});

test("unclosedVarOpenIndex returns -1 when } closes the var", () => {
  assert.equal(unclosedVarOpenIndex("hello ${color}"), -1);
});

test("unclosedVarOpenIndex returns -1 with no ${", () => {
  assert.equal(unclosedVarOpenIndex("hello world"), -1);
});

test("definedVarNames extracts unique names from ${name=...} occurrences", () => {
  const names = definedVarNames("${a=red}${b=blue}");
  assert.deepEqual(names, ["a", "b"]);
});

test("definedVarNames de-duplicates names that appear twice", () => {
  // Re-definition is allowed at runtime; the completer should not
  // suggest the same name twice.
  const names = definedVarNames("${color=red}${color=blue}");
  assert.deepEqual(names, ["color"]);
});

test("definedVarNames returns empty list when nothing is defined", () => {
  assert.deepEqual(definedVarNames("a quiet garden"), []);
});

test("definedVarNames ignores plain references (no =)", () => {
  // `${color}` is a reference, not a definition — must not appear.
  assert.deepEqual(definedVarNames("${color} something else"), []);
});

test("variable suggestions are returned when cursor is inside ${", () => {
  // Setup: a definition earlier in the line, cursor partway through a
  // reference. updateSuggest in app.js is responsible for re-aiming
  // tokStart at the `${`; here we simulate that by passing the narrowed
  // token directly.
  const value = "${color=red|blue}${c";
  const tokStart = 17;             // position of `$` in second `${`
  const tokText = "${c";
  const items = completionItems(value, tokStart, tokText, {});
  assert.deepEqual(items.map((it) => it.label), ["${color}"]);
});

test("variable suggestion includes a closing brace so insertion is complete", () => {
  // The label, when used to replace tokText, must produce a valid
  // closed `${name}` — otherwise the user is left with an unclosed
  // var and the parser would error.
  const value = "${color=red|blue}${";
  const items = completionItems(value, 17, "${", {});
  assert.deepEqual(items.map((it) => it.label), ["${color}"]);
});

test("variable suggestions are filtered by the typed prefix", () => {
  const value = "${apple=red}${banana=yellow}${a";
  const items = completionItems(value, 28, "${a", {});
  assert.deepEqual(items.map((it) => it.label), ["${apple}"]);
});

test("forward references are not suggested", () => {
  // The user is typing `${c` BEFORE any definition; nothing to offer.
  const value = "${c";
  const items = completionItems(value, 0, "${c", {});
  assert.deepEqual(items, []);
});

test("definedVarNames extracts composite field paths", () => {
  // User's exact syntax: `${p={hair=long|short}, {clothes=red|blue}}`
  // should yield ["p.hair", "p.clothes"] — both flat-dotted leaves.
  const names = definedVarNames(
    "${p={hair=long|short}, {clothes=red|blue}}"
  );
  assert.deepEqual(names, ["p.hair", "p.clothes"]);
});

test("definedVarNames extracts nested composite field paths", () => {
  // Nested object literal: outfit.shirt and outfit.pants.
  const names = definedVarNames(
    "${p={outfit={shirt=red}, {pants=blue}}, {hair=long}}"
  );
  assert.deepEqual(names, ["p.outfit.shirt", "p.outfit.pants", "p.hair"]);
});

test("definedVarNames flat-dotted def is accepted directly", () => {
  // `${a.b=...}` is equivalent to `${a={b=...}}` — the regex captures
  // the dotted form too.
  const names = definedVarNames("${a.b=red}${a.c=blue}");
  assert.deepEqual(names, ["a.b", "a.c"]);
});

test("dotted reference completion suggests composite fields", () => {
  // User types `${personA.h<TAB>` after defining the composite.
  const value =
    "${personA={hair=long|short}, {clothes=red|blue}}" +
    "${personA.h";
  // tokStart was narrowed by updateSuggest to point at the second `$`.
  const tokStart = "${personA={hair=long|short}, {clothes=red|blue}}".length;
  const tokText = "${personA.h";
  const items = completionItems(value, tokStart, tokText, {});
  assert.deepEqual(
    items.map((it) => it.label),
    ["${personA.hair}"],
  );
});

test("dotted reference completion without typed field suggests all fields", () => {
  // `${personA.<TAB>` with nothing after the dot — both fields offered.
  const value =
    "${personA={hair=long|short}, {clothes=red|blue}}" +
    "${personA.";
  const tokStart = "${personA={hair=long|short}, {clothes=red|blue}}".length;
  const items = completionItems(value, tokStart, "${personA.", {});
  assert.deepEqual(
    items.map((it) => it.label),
    ["${personA.hair}", "${personA.clothes}"],
  );
});

test("variable suggestions exclude definitions only inside future text", () => {
  // Definition appears AFTER the cursor — must not appear in
  // suggestions because forward refs raise at expansion time.
  const value = "${c${later=blue}";
  // tokStart is at the FIRST `${`, tokText covers up to a hypothetical
  // cursor right after the `c`. The trailing `${later=...}` is part of
  // value but comes after tokStart, so it shouldn't surface.
  const items = completionItems(value, 0, "${c", {});
  assert.deepEqual(items, []);
});
