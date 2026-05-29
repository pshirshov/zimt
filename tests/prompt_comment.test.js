const assert = require("node:assert/strict");
const test = require("node:test");

const { toggleComment } = require("../static/prompt_comment.js");

test("comments a single line with no selection", () => {
  const v = "hello world";
  const r = toggleComment(v, 3, 3);
  assert.equal(r.value, "<!-- hello world -->");
  assert.equal(r.selectionStart, 0);
  assert.equal(r.selectionEnd, r.value.length);
});

test("uncomments a commented line back to the original", () => {
  const v = "hello world";
  const commented = toggleComment(v, 3, 3).value;
  const r = toggleComment(commented, 0, 0);
  assert.equal(r.value, "hello world");
});

test("toggle is lossless round-trip", () => {
  const v = "${hair=[red|blonde]}";
  const once = toggleComment(v, 0, 0).value;
  const twice = toggleComment(once, 0, 0).value;
  assert.equal(twice, v);
});

test("preserves leading indentation", () => {
  const v = "    indented";
  const r = toggleComment(v, 0, 0);
  assert.equal(r.value, "    <!-- indented -->");
  assert.equal(toggleComment(r.value, 0, 0).value, v);
});

test("comments every selected line", () => {
  const v = "a\nb\nc";
  const r = toggleComment(v, 0, v.length);
  assert.equal(r.value, "<!-- a -->\n<!-- b -->\n<!-- c -->");
  assert.equal(r.selectionStart, 0);
  assert.equal(r.selectionEnd, r.value.length);
});

test("uncomments only when every non-blank line is already commented", () => {
  const v = "<!-- a -->\nb";
  // Mixed state -> comment all (b becomes commented, a gets double-wrapped).
  const r = toggleComment(v, 0, v.length);
  assert.equal(r.value, "<!-- <!-- a --> -->\n<!-- b -->");
});

test("fully-commented block uncomments", () => {
  const v = "<!-- a -->\n<!-- b -->";
  const r = toggleComment(v, 0, v.length);
  assert.equal(r.value, "a\nb");
});

test("leaves blank lines untouched when commenting", () => {
  const v = "a\n\nb";
  const r = toggleComment(v, 0, v.length);
  assert.equal(r.value, "<!-- a -->\n\n<!-- b -->");
});

test("blank lines do not block uncommenting", () => {
  const v = "<!-- a -->\n\n<!-- b -->";
  const r = toggleComment(v, 0, v.length);
  assert.equal(r.value, "a\n\nb");
});

test("selection ending at a line boundary excludes the next line", () => {
  const v = "a\nb\nc";
  // Select "a\n" only (offsets 0..2). The trailing newline lands on line 2's
  // start, which must NOT be commented.
  const r = toggleComment(v, 0, 2);
  assert.equal(r.value, "<!-- a -->\nb\nc");
});

test("only-whitespace selection is a no-op (no meaningful lines)", () => {
  const v = "   ";
  const r = toggleComment(v, 0, v.length);
  assert.equal(r.value, "   ");
});
