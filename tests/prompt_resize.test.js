"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { clampToMaxHeight } = require("../static/prompt_resize.js");

test("returns content height when under the cap", () => {
  assert.equal(clampToMaxHeight(100, 200), 100);
});

test("clamps to the cap when content exceeds it", () => {
  // The original bug: a 1000px prompt grew the wrap past max-height,
  // letting the highlight underlay extend below the visible editor.
  assert.equal(clampToMaxHeight(1000, 224), 224);
});

test("returns content unchanged when max is NaN (missing CSS rule)", () => {
  // `getComputedStyle().maxHeight` returns "none" for no cap; parseFloat
  // turns that into NaN. We must not collapse the editor to 0 in that
  // case — pass content through.
  assert.equal(clampToMaxHeight(150, NaN), 150);
});

test("returns content unchanged when max is zero or negative", () => {
  // Defensive: a misconfigured CSS rule shouldn't make the editor
  // disappear.
  assert.equal(clampToMaxHeight(150, 0), 150);
  assert.equal(clampToMaxHeight(150, -10), 150);
});

test("returns content unchanged when max is Infinity", () => {
  assert.equal(clampToMaxHeight(150, Infinity), 150);
});

test("equal content and cap returns the cap", () => {
  assert.equal(clampToMaxHeight(224, 224), 224);
});
