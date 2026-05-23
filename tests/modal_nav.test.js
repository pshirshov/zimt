"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { neighborInOutputs } = require("../static/modal_nav.js");

const E = (name, fav = false) => ({ name, fav });

test("returns null when outputs is empty", () => {
  assert.equal(neighborInOutputs([], "a.png", +1), null);
  assert.equal(neighborInOutputs([], "a.png", -1), null);
});

test("returns null when the current entry is not in outputs", () => {
  const xs = [E("a.png"), E("b.png")];
  assert.equal(neighborInOutputs(xs, "c.png", +1), null);
});

test("returns next entry when not at the end", () => {
  const xs = [E("a.png"), E("b.png"), E("c.png")];
  assert.equal(neighborInOutputs(xs, "a.png", +1).name, "b.png");
  assert.equal(neighborInOutputs(xs, "b.png", +1).name, "c.png");
});

test("returns previous entry when not at the start", () => {
  const xs = [E("a.png"), E("b.png"), E("c.png")];
  assert.equal(neighborInOutputs(xs, "c.png", -1).name, "b.png");
  assert.equal(neighborInOutputs(xs, "b.png", -1).name, "a.png");
});

test("returns null at start when navigating backward", () => {
  const xs = [E("a.png"), E("b.png")];
  assert.equal(neighborInOutputs(xs, "a.png", -1), null);
});

test("returns null at end when navigating forward", () => {
  const xs = [E("a.png"), E("b.png")];
  assert.equal(neighborInOutputs(xs, "b.png", +1), null);
});

test("rejects invalid direction values", () => {
  const xs = [E("a.png"), E("b.png")];
  assert.equal(neighborInOutputs(xs, "a.png", 0), null);
  assert.equal(neighborInOutputs(xs, "a.png", +2), null);
  assert.equal(neighborInOutputs(xs, "a.png", "next"), null);
});

test("filter-awareness is intrinsic: a fav-only outputs list "
     + "navigates only within favorites", () => {
  // The caller is expected to pass the already-filtered outputs list —
  // which the app does, since `outputs` always reflects the active tab.
  // Here we exercise that contract: given only favorites, prev/next
  // visit only favorites.
  const favs = [E("img-2.png", true), E("img-5.png", true), E("img-9.png", true)];
  assert.equal(neighborInOutputs(favs, "img-2.png", +1).name, "img-5.png");
  assert.equal(neighborInOutputs(favs, "img-5.png", +1).name, "img-9.png");
  assert.equal(neighborInOutputs(favs, "img-9.png", +1), null);
});

test("tolerates null/undefined entries defensively", () => {
  const xs = [null, E("a.png"), undefined, E("b.png")];
  // Sparse entries shouldn't crash the scan; finding `a.png` returns
  // its real index, and stepping past it reaches the next real entry
  // even though there's a gap.
  assert.equal(neighborInOutputs(xs, "a.png", +1), undefined);
  // Skipping the undefined and reaching b.png requires the caller to
  // not leave gaps; we don't try to be smarter than the data. The
  // contract is: "neighbor at index ± 1".
  assert.equal(neighborInOutputs(xs, "b.png", -1), undefined);
});
