const assert = require("node:assert/strict");
const test = require("node:test");

const {
  popupArrowSelection,
  promptArrowAction,
} = require("../static/prompt_navigation.js");

test("ArrowUp uses history only at absolute start", () => {
  assert.equal(promptArrowAction("first\nsecond", 0, 0, "ArrowUp"), "history-prev");
  assert.equal(promptArrowAction("first\nsecond", 3, 3, "ArrowUp"), "move-start");
  assert.equal(promptArrowAction("first\nsecond", 8, 8, "ArrowUp"), "native");
});

test("ArrowDown uses history only at absolute end", () => {
  const value = "first\nsecond";
  assert.equal(promptArrowAction(value, value.length, value.length, "ArrowDown"), "history-next");
  assert.equal(promptArrowAction(value, 8, 8, "ArrowDown"), "move-end");
  assert.equal(promptArrowAction(value, 3, 3, "ArrowDown"), "native");
});

test("selected text keeps native arrow behavior", () => {
  assert.equal(promptArrowAction("first\nsecond", 0, 5, "ArrowUp"), "native");
  assert.equal(promptArrowAction("first\nsecond", 6, 12, "ArrowDown"), "native");
});

test("popup arrows move selected suggestion with wraparound", () => {
  assert.equal(popupArrowSelection(0, 3, "ArrowDown"), 1);
  assert.equal(popupArrowSelection(2, 3, "ArrowDown"), 0);
  assert.equal(popupArrowSelection(0, 3, "ArrowUp"), 2);
  assert.equal(popupArrowSelection(1, 3, "ArrowLeft"), 1);
  assert.equal(popupArrowSelection(1, 0, "ArrowDown"), 1);
});
