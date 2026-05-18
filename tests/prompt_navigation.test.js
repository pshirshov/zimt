const assert = require("node:assert/strict");
const test = require("node:test");

const { promptArrowAction } = require("../static/prompt_navigation.js");

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
