const assert = require("node:assert/strict");
const test = require("node:test");

const { COMMANDS, completionItems, tokenAtCursor } = require("../static/completion.js");

const appState = {
  loaded: true,
  model: "pony-v6-xl",
  models: [
    {
      name: "pony-v6-xl",
      description: "Pony Diffusion V6 XL",
      samplers: ["euler", "euler-a", "dpmpp-2m"],
      resolutions: [{ w: 1024, h: 1024, label: "1:1 square" }],
    },
  ],
};

test("/sampler is present in slash command completion", () => {
  assert.ok(COMMANDS.includes("/sampler"));
  const token = tokenAtCursor("/sam", 4);
  const labels = completionItems("/sam", token.start, token.text, appState)
    .map((item) => item.label);
  assert.deepEqual(labels, ["/sampler"]);
});

test("/sampler argument completion uses loaded model samplers", () => {
  const value = "/sampler eu";
  const token = tokenAtCursor(value, value.length);
  const labels = completionItems(value, token.start, token.text, appState)
    .map((item) => item.label);
  assert.deepEqual(labels, ["euler", "euler-a"]);
});
