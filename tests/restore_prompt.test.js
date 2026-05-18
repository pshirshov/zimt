const assert = require("node:assert/strict");
const test = require("node:test");

const { buildRestorePromptLine } = require("../static/restore_prompt.js");

const metadata = {
  model: "pony-v6-xl",
  cfg: "7.0",
  steps: "25",
  sampler: "euler-a",
  clip_skip: "2",
  width: "1024",
  height: "1024",
  negative_prompt: "low quality",
  seed: "123",
  prompt: "score_9, portrait",
  raw_prompt: "portrait",
};

test("restore line includes seed by default", () => {
  assert.equal(
    buildRestorePromptLine(metadata),
    "/model pony-v6-xl /cfg 7.0 /steps 25 /sampler euler-a "
      + "/clip_skip 2 /size 1024 1024 /negprompt low quality "
      + "/seed 123 /raw score_9, portrait",
  );
});

test("restore line can omit seed while preserving prompt and settings", () => {
  assert.equal(
    buildRestorePromptLine(metadata, { includeSeed: false }),
    "/model pony-v6-xl /cfg 7.0 /steps 25 /sampler euler-a "
      + "/clip_skip 2 /size 1024 1024 /negprompt low quality "
      + "/raw score_9, portrait",
  );
});
