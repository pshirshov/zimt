(function (root) {
  "use strict";

  function buildRestorePromptLine(metadata, options = {}) {
    const includeSeed = options.includeSeed !== false;
    const meta = metadata || {};
    const parts = [];
    if (meta.model) parts.push(`/model ${meta.model}`);
    if (meta.cfg) parts.push(`/cfg ${meta.cfg}`);
    if (meta.steps) parts.push(`/steps ${meta.steps}`);
    if (meta.sampler) parts.push(`/sampler ${meta.sampler}`);
    if (meta.clip_skip && Number(meta.clip_skip) > 0) {
      parts.push(`/clip_skip ${meta.clip_skip}`);
    }
    if (meta.width && meta.height) parts.push(`/size ${meta.width} ${meta.height}`);
    if (meta.negative_prompt != null) parts.push(`/negprompt ${meta.negative_prompt}`);
    if (includeSeed && meta.seed) parts.push(`/seed ${meta.seed}`);
    if (meta.loras) parts.push(`/lora ${meta.loras.replace(/,/g, " ")}`);

    // Use the full composed prompt so restore does not apply score tags twice
    // or drop them when replaying with /raw.
    const text = meta.prompt ?? meta.raw_prompt ?? "";
    if (text) parts.push("/raw", text);
    return parts.join(" ");
  }

  root.buildRestorePromptLine = buildRestorePromptLine;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { buildRestorePromptLine };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
