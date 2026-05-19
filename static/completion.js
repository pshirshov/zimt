(function (root) {
  "use strict";

  const COMMANDS = [
    "/help", "/?", "/raw", "/many", "/seed", "/negprompt", "/cfg", "/steps",
    "/size", "/res", "/sampler", "/clip_skip", "/model", "/lora", "/tokenize",
    "/quit", "/exit", "/q",
  ].sort();

  const COMMAND_HELP = {
    "/help": "show command list",
    "/?": "show command list",
    "/quit": "exit (REPL only)",
    "/exit": "exit (REPL only)",
    "/q": "exit (REPL only)",
    "/raw": "skip the model's auto-prefix",
    "/many": "generate N images with sequential seeds",
    "/seed": "pin seed for next gen",
    "/cfg": "guidance scale",
    "/steps": "num inference steps",
    "/size": "set W H (free-form)",
    "/res": "pick a model-preset resolution",
    "/sampler": "switch scheduler",
    "/clip_skip": "SDXL only; skip top N CLIP layers",
    "/model": "load a model",
    "/lora": "add/remove LoRAs (name | name:0.8 | -name | -)",
    "/tokenize": "per-encoder token analysis",
    "/negprompt": "set/clear negative prompt",
  };

  function tokenAtCursor(value, cursor) {
    let start = cursor;
    while (start > 0 && !/\s/.test(value[start - 1])) start -= 1;
    let end = cursor;
    while (end < value.length && !/\s/.test(value[end])) end += 1;
    return { start, end, text: value.slice(start, end) };
  }

  function tokensBefore(value, end) {
    return value.slice(0, end).split(/\s+/).filter(Boolean);
  }

  function modelForContext(tokens, appState) {
    const models = appState?.models ?? [];
    for (let i = tokens.length - 2; i >= 0; i -= 1) {
      if (tokens[i] !== "/model") continue;
      const name = tokens[i + 1];
      if (!name || name.startsWith("/")) continue;
      const explicit = models.find((m) => m.name === name);
      if (explicit) return explicit;
    }
    if (!appState?.model) return null;
    return models.find((m) => m.name === appState.model) ?? null;
  }

  function completionItems(value, tokStart, tokText, appState) {
    const before = tokensBefore(value, tokStart);
    const prev = before.length ? before[before.length - 1] : "";
    const lower = tokText.toLowerCase();

    if (prev === "/model") {
      const models = appState?.models ?? [];
      return models
        .filter((m) => m.name.toLowerCase().startsWith(lower))
        .map((m) => ({ label: m.name, desc: m.description }));
    }

    // /lora is greedy — every subsequent token until the next /cmd is a
    // LoRA-name argument. Walk back to find the most-recent /cmd; if it's
    // /lora, complete LoRA names filtered by compatibility with the
    // active base model (or the most recent /model arg in this line).
    const knownCmds = new Set(COMMANDS);
    let mostRecentCmd = null;
    for (let i = before.length - 1; i >= 0; i -= 1) {
      if (knownCmds.has(before[i])) { mostRecentCmd = before[i]; break; }
    }
    if (mostRecentCmd === "/lora") {
      const loras = appState?.loras ?? [];
      const base = modelForContext(before, appState);
      const baseTags = new Set(base?.compatibility_tags ?? []);
      let core = lower;
      if (core.startsWith("-")) core = core.slice(1);
      if (core.includes(":")) return [];   // user typing a weight
      return loras
        .filter((l) => l.name.toLowerCase().startsWith(core))
        .filter((l) => {
          if (!baseTags.size) return true;
          return (l.compatible_with || []).some((t) => baseTags.has(t));
        })
        .map((l) => ({
          label: l.name,
          desc: l.trigger_tags ? `trigger: ${l.trigger_tags}` : l.description,
        }));
    }

    if (prev === "/sampler") {
      const model = modelForContext(before, appState);
      return (model?.samplers ?? [])
        .filter((s) => s.toLowerCase().startsWith(lower))
        .map((s) => ({ label: s, desc: "" }));
    }

    if (prev === "/res") {
      const model = modelForContext(before, appState);
      const presets = model?.resolutions ?? [];
      const byArea = (a, b) => (b.w * b.h) - (a.w * a.h);
      const square = [...presets].sort(byArea).find((r) => r.w === r.h);
      const landscape = [...presets].sort(byArea).find((r) => r.w > r.h);
      const portrait = [...presets].sort(byArea).find((r) => r.w < r.h);
      const orientations = [
        square && { label: "square", desc: `largest 1:1 (${square.w}x${square.h})` },
        landscape && { label: "landscape", desc: `largest W>H (${landscape.w}x${landscape.h})` },
        portrait && { label: "portrait", desc: `largest W<H (${portrait.w}x${portrait.h})` },
      ].filter(Boolean);
      const explicit = presets.map((r) => ({ label: `${r.w}x${r.h}`, desc: r.label }));
      return [...orientations, ...explicit]
        .filter((it) => it.label.toLowerCase().startsWith(lower));
    }

    if (tokText.startsWith("/")) {
      return COMMANDS
        .filter((c) => c.toLowerCase().startsWith(lower))
        .map((c) => ({ label: c, desc: COMMAND_HELP[c] ?? "" }));
    }

    return [];
  }

  root.tokenAtCursor = tokenAtCursor;
  root.completionItems = completionItems;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { COMMANDS, COMMAND_HELP, tokenAtCursor, completionItems };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
