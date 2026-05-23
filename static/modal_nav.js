// Pure neighbor lookup for the outputs preview modal.
//
// `outputs` is the list currently shown in the gallery — already filtered
// by the active tab (all/fav). So returning a neighbor from it automatically
// respects the user's filter; we don't need a separate "filter mode" arg.
//
// Returns the adjacent entry or null when at a boundary or when the
// current entry isn't in the list (e.g. it was removed by a cleanup
// while the modal was open).
(function (root) {
  "use strict";

  function neighborInOutputs(outputs, currentName, direction) {
    if (!Array.isArray(outputs) || outputs.length === 0) return null;
    if (direction !== 1 && direction !== -1) return null;
    const idx = outputs.findIndex((o) => o && o.name === currentName);
    if (idx < 0) return null;
    const next = idx + direction;
    if (next < 0 || next >= outputs.length) return null;
    return outputs[next];
  }

  root.neighborInOutputs = neighborInOutputs;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { neighborInOutputs };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
