// Helpers used by app.js autoResize().
//
// The full autoResize() is DOM glue (read scrollHeight, write style on
// three elements), but the cap math has enough edge cases — NaN from a
// missing/invalid CSS max-height rule, zero or negative max, content
// shorter than max — that it's worth pinning down with unit tests.
(function (root) {
  "use strict";

  // Returns the height to apply to the textarea wrap given the measured
  // content height and an optional CSS max-height (in pixels). When
  // max-height is not a positive finite number — e.g. the CSS rule was
  // removed or `getComputedStyle().maxHeight` returned `"none"` — the
  // content height passes through untouched.
  function clampToMaxHeight(contentHeight, maxHeight) {
    if (!isFinite(maxHeight) || maxHeight <= 0) return contentHeight;
    return Math.min(contentHeight, maxHeight);
  }

  root.clampToMaxHeight = clampToMaxHeight;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { clampToMaxHeight };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
