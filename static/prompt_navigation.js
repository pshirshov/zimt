(function (root) {
  "use strict";

  function promptArrowAction(value, selectionStart, selectionEnd, key) {
    if (selectionStart !== selectionEnd) return "native";

    if (key === "ArrowUp") {
      if (selectionStart === 0) return "history-prev";
      const lineStart = value.lastIndexOf("\n", selectionStart - 1) + 1;
      return lineStart === 0 ? "move-start" : "native";
    }

    if (key === "ArrowDown") {
      if (selectionStart === value.length) return "history-next";
      const lineEnd = value.indexOf("\n", selectionStart);
      return lineEnd === -1 ? "move-end" : "native";
    }

    return "native";
  }

  root.promptArrowAction = promptArrowAction;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { promptArrowAction };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
