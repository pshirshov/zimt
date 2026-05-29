(function (root) {
  "use strict";

  // Line-comment toggle for the dynamics DSL. The only comment form the
  // parser understands is the HTML-style `<!-- ... -->` block (stripped by
  // `_COMMENT_RE` before parsing), so a "line comment" wraps each line's
  // content in those markers, preserving leading indentation.
  const OPEN = "<!-- ";
  const CLOSE = " -->";
  // Matches an already-commented line: optional indent, the markers, and
  // the content between them. A single optional space after `<!--` and
  // before `-->` is absorbed so round-tripping is lossless.
  const COMMENTED_RE = /^(\s*)<!--\s?([\s\S]*?)\s?-->(\s*)$/;
  const BLANK_RE = /^\s*$/;

  function isCommented(line) {
    return COMMENTED_RE.test(line);
  }

  function commentLine(line) {
    if (BLANK_RE.test(line)) return line; // leave blank lines untouched
    const indent = line.match(/^\s*/)[0];
    return indent + OPEN + line.slice(indent.length) + CLOSE;
  }

  function uncommentLine(line) {
    const m = line.match(COMMENTED_RE);
    if (!m) return line;
    return m[1] + m[2] + m[3];
  }

  // Returns the [start, end) offsets of the full lines spanned by a
  // selection. A selection that ends exactly at a line boundary (right
  // after a newline) does not pull in the following line — matching the
  // behaviour of common code editors.
  function spannedLineBounds(value, selStart, selEnd) {
    const firstLineStart = value.lastIndexOf("\n", selStart - 1) + 1;
    let end = selEnd;
    if (selEnd > selStart && selEnd > firstLineStart && value[selEnd - 1] === "\n") {
      end = selEnd - 1;
    }
    let lastLineEnd = value.indexOf("\n", end);
    if (lastLineEnd === -1) lastLineEnd = value.length;
    return { firstLineStart, lastLineEnd };
  }

  // Toggle line comments over the selection (or the cursor's line when the
  // selection is empty). If every non-blank spanned line is already
  // commented, all are uncommented; otherwise all non-blank lines are
  // commented. Returns the new value plus a selection covering the
  // rewritten block.
  function toggleComment(value, selStart, selEnd) {
    const { firstLineStart, lastLineEnd } = spannedLineBounds(value, selStart, selEnd);
    const block = value.slice(firstLineStart, lastLineEnd);
    const lines = block.split("\n");
    const meaningful = lines.filter((l) => !BLANK_RE.test(l));
    const allCommented = meaningful.length > 0 && meaningful.every(isCommented);
    const newLines = lines.map(allCommented ? uncommentLine : commentLine);
    const newBlock = newLines.join("\n");
    return {
      value: value.slice(0, firstLineStart) + newBlock + value.slice(lastLineEnd),
      selectionStart: firstLineStart,
      selectionEnd: firstLineStart + newBlock.length,
    };
  }

  root.toggleComment = toggleComment;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { toggleComment };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
