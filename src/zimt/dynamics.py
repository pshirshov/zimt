"""Dynamic prompt syntax — JSON-like with two-pass verbatim expansion.

Syntax
------
``[a|b|c]``                   alternation: pick one option uniformly
``[2::a|1::b]``               weighted alternation: 2/3 a, 1/3 b
``[a [b|c]|d]``               nesting (alternations nest to any depth)
``[|a|b]``                    empty option allowed → "", "a", or "b"
``${name=value}``             scalar variable definition (silent)
``${name}``                   variable reference (renders bound value)
``${obj={k1=v1, k2=v2}}``     composite definition; binds ``obj.k1`` etc.
``${obj.k1}``                 access a composite field (nested OK: ``obj.a.b``)
```${name=`raw text`}```      verbatim definition — value stored as raw
                              text, re-parsed when referenced (see below)
``\\[ \\] \\{ \\} \\| \\$ \\` \\\\``  escape literals
``<!-- foo -->``              HTML-style comment, stripped before parsing

Alternation lookahead
---------------------
``[...]`` is parsed as alternation only when a top-level ``|`` appears
between the matched brackets. ``[word]`` stays literal text — that
matters because compel's negative-weighting syntax (``[word]-``) needs
to survive dynamics expansion unchanged. To get a literal ``|``-bearing
bracket pair (``[a|b]`` typed verbatim), escape with ``\\[`` ``\\]``.

Two-pass expansion
------------------
**Pass 1 — verbatim preprocessor.** A text-level scan finds every
``${name=`...`}`` definition, stores the raw inner text in an env, and
removes the definition from the text. Every ``${name}`` reference is
then substituted with the env value IF that name was bound to a
verbatim. Non-verbatim ``${...}`` constructs (``${color=red|blue}``,
``${obj={...}}``, references to non-verbatim names) pass through pass 1
unchanged. Pass 1 does **not** consume any RNG — it never picks an
alternation.

**Pass 2 — full evaluator.** The pass-1 output is parsed and rendered
with the RNG. Each ``${name=...}`` definition is evaluated and bound;
``${name}`` references resolve from a fresh env; ``[...]`` alternations
pick using the RNG. Verbatim substitutions from pass 1 are now part of
the source text, so each reference site to a verbatim variable
RE-EVALUATES its inlined contents — meaning two references to
```${tt=`[shorts|skirt]`}``` produce two independent picks.

Both passes are deterministic per seed: the RNG state is fresh for
pass 2 (pass 1 never touches it).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*::")
# Variable / reference names — dots allowed for flat-dotted refs and
# composite-field access. First char must be a letter or underscore.
_VAR_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
# Field names inside object literals — same as var names but no dots
# (the dotted path is constructed at parse time from the surrounding
# composite definition's name).
_FIELD_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class DynamicsSyntaxError(ValueError):
    """User typed a malformed template (unclosed bracket, undefined ref, ...)."""


# ---------- AST nodes ----------

@dataclass(frozen=True)
class Literal:
    """A run of literal text."""
    text: str


@dataclass(frozen=True)
class Option:
    """One branch of an :class:`Alternation`. ``weight`` defaults to 1.0."""
    weight: float
    parts: tuple["Node", ...]


@dataclass(frozen=True)
class Alternation:
    """``[a|b]`` — picked uniformly (or by weights) at render time."""
    options: tuple[Option, ...]


@dataclass(frozen=True)
class VarRef:
    """``${name}`` — renders the value bound to ``name`` (env lookup)."""
    name: str


@dataclass(frozen=True)
class VarDef:
    """``${name=value}`` — silent; binds ``name`` to the rendered RHS."""
    name: str
    parts: tuple["Node", ...]


@dataclass(frozen=True)
class MultiVarDef:
    """``${obj={k=v, k=v}}`` — flat-dotted bindings from one block.

    Built at parse time by flattening composite literals into individual
    ``(name.field, parts)`` pairs. Renders silently, binding each pair
    against the shared env in source order.
    """
    bindings: tuple[tuple[str, tuple["Node", ...]], ...]


Node = Literal | Alternation | VarRef | VarDef | MultiVarDef


# ---------- public API ----------

def has_dynamics(text: str) -> bool:
    """Cheap precheck: does this text contain any of the special syntax?

    Used to skip both passes for plain prompts. Conservative — false
    positives (e.g. a stray ``[`` that doesn't pair up) just cause the
    parser to run and find no alternations. Backslashes count because
    even a plain text prompt with ``\\X`` escape sequences needs the
    parser pass to materialise them.
    """
    if "${" in text or "<!--" in text or "`" in text or "\\" in text:
        return True
    if "[" in text and "|" in text:
        return True
    return False


def validate(text: str) -> None:
    """Parse ``text`` and raise :class:`DynamicsSyntaxError` if malformed.

    Sacrificial render with a fixed seed proves syntactic validity
    without exposing AST internals. Used at submission time to
    surface a template error once rather than once per ``/many`` job.
    """
    if not has_dynamics(text):
        return
    expand(text, random.Random(0))


def expand(text: str, rng: random.Random) -> str:
    """Two-pass expand: verbatim preprocessor → full evaluator.

    The function is pure — same ``text`` + same RNG state yields the
    same output. Pass 1 doesn't touch the RNG; pass 2 consumes from it
    deterministically. The env is fresh per call; nothing leaks between
    successive ``expand()`` invocations.
    """
    stripped = _COMMENT_RE.sub("", text)
    pass1 = _pass1_verbatim(stripped)
    if pass1 == stripped and not has_dynamics(pass1):
        # Nothing to evaluate — pass 1 produced no changes and there's
        # no remaining dynamics syntax. Common-case fast path.
        return pass1
    ast, end = _parse_top(pass1, 0)
    if end != len(pass1):
        # Defensive — _parse_top with empty boundary should consume
        # everything. A residual position means an internal bug.
        raise DynamicsSyntaxError(
            f"internal: residual input at {end}/{len(pass1)}"
        )
    return _render(ast, rng, env={})


# ---------- pass 1: verbatim preprocessor ----------

def _pass1_verbatim(text: str, env: dict[str, str] | None = None) -> str:
    """Substitute every ``${name=`raw`}`` def and every ``${name}`` ref
    that resolves to a verbatim binding. Everything else passes through
    unchanged so pass 2's full parser sees it as the user typed.

    Recurses into non-verbatim ``${...}`` bodies (composite defs, scalar
    defs whose RHS is a regular expression, etc.) so verbatim references
    nested inside — e.g. ``${boyA={type=${clothesType}}}`` — are
    substituted before pass 2 attempts to render them. The shared
    ``env`` argument carries verbatim bindings across the recursion.

    No alternation picking happens here — pass 1 is RNG-free and
    semantically a textual transform.
    """
    if env is None:
        env = {}
    out: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        c = text[i]
        if c == "\\":
            # Preserve escape so pass 2 sees it.
            if i + 1 < n:
                out.append(text[i:i + 2])
                i += 2
            else:
                out.append("\\")
                i += 1
            continue
        if c == "`":
            # Standalone backtick (outside ${name=...}) — strip the
            # delimiters and pass the content through. Unclosed: treat
            # the rest as raw and drop the leading backtick (lenient,
            # avoids a hard error on a typo).
            end = _find_verbatim_end(text, i + 1)
            if end < 0:
                out.append(text[i + 1:])
                break
            out.append(text[i + 1:end])
            i = end + 1
            continue
        if c == "$" and i + 1 < n and text[i + 1] == "{":
            close = _find_var_close(text, i)
            if close < 0:
                # Unclosed ${...} — let pass 2 report a useful error.
                out.append(text[i:])
                break
            body = text[i + 2:close]
            replacement, consumed = _pass1_handle_var_body(body, env)
            if consumed:
                out.append(replacement)
                i = close + 1
                continue
            # Non-verbatim def or unbindable ref. Recurse on the body
            # so verbatim refs nested inside (composite field values,
            # scalar RHS, etc.) get substituted using the same env.
            # Re-wrap the transformed body in ${...} for pass 2.
            transformed_body = _pass1_verbatim(body, env)
            out.append("${")
            out.append(transformed_body)
            out.append("}")
            i = close + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _find_verbatim_end(text: str, start: int) -> int:
    """Find the next un-escaped backtick at or after ``start``. -1 if none."""
    i = start
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            i += 2
            continue
        if c == "`":
            return i
        i += 1
    return -1


def _find_var_close(text: str, start: int) -> int:
    """Given ``start`` pointing at ``$`` of a ``${...}`` block, return the
    position of the matching ``}``. Accounts for nested ``${...}``,
    object-literal ``{...}``, backtick-quoted verbatim, and escapes.
    Returns -1 if unmatched.
    """
    i = start + 2  # past `${`
    depth = 0  # nested `{` depth WITHIN the outer ${...}
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            i += 2
            continue
        if c == "`":
            end = _find_verbatim_end(text, i + 1)
            if end < 0:
                return -1
            i = end + 1
            continue
        if c == "$" and i + 1 < len(text) and text[i + 1] == "{":
            nested = _find_var_close(text, i)
            if nested < 0:
                return -1
            i = nested + 1
            continue
        if c == "{":
            depth += 1
            i += 1
        elif c == "}":
            if depth == 0:
                return i
            depth -= 1
            i += 1
        else:
            i += 1
    return -1


_PASS1_VERBATIM_DEF = re.compile(
    r"\s*([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*`",
)
_PASS1_REF_ONLY = re.compile(
    r"\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\Z",
)


def _pass1_handle_var_body(
    body: str, env: dict[str, str],
) -> tuple[str, bool]:
    """Decide what to do with the contents of a ``${...}`` block.

    Returns ``(replacement, consumed)``:
      * verbatim def — register env, return ("", True) so the block is
        removed from the text.
      * ref to a name with a verbatim binding — return (raw_value, True).
      * anything else (non-verbatim def, ref to undefined / non-verbatim,
        composite def, scalar def with regular RHS) — return ("", False)
        so the caller keeps the block intact for pass 2.
    """
    m = _PASS1_VERBATIM_DEF.match(body)
    if m is not None:
        # Looks like `name = ` followed by an opening backtick. The body
        # MUST then be: name=`...raw...` with optional trailing ws.
        name = m.group(1)
        verb_start = m.end()  # just past the opening backtick
        # Find the closing backtick. The content runs until the last
        # un-escaped backtick; trailing chars after it would mean the
        # body wasn't a pure verbatim def, so we hand it back to pass 2.
        end = _find_verbatim_end(body, verb_start)
        if end < 0:
            return "", False
        trailing = body[end + 1:].strip()
        if trailing != "":
            return "", False
        env[name] = body[verb_start:end]
        return "", True
    m = _PASS1_REF_ONLY.match(body)
    if m is not None:
        name = m.group(1)
        if name in env:
            return env[name], True
    return "", False


# ---------- pass 2: full parser ----------

def _parse_top(s: str, i: int, until_chars: str = "") -> tuple[tuple[Node, ...], int]:
    """Parse a flat sequence of literals + alternations + var blocks.

    ``until_chars`` is the set of characters that end the sequence
    (``"|]"`` inside an alternation option; ``",}"`` inside a scalar
    field value; empty at top level).
    """
    parts: list[Node] = []
    buf: list[str] = []
    n = len(s)
    while i < n:
        c = s[i]
        if c in until_chars:
            break
        if c == "\\":
            if i + 1 < n:
                buf.append(s[i + 1])
                i += 2
            else:
                buf.append("\\")
                i += 1
            continue
        if c == "`":
            # Pass-1 already stripped verbatim defs / refs. A remaining
            # backtick is just a literal delimiter — strip it and emit
            # the content. Same lenient policy on unclosed as pass 1:
            # drop the leading backtick, treat the rest as literal text.
            end = _find_verbatim_end(s, i + 1)
            if end < 0:
                buf.append(s[i + 1:])
                i = n
                continue
            buf.append(s[i + 1:end])
            i = end + 1
            continue
        if c == "$" and i + 1 < n and s[i + 1] == "{":
            if buf:
                parts.append(Literal("".join(buf)))
                buf.clear()
            node, i = _parse_var(s, i + 2, open_pos=i)
            parts.append(node)
            continue
        if c == "[":
            # Alternation? Only when a top-level `|` exists between this
            # `[` and its matching `]`. Otherwise it's literal — compel
            # weighting like `[word]-` survives this parser intact.
            close_pos, has_pipe = _check_alternation(s, i)
            if close_pos > 0 and has_pipe:
                if buf:
                    parts.append(Literal("".join(buf)))
                    buf.clear()
                alt, i = _parse_alternation(s, i + 1, open_pos=i)
                parts.append(alt)
                continue
            if has_pipe and close_pos < 0:
                # User typed something like `[red|blue` — the pipe shows
                # intent to write an alternation, the missing `]` is a
                # typo. Silent fallthrough would hide the mistake;
                # report it clearly.
                raise DynamicsSyntaxError(
                    f"unclosed '[' alternation at position {i}"
                )
            buf.append("[")
            i += 1
            continue
        buf.append(c)
        i += 1
    if buf:
        parts.append(Literal("".join(buf)))
    return tuple(parts), i


def _check_alternation(s: str, start: int) -> tuple[int, bool]:
    """Look ahead from ``start`` (the ``[`` char) for the matching ``]``.

    Returns ``(close_pos, has_pipe)`` — ``close_pos == -1`` for
    unmatched, ``has_pipe`` indicates whether at least one top-level
    ``|`` exists between the brackets. Both checks happen in one scan
    so the parser can decide alternation-vs-literal cheaply.
    """
    n = len(s)
    if start >= n or s[start] != "[":
        return -1, False
    i = start + 1
    depth = 1
    has_pipe = False
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c == "`":
            end = _find_verbatim_end(s, i + 1)
            if end < 0:
                return -1, False
            i = end + 1
            continue
        if c == "$" and i + 1 < n and s[i + 1] == "{":
            close = _find_var_close(s, i)
            if close < 0:
                return -1, False
            i = close + 1
            continue
        if c == "[":
            depth += 1
            i += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i, has_pipe
            i += 1
        elif c == "|" and depth == 1:
            has_pipe = True
            i += 1
        else:
            i += 1
    # Unmatched — preserve the has_pipe signal so the caller can
    # distinguish "literal `[` typo'd `]`" from "user clearly meant
    # alternation but forgot the closing bracket".
    return -1, has_pipe


def _parse_alternation(s: str, i: int, *, open_pos: int) -> tuple[Alternation, int]:
    """Parse the body of a ``[...]`` starting just past the opening bracket."""
    options: list[Option] = []
    while True:
        weight, i = _parse_weight(s, i)
        parts, i = _parse_top(s, i, until_chars="|]")
        options.append(Option(weight, parts))
        if i >= len(s):
            raise DynamicsSyntaxError(
                f"unclosed '[' starting at position {open_pos}"
            )
        if s[i] == "]":
            return Alternation(tuple(options)), i + 1
        i += 1  # consume '|'


def _parse_weight(s: str, i: int) -> tuple[float, int]:
    """Look for a ``<number>::`` prefix at position ``i``. Default 1.0."""
    m = _WEIGHT_RE.match(s, i)
    if m is None:
        return 1.0, i
    return float(m.group(1)), m.end()


def _parse_var(s: str, i: int, *, open_pos: int) -> tuple[Node, int]:
    """Parse the body of a ``${...}`` starting just past the ``${``.

    Three shapes:
      * ``${name}``                 — reference. Returns :class:`VarRef`.
      * ``${name=<expr>}``         — scalar def. Returns :class:`VarDef`.
      * ``${name={k=v, k=v}}``     — composite def. Returns
                                      :class:`MultiVarDef` with the
                                      object literal flattened to flat
                                      dotted bindings.
    """
    m = _VAR_NAME_RE.match(s, i)
    if m is None:
        raise DynamicsSyntaxError(
            f"expected variable name after '${{' at position {open_pos}"
        )
    name = m.group(0)
    i = m.end()
    if i >= len(s):
        raise DynamicsSyntaxError(
            f"unclosed '${{' at position {open_pos}"
        )
    if s[i] == "}":
        return VarRef(name), i + 1
    if s[i] != "=":
        raise DynamicsSyntaxError(
            f"expected '=' or '}}' after variable {name!r} at position {open_pos}"
        )
    i += 1  # consume '='
    if _is_object_literal(s, i):
        bindings, i = _parse_object_literal(s, i, prefix=name, open_pos=open_pos)
        i = _skip_ws(s, i)
        if i >= len(s) or s[i] != "}":
            raise DynamicsSyntaxError(
                f"unclosed '${{' at position {open_pos}"
            )
        return MultiVarDef(tuple(bindings)), i + 1
    parts, i = _parse_top(s, i, until_chars="}")
    if i >= len(s) or s[i] != "}":
        raise DynamicsSyntaxError(
            f"unclosed '${{' at position {open_pos}"
        )
    return VarDef(name, parts), i + 1


def _is_object_literal(s: str, i: int) -> bool:
    """Peek: does ``s[i:]`` start with ``{<ident>=`` (an object literal)?"""
    j = _skip_ws(s, i)
    if j >= len(s) or s[j] != "{":
        return False
    j = _skip_ws(s, j + 1)
    m = _FIELD_NAME_RE.match(s, j)
    if m is None:
        return False
    j = _skip_ws(s, m.end())
    return j < len(s) and s[j] == "="


def _parse_object_literal(
    s: str, i: int, *, prefix: str, open_pos: int,
) -> tuple[list[tuple[str, tuple[Node, ...]]], int]:
    """Parse ``{k1=v1, k2=v2, ...}`` and return flat bindings.

    Each value is parsed as a scalar RHS (recursing into nested object
    literals if the value itself looks like one). Field-paths are
    joined with ``.`` so a binding's full name reflects its position in
    the source tree.
    """
    i = _skip_ws(s, i)
    if i >= len(s) or s[i] != "{":
        raise DynamicsSyntaxError(
            f"expected '{{' for object literal at position {i}"
        )
    i += 1  # consume '{'
    bindings: list[tuple[str, tuple[Node, ...]]] = []
    while True:
        i = _skip_ws(s, i)
        if i >= len(s):
            raise DynamicsSyntaxError(
                f"unclosed object literal opened at {open_pos}"
            )
        if s[i] == "}":
            return bindings, i + 1
        m = _FIELD_NAME_RE.match(s, i)
        if m is None:
            raise DynamicsSyntaxError(
                f"expected field name at position {i}"
            )
        field_name = m.group(0)
        i = _skip_ws(s, m.end())
        if i >= len(s) or s[i] != "=":
            raise DynamicsSyntaxError(
                f"expected '=' after field {field_name!r} at position {i}"
            )
        i += 1  # consume '='
        full_name = f"{prefix}.{field_name}"
        if _is_object_literal(s, i):
            sub_bindings, i = _parse_object_literal(
                s, i, prefix=full_name, open_pos=open_pos,
            )
            bindings.extend(sub_bindings)
        else:
            value_parts, i = _parse_top(s, i, until_chars=",}")
            bindings.append((full_name, value_parts))
        i = _skip_ws(s, i)
        if i >= len(s):
            raise DynamicsSyntaxError(
                f"unclosed object literal opened at {open_pos}"
            )
        if s[i] == ",":
            i += 1
            continue
        if s[i] == "}":
            return bindings, i + 1
        raise DynamicsSyntaxError(
            f"expected ',' or '}}' in object literal at position {i}"
        )


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\n\r":
        i += 1
    return i


# ---------- pass 2: render ----------

def _render(
    parts: tuple[Node, ...],
    rng: random.Random,
    env: dict[str, str],
) -> str:
    """Render an AST against a seeded RNG and a mutable env.

    ``env`` is mutated by :class:`VarDef` / :class:`MultiVarDef` nodes —
    intentional, that's how variables flow left-to-right within a
    single expansion. The dict is fresh per top-level ``expand()`` call.
    """
    out: list[str] = []
    for p in parts:
        if isinstance(p, Literal):
            out.append(p.text)
        elif isinstance(p, Alternation):
            if not p.options:
                continue
            weights = [o.weight for o in p.options]
            chosen = rng.choices(p.options, weights=weights, k=1)[0]
            out.append(_render(chosen.parts, rng, env))
        elif isinstance(p, VarDef):
            env[p.name] = _render(p.parts, rng, env)
        elif isinstance(p, MultiVarDef):
            for binding_name, binding_parts in p.bindings:
                env[binding_name] = _render(binding_parts, rng, env)
        elif isinstance(p, VarRef):
            if p.name not in env:
                raise DynamicsSyntaxError(
                    f"undefined variable {p.name!r} — either it was never "
                    f"defined or it was defined only inside an alternation "
                    f"branch that this seed didn't pick"
                )
            out.append(env[p.name])
    return "".join(out)
