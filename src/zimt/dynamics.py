"""Dynamic prompt syntax — A1111/ComfyUI-style alternation and friends.

Syntax
------
``{a|b|c}``         alternation: pick one option uniformly
``{2::a|1::b}``     weighted alternation: 2/3 a, 1/3 b
``{a {b|c}|d}``     nesting: choices nest to any depth
``{|a|b}``          empty option allowed → "", "a", or "b"
``${name=expr}``    variable: pick once, bind to ``name``, render nothing
``${name}``         variable reference: render the bound value
``\\{ \\| \\}``     backslash-escape for literal braces / pipe / dollar
``<!-- foo -->``    HTML-style comments stripped before parsing

Variables bind silently — the definition site itself renders to "" so the
prompt reads cleanly when you don't want to display the chosen value
inline. To show the value where you define it, follow with a reference:
``${color=red|green}${color} hair``. References are scoped to one
``expand()`` call; nothing leaks between successive generations.

Inside ``${name=...}``, ``|`` is sugar for an implicit choice — so
``${color=red|green|blue}`` is equivalent to ``${color={red|green|blue}}``.
Forward references and references to a variable that wasn't defined
along the actually-chosen branch raise :class:`DynamicsSyntaxError` at
render time.

Determinism
-----------
Every expansion runs against a caller-supplied :class:`random.Random`,
so a fixed seed always yields the same expansion. The image generator
uses its own :class:`torch.Generator`; the two RNG streams are independent
so template determinism and image-sampling determinism don't interfere
with each other.

Wired into :func:`zimt.generate.generate` after the seed is picked and
before :func:`zimt.generate.compose_prompt` prepends any score-tag
prefix — so the model's score tags themselves never get
template-expanded by accident.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*::")
_VAR_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class DynamicsSyntaxError(ValueError):
    """User typed a malformed template (unclosed ``{``, etc.)."""


@dataclass(frozen=True)
class Literal:
    """A run of literal text with no choices in it."""
    text: str


@dataclass(frozen=True)
class Option:
    """One branch of a :class:`Choice`. ``weight`` defaults to 1.0."""
    weight: float
    parts: tuple["Node", ...]


@dataclass(frozen=True)
class Choice:
    """An alternation. ``options`` may be empty (renders to "")."""
    options: tuple[Option, ...]


@dataclass(frozen=True)
class VarDef:
    """``${name=expr}`` — bind ``name`` to the rendered ``parts``. Renders ""."""
    name: str
    parts: tuple["Node", ...]


@dataclass(frozen=True)
class VarRef:
    """``${name}`` — render the value previously bound to ``name``."""
    name: str


Node = Literal | Choice | VarDef | VarRef


def validate(text: str) -> None:
    """Parse ``text`` and raise :class:`DynamicsSyntaxError` if malformed.

    Used by callers that want to surface a template error once at
    submission time rather than once per ``/many`` iteration. The parser
    output is discarded; rendering is not performed.
    """
    if not has_dynamics(text):
        return
    # Parsing and rendering share the same code path, so a sacrificial
    # render with a fixed seed proves syntactic validity without leaking
    # implementation details about the AST.
    expand(text, random.Random(0))


def has_dynamics(text: str) -> bool:
    """Cheap pre-check: does this text plausibly contain template syntax?

    Used to skip parsing for the common case of a literal prompt. Returns
    True conservatively — a ``{`` inside a comment still triggers parsing,
    which then strips the comment correctly anyway.
    """
    return "{" in text or "<!--" in text


def expand(text: str, rng: random.Random) -> str:
    """Parse ``text`` and render one expansion using ``rng``.

    Raises :class:`DynamicsSyntaxError` for malformed templates (unclosed
    brace, undefined variable, ...). The function is pure — same ``text``
    + same RNG state yields the same string. The variable environment is
    fresh per call; nothing leaks between successive ``expand()``s.
    """
    stripped = _COMMENT_RE.sub("", text)
    ast, end = _parse_seq(stripped, 0, until_chars="")
    if end != len(stripped):
        # Defensive: _parse_seq with empty boundary should consume
        # everything. A residual position means an internal bug, not a
        # user-input issue.
        raise DynamicsSyntaxError(
            f"internal: residual input at {end}/{len(stripped)}"
        )
    return _render(ast, rng, env={})


def _parse_seq(
    s: str, i: int, until_chars: str
) -> tuple[tuple[Node, ...], int]:
    """Parse a flat sequence of literals + choices until end or boundary.

    Returns (ast, position) where ``position`` is the index of the
    terminating character (still unconsumed) or ``len(s)``. ``until_chars``
    is the set of characters that end the sequence (``"|}"`` inside a
    choice; ``""`` at the top level).
    """
    parts: list[Node] = []
    buf: list[str] = []
    n = len(s)
    while i < n:
        c = s[i]
        if c in until_chars:
            break
        if c == "\\":
            # Backslash escapes the next char (any char). At end of string
            # a trailing backslash is a literal backslash.
            if i + 1 < n:
                buf.append(s[i + 1])
                i += 2
            else:
                buf.append("\\")
                i += 1
        elif c == "$" and i + 1 < n and s[i + 1] == "{":
            # Variable definition or reference. Plain `$` (not followed
            # by `{`) is literal so prices and shell-like syntax survive.
            if buf:
                parts.append(Literal("".join(buf)))
                buf.clear()
            node, i = _parse_var(s, i + 2, open_pos=i)
            parts.append(node)
        elif c == "{":
            if buf:
                parts.append(Literal("".join(buf)))
                buf.clear()
            choice, i = _parse_choice(s, i + 1, open_pos=i)
            parts.append(choice)
        elif c == "}":
            # Stray `}` outside a choice. Be lenient — treat as literal.
            # Prompts naturally contain `}` ("she said {hello|hi} to him.
            # also: }}}"), and forcing escapes here annoys users.
            buf.append("}")
            i += 1
        else:
            buf.append(c)
            i += 1
    if buf:
        parts.append(Literal("".join(buf)))
    return tuple(parts), i


def _parse_choice(s: str, i: int, *, open_pos: int) -> tuple[Choice, int]:
    """Parse the body of a ``{...}`` starting just after the opening brace."""
    options: list[Option] = []
    while True:
        weight, i = _parse_weight(s, i)
        parts, i = _parse_seq(s, i, until_chars="|}")
        options.append(Option(weight, parts))
        if i >= len(s):
            raise DynamicsSyntaxError(
                f"unclosed '{{' starting at position {open_pos}"
            )
        if s[i] == "}":
            return Choice(tuple(options)), i + 1
        # s[i] is '|' — consume separator, continue with the next option.
        i += 1


def _parse_var(s: str, i: int, *, open_pos: int) -> tuple[Node, int]:
    """Parse the body of a ``${...}`` starting just after the ``${``.

    Two shapes:
      * ``${name}``           — reference. Returns :class:`VarRef`.
      * ``${name=<expr>}``   — definition. Inside the RHS, ``|`` is sugar
        for an implicit choice so ``${color=red|green}`` works as
        intuition suggests; explicit braces still nest as expected.
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
    # `${name=...}` — parse the RHS with `|` acting as an implicit
    # alternation separator. One option → bind to that sequence; many →
    # wrap in a Choice. Weights work like inside a regular ``{...}``.
    i += 1  # consume '='
    options: list[Option] = []
    while True:
        weight, i = _parse_weight(s, i)
        rhs_parts, i = _parse_seq(s, i, until_chars="|}")
        options.append(Option(weight, rhs_parts))
        if i >= len(s):
            raise DynamicsSyntaxError(
                f"unclosed '${{' at position {open_pos}"
            )
        if s[i] == "}":
            i += 1  # consume '}'
            break
        i += 1  # consume '|'
    if len(options) == 1:
        return VarDef(name, options[0].parts), i
    return VarDef(name, (Choice(tuple(options)),)), i


def _parse_weight(s: str, i: int) -> tuple[float, int]:
    """Look for a ``<number>::`` prefix at position ``i``.

    Returns ``(weight, position_after_prefix)``. Missing → ``(1.0, i)``.
    Whitespace between the number and ``::`` is allowed (``2 :: a`` works).
    """
    m = _WEIGHT_RE.match(s, i)
    if m is None:
        return 1.0, i
    return float(m.group(1)), m.end()


def _render(
    parts: tuple[Node, ...],
    rng: random.Random,
    env: dict[str, str],
) -> str:
    """Render a parsed sequence with seeded randomness and a shared env.

    ``env`` is mutated in place by :class:`VarDef` nodes; that mutation
    is intentional and the only way variables flow forward in the
    sequence. The dict is shared across nested ``_render`` calls inside
    one ``expand()`` invocation but is fresh per top-level call.
    """
    out: list[str] = []
    for p in parts:
        if isinstance(p, Literal):
            out.append(p.text)
        elif isinstance(p, Choice):
            if not p.options:
                # `{}` — degenerate but well-defined: renders to "".
                # Don't consume any RNG state.
                continue
            weights = [o.weight for o in p.options]
            chosen = rng.choices(p.options, weights=weights, k=1)[0]
            out.append(_render(chosen.parts, rng, env))
        elif isinstance(p, VarDef):
            # Evaluate the RHS, bind the name. Silent — definition site
            # renders to nothing. Redefinition is allowed; the latest
            # value wins.
            env[p.name] = _render(p.parts, rng, env)
        elif isinstance(p, VarRef):
            if p.name not in env:
                raise DynamicsSyntaxError(
                    f"undefined variable {p.name!r} — either it was never "
                    f"defined or it was defined only inside a branch that "
                    f"this seed didn't pick"
                )
            out.append(env[p.name])
    return "".join(out)
