"""Dynamic prompt syntax — A1111/ComfyUI-style alternation and friends.

Syntax
------
``{a|b|c}``                   alternation: pick one option uniformly
``{2::a|1::b}``               weighted alternation: 2/3 a, 1/3 b
``{a {b|c}|d}``               nesting: choices nest to any depth
``{|a|b}``                    empty option allowed → "", "a", or "b"
``${name=expr}``              variable: pick once, bind to ``name``, render nothing
``${name}``                   variable reference: render the bound value
``${obj={k1=v1}, {k2=v2}}``   composite variable: each field binds at ``obj.k1`` etc.
``${obj.k1}``                 access a composite field
``\\{ \\| \\$``               backslash-escape for literal braces / pipe / dollar
``<!-- foo -->``              HTML-style comments stripped before parsing

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

Composite variables
-------------------
``${person={hair=long|short}, {clothes=red|blue}}`` binds two keys,
``person.hair`` and ``person.clothes``, each evaluated independently
from the same RNG stream. Fields nest arbitrarily — a field value can
itself be another ``{name=value}, {name=value}`` object literal — and
the flat-dotted equivalent ``${person.hair=long|short}`` works too.

Storage is flat: ``env`` holds ``"person.hair"`` and ``"person.clothes"``
as separate keys. ``${person}`` (no dot) raises like any other
undefined variable — composite parents don't render as scalars.

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
# Variable names can include `.` so flat-dotted forms like ``${a.b=...}``
# and references like ``${a.b.c}`` work alongside the composite-literal
# syntax. The first character must still be a letter or underscore.
_VAR_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
# Field names inside ``{name=value}`` object-literal blocks. No dots —
# the dotted path is built by combining the outer name with the field
# name at parse time.
_FIELD_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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


@dataclass(frozen=True)
class MultiVarDef:
    """``${obj={k1=v1}, {k2=v2}}`` — emit multiple flat-dotted bindings.

    The object-literal syntax is parser-only sugar; at the AST level we
    just have an ordered sequence of ``(full_dotted_name, value_parts)``
    pairs that bind in order against the shared env. Each binding is
    silent like a regular :class:`VarDef`.

    Nested object literals (``${a={b={c=...}}}``) flatten during parse
    into deeper dotted paths, so the AST stays shallow no matter how
    nested the syntactic form is.
    """
    bindings: tuple[tuple[str, tuple["Node", ...]], ...]


Node = Literal | Choice | VarDef | VarRef | MultiVarDef


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

    Three shapes:
      * ``${name}``                   — reference. Returns :class:`VarRef`.
      * ``${name=<expr>}``           — scalar definition. ``|`` inside the
        RHS is implicit choice sugar.
      * ``${name={k=v}, {k=v}, ...}`` — composite definition. Each field
        binds at the flat-dotted path ``name.k``; nested object literals
        flatten further. Returns :class:`MultiVarDef`.
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
    # Composite ahead? Detect by peeking — the first non-whitespace must
    # be ``{name=`` for object-literal mode. A bare ``{`` without an
    # equals inside is still a regular Choice in scalar mode.
    if _looks_like_object_literal(s, i):
        bindings, i = _parse_object_literal_body(s, i, prefix=name, open_pos=open_pos)
        i = _skip_ws(s, i)
        if i >= len(s) or s[i] != "}":
            raise DynamicsSyntaxError(
                f"unclosed '${{' at position {open_pos}"
            )
        return MultiVarDef(tuple(bindings)), i + 1
    # Scalar definition — `|` in RHS is implicit Choice sugar.
    parts, i = _parse_scalar_rhs(s, i, until_chars="}", open_pos=open_pos)
    if i >= len(s) or s[i] != "}":
        raise DynamicsSyntaxError(
            f"unclosed '${{' at position {open_pos}"
        )
    return VarDef(name, parts), i + 1


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\n\r":
        i += 1
    return i


def _looks_like_object_literal(s: str, i: int) -> bool:
    """Lookahead: does the value at ``s[i:]`` start with a field block?

    A field block is ``{<ident>=...}`` — the presence of ``=`` after the
    opening ``{`` and an identifier is the distinguishing signal. Any
    other shape (bare alternation ``{a|b}``, literal text) is scalar.
    """
    j = _skip_ws(s, i)
    if j >= len(s) or s[j] != "{":
        return False
    j = _skip_ws(s, j + 1)
    m = _FIELD_NAME_RE.match(s, j)
    if m is None:
        return False
    j = _skip_ws(s, m.end())
    return j < len(s) and s[j] == "="


def _parse_object_literal_body(
    s: str, i: int, *, prefix: str, open_pos: int,
) -> tuple[list[tuple[str, tuple[Node, ...]]], int]:
    """Parse ``{k1=v1}, {k2=v2}, ...`` and return flat bindings.

    Each block's value is itself parsed in RHS mode, so it can be either
    a scalar (with ``|`` alternation) or another nested object literal.
    The returned bindings list is flat: nested objects' field paths are
    already joined with the parent prefix.
    """
    bindings: list[tuple[str, tuple[Node, ...]]] = []
    while True:
        i = _skip_ws(s, i)
        if i >= len(s) or s[i] != "{":
            return bindings, i
        i += 1  # consume '{'
        i = _skip_ws(s, i)
        m = _FIELD_NAME_RE.match(s, i)
        if m is None:
            raise DynamicsSyntaxError(
                f"expected field name in object literal at position {i}"
            )
        field_name = m.group(0)
        i = _skip_ws(s, m.end())
        if i >= len(s) or s[i] != "=":
            raise DynamicsSyntaxError(
                f"expected '=' after field {field_name!r} at position {i}"
            )
        i += 1  # consume '='
        full_name = f"{prefix}.{field_name}"
        if _looks_like_object_literal(s, i):
            # Nested composite: recurse, extending our binding list
            # with the deeper paths.
            sub_bindings, i = _parse_object_literal_body(
                s, i, prefix=full_name, open_pos=open_pos,
            )
            bindings.extend(sub_bindings)
        else:
            parts, i = _parse_scalar_rhs(
                s, i, until_chars="}", open_pos=open_pos,
            )
            bindings.append((full_name, parts))
        i = _skip_ws(s, i)
        if i >= len(s) or s[i] != "}":
            raise DynamicsSyntaxError(
                f"expected '}}' to close field {field_name!r} at position {i}"
            )
        i += 1  # consume '}' of the field block
        i = _skip_ws(s, i)
        if i < len(s) and s[i] == ",":
            i += 1  # consume ',' and look for another field
            continue
        return bindings, i


def _parse_scalar_rhs(
    s: str, i: int, *, until_chars: str, open_pos: int,
) -> tuple[tuple[Node, ...], int]:
    """Parse a scalar var/field RHS with ``|`` as implicit alternation.

    Returns parts for the value. A single option flattens to its parts;
    multiple options wrap in a :class:`Choice` so the renderer picks one.
    """
    options: list[Option] = []
    while True:
        weight, i = _parse_weight(s, i)
        rhs_parts, i = _parse_seq(s, i, until_chars="|" + until_chars)
        options.append(Option(weight, rhs_parts))
        if i >= len(s):
            raise DynamicsSyntaxError(
                f"unclosed '${{' at position {open_pos}"
            )
        if s[i] in until_chars:
            break
        i += 1  # consume '|'
    if len(options) == 1:
        return options[0].parts, i
    return (Choice(tuple(options)),), i


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
        elif isinstance(p, MultiVarDef):
            # Composite literal: bind each pre-flattened path. Same
            # silent / latest-wins semantics as VarDef, just emitting
            # several env entries from one syntactic block. Bindings are
            # evaluated in source order so a later field can reference
            # an earlier one ``${${p={a=red}, {b=${p.a}}}``.
            for binding_name, binding_parts in p.bindings:
                env[binding_name] = _render(binding_parts, rng, env)
        elif isinstance(p, VarRef):
            if p.name not in env:
                raise DynamicsSyntaxError(
                    f"undefined variable {p.name!r} — either it was never "
                    f"defined or it was defined only inside a branch that "
                    f"this seed didn't pick"
                )
            out.append(env[p.name])
    return "".join(out)
