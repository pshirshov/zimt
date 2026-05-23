"""Unit tests for zimt.dynamics — JSON-like template syntax with two-pass
verbatim expansion.

Syntax under test:
  [a|b]                    alternation (lookahead: only when `|` present)
  ${name=...}              scalar definition
  ${name}                  reference
  ${obj={k=v, k=v}}        composite definition (flattens to obj.k bindings)
  ${obj.k}                 composite access
  `verbatim text`          stored as raw, re-evaluated at reference site
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from zimt.dynamics import (  # noqa: E402
    DynamicsSyntaxError,
    expand,
    has_dynamics,
    validate,
)


def _r(seed: int) -> random.Random:
    return random.Random(seed)


# ---------- has_dynamics ----------

class HasDynamicsTests(unittest.TestCase):
    def test_plain_text_has_no_dynamics(self) -> None:
        self.assertFalse(has_dynamics(""))
        self.assertFalse(has_dynamics("a cat on a rug"))

    def test_brackets_alone_no_pipe_is_not_dynamics(self) -> None:
        # `[word]` is compel-weighting syntax — must not trigger our parser.
        self.assertFalse(has_dynamics("[word]"))

    def test_brackets_with_pipe_triggers_dynamics(self) -> None:
        self.assertTrue(has_dynamics("a [red|blue] cat"))

    def test_var_syntax_triggers_dynamics(self) -> None:
        self.assertTrue(has_dynamics("a ${color} cat"))

    def test_verbatim_backtick_triggers_dynamics(self) -> None:
        # Standalone backticks in body don't change behaviour but are
        # cheap to scan; flagging them keeps the precheck conservative.
        self.assertTrue(has_dynamics("a `literal` thing"))

    def test_comment_triggers_dynamics(self) -> None:
        self.assertTrue(has_dynamics("a <!-- note --> cat"))


# ---------- identity / plain text ----------

class IdentityTests(unittest.TestCase):
    def test_empty_round_trip(self) -> None:
        self.assertEqual(expand("", _r(0)), "")

    def test_plain_text_round_trip(self) -> None:
        s = "a serene mountain lake at dawn"
        self.assertEqual(expand(s, _r(0)), s)

    def test_unbracketed_pipe_is_literal(self) -> None:
        self.assertEqual(expand("a|b", _r(0)), "a|b")


# ---------- alternation ----------

class AlternationTests(unittest.TestCase):
    def test_two_option_alternation_is_deterministic_per_seed(self) -> None:
        self.assertEqual(expand("[red|blue]", _r(42)), expand("[red|blue]", _r(42)))

    def test_two_option_alternation_covers_both_options(self) -> None:
        seen: set[str] = set()
        for seed in range(20):
            seen.add(expand("[red|blue]", _r(seed)))
        self.assertEqual(seen, {"red", "blue"})

    def test_brackets_without_pipe_are_literal(self) -> None:
        # Compel-style `[word]-` must survive expansion intact.
        self.assertEqual(expand("[word]-", _r(0)), "[word]-")
        self.assertEqual(expand("[bad quality]", _r(0)), "[bad quality]")

    def test_empty_option_allowed(self) -> None:
        # `[|a]` means "a or nothing" — both reachable across seeds.
        seen: set[str] = set()
        for seed in range(30):
            seen.add(expand("hair[ red|]", _r(seed)))
        self.assertIn("hair red", seen)
        self.assertIn("hair", seen)

    def test_nested_alternation(self) -> None:
        results: set[str] = set()
        for seed in range(40):
            results.add(expand("[a [b|c]|d]", _r(seed)))
        self.assertEqual(results, {"a b", "a c", "d"})

    def test_weighted_alternation_favours_heavier(self) -> None:
        # 19::a vs 1::b — generous threshold to avoid flakes.
        a_count = sum(
            1 for s in range(200) if expand("[19::a|1::b]", _r(s)) == "a"
        )
        self.assertGreater(a_count, 150)


# ---------- variables ----------

class VariableTests(unittest.TestCase):
    def test_definition_is_silent(self) -> None:
        self.assertEqual(expand("${c=red}", _r(0)), "")

    def test_reference_returns_bound_value(self) -> None:
        self.assertEqual(expand("${c=red}${c}", _r(0)), "red")

    def test_reference_reuses_same_pick_across_references(self) -> None:
        out = expand("${c=[red|blue]} ${c} ${c} ${c}", _r(0))
        word = out.strip().split()[0]
        self.assertEqual(out, f" {word} {word} {word}")
        self.assertIn(word, {"red", "blue"})

    def test_undefined_reference_raises(self) -> None:
        with self.assertRaises(DynamicsSyntaxError):
            expand("hello ${c}", _r(0))

    def test_redefinition_uses_latest_value(self) -> None:
        self.assertEqual(
            expand("${c=red}${c}${c=blue}${c}", _r(0)),
            "redblue",
        )

    def test_alternation_in_rhs(self) -> None:
        # `${c=[red|blue]}` — alternation as scalar value. Single pick.
        seen: set[str] = set()
        for seed in range(20):
            seen.add(expand("${c=[red|green|blue]}${c}", _r(seed)))
        self.assertEqual(seen, {"red", "green", "blue"})

    def test_cross_reference_in_rhs(self) -> None:
        self.assertEqual(
            expand("${a=foo}${b=${a}bar}${b}", _r(0)),
            "foobar",
        )


# ---------- composite (object literal) ----------

class CompositeTests(unittest.TestCase):
    def test_single_field_object(self) -> None:
        self.assertEqual(
            expand("${p={hair=long}}${p.hair}", _r(0)),
            "long",
        )

    def test_user_example_two_fields(self) -> None:
        # Exact shape from the redo brief:
        # `${person={clothes={type=[shorts|skirt]}, hair=[blond|brown]}}`
        out = expand(
            "${person={clothes={type=[shorts|skirt]}, hair=[blond|brown]}}"
            "hair: ${person.hair}, clothes: ${person.clothes.type}",
            _r(0),
        )
        self.assertRegex(
            out,
            r"^hair: (blond|brown), clothes: (shorts|skirt)$",
        )

    def test_nested_composite_flattens(self) -> None:
        self.assertEqual(
            expand("${a={b={c=val}}}${a.b.c}", _r(0)),
            "val",
        )

    def test_field_value_with_alternation_reaches_all_options(self) -> None:
        seen: set[str] = set()
        for seed in range(20):
            seen.add(expand("${p={c=[red|blue]}}${p.c}", _r(seed)))
        self.assertEqual(seen, {"red", "blue"})

    def test_field_can_reference_earlier_field(self) -> None:
        # Left-to-right evaluation — `echo` reads the already-bound `base`.
        self.assertEqual(
            expand("${p={base=red, echo=${p.base} shirt}}${p.echo}", _r(0)),
            "red shirt",
        )

    def test_flat_dotted_def_equivalent(self) -> None:
        a = expand("${p.hair=long}${p.hair}", _r(0))
        b = expand("${p={hair=long}}${p.hair}", _r(0))
        self.assertEqual(a, b)
        self.assertEqual(a, "long")

    def test_parent_reference_without_dot_raises(self) -> None:
        with self.assertRaises(DynamicsSyntaxError):
            expand("${p={hair=long}}${p}", _r(0))

    def test_missing_field_reference_raises(self) -> None:
        with self.assertRaises(DynamicsSyntaxError):
            expand("${p={hair=long}}${p.clothes}", _r(0))


# ---------- verbatim ----------

class VerbatimTests(unittest.TestCase):
    def test_verbatim_def_stores_raw_text(self) -> None:
        # Definition itself renders to "" (silent, like any var def).
        self.assertEqual(expand("${tt=`raw text`}", _r(0)), "")

    def test_verbatim_def_then_ref_substitutes_raw_text(self) -> None:
        # Reference inlines the raw text; for non-dynamics content the
        # output equals the raw text.
        self.assertEqual(
            expand("${tt=`hello world`}-${tt}-", _r(0)),
            "-hello world-",
        )

    def test_verbatim_value_with_alternation_is_re_picked_per_ref(self) -> None:
        # The user's marquee case: each `${tt}` re-rolls because pass 1
        # substituted raw text and pass 2 sees two independent
        # alternations.
        result_pairs: set[tuple[str, str]] = set()
        for seed in range(40):
            out = expand(
                "${tt=`[shorts|skirt]`}First: ${tt}; Second: ${tt}",
                _r(seed),
            )
            # Both picks should appear; verify by collecting their (a,b) tuple.
            parts = out.replace("First: ", "").split("; Second: ")
            self.assertEqual(len(parts), 2)
            result_pairs.add((parts[0], parts[1]))
        # We should see at least one case where the two picks differ.
        self.assertTrue(
            any(a != b for a, b in result_pairs),
            f"verbatim refs never differed: {result_pairs}",
        )
        # All values must be one of the alternation options.
        for a, b in result_pairs:
            self.assertIn(a, {"shorts", "skirt"})
            self.assertIn(b, {"shorts", "skirt"})

    def test_verbatim_def_does_not_consume_rng(self) -> None:
        # Pass 1 must not touch the RNG, so two equivalent expansions
        # — one with verbatim plumbing, one without — produce the same
        # downstream pick from the SAME seed.
        with_verbatim = expand(
            "${tt=`[red|blue]`}${tt}", _r(7),
        )
        without_verbatim = expand("[red|blue]", _r(7))
        self.assertEqual(with_verbatim, without_verbatim)

    def test_verbatim_ref_inside_text(self) -> None:
        # Pass-1 substitution happens inside arbitrary text.
        self.assertEqual(
            expand("${color=`red`}A ${color} shirt and a ${color} hat", _r(0)),
            "A red shirt and a red hat",
        )

    def test_unclosed_backtick_is_lenient(self) -> None:
        # Pass 1 leaves it; pass 2 strips the backtick and treats the
        # rest as literal text rather than blowing up.
        self.assertEqual(expand("hello `unclosed", _r(0)), "hello unclosed")

    def test_backtick_inside_normal_text_is_stripped(self) -> None:
        # A `literal` substring renders without the backticks — they're
        # transparent delimiters. (Useful if a future parser feature
        # would interpret the content; today it's a no-op.)
        self.assertEqual(expand("a `b` c", _r(0)), "a b c")

    def test_escaped_backtick_inside_verbatim(self) -> None:
        # `\\\`` inside a verbatim escapes the backtick, letting raw
        # content contain literal backticks.
        self.assertEqual(
            expand(r"${t=`back\`tick`}${t}", _r(0)),
            "back`tick",
        )


# ---------- escapes ----------

class EscapeTests(unittest.TestCase):
    def test_escaped_bracket_is_literal(self) -> None:
        self.assertEqual(expand(r"\[a|b\]", _r(0)), "[a|b]")

    def test_escaped_dollar_is_literal(self) -> None:
        # `\$` makes `$` literal; the following `{c}` is just text since
        # `{` is no longer a Choice opener in the new syntax.
        self.assertEqual(expand(r"\${c}", _r(0)), "${c}")

    def test_escaped_brace_is_literal(self) -> None:
        # `{` and `}` outside an object-literal context render as text;
        # explicit escape just removes any future ambiguity.
        self.assertEqual(expand(r"\{ \}", _r(0)), "{ }")

    def test_backslash_at_end_is_literal(self) -> None:
        self.assertEqual(expand("end\\", _r(0)), "end\\")


# ---------- comments ----------

class CommentTests(unittest.TestCase):
    def test_comment_stripped_before_parse(self) -> None:
        self.assertEqual(
            expand("hello <!-- note --> world", _r(0)),
            "hello  world",
        )

    def test_comment_can_hide_special_syntax(self) -> None:
        self.assertEqual(
            expand("a <!-- ${ignored} --> b", _r(0)),
            "a  b",
        )


# ---------- error reporting ----------

class ErrorTests(unittest.TestCase):
    def test_unclosed_alternation_raises(self) -> None:
        with self.assertRaises(DynamicsSyntaxError):
            expand("a [red|blue", _r(0))

    def test_unclosed_var_raises(self) -> None:
        with self.assertRaises(DynamicsSyntaxError):
            expand("${color=red", _r(0))

    def test_unclosed_object_literal_raises(self) -> None:
        with self.assertRaises(DynamicsSyntaxError):
            expand("${p={hair=long, clothes=red", _r(0))

    def test_invalid_var_name_raises(self) -> None:
        with self.assertRaises(DynamicsSyntaxError):
            expand("${1=red}", _r(0))


# ---------- validate ----------

class ValidateTests(unittest.TestCase):
    def test_validate_noop_for_plain_text(self) -> None:
        validate("hello world")  # should not raise

    def test_validate_passes_on_good_template(self) -> None:
        validate("${c=red}${c} [a|b|c]")  # should not raise

    def test_validate_raises_on_bad_template(self) -> None:
        with self.assertRaises(DynamicsSyntaxError):
            validate("${unclosed")


# ---------- integration (realistic prompts) ----------

class IntegrationTests(unittest.TestCase):
    def test_typical_outfit_prompt(self) -> None:
        out = expand(
            "${person={clothes={type=[shorts|skirt]}, hair=[blond|brown]}}"
            "a ${person.hair}-haired girl wearing ${person.clothes.type}",
            _r(3),
        )
        self.assertRegex(
            out, r"^a (blond|brown)-haired girl wearing (shorts|skirt)$",
        )

    def test_verbatim_template_for_two_people(self) -> None:
        # User's marquee case: two independent picks for two characters
        # from one verbatim definition.
        out = expand(
            "${outfit=`[shorts|skirt]`}"
            "first wears ${outfit}, second wears ${outfit}",
            _r(11),
        )
        # Both picks valid; might be same or different.
        for token in ("first wears", "second wears"):
            self.assertIn(token, out)
        choices = {"shorts", "skirt"}
        first_pick = out.split("first wears ")[1].split(",")[0]
        second_pick = out.split("second wears ")[1].strip()
        self.assertIn(first_pick, choices)
        self.assertIn(second_pick, choices)

    def test_compel_negative_weighting_passes_through(self) -> None:
        # Real-world prompt that uses compel `[word]-` — must NOT be
        # interpreted as alternation. The output should preserve the
        # brackets so the SDXL pipeline's compel pass can read them.
        self.assertEqual(
            expand("a [bad anatomy]- (masterpiece:1.2) cat", _r(0)),
            "a [bad anatomy]- (masterpiece:1.2) cat",
        )


if __name__ == "__main__":
    unittest.main()
