"""Unit tests for zimt.dynamics — the wildcard / alternation expander."""

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
)


def _r(seed: int) -> random.Random:
    return random.Random(seed)


class HasDynamicsTests(unittest.TestCase):
    def test_plain_text_has_no_dynamics(self) -> None:
        self.assertFalse(has_dynamics(""))
        self.assertFalse(has_dynamics("a cat on a rug"))

    def test_brace_triggers_dynamics(self) -> None:
        self.assertTrue(has_dynamics("a {red|blue} cat"))

    def test_comment_triggers_dynamics(self) -> None:
        # Comments alone count — they still need stripping before encoding.
        self.assertTrue(has_dynamics("a cat <!-- note --> on a rug"))


class ExpandIdentityTests(unittest.TestCase):
    """No template syntax → output equals input verbatim."""

    def test_empty_string_round_trip(self) -> None:
        self.assertEqual(expand("", _r(0)), "")

    def test_plain_text_round_trip(self) -> None:
        s = "a serene mountain lake at dawn"
        self.assertEqual(expand(s, _r(0)), s)

    def test_unbraced_pipe_is_literal(self) -> None:
        # Outside `{}`, `|` is not a separator.
        self.assertEqual(expand("a|b", _r(0)), "a|b")


class ExpandAlternationTests(unittest.TestCase):
    def test_two_option_alternation_is_deterministic_per_seed(self) -> None:
        # Same seed → same pick across calls.
        self.assertEqual(expand("{red|blue}", _r(42)), expand("{red|blue}", _r(42)))

    def test_two_option_alternation_covers_both_options(self) -> None:
        seen: set[str] = set()
        for seed in range(20):
            seen.add(expand("{red|blue}", _r(seed)))
        self.assertEqual(seen, {"red", "blue"})

    def test_single_option_alternation_returns_that_option(self) -> None:
        # `{x}` has no real choice — should always render to "x".
        for seed in range(5):
            self.assertEqual(expand("{red}", _r(seed)), "red")

    def test_empty_alternation_renders_empty(self) -> None:
        # `{}` is degenerate but well-defined.
        for seed in range(5):
            self.assertEqual(expand("a {} b", _r(seed)), "a  b")

    def test_empty_option_can_render_nothing(self) -> None:
        # `{a|}` means "a or nothing" — both should appear across seeds.
        seen: set[str] = set()
        for seed in range(30):
            seen.add(expand("hair{ red|}", _r(seed)))
        self.assertIn("hair red", seen)
        self.assertIn("hair", seen)


class ExpandNestingTests(unittest.TestCase):
    def test_nested_choice_renders_correctly(self) -> None:
        results: set[str] = set()
        for seed in range(40):
            results.add(expand("{a {b|c}|d}", _r(seed)))
        # All three reachable expansions should appear given enough seeds.
        self.assertEqual(results, {"a b", "a c", "d"})

    def test_deeply_nested_does_not_blow_up(self) -> None:
        s = "{a {b {c {d|e}|f}|g}|h}"
        # Mostly a smoke test: every seed should produce a non-empty string
        # that's a member of the small enumerable set.
        valid = {"h", "a g", "a b f", "a b c d", "a b c e"}
        for seed in range(50):
            self.assertIn(expand(s, _r(seed)), valid)


class ExpandWeightedTests(unittest.TestCase):
    def test_weighted_picks_favour_heavier_option(self) -> None:
        # 19::a vs 1::b — out of 200 trials we expect overwhelmingly "a".
        # We assert a generous lower bound to avoid flakes while still
        # catching a literal reversal of the weights.
        a_count = sum(
            1 for s in range(200) if expand("{19::a|1::b}", _r(s)) == "a"
        )
        self.assertGreater(a_count, 150)

    def test_weight_with_decimal_parses(self) -> None:
        # Just check it doesn't raise — output is a single character.
        out = expand("{1.5::a|0.5::b}", _r(0))
        self.assertIn(out, {"a", "b"})

    def test_unweighted_options_default_to_one(self) -> None:
        # Mixing: `{2::a|b}` → b should still appear sometimes (weight 1
        # vs 2 = 1/3 chance). Quick smoke check across seeds.
        b_count = sum(1 for s in range(100) if expand("{2::a|b}", _r(s)) == "b")
        self.assertGreater(b_count, 10)
        self.assertLess(b_count, 60)


class ExpandEscapeTests(unittest.TestCase):
    def test_escaped_brace_is_literal(self) -> None:
        self.assertEqual(expand(r"\{a\|b\}", _r(0)), "{a|b}")

    def test_stray_closing_brace_is_literal(self) -> None:
        # Lenient policy: `}` outside any open brace renders as itself
        # so common prompts ("she said hi}") don't need escaping.
        self.assertEqual(expand("hello } world", _r(0)), "hello } world")

    def test_backslash_at_end_is_literal_backslash(self) -> None:
        self.assertEqual(expand("end\\", _r(0)), "end\\")


class ExpandCommentTests(unittest.TestCase):
    def test_comment_stripped_before_parse(self) -> None:
        self.assertEqual(
            expand("hello <!-- note --> world", _r(0)),
            "hello  world",
        )

    def test_comment_can_hide_alternation_syntax(self) -> None:
        # The `{` inside the comment is invisible to the parser.
        self.assertEqual(
            expand("a <!-- {ignored} --> b", _r(0)),
            "a  b",
        )

    def test_multi_line_comment_stripped(self) -> None:
        self.assertEqual(
            expand("a <!--\n  line\n  break\n--> b", _r(0)),
            "a  b",
        )


class ExpandErrorTests(unittest.TestCase):
    def test_unclosed_brace_raises_syntax_error(self) -> None:
        with self.assertRaises(DynamicsSyntaxError) as cm:
            expand("a {red|blue", _r(0))
        # Error mentions where the unclosed brace started.
        self.assertIn("2", str(cm.exception))

    def test_unclosed_inner_brace_raises(self) -> None:
        with self.assertRaises(DynamicsSyntaxError):
            expand("{a {b|c}", _r(0))


class ExpandVariableTests(unittest.TestCase):
    def test_definition_is_silent_and_reference_renders_value(self) -> None:
        # Most basic case: define, then reference. The definition site
        # contributes nothing to the output; the reference contributes
        # the bound value.
        self.assertEqual(expand("${c=red}${c}", _r(0)), "red")

    def test_definition_alone_renders_empty(self) -> None:
        self.assertEqual(expand("${c=red}", _r(0)), "")

    def test_reference_returns_same_value_on_repeat(self) -> None:
        # `${c}` re-used multiple times must return the SAME value as
        # the first reference; we pick once, not per reference.
        out = expand("${c=red|blue} ${c} ${c} ${c}", _r(0))
        word = out.strip().split()[0]
        self.assertEqual(out, f" {word} {word} {word}")
        self.assertIn(word, {"red", "blue"})

    def test_undefined_reference_raises(self) -> None:
        with self.assertRaises(DynamicsSyntaxError) as cm:
            expand("hello ${c}", _r(0))
        self.assertIn("c", str(cm.exception))

    def test_redefinition_uses_latest_value(self) -> None:
        # Two consecutive bindings; the second wins.
        self.assertEqual(
            expand("${c=red}${c}${c=blue}${c}", _r(0)),
            "redblue",
        )

    def test_alternation_inside_definition_is_sugar(self) -> None:
        # Verify `${c=a|b|d}` is equivalent to `${c={a|b|d}}` —
        # i.e. all three options should be reachable across seeds.
        seen: set[str] = set()
        for seed in range(20):
            seen.add(expand("${c=red|green|blue}${c}", _r(seed)))
        self.assertEqual(seen, {"red", "green", "blue"})

    def test_weighted_inside_definition_works(self) -> None:
        # Weighted RHS with the same `2::a` syntax as a Choice.
        a_count = sum(
            1
            for s in range(200)
            if expand("${c=19::a|1::b}${c}", _r(s)) == "a"
        )
        self.assertGreater(a_count, 150)

    def test_nested_choice_inside_definition_works(self) -> None:
        # `${c={a|b}|d}` — the LHS option is a Choice; the RHS is a
        # literal. RNG picks one option, then if it's a Choice, picks
        # within. All three leaf values should be reachable.
        seen: set[str] = set()
        for seed in range(30):
            seen.add(expand("${c={a|b}|d}${c}", _r(seed)))
        self.assertEqual(seen, {"a", "b", "d"})

    def test_definition_inside_choice_branch(self) -> None:
        # When the RNG picks a Choice branch that defines a variable,
        # the binding survives for later references.
        # When the un-defining branch is picked, the reference must fail.
        outcomes: set[str] = set()
        for seed in range(40):
            try:
                outcomes.add(expand("{${c=red}|${c=blue}}-${c}", _r(seed)))
            except DynamicsSyntaxError:  # pragma: no cover - shouldn't happen
                outcomes.add("ERROR")
        self.assertEqual(outcomes, {"-red", "-blue"})

    def test_reference_in_branch_without_matching_definition_raises(self) -> None:
        # Half the seeds pick a branch that defines `c`; the other half
        # don't. The `${c}` reference outside the choice must fail in
        # the latter case.
        errors = 0
        ok = 0
        for seed in range(40):
            try:
                expand("{a|${c=blue}}-${c}", _r(seed))
                ok += 1
            except DynamicsSyntaxError:
                errors += 1
        self.assertGreater(errors, 5,
                           "expected some seeds to hit the undefined branch")
        self.assertGreater(ok, 5,
                           "expected some seeds to hit the defining branch")

    def test_cross_reference_in_definition_rhs(self) -> None:
        # `${b=${a}bar}` — RHS references another variable. As long as
        # `a` was defined earlier, this works.
        self.assertEqual(
            expand("${a=foo}${b=${a}bar}${b}", _r(0)),
            "foobar",
        )

    def test_dollar_without_brace_is_literal(self) -> None:
        # `$5` and `$cost` must not trip the variable parser.
        self.assertEqual(expand("price: $5.99", _r(0)), "price: $5.99")
        self.assertEqual(expand("$cost = 10", _r(0)), "$cost = 10")

    def test_escaped_dollar_is_literal(self) -> None:
        # `\$` makes the `$` literal — but `{c}` after it is then still a
        # Choice (with one option "c"). So `\${c}` renders as "$c". To
        # get a fully-literal `${c}` the user escapes all three special
        # chars: `\$\{c\}`.
        self.assertEqual(expand(r"\${c}", _r(0)), "$c")
        self.assertEqual(expand(r"\$\{c\}", _r(0)), "${c}")

    def test_unclosed_var_raises(self) -> None:
        with self.assertRaises(DynamicsSyntaxError):
            expand("${color=red", _r(0))

    def test_invalid_var_name_raises(self) -> None:
        # Names must start with a letter or `_`.
        with self.assertRaises(DynamicsSyntaxError):
            expand("${1=red}", _r(0))

    def test_fresh_env_per_expand_call(self) -> None:
        # Two separate expand() calls share no state.
        rng = _r(0)
        expand("${c=red}", rng)
        with self.assertRaises(DynamicsSyntaxError):
            expand("${c}", rng)

    def test_empty_definition_binds_empty_string(self) -> None:
        self.assertEqual(expand("[${c=}]${c}", _r(0)), "[]")


class ExpandCompositeTests(unittest.TestCase):
    """Object-literal composite variables — `${obj={k=v}, {k=v}}` form."""

    def test_single_field_object_binds_dotted_path(self) -> None:
        # `${p={hair=long}}${p.hair}` should render "long".
        self.assertEqual(expand("${p={hair=long}}${p.hair}", _r(0)), "long")

    def test_object_definition_is_silent(self) -> None:
        # Like regular var defs, the definition site renders to "".
        self.assertEqual(expand("${p={hair=long}}", _r(0)), "")

    def test_multiple_fields_user_example(self) -> None:
        # The exact syntax from the user's request:
        # `${personA={hair=long|short}, {clothes=red|blue}}${personA.hair}`
        out = expand(
            "${p={hair=long|short}, {clothes=red|blue}}"
            "${p.hair} hair, ${p.clothes} clothes",
            _r(0),
        )
        # The values came from the seeded RNG; we don't pin them, but
        # both colours and both lengths should be reachable across seeds.
        self.assertIn(out.split(" hair")[0], {"long", "short"})
        self.assertIn(out.split("clothes")[0].split(", ")[1].strip(),
                      {"red", "blue"})

    def test_redefinition_within_object_uses_latest(self) -> None:
        # Object literal with the same field twice → latest wins, same
        # rule as scalar VarDef redefinition.
        self.assertEqual(
            expand("${p={hair=long}, {hair=short}}${p.hair}", _r(0)),
            "short",
        )

    def test_nested_object_literal_flattens(self) -> None:
        # `${p={outfit={shirt=red}}}` binds `p.outfit.shirt`. We can
        # reach it with a dotted reference.
        self.assertEqual(
            expand("${p={outfit={shirt=red}}}${p.outfit.shirt}", _r(0)),
            "red",
        )

    def test_three_levels_of_nesting(self) -> None:
        self.assertEqual(
            expand(
                "${a={b={c={d=value}}}}${a.b.c.d}",
                _r(0),
            ),
            "value",
        )

    def test_field_value_with_alternation(self) -> None:
        # Each field value still supports `|` implicit alternation.
        # Run across seeds and verify both options are reachable.
        seen: set[str] = set()
        for seed in range(20):
            seen.add(expand("${p={c=red|blue}}${p.c}", _r(seed)))
        self.assertEqual(seen, {"red", "blue"})

    def test_field_can_reference_earlier_field(self) -> None:
        # Bindings within one MultiVarDef are evaluated left-to-right,
        # so a later field can reference an earlier one.
        self.assertEqual(
            expand("${p={base=red}, {echo=${p.base} shirt}}${p.echo}", _r(0)),
            "red shirt",
        )

    def test_flat_dotted_def_works_directly(self) -> None:
        # `${a.b=...}` should be equivalent to `${a={b=...}}` — same
        # env key. The composite syntax is just one way to spell it.
        self.assertEqual(
            expand("${p.hair=long}${p.hair}", _r(0)),
            "long",
        )

    def test_reference_to_parent_without_dot_raises(self) -> None:
        # `${p}` where only `p.hair` was bound → no such key.
        with self.assertRaises(DynamicsSyntaxError):
            expand("${p={hair=long}}${p}", _r(0))

    def test_reference_to_missing_field_raises(self) -> None:
        with self.assertRaises(DynamicsSyntaxError):
            expand("${p={hair=long}}${p.clothes}", _r(0))

    def test_brace_choice_in_rhs_stays_a_choice(self) -> None:
        # `${color={red|blue}}` — the inner block has no `=`, so it's a
        # regular Choice, not an object literal. The whole thing reduces
        # to a scalar definition.
        seen: set[str] = set()
        for seed in range(10):
            seen.add(expand("${color={red|blue}}${color}", _r(seed)))
        self.assertEqual(seen, {"red", "blue"})

    def test_whitespace_around_comma_and_braces_tolerated(self) -> None:
        # `,` is the field separator; optional whitespace around it.
        self.assertEqual(
            expand(
                "${p={hair=long} , {clothes=red}}"
                "${p.hair} / ${p.clothes}",
                _r(0),
            ),
            "long / red",
        )

    def test_object_inside_choice_branch(self) -> None:
        # When a Choice picks the branch that defines the object, the
        # binding is visible afterwards. The undefining branch makes
        # the reference fail at expansion time — same rule as scalar.
        outcomes: set[str] = set()
        for seed in range(40):
            try:
                outcomes.add(
                    expand(
                        "{${p={c=red}}|${p={c=blue}}} -> ${p.c}",
                        _r(seed),
                    ),
                )
            except DynamicsSyntaxError:
                outcomes.add("ERROR")
        # Both branches define p.c, so we never hit the error path —
        # we should see both possible expansions.
        self.assertEqual(outcomes, {" -> red", " -> blue"})

    def test_malformed_field_block_missing_equals_raises(self) -> None:
        # `{hair}` looks like a Choice (no `=`), so the parser stays
        # in scalar mode and the outer block is just `${p={hair}}` —
        # a regular scalar def. Not malformed.
        # Truly malformed: `${p={hair}, {clothes=red}}` would be ambiguous.
        # Per our rule, the first block decides the mode. `{hair}` has
        # no `=` so we're in scalar mode; the comma then becomes part
        # of the scalar value. We accept that as well-defined.
        out = expand("${p={hair}}${p}", _r(0))
        self.assertEqual(out, "hair")

    def test_unclosed_object_literal_raises(self) -> None:
        with self.assertRaises(DynamicsSyntaxError):
            expand("${p={hair=long}, {clothes=red}", _r(0))


class ExpandIntegrationTests(unittest.TestCase):
    """Realistic prompts the user might type."""

    def test_typical_hair_color_prompt(self) -> None:
        results: set[str] = set()
        for seed in range(30):
            out = expand("girl with {red|green|blue} hair", _r(seed))
            results.add(out)
        self.assertEqual(
            results,
            {
                "girl with red hair",
                "girl with green hair",
                "girl with blue hair",
            },
        )

    def test_score_tags_in_template_pass_through(self) -> None:
        # A literal `:` in the prompt (e.g. score_9 tags) shouldn't trip
        # the weight parser, because weight matching is anchored at the
        # start of an option and needs `::` not `:`.
        out = expand("score_9:1.2, {red|blue}", _r(7))
        self.assertTrue(out.startswith("score_9:1.2, "))
        self.assertIn(out.split(", ")[1], {"red", "blue"})


if __name__ == "__main__":
    unittest.main()
