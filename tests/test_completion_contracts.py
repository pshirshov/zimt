from __future__ import annotations

import json
import subprocess
import unittest

from zimt.models.registry import MODELS
from zimt.repl.commands import COMMANDS
from zimt.repl.history import _completion_options


class CompletionContractTests(unittest.TestCase):
    def test_tui_exposes_every_parser_command_anywhere_in_line(self) -> None:
        self.assertEqual(_completion_options("", 0, "/"), COMMANDS)
        opts = _completion_options("prompt text ", len("prompt text "), "/")
        self.assertEqual(opts, COMMANDS)

    def test_web_exposes_every_parser_command(self) -> None:
        script = "console.log(JSON.stringify(require('./static/completion.js').COMMANDS))"
        completed = subprocess.run(
            ["node", "-e", script],
            check=True,
            cwd=".",
            text=True,
            capture_output=True,
        )
        self.assertEqual(json.loads(completed.stdout), COMMANDS)

    def test_tui_model_sampler_and_resolution_arguments_complete(self) -> None:
        self.assertIn("pony-v6-xl", _completion_options("/model ", len("/model "), "pony"))

        union_samplers = _completion_options("/sampler ", len("/sampler "), "euler")
        self.assertIn("euler", union_samplers)
        self.assertIn("euler-a", union_samplers)

        pony_line = "/model pony-v6-xl /sampler "
        self.assertEqual(
            _completion_options(pony_line, len(pony_line), "euler"),
            [s for s in sorted(MODELS["pony-v6-xl"].samplers) if s.startswith("euler")],
        )

        res_line = "/model pony-v6-xl /res "
        pony_res = _completion_options(res_line, len(res_line), "")
        self.assertIn("1024x1024", pony_res)
        self.assertIn("landscape", pony_res)
        self.assertNotIn("1280x720", pony_res)

    def test_web_argument_completions_cover_finite_argument_sets(self) -> None:
        script = r"""
const { completionItems, tokenAtCursor } = require('./static/completion.js');
const state = {
  loaded: true,
  model: 'pony-v6-xl',
  models: [
    {
      name: 'pony-v6-xl',
      description: 'Pony',
      samplers: ['euler', 'euler-a'],
      resolutions: [{w: 1024, h: 1024, label: 'square'}],
    },
    {
      name: 'z-image-turbo',
      description: 'Z-Image',
      samplers: ['flow-match-euler'],
      resolutions: [{w: 1280, h: 720, label: 'wide'}],
    },
  ],
};
function labels(value) {
  const token = tokenAtCursor(value, value.length);
  return completionItems(value, token.start, token.text, state).map((item) => item.label);
}
console.log(JSON.stringify({
  model: labels('/model po'),
  samplerLoaded: labels('/sampler eu'),
  samplerInline: labels('/model z-image-turbo /sampler '),
  resLoaded: labels('/res '),
  resInline: labels('/model z-image-turbo /res '),
}));
"""
        completed = subprocess.run(
            ["node", "-e", script],
            check=True,
            cwd=".",
            text=True,
            capture_output=True,
        )
        actual = json.loads(completed.stdout)
        self.assertEqual(actual["model"], ["pony-v6-xl"])
        self.assertEqual(actual["samplerLoaded"], ["euler", "euler-a"])
        self.assertEqual(actual["samplerInline"], ["flow-match-euler"])
        self.assertIn("1024x1024", actual["resLoaded"])
        self.assertNotIn("1280x720", actual["resLoaded"])
        self.assertEqual(actual["resInline"], ["landscape", "1280x720"])
