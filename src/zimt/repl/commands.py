"""Multi-command line parser shared by the REPL and the web ``/api/exec``.

Users can type ``/model X /cfg 5 /steps 25 a prompt here`` in a single line.
Greedy commands (``negprompt``, ``tokenize``, ``many``) consume tokens until
the next known ``/cmd`` boundary so multi-step pipelines compose cleanly.

A token that exactly matches a key of :data:`COMMAND_ARITY` is a command
boundary. Anything before the first command and any tail after the last
known command form a synthetic ``("/_prompt", [text])`` entry — the
trailing prompt the user wants to send to the model.
"""

from __future__ import annotations

from typing import Literal

Arity = int | Literal["GREEDY", "MANY"]

COMMAND_ARITY: dict[str, Arity] = {
    "/help": 0, "/?": 0, "/quit": 0, "/exit": 0, "/q": 0,
    "/model": 1, "/cfg": 1, "/steps": 1, "/seed": 1, "/res": 1,
    "/clip_skip": 1,
    "/sampler": 1,
    "/size": 2,
    "/raw": 0,
    "/many": "MANY",
    "/negprompt": "GREEDY",
    "/tokenize": "GREEDY",
    "/lora": "GREEDY",
}

KNOWN_COMMANDS: frozenset[str] = frozenset(COMMAND_ARITY)

# Convenience for the readline completer — alphabetically sorted so cycling
# Tab presses goes in a stable order.
COMMANDS: list[str] = sorted(COMMAND_ARITY)


def parse_commands(line: str) -> list[tuple[str, list[str]]]:
    """Tokenize a line into ``[(command, args)]``.

    ``/cmd`` tokens may appear *anywhere* on the line — before the prompt,
    after it, or interleaved. Every non-command token that isn't consumed
    by a command's args is collected into a single trailing
    ``("/_prompt", [text])`` entry. This way users can write
    ``a cute girl /many 3`` and have ``/many 3`` interpreted as a command
    rather than swallowed into the prompt.

    Greedy commands (``GREEDY`` / ``MANY``) still stop at the next known
    ``/cmd`` boundary.
    """
    tokens = line.split()
    out: list[tuple[str, list[str]]] = []
    prompt_acc: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok not in KNOWN_COMMANDS:
            prompt_acc.append(tok)
            i += 1
            continue
        arity = COMMAND_ARITY[tok]

        if arity == 0:
            out.append((tok, []))
            i += 1
        elif arity == "GREEDY":
            end = i + 1
            while end < len(tokens) and tokens[end] not in KNOWN_COMMANDS:
                end += 1
            out.append((tok, [" ".join(tokens[i + 1:end])]))
            i = end
        elif arity == "MANY":
            # `/many N <greedy prompt>` — N is positional, prompt is greedy.
            if i + 1 >= len(tokens):
                out.append((tok, []))
                i += 1
                continue
            n_arg = tokens[i + 1]
            end = i + 2
            while end < len(tokens) and tokens[end] not in KNOWN_COMMANDS:
                end += 1
            out.append((tok, [n_arg, " ".join(tokens[i + 2:end])]))
            i = end
        else:  # fixed int arity
            args = tokens[i + 1:i + 1 + arity]
            out.append((tok, args))
            i += 1 + arity

    if prompt_acc:
        out.append(("/_prompt", [" ".join(prompt_acc)]))
    return out
