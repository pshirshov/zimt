"""``python -m zimt`` entry point.

The env-affecting flags must be applied before the heavy imports — see
:func:`zimt.cli.early_env_args` for why.
"""

from __future__ import annotations

import sys

from .cli import early_env_args, main

early_env_args()

if __name__ == "__main__":
    sys.exit(main())
