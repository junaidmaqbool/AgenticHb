"""Enable ``python -m adaptivehb`` to invoke the command-line interface."""

from __future__ import annotations

import sys

from adaptivehb.cli import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
