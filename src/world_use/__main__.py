from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":        # a spawned worker process re-imports the main module: it must not run the CLI
    sys.exit(main())
