#!/usr/bin/env python3
"""Real layer-by-layer oracle-routed continuation experiment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from routed_lm.cli import main

if __name__ == "__main__":
    main()
