#!/usr/bin/env python3
"""Bootstrap without importing torch before physical GPU isolation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from layerwise_lm.cli import main

if __name__ == "__main__":
    main()
