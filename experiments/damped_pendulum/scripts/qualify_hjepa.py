#!/usr/bin/env python3
"""Read-only H-JEPA checkpoint acceptance; exit 0 passes, exit 2 stops continuation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dpbench.hjepa_acceptance import main

if __name__ == "__main__":
    raise SystemExit(main())
