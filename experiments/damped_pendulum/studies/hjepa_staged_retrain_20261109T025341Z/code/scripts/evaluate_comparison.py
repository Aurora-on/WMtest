#!/usr/bin/env python3
"""Evaluate a locally trained comparison checkpoint with its own K0 readout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dpbench.evaluate_comparison import main

if __name__ == "__main__":
    main()
