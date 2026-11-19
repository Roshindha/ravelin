#!/usr/bin/env python3
"""Status of the v2 grid (B14): per-job state, step, validation score, steps/s, ETA. Thin wrapper around `scripts/b14_grid.py status`
(the prototype's PPO-grid status table this file used to print is gone with the prototype trainer).

    .venv/bin/python scripts/status.py [--watch]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import b14_grid as G  # noqa: E402

if __name__ == "__main__":
    sys.exit(G.main(["status", *sys.argv[1:]]))
