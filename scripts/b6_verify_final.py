"""B6.1: MANUAL re-check of acceptance on the FINAL / transfer layouts of the frozen pool. Not part of the pytest suite.

Refuses to run if data/v2/final_access.log contains any 'eval' entry. Appends ONE entry tagged 'build' to the real ledger.
Prints only per-family accept counts and per-rule rejection counts. Usage: python scripts/b6_verify_final.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # the repo root is importable when run as `python scripts/<name>.py`
import sys

from breachlens import pool as P

if __name__ == "__main__":
    try:
        out = P.verify_final_acceptance()
    except PermissionError as e:
        sys.exit(str(e))
    print("FINAL accepted / layouts per family:")
    for fam, d in sorted(out["per_family"].items()):
        print(f"  {fam:12s} {d['accepted']} / {d['n']}")
    print("rejections per rule:", dict(sorted(out["rejections_per_rule"].items())) or "none")
