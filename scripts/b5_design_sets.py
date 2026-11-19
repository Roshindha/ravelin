"""B5: rebuild TUNE / LAB-DESIGN with the stat-dependent acceptance rules (too_hard, single_type_trivial) from streams T and L.

Writes data/v2/design_sets.json (schema of B4 + `stat_acceptance`). Never touches stream F / FINAL.  Usage: python scripts/b5_design_sets.py
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # the repo root is importable when run as `python scripts/<name>.py`
import json
import time

from breachlens import acceptance as A
from breachlens import layouts as Y

if __name__ == "__main__":
    t0 = time.time()
    d = A.build_accepted_design_sets()
    Y.DESIGN_SETS_PATH.write_text(json.dumps(d, indent=1) + "\n")
    print("counts", d["counts"], f"({time.time() - t0:.0f}s)")
    for fam, s in d["stat_acceptance"].items():
        print(f"{fam:12s} evaluated={s['stat_evaluated']:4d} trivial={s['single_type_trivial']:3d} too_hard={s['too_hard']:3d} "
              f"rejected={s['stat_rejected']:3d} rate={s['stat_rejection_rate']:.1%} {'FLAG >50%' if s['flag_above_50pct'] else ''}")
