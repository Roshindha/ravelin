"""B3 smoke D4 check (NOT a substitute for B4): random battles on the PROVISIONAL layouts under all 8 symmetries.

For each random (layout, army with Menders) the battle is run in the original frame and in the 7 transformed frames;
ticks, reason, D/C, hp totals and the destroyed set (mapped back to the original frame) must be identical.
Usage: python scripts/b3_smoke_d4.py [n_battles] [seed]
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # the repo root is importable when run as `python scripts/<name>.py`
import sys

import numpy as np

from breachlens import _d4check as D
from breachlens._bench_layouts import FAMILIES, make_layout, random_army
from breachlens.sim import Prepared, simulate


def run(n=200, seed=3030, verbose=True):
    rng = np.random.default_rng(seed)
    bad, strict_extra, with_mender, fb, eq = [], 0, 0, 0, 0
    for b in range(n):
        fam = FAMILIES[int(rng.integers(4))]
        ls = int(rng.integers(25))
        L = make_layout(fam, ls)
        army = random_army(rng, L, min_units=1, max_units=20, n_types=5)
        with_mender += any(t == 4 for t, _ in army)
        base = None
        for k in range(8):
            Lk = D.transform_layout(L, k)
            ak = D.transform_army(army, k)
            r = simulate(Prepared(Lk), ak)
            fb += r.fallback_calls; eq += r.canon_equal_picks
            s = D.summary(r, Lk, k)
            if base is None:
                base = s
                continue
            bad_fields = D.diff(base, s, D.PRIMARY)
            if bad_fields:
                bad.append({"battle": b, "family": fam, "layout_seed": ls, "sym": k, "fields": bad_fields,
                            "base": {f: base[f] for f in bad_fields}, "got": {f: s[f] for f in bad_fields},
                            "stabilizer": D.stabilizer(L, army), "army": army})
            elif D.diff(base, s, ("struct_hp_by_cell",)):
                strict_extra += 1
    if verbose:
        print(f"battles={n} x 8 symmetries; armies with a Mender={with_mender}/{n}; fallback_calls={fb}; canon_equal_picks={eq}")
        print(f"MISMATCHES (primary fields)={len(bad)}; extra per-cell-hp-only mismatches={strict_extra}")
        for c in bad[:3]:
            print(c)
    return bad


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 200, int(sys.argv[2]) if len(sys.argv) > 2 else 3030)
