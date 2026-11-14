"""B4 full-scale report on the REAL design layouts (TUNE-train/val, LAB-DESIGN): D4 equivariance, battle-length distributions.

Writes data/v2/b4_report.json. Usage: python scripts_b4_report.py
"""
import json
import time
from pathlib import Path

import numpy as np

from breachlens import _d4check as D
from breachlens import config
from breachlens import layouts as Y
from breachlens.armies import random_army, scripted_army
from breachlens.sim import Prepared, simulate

OUT = Path(__file__).parent / "data" / "v2" / "b4_report.json"
PLAIN = {"labyrinth": 2000, "spine_pods": 700, "archipelago": 700, "bunker_nest": 700}     # 4100 battles x 8, trace-checked
SHUF = {"labyrinth": 300, "spine_pods": 170, "archipelago": 170, "bunker_nest": 170}        # 810 battles x 8, adversarial shuffles


def d4():
    rows = {}
    for fam, n in PLAIN.items():
        rows[(fam, "plain")] = D.run_parallel_real(fam, n, 7000 + Y.FAMILIES.index(fam), False, True)
    for fam, n in SHUF.items():
        rows[(fam, "shuffle")] = D.run_parallel_real(fam, n, 8000 + Y.FAMILIES.index(fam), True, True)
    out = []
    for (fam, mode), r in rows.items():
        out.append({"family": fam, "mode": mode, "battles": r["n"], "symmetries": 8, "sims": r["sims"],
                    "armies_with_mender": r["menders"], "mismatches": len(r["bad"]), "trace_mismatches": len(r["trace_bad"]),
                    "per_cell_hp_only_mismatches": r["extra"],
                    "fallback_calls_per_battle": r["fallback_calls"] / r["sims"],
                    "canon_equal_picks_per_battle": r["canon_equal_picks"] / r["sims"],
                    "first_failures": (r["bad"] + r["trace_bad"])[:3]})
    return out


def lengths():
    rng = np.random.default_rng(4242)
    res = {}
    for fam in Y.FAMILIES:
        pool = D.real_layouts(fam)
        prep = [(rec, L, Prepared(L)) for rec, L in pool]
        for pol in ("random", "scripted"):
            ticks, reasons, units = [], {"horizon": 0, "stalemate": 0, "cleared": 0, "all_dead": 0, "no_attackers": 0}, []
            n = 500 if pol == "random" else len(prep)
            for i in range(n):
                rec, L, P = prep[int(rng.integers(len(prep)))] if pol == "random" else prep[i]
                army = random_army(rng, L, min_units=1, max_units=20, n_types=5) if pol == "random" else scripted_army(L)
                r = simulate(P, army)
                ticks.append(r.ticks); reasons[r.reason] += 1; units.append(len(army))
            t = np.array(ticks)
            res[f"{fam}/{pol}"] = {"n": n, "mean": float(t.mean()), "p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
                                   "max": int(t.max()), "mean_units": float(np.mean(units)),
                                   **{f"share_{k}": v / n for k, v in reasons.items()}}
    return res


if __name__ == "__main__":
    t0 = time.time()
    out = {"sim_version": config.sim_version(), "d4": d4(), "d4_seconds": time.time() - t0, "battle_lengths": lengths()}
    OUT.write_text(json.dumps(out, indent=1))
    for r in out["d4"]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "first_failures"})
        for f in r["first_failures"]:
            print("  FAIL", f)
    for k, v in out["battle_lengths"].items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
