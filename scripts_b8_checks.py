"""B8 checks (no FINAL, no ledger write): (1) B5 consistency of Oracle / Greedy on TUNE; (2) Oracle >= every other baseline per layout;
(3) boundary check: re-run DEV acceptance walks under the current score and compare every bit with the B6 walk records; count candidates whose exact
best single-unit score ratio is 99/100 (DEV accepted + stat-evaluated rejected, TUNE, LAB-DESIGN). Run once, after scripts_b8_baselines.py."""
import csv
import json
import multiprocessing as mp
import sys
from datetime import date
from pathlib import Path

import numpy as np

from breachlens import layouts as Y
from breachlens import pool as P
from breachlens.armies import place_on_border, single_unit_types
from breachlens.score import result_key
from breachlens.sim import Prepared, simulate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "v2" / "baselines"


def best_single_ratio(args):
    family, seed = args
    L = Y.candidate(family, seed)
    Pp = Prepared(L)
    best = None
    for t in range(5):
        num, den = result_key(simulate(Pp, place_on_border(L, single_unit_types(t))))
        if best is None or num * best[1] > best[0] * den:
            best = (num, den)
    return seed, best[0], best[1], (100 * best[0] == 99 * best[1])


def b5_consistency():
    raw = json.loads((ROOT / "data/v2/b5_raw.json").read_text())
    rows = list(csv.DictReader(open(OUT / "b8_layout_rows.csv")))
    new = {(r["policy"], r["id"]): r for r in rows}
    out = {"b5_raw_sim_version": raw["sim_version"], "per_set": {}}
    for s in ("TUNE-train", "TUNE-val"):
        for pol, getter in (("Oracle", lambda l: l["oracle"]["all"]), ("Greedy", lambda l: l["greedy"])):
            d_score, n, breach_diff, units_diff = [], 0, 0, 0
            old_mean, new_mean = {}, {}
            for l in raw["layouts"]:
                if l["set"] != s:
                    continue
                o, nrow = getter(l), new[(pol, l["id"])]
                d_score.append(float(nrow["score"]) - o["score"])
                breach_diff += int(float(nrow["breach"]) != float(o["breach"]))
                n += 1
                old_mean.setdefault(l["family"], []).append(o["score"]); new_mean.setdefault(l["family"], []).append(float(nrow["score"]))
            d = np.array(d_score)
            out["per_set"][f"{s}/{pol}"] = {"n": n, "max_abs_score_diff": float(np.abs(d).max()), "n_exact_equal": int((d == 0).sum()), "breach_flag_mismatches": breach_diff,
                "family_mean_b5": {f: float(np.mean(v)) for f, v in old_mean.items()}, "family_mean_b8": {f: float(np.mean(v)) for f, v in new_mean.items()}}
    return out


def oracle_dominance():
    rows = list(csv.DictReader(open(OUT / "b8_layout_rows.csv")))
    by = {}
    for r in rows:
        by.setdefault((r["set"], r["id"]), {})[r["policy"]] = float(r["score"])
    viol = {}
    n = 0
    for k, v in by.items():
        n += 1
        for p, sc in v.items():
            if p != "Oracle" and sc > v["Oracle"]:
                viol.setdefault(p, []).append([k[1], sc, v["Oracle"]])
    return {"layouts": n, "violations_by_policy": {p: len(x) for p, x in viol.items()}, "violation_examples": {p: x[:3] for p, x in viol.items()}}


def main():
    res = {"date": str(date.today()), "git_commit": P._git("rev-parse", "HEAD")}
    res["b5_consistency"] = b5_consistency()
    res["oracle_dominance"] = oracle_dominance()
    ledger_before = P.read_ledger()
    pool_doc = json.loads(P.POOL_PATH.read_text())
    taken0 = set(P.design_set_hashes())
    ctx = mp.get_context("spawn")
    with ctx.Pool(7) as pl:
        # (a) DEV walks, bit for bit (walk() is the B6 builder; group "DEV" never touches FINAL)
        taken = set(taken0)
        walks = {}
        n_total_bits = n_mismatch = 0
        for fam in Y.FAMILIAR:
            recs, rep = P.walk("DEV", fam, P.PER_FAMILY["DEV-train"] + P.PER_FAMILY["DEV-validation"], taken, pl)
            stored = pool_doc["walk"][f"DEV/{fam}"]["candidate_bits"]
            mism = [[a, b] for a, b in zip(rep["bits"], stored) if a != b]
            n_total_bits += len(stored)
            n_mismatch += len(mism) + abs(len(rep["bits"]) - len(stored))
            walks[fam] = {"candidates": len(stored), "recomputed": len(rep["bits"]), "mismatches": len(mism) + abs(len(rep["bits"]) - len(stored)),
                          "rejected_stat_rules": rep["stat_rejected"], "accepted": rep["accepted"], "mismatch_examples": mism[:3]}
            stored_ids = [r["id"] for r in pool_doc["records"] if r["family"] == fam and r["split"].startswith("DEV")]
            walks[fam]["records_identical"] = stored_ids == [r["id"] for r in recs]
        res["boundary_dev_acceptance"] = {"per_family": walks, "bits_compared": n_total_bits, "bit_mismatches": n_mismatch}
        # (b) exact 99/100 count
        counts = {}
        for fam in Y.FAMILIAR:
            bits = pool_doc["walk"][f"DEV/{fam}"]["candidate_bits"]
            # structurally valid, not a D4 duplicate: accepted, or rejected by a stat rule
            evald = [s for s, a, rules in bits if (a == 1) or set(rules) <= {"single_type_trivial", "too_hard"} and rules]
            r = pl.map(best_single_ratio, [(fam, s) for s in evald], chunksize=4)
            accepted = {s for s, a, _ in bits if a == 1}
            hit = [x for x in r if x[3]]
            counts[f"DEV/{fam}"] = {"evaluated": len(r), "n_99_100": len(hit), "of_which_accepted": sum(x[0] in accepted for x in hit), "seeds": [x[0] for x in hit]}
        meta = Y.load_design_sets_meta()
        for name, fams in (("TUNE", Y.FAMILIAR), ("LAB-DESIGN", ("labyrinth",))):
            for fam in fams:
                acc = [r["seed"] for s, v in meta["sets"].items() if (s.startswith("TUNE") if name == "TUNE" else s == "LAB-DESIGN") for r in v if r["family"] == fam]
                rej = meta["stat_acceptance"][fam]["stat_rejected_seeds"]
                r = pl.map(best_single_ratio, [(fam, s) for s in acc + rej], chunksize=4)
                hit = [x for x in r if x[3]]
                counts[f"{name}/{fam}"] = {"evaluated": len(r), "accepted_in_sets": len(acc), "stat_rejected": len(rej), "n_99_100": len(hit),
                                           "of_which_accepted": sum(x[0] in set(acc) for x in hit), "seeds": [x[0] for x in hit]}
    res["exact_99_over_100"] = {"per_group": counts, "total": sum(c["n_99_100"] for c in counts.values()),
                                "total_evaluated": sum(c["evaluated"] for c in counts.values()),
                                "note": "exact ratio test: 100 * num == 99 * den for the best of the 5 single-unit armies (scripted deployment); candidates failing structural rules or D4-duplicates are not simulated and not counted"}
    res["ledger_after"] = {"entries": len(P.read_ledger()), "build": sum(e["purpose"] == "build" for e in P.read_ledger()), "eval": sum(e["purpose"] == "eval" for e in P.read_ledger()),
                           "unchanged": P.read_ledger() == ledger_before}
    (OUT / "b8_checks.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    sys.exit(main())
