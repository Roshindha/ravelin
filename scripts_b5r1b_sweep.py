"""B5-R1b exploration sweep (NOT a revision; docs/SPEC_V2.md 3.4 item 8). TUNE-train first 20 accepted layouts per familiar family of each arm's own
walk through stream T + first 20 accepted LAB-DESIGN (stream L). TUNE-val and stream F are never touched.

  python scripts_b5r1b_sweep.py               # every arm (own process each, env overrides) then the pre-registered selection rule
  python scripts_b5r1b_sweep.py --arm R1b     # one arm (writes data/v2/b5r1b_sweep/arm_R1b.json)
  python scripts_b5r1b_sweep.py --select      # selection rule + tables from the stored arm files (no simulation)

Arms (cumulative on B5-R1 arm B = building / core / defender hp x1.4 + Lancer share ~40 %):
  R1a = B + Sniper 40 % vs buildings/walls/core (100 % vs turrets)        R1b = R1a + sealed cores (50 %) + core hp x3
  R1c = R1b + unit cap 14                                                 R1d = R1b + Mender heal 14, hp 60
  R0  = revision-0 stats / generator, re-run with this evaluator (reference for the old-vs-monotone ban-loss comparison; not a candidate)
"""
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import scripts_b5r1_sweep as OLD

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "v2" / "b5r1b_sweep"
BASE_STATS = OLD.BASE_STATS
N_PER_FAMILY = 20
FAMILIES, FAMILIAR = OLD.FAMILIES, OLD.FAMILIAR
UNITS = ("scout", "sniper", "bulwark", "blaster", "mender")

ARM_B = OLD.ARMS["B"]
ARMS = {
    "R1a": {"n_changes": 1, "sniper_pct": True, "sealed": 0.0, "core_mult": 1, "cap": None, "mender": False,
            "desc": "B + Sniper 40% vs bldg/wall/core"},
    "R1b": {"n_changes": 3, "sniper_pct": True, "sealed": 0.5, "core_mult": 3, "cap": None, "mender": False, "desc": "R1a + sealed cores 50% + core hp x3"},
    "R1c": {"n_changes": 4, "sniper_pct": True, "sealed": 0.5, "core_mult": 3, "cap": 14, "mender": False, "desc": "R1b + cap 14"},
    "R1d": {"n_changes": 5, "sniper_pct": True, "sealed": 0.5, "core_mult": 3, "cap": None, "mender": True, "desc": "R1b + Mender heal 14, hp 60"},
}
REF = {"R0": {"n_changes": 0, "desc": "R0 (revision 0)"}}


def arm_stats(arm: dict) -> dict:
    s = OLD.arm_stats(ARM_B)                                    # hp x1.4 on buildings / core / defenders
    if arm["sniper_pct"]:
        s["units"]["sniper"]["class_damage_pct"] = {"building": 40, "wall": 40, "turret": 100}   # the core counts as a building
    s["structures"]["core_hp"] = int(s["structures"]["core_hp"] * arm["core_mult"])
    if arm["cap"] is not None:
        s["rules"]["unit_cap"] = int(arm["cap"])
    if arm["mender"]:
        s["units"]["mender"]["heal"] = 14
        s["units"]["mender"]["hp"] = 60
    return s


def arm_gen(arm: dict) -> dict:
    return {"lancer_share": ARM_B["lancer_share"], "defender_count_mult": 1.0, "sealed_fraction": arm["sealed"]}


# ------------------------------------------------------------------ worker: one accepted layout
def sweep_eval(task: dict) -> dict:
    from breachlens import layouts as Y
    from breachlens import oracle as O
    from breachlens.acceptance import warm_starts
    from breachlens.armies import greedy_types, place_on_border, single_unit_types
    from breachlens.sim import Prepared, simulate
    fam, seed = task["family"], task["seed"]
    L = Y.generate(fam, seed)
    P = Prepared(L)
    single = [simulate(P, place_on_border(L, single_unit_types(t))).score for t in range(5)]
    greedy = simulate(P, place_on_border(L, greedy_types())).score
    ws = warm_starts()
    full = O.oracle_search(L, O.oracle_rng(seed), warm_starts=ws, prepared=P)
    oracle = {"all": {"score": full.score, "breach": bool(full.breach), "units": full.units, "counts": [full.types.count(t) for t in range(5)]}}
    indep, mono = {}, {}
    for u in range(5):
        ab = O.oracle_search(L, O.oracle_rng(seed), allowed=tuple(t for t in range(5) if t != u), warm_starts=ws, prepared=P)
        indep[str(u)] = full.score - ab.score
        m = O.monotone_ban_pair(L, seed, u, ab, warm_starts=ws, prepared=P)
        mono[str(u)] = {"full": m["full"].score, "ablated": m["ablated"].score, "loss": m["loss"]}
    types = [d["type"] for d in L["defenders"]]
    return {"id": task["id"], "family": fam, "seed": seed, "single": [{"score": s} for s in single], "greedy": {"score": greedy}, "oracle": oracle,
            "ban_independent": indep, "ban_monotone": mono, "sealed": bool(Y.is_sealed(L)), "n_walls": len(L["walls"]),
            "n_defenders": len(types), "n_lancer": types.count("lancer")}


def run_arm(name: str) -> dict:
    from breachlens import acceptance as A
    from breachlens import layouts as Y
    from breachlens.config import CFG, sim_version
    t0 = time.time()
    pool = mp.get_context("spawn").Pool(min(8, os.cpu_count() or 1))
    taken: set = set()
    recs, rej = {}, {}
    try:
        for fam in FAMILIAR:
            r, used, rj = A._walk("T", fam, N_PER_FAMILY, taken, pool)
            recs[fam], rej[fam] = r, {**rj, "seeds_consumed": used}
        r, used, rj = A._walk("L", "labyrinth", N_PER_FAMILY, taken, pool)
        recs["labyrinth"], rej["labyrinth"] = r, {**rj, "seeds_consumed": used}
        tasks = [{"id": x["id"], "family": f, "seed": x["seed"]} for f in FAMILIES for x in recs[f]]
        rows = pool.map(sweep_eval, tasks, chunksize=1)
    finally:
        pool.close(); pool.join()
    return {"arm": name, "spec": ARMS.get(name), "sim_version": sim_version(), "gen_params": dict(Y.GEN_PARAMS), "unit_cap": CFG.unit_cap,
            "seconds": time.time() - t0, "rejection": {f: {k: v for k, v in x.items() if k != "stat_rejected_seeds"} for f, x in rej.items()}, "rows": rows}


# ------------------------------------------------------------------ summary + pre-registered selection rule
def summarize(rows: list, rejection: dict) -> dict:
    from breachlens import gates as G
    out = {"by_family": {}}
    for f in FAMILIES:
        R = [r for r in rows if r["family"] == f]
        S = np.array([[x["score"] for x in r["single"]] for r in R])
        means = S.mean(axis=0)
        bt = int(np.argmax(means))
        sh = OLD._share_best(S)
        d = {"n": len(R), "oracle_score": float(np.mean([r["oracle"]["all"]["score"] for r in R])),
             "oracle_breach": float(np.mean([r["oracle"]["all"]["breach"] for r in R])),
             "best_single_type": UNITS[bt], "best_single_mean": float(means[bt]), "single_means": means.tolist(),
             "sniper_share_best_single": float(sh[1]), "best_single_share": sh.tolist(),
             "ban_loss_monotone": {UNITS[u]: G.bootstrap_mean([r["ban_monotone"][str(u)]["loss"] for r in R]) for u in range(5)},
             "ban_loss_independent": {UNITS[u]: G.bootstrap_mean([r["ban_independent"][str(u)] for r in R]) for u in range(5)},
             "negative_independent_layouts": {UNITS[u]: int(sum(r["ban_independent"][str(u)] < -1e-12 for r in R)) for u in range(5)},
             "greedy_score": float(np.mean([r["greedy"]["score"] for r in R])),
             "share_oracle_over_12_units": float(np.mean([r["oracle"]["all"]["units"] > 12 for r in R])),
             "oracle_mean_units": float(np.mean([r["oracle"]["all"]["units"] for r in R])),
             "sealed_share": float(np.mean([r["sealed"] for r in R])), "mean_walls": float(np.mean([r["n_walls"] for r in R]))}
        if rejection and f in rejection:
            q = rejection[f]
            d.update({"rejection_rate": q["stat_rejection_rate"], "single_type_trivial": q["single_type_trivial"], "too_hard": q["too_hard"],
                      "stat_evaluated": q["stat_evaluated"], "structural_rejected": q["structural_rejected"]})
        out["by_family"][f] = d
    Sf = np.array([[x["score"] for x in r["single"]] for r in rows if r["family"] in FAMILIAR])
    out["sniper_share_best_single_pooled_familiar"] = float(OLD._share_best(Sf)[1])
    return out


def criteria(sm: dict, families=FAMILIES) -> dict:
    """The five pre-registered criteria (spec 3.4 item 8)."""
    bf = sm["by_family"]
    mender_ok = any(bf[f]["ban_loss_monotone"]["mender"]["mean"] >= 0.02 and bf[f]["ban_loss_monotone"]["mender"]["lo95"] > 0 for f in families)
    return {
        "i_sniper_best_single_share_le_0.60": sm["sniper_share_best_single_pooled_familiar"] <= 0.60,
        "ii_best_single_in_0.25_0.65": all(0.25 <= bf[f]["best_single_mean"] <= 0.65 for f in families),
        "iii_rejection_le_0.40": all(bf[f]["rejection_rate"] <= 0.40 for f in families),
        "iv_oracle_score_in_0.80_0.97": all(0.80 <= bf[f]["oracle_score"] <= 0.97 for f in families),
        "v_mender_monotone_ban_loss": bool(mender_ok),
    }


def select_arm(summaries: dict, specs: dict, families=FAMILIES) -> dict:
    """Most criteria met; ties -> fewest changed parameters, then earlier letter; nothing applied unless >= 4 of 5."""
    res = {k: {"criteria": (c := criteria(sm, families)), "n_met": int(sum(c.values()))} for k, sm in summaries.items()}
    order = sorted(res, key=lambda k: (-res[k]["n_met"], specs[k]["n_changes"], k))
    best = order[0]
    return {"per_arm": res, "ranking": order, "chosen": best if res[best]["n_met"] >= 4 else None, "best_n_met": res[best]["n_met"]}


# ------------------------------------------------------------------ tables
def fmt_table(summaries: dict, labels: dict) -> str:
    L = []
    hdr = "| arm | " + " | ".join(FAMILIES) + " |\n|---|" + "---|" * len(FAMILIES)

    def tab(title, fn):
        L.append(f"\n**{title}**\n\n{hdr}")
        for k, sm in summaries.items():
            L.append(f"| {labels[k]} | " + " | ".join(fn(sm["by_family"][f]) for f in FAMILIES) + " |")

    ci = lambda b: f"{b['mean']:.3f} [{b['lo95']:.3f}, {b['hi95']:.3f}]"
    tab("Oracle score / breach", lambda d: f"{d['oracle_score']:.3f} / {d['oracle_breach']:.2f}")
    tab("Best single-unit army (type, mean score)", lambda d: f"{d['best_single_type']} {d['best_single_mean']:.3f}")
    tab("Sniper share of best-single (per layout, ties split)", lambda d: f"{d['sniper_share_best_single']:.2f}")
    tab("Single-type rejection / stat rejection rate (trivial / too_hard / evaluated)",
        lambda d: f"{d['rejection_rate']:.1%} ({d['single_type_trivial']}/{d['too_hard']}/{d['stat_evaluated']})")
    tab("Greedy score", lambda d: f"{d['greedy_score']:.3f}")
    tab("Share of Oracle armies using > 12 units", lambda d: f"{d['share_oracle_over_12_units']:.2f}")
    tab("Sealed-core share / mean walls", lambda d: f"{d['sealed_share']:.2f} / {d['mean_walls']:.1f}")
    for u in UNITS:
        tab(f"Monotone ban-loss, {u} (mean [95% CI])", lambda d, u=u: ci(d["ban_loss_monotone"][u]))
    return "\n".join(L)


def estimator_comparison(summaries: dict, labels: dict, key="R0") -> str:
    sm = summaries[key]
    L = [f"\n**Ban-loss estimators side by side, {labels[key]}** (independent = two independent searches; monotone = full warm-started from the ablated best; mean [95% CI]; "
         "'neg' = layouts with a negative independent ban-loss)\n", "| unit / family | " + " | ".join(FAMILIES) + " |\n|---|" + "---|" * len(FAMILIES)]
    for u in UNITS:
        for kind, nm in (("ban_loss_independent", "independent"), ("ban_loss_monotone", "monotone")):
            cells = []
            for f in FAMILIES:
                b = sm["by_family"][f][kind][u]
                cells.append(f"{b['mean']:.3f} [{b['lo95']:.3f}, {b['hi95']:.3f}]" + (f" neg {sm['by_family'][f]['negative_independent_layouts'][u]}" if kind == "ban_loss_independent" else ""))
            L.append(f"| {u} {nm} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def main_select():
    arms = {k: json.loads((OUT / f"arm_{k}.json").read_text()) for k in (*REF, *ARMS) if (OUT / f"arm_{k}.json").exists()}
    summaries = {k: summarize(a["rows"], a["rejection"]) for k, a in arms.items()}
    specs = {**REF, **ARMS}
    labels = {k: f"{k}: {specs[k]['desc']}" for k in arms}
    cand = {k: v for k, v in summaries.items() if k in ARMS}
    sel = select_arm(cand, ARMS)
    sel3 = select_arm(cand, ARMS, families=FAMILIAR)
    ref_c = {k: criteria(summaries[k]) for k in REF if k in summaries}
    report = {"summaries": summaries, "selection": sel, "selection_sensitivity_3_familiar_families_only": sel3, "reference_criteria": ref_c}
    (OUT / "sweep_summary.json").write_text(json.dumps(report, indent=1) + "\n")
    md = "# B5-R1b exploration sweep (not a revision)\n" + fmt_table(summaries, labels) + "\n" + (estimator_comparison(summaries, labels) if "R0" in summaries else "")
    md += "\n\n## Selection rule (4 families)\n\n| arm | " + " | ".join(next(iter(sel["per_arm"].values()))["criteria"]) + " | met |\n|---|" + "---|" * 6 + "\n"
    for k in summaries:
        c = ref_c[k] if k in REF else sel["per_arm"][k]["criteria"]
        md += f"| {labels[k]} | " + " | ".join("yes" if v else "no" for v in c.values()) + f" | {sum(c.values())}/5 |\n"
    md += (f"\nRanking: {sel['ranking']}; chosen: **{sel['chosen']}** (best meets {sel['best_n_met']}/5). "
           f"Sensitivity, 3 familiar families only: chosen {sel3['chosen']} ({sel3['best_n_met']}/5), ranking {sel3['ranking']}.\n")
    (OUT / "sweep_summary.md").write_text(md)
    print(md)
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--select" in sys.argv:
        return main_select()
    if "--arm" in sys.argv:
        name = sys.argv[sys.argv.index("--arm") + 1]
        res = run_arm(name)
        (OUT / f"arm_{name}.json").write_text(json.dumps(res) + "\n")
        print(f"arm {name}: {res['seconds']:.0f}s, rows {len(res['rows'])}")
        return
    for name in ("R0", *ARMS):
        env = dict(os.environ)
        for k in ("BREACHLENS_STATS_PATH", "BREACHLENS_GEN_PARAMS"):
            env.pop(k, None)
        if name in ARMS:
            sp = OUT / f"stats_{name}.json"
            sp.write_text(json.dumps(arm_stats(ARMS[name]), indent=1) + "\n")
            env.update(BREACHLENS_STATS_PATH=str(sp), BREACHLENS_GEN_PARAMS=json.dumps(arm_gen(ARMS[name])))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--arm", name], env=env, check=True, cwd=ROOT)
    main_select()


if __name__ == "__main__":
    main()
