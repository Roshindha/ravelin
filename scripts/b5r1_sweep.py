"""B5-R1 exploration sweep (NOT a revision). TUNE-train only (first 20 accepted layouts per familiar family of each arm's own walk
through stream T) + the first 20 accepted LAB-DESIGN layouts (stream L). TUNE-val is never touched; stream F / FINAL never touched.

  python scripts/b5r1_sweep.py              # run every arm (each in its own process, env overrides), then the selection rule
  python scripts/b5r1_sweep.py --arm B      # one arm (writes data/v2/b5r1_sweep/arm_B.json)
  python scripts/b5r1_sweep.py --baseline   # revision-0 reference rows from the stored B5 data (no simulation)
  python scripts/b5r1_sweep.py --select     # re-run the pre-registered selection rule on the stored arm files (no simulation)

Arms change ONLY what is listed (unit attack stats and defender damage are never touched; walls are never touched):
  A: building / core / defender hp x1.4                      B: A + Lancer share ~40 % of defenders (Guns stay >= 50 % in familiar families)
  C: B + Sniper cost 24 -> 28                                D: defender count x1.25 + Lancer share ~40 %, hp unchanged
  E: building / core / defender hp x1.8 + Lancer share ~40 %
Selection rule (docs/SPEC_V2.md 3.4 item 7, written before any sweep result existed) is `select_arm` below.
The overrides are read by breachlens.config (BREACHLENS_STATS_PATH) and breachlens.layouts (BREACHLENS_GEN_PARAMS); the real gate run asserts neither is set.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # the repo root is importable when run as `python scripts/<name>.py`
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2" / "b5r1_sweep"
BASE_STATS = OUT / "stats_rev0.json"          # a verbatim copy of data/v2/unit_stats.json at revision 0
N_PER_FAMILY = 20
FAMILIES = ("spine_pods", "archipelago", "bunker_nest", "labyrinth")
FAMILIAR = FAMILIES[:3]

ARMS = {
    "A": {"hp": 1.4, "lancer_share": None, "count": 1.0, "sniper_cost": None, "n_changes": 1, "desc": "building/core/defender hp x1.4"},
    "B": {"hp": 1.4, "lancer_share": 0.4, "count": 1.0, "sniper_cost": None, "n_changes": 2, "desc": "A + Lancer share ~40 %"},
    "C": {"hp": 1.4, "lancer_share": 0.4, "count": 1.0, "sniper_cost": 28, "n_changes": 3, "desc": "B + Sniper cost 28"},
    "D": {"hp": 1.0, "lancer_share": 0.4, "count": 1.25, "sniper_cost": None, "n_changes": 2, "desc": "defender count x1.25 + Lancer share ~40 %"},
    "E": {"hp": 1.8, "lancer_share": 0.4, "count": 1.0, "sniper_cost": None, "n_changes": 2, "desc": "building/core/defender hp x1.8 + Lancer share ~40 %"},
}


# ------------------------------------------------------------------ arm -> stats file / generator params
def arm_stats(arm: dict) -> dict:
    s = json.loads(BASE_STATS.read_text())
    m = arm["hp"]
    if m != 1.0:
        s["structures"]["building_hp_default"] = int(round(s["structures"]["building_hp_default"] * m))
        s["structures"]["core_hp"] = int(round(s["structures"]["core_hp"] * m))
        for d in s["defenders"].values():
            d["hp"] = int(round(d["hp"] * m))        # walls (structures.wall_hp) and every damage / range / cost field stay untouched
    if arm["sniper_cost"] is not None:
        s["units"]["sniper"]["cost"] = int(arm["sniper_cost"])
    return s


def arm_gen(arm: dict) -> dict:
    return {"lancer_share": arm["lancer_share"], "defender_count_mult": arm["count"]}


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
    oracle = {}
    for key, allowed in (("all", tuple(range(5))), ("ban4", (0, 1, 2, 3))):    # ban4 = Oracle without Menders
        r = O.oracle_search(L, O.oracle_rng(seed), allowed=allowed, warm_starts=ws, prepared=P)
        oracle[key] = {"score": r.score, "breach": bool(r.breach), "units": r.units, "counts": [r.types.count(t) for t in range(5)]}
    types = [d["type"] for d in L["defenders"]]
    return {"id": task["id"], "family": fam, "seed": seed, "single": [{"score": s} for s in single], "greedy": {"score": greedy}, "oracle": oracle,
            "n_defenders": len(types), "n_lancer": types.count("lancer"), "n_gun": types.count("gun"), "n_flak": types.count("flak")}


def run_arm(name: str) -> dict:
    from breachlens import acceptance as A
    from breachlens import layouts as Y
    from breachlens.config import sim_version
    assert Y.stream_of(Y.stream_range("T", "spine_pods")[0]) == "T"
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
    return {"arm": name, "spec": ARMS.get(name), "sim_version": sim_version(), "gen_params": dict(Y.GEN_PARAMS), "seconds": time.time() - t0,
            "rejection": {f: {k: v for k, v in x.items() if k != "stat_rejected_seeds"} for f, x in rej.items()}, "rows": rows}


# ------------------------------------------------------------------ summary + pre-registered selection rule
def _share_best(S):
    sh = np.zeros(5)
    for s in S:
        b = np.flatnonzero(s == s.max())
        sh[b] += 1 / len(b)
    return sh / len(S)


def summarize(rows: list, rejection: dict) -> dict:
    """rows: dicts with family, single[5].score, greedy.score, oracle.all/ban4 {score, breach, units, counts} (+ optional defender counts)."""
    from breachlens import gates as G
    out = {"by_family": {}}
    for f in FAMILIES:
        R = [r for r in rows if r["family"] == f]
        S = np.array([[x["score"] for x in r["single"]] for r in R])
        means = S.mean(axis=0)
        bt = int(np.argmax(means))
        loss = np.array([r["oracle"]["all"]["score"] - r["oracle"]["ban4"]["score"] for r in R])
        d = {"n": len(R), "oracle_score": float(np.mean([r["oracle"]["all"]["score"] for r in R])),
             "oracle_breach": float(np.mean([r["oracle"]["all"]["breach"] for r in R])),
             "best_single_type": ["scout", "sniper", "bulwark", "blaster", "mender"][bt], "best_single_mean": float(means[bt]),
             "single_means": means.tolist(), "sniper_share_best_single": float(_share_best(S)[1]), "best_single_share": _share_best(S).tolist(),
             "mender_ban_loss": G.bootstrap_mean(loss), "greedy_score": float(np.mean([r["greedy"]["score"] for r in R])),
             "share_oracle_over_12_units": float(np.mean([r["oracle"]["all"]["units"] > 12 for r in R])),
             "oracle_mean_units": float(np.mean([r["oracle"]["all"]["units"] for r in R])),
             "oracle_mean_counts": np.mean([r["oracle"]["all"]["counts"] for r in R], axis=0).tolist()}
        if "n_defenders" in R[0]:
            nd = sum(r["n_defenders"] for r in R)
            d.update({"mean_defenders": nd / len(R), "lancer_share": sum(r["n_lancer"] for r in R) / nd, "gun_share": sum(r["n_gun"] for r in R) / nd,
                      "flak_share": sum(r["n_flak"] for r in R) / nd})
        if rejection and f in rejection:
            q = rejection[f]
            d.update({"rejection_rate": q["stat_rejection_rate"], "single_type_trivial": q["single_type_trivial"], "too_hard": q["too_hard"],
                      "stat_evaluated": q["stat_evaluated"]})
        out["by_family"][f] = d
    Sf = np.array([[x["score"] for x in r["single"]] for r in rows if r["family"] in FAMILIAR])
    out["sniper_share_best_single_pooled_familiar"] = float(_share_best(Sf)[1])
    return out


def criteria(sm: dict, families=FAMILIES) -> dict:
    """The five pre-registered criteria (spec 3.4 item 7). `families` = the families that count for the 'each family' criteria."""
    bf = sm["by_family"]
    c = {
        "1_oracle_not_saturated": all(bf[f]["oracle_breach"] <= 0.95 and bf[f]["oracle_score"] <= 0.97 for f in families),
        "2_best_single_in_0.25_0.65": all(0.25 <= bf[f]["best_single_mean"] <= 0.65 for f in families),
        "3_sniper_best_single_share_le_0.60": sm["sniper_share_best_single_pooled_familiar"] <= 0.60,
        "4_rejection_rate_le_0.40": all(bf[f]["rejection_rate"] <= 0.40 for f in families),
        "5_greedy_labyrinth_le_0.45": bf["labyrinth"]["greedy_score"] <= 0.45,
    }
    return c


def select_arm(summaries: dict, specs: dict, families=FAMILIES) -> dict:
    """Most criteria met; ties -> fewest changed parameters, then earlier letter. None applied unless >= 4 of 5."""
    res = {}
    for k, sm in summaries.items():
        c = criteria(sm, families)
        res[k] = {"criteria": c, "n_met": int(sum(c.values()))}
    order = sorted(res, key=lambda k: (-res[k]["n_met"], specs[k]["n_changes"], k))
    best = order[0]
    return {"per_arm": res, "ranking": order, "chosen": best if res[best]["n_met"] >= 4 else None, "best_n_met": res[best]["n_met"]}


def build_baseline():
    """Revision-0 reference from the stored B5 data (first 20 accepted per family of the same seed walk); no simulation. Writes baseline_rev0.json."""
    from breachlens import layouts as Y
    meta = json.loads((ROOT / "data" / "v2" / "design_sets.json").read_text())
    raw = {r["id"]: r for r in json.loads((ROOT / "data" / "v2" / "b5_raw.json").read_text())["layouts"]}
    rows, rejection = [], {}
    for fam in FAMILIES:
        recs = [r for r in meta["sets"]["LAB-DESIGN" if fam == "labyrinth" else "TUNE-train"] if r["family"] == fam][:N_PER_FAMILY]
        last = max(r["seed"] for r in recs)
        sa = meta["stat_acceptance"][fam]
        rb = sum(1 for sd in sa["stat_rejected_seeds"] if sd < last)
        rejection[fam] = {"stat_evaluated": N_PER_FAMILY + rb, "stat_rejected": rb, "stat_rejection_rate": rb / (N_PER_FAMILY + rb),
                          "single_type_trivial": rb if sa["too_hard"] == 0 else None, "too_hard": 0 if sa["too_hard"] == 0 else None}
        for r in recs:
            x = raw[r["id"]]
            types = [d["type"] for d in Y.generate(fam, r["seed"])["defenders"]]
            rows.append({"id": r["id"], "family": fam, "seed": r["seed"], "single": [{"score": q["score"]} for q in x["single"]], "greedy": {"score": x["greedy"]["score"]},
                         "oracle": {k: {"score": x["oracle"][k]["score"], "breach": x["oracle"][k]["breach"], "units": x["oracle"][k]["units"], "counts": x["oracle"][k]["counts"]}
                                    for k in ("all", "ban4")},
                         "n_defenders": len(types), "n_lancer": types.count("lancer"), "n_gun": types.count("gun"), "n_flak": types.count("flak")})
    (OUT / "baseline_rev0.json").write_text(json.dumps({"rows": rows, "rejection": rejection}) + "\n")
    return rows


# ------------------------------------------------------------------ table output
def fmt_table(summaries: dict, labels: dict) -> str:
    L = []
    hdr = "| arm | " + " | ".join(FAMILIES) + " |\n|---|" + "---|" * len(FAMILIES)

    def tab(title, fn):
        L.append(f"\n**{title}**\n\n{hdr}")
        for k, sm in summaries.items():
            L.append(f"| {labels[k]} | " + " | ".join(fn(sm["by_family"][f]) for f in FAMILIES) + " |")
    tab("Oracle score / breach", lambda d: f"{d['oracle_score']:.3f} / {d['oracle_breach']:.2f}")
    tab("Best single-unit army (type, mean score)", lambda d: f"{d['best_single_type']} {d['best_single_mean']:.3f}")
    tab("Sniper share of best-single (per layout, ties split)", lambda d: f"{d['sniper_share_best_single']:.2f}")
    tab("Mender Oracle ban-loss (paired mean [95% CI])", lambda d: f"{d['mender_ban_loss']['mean']:.3f} [{d['mender_ban_loss']['lo95']:.3f}, {d['mender_ban_loss']['hi95']:.3f}]")
    tab("single_type_trivial rejection rate (any stat rejection; trivial / too_hard / evaluated)",
        lambda d: f"{d['rejection_rate']:.1%} ({d['single_type_trivial']}/{d['too_hard']}/{d['stat_evaluated']})" if "rejection_rate" in d else "n/a")
    tab("Greedy score", lambda d: f"{d['greedy_score']:.3f}")
    tab("Oracle armies using > 12 units", lambda d: f"{d['share_oracle_over_12_units']:.2f}")
    tab("Defenders per layout / Lancer share", lambda d: f"{d['mean_defenders']:.1f} / {d['lancer_share']:.0%}" if "mean_defenders" in d else "n/a")
    L.append("\nSniper best-single share pooled over the 3 familiar families: " + ", ".join(f"{labels[k]} {sm['sniper_share_best_single_pooled_familiar']:.2f}" for k, sm in summaries.items()))
    return "\n".join(L)


def main_select():
    arms = {k: json.loads((OUT / f"arm_{k}.json").read_text()) for k in ARMS if (OUT / f"arm_{k}.json").exists()}
    summaries = {k: summarize(a["rows"], a["rejection"]) for k, a in arms.items()}
    base = json.loads((OUT / "baseline_rev0.json").read_text())
    summaries_all = {"R0": summarize(base["rows"], base["rejection"]), **summaries}
    labels = {"R0": "R0 (reference)", **{k: f"{k}: {ARMS[k]['desc']}" for k in arms}}
    sel = select_arm(summaries, ARMS)
    sel3 = select_arm(summaries, ARMS, families=FAMILIAR)
    report = {"summaries": summaries_all, "selection": sel, "selection_sensitivity_3_familiar_families_only": sel3,
              "criteria_R0_reference": criteria(summaries_all["R0"])}
    (OUT / "sweep_summary.json").write_text(json.dumps(report, indent=1) + "\n")
    md = "# B5-R1 exploration sweep (not a revision)\n" + fmt_table(summaries_all, labels) + "\n\n## Selection rule (4 families)\n\n| arm | " + " | ".join(
        next(iter(sel["per_arm"].values()))["criteria"]) + " | met |\n|---|" + "---|" * 6 + "\n"
    for k in ["R0", *summaries]:
        c = criteria(summaries_all[k]) if k == "R0" else sel["per_arm"][k]["criteria"]
        md += f"| {labels[k]} | " + " | ".join("yes" if v else "no" for v in c.values()) + f" | {sum(c.values())}/5 |\n"
    md += f"\nRanking: {sel['ranking']}; chosen: **{sel['chosen']}** (best meets {sel['best_n_met']}/5). Sensitivity, 3 familiar families only: chosen {sel3['chosen']} ({sel3['best_n_met']}/5), ranking {sel3['ranking']}.\n"
    (OUT / "sweep_summary.md").write_text(md)
    print(md)
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--baseline" in sys.argv:
        build_baseline(); return
    if "--select" in sys.argv:
        return main_select()
    if "--arm" in sys.argv:
        name = sys.argv[sys.argv.index("--arm") + 1]
        res = run_arm(name)
        if name in ARMS:
            (OUT / f"arm_{name}.json").write_text(json.dumps(res) + "\n")
        print(f"arm {name}: {res['seconds']:.0f}s, rows {len(res['rows'])}")
        return
    build_baseline()
    for name, arm in ARMS.items():
        sp = OUT / f"stats_{name}.json"
        sp.write_text(json.dumps(arm_stats(arm), indent=1) + "\n")
        env = dict(os.environ, BREACHLENS_STATS_PATH=str(sp), BREACHLENS_GEN_PARAMS=json.dumps(arm_gen(arm)))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--arm", name], env=env, check=True, cwd=ROOT)
    main_select()


if __name__ == "__main__":
    main()
