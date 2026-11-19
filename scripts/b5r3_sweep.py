"""B5-R3 exploration sweep (NOT a revision; docs/SPEC_V2.md 3.4 item 10). Same sample / seeds as the R1b and R2 sweeps. Base = revision 1 (live stats file).

  python scripts/b5r3_sweep.py              # every arm (own process each) then the pre-registered selection rule
  python scripts/b5r3_sweep.py --arm R3a    # one arm
  python scripts/b5r3_sweep.py --select     # tables + selection from stored arm files (no simulation)

R3a = Blaster cost 30, hp 75, splash_pct 70       R3b = R3a + splash radius 2.0       R3c = R3a + Blaster range 3.0       R3d = R3b + Scout hp 20
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # the repo root is importable when run as `python scripts/<name>.py`
import json, multiprocessing as mp, os, subprocess, sys, time
from pathlib import Path

import numpy as np

from scripts import b5r1b_sweep as B

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2" / "b5r3_sweep"
LIVE_STATS = ROOT / "data" / "v2" / "unit_stats.json"
R2A_ARM = ROOT / "data" / "v2" / "b5r2_sweep" / "arm_R2a.json"      # revision-1 sweep arm on the same sample (baseline for preference P3)
FAMILIES, FAMILIAR, UNITS, N = B.FAMILIES, B.FAMILIAR, B.UNITS, B.N_PER_FAMILY
TEMPL = ("scout", "sniper", "bulwark", "blaster", "bulwark_mender", "mender_only")
ARMS = {
    "R3a": {"n_changes": 3, "desc": "Blaster cost 30, hp 75, splash 70%", "blaster": {"cost": 30, "hp": 75, "splash_pct": 70}},
    "R3b": {"n_changes": 4, "desc": "R3a + splash radius 2.0", "blaster": {"cost": 30, "hp": 75, "splash_pct": 70, "splash_r_x10": 20}},
    "R3c": {"n_changes": 4, "desc": "R3a + Blaster range 3.0", "blaster": {"cost": 30, "hp": 75, "splash_pct": 70, "range_x10": 30}},
    "R3d": {"n_changes": 5, "desc": "R3b + Scout hp 20", "blaster": {"cost": 30, "hp": 75, "splash_pct": 70, "splash_r_x10": 20}, "scout": {"hp": 20}},
}


def arm_stats(arm):
    s = json.loads(LIVE_STATS.read_text())
    s["units"]["blaster"].update(arm["blaster"])
    if "scout" in arm:
        s["units"]["scout"].update(arm["scout"])
    return s


def sweep_eval3(task):
    from breachlens.acceptance import TEMPLATE_BULWARK_MENDER
    from breachlens.armies import place_on_border
    from breachlens import layouts as Y
    from breachlens.sim import Prepared, simulate
    row = B.sweep_eval(task)
    L = Y.generate(task["family"], task["seed"])
    row["template_bulwark_mender"] = simulate(Prepared(L), place_on_border(L, TEMPLATE_BULWARK_MENDER)).score
    return row


def run_arm(name):
    from breachlens import acceptance as A
    from breachlens import layouts as Y
    from breachlens.config import CFG, sim_version
    assert not os.environ.get("BREACHLENS_GEN_PARAMS"), "generator defaults (revision 1) must apply"
    t0 = time.time()
    pool = mp.get_context("spawn").Pool(min(8, os.cpu_count() or 1))
    taken, recs, rej = set(), {}, {}
    try:
        for fam in B.FAMILIAR:
            r, used, rj = A._walk("T", fam, N, taken, pool); recs[fam], rej[fam] = r, {**rj, "seeds_consumed": used}
        r, used, rj = A._walk("L", "labyrinth", N, taken, pool); recs["labyrinth"], rej["labyrinth"] = r, {**rj, "seeds_consumed": used}
        tasks = [{"id": x["id"], "family": f, "seed": x["seed"]} for f in FAMILIES for x in recs[f]]
        rows = pool.map(sweep_eval3, tasks, chunksize=1)
    finally:
        pool.close(); pool.join()
    return {"arm": name, "sim_version": sim_version(), "gen_params": dict(Y.GEN_PARAMS), "unit_cap": CFG.unit_cap, "seconds": time.time() - t0,
            "rejection": {f: {k: v for k, v in x.items() if k != "stat_rejected_seeds"} for f, x in rej.items()}, "rows": rows}


def niche(rows):
    """G3 niche share per template owner (margin >= 0.03 over the runner-up of 6 templates), pooled + per family."""
    def T(rs):
        return np.array([[x["score"] for x in r["single"][:4]] + [r["template_bulwark_mender"], r["single"][4]["score"]] for r in rs])
    def share(rs):
        M = T(rs); s = np.sort(M, axis=1); ok = (s[:, -1] - s[:, -2]) >= 0.03; b = np.argmax(M, axis=1)
        return {UNITS[u]: float(np.mean(ok & (b == u))) for u in range(5)}
    return {"pooled": share(rows), **{f: share([r for r in rows if r["family"] == f]) for f in FAMILIES}}


def gold(rows, costs):
    out = {}
    for f in FAMILIES:
        R = [r for r in rows if r["family"] == f]
        g = np.array([[r["oracle"]["all"]["counts"][u] * costs[u] for u in range(5)] for r in R], float)
        out[f] = (g.sum(axis=0) / max(g.sum(), 1)).tolist()
    return out


def evaluate(arm_json, costs):
    sm = B.summarize(arm_json["rows"], arm_json["rejection"])
    sm["niche"] = niche(arm_json["rows"]); sm["gold_share"] = gold(arm_json["rows"], costs)
    return sm


def decide(sm, base):
    bf = sm["by_family"]
    bl = lambda u, f: bf[f]["ban_loss_monotone"][u]
    req = {"R1_blaster_ban_loss": any(bl("blaster", f)["mean"] >= 0.02 and bl("blaster", f)["lo95"] > 0 for f in FAMILIES),
           "R2_oracle_in_0.80_0.97": all(0.80 <= bf[f]["oracle_score"] <= 0.97 for f in FAMILIES),
           "R3_best_single_in_0.25_0.65_and_sniper_share_le_0.60": all(0.25 <= bf[f]["best_single_mean"] <= 0.65 for f in FAMILIES) and sm["sniper_share_best_single_pooled_familiar"] <= 0.60}
    keep = all(bl(u, f)["mean"] >= 0.02 for u in ("mender", "bulwark") for f in FAMILIES if base["by_family"][f]["ban_loss_monotone"][u]["mean"] >= 0.02)
    pref = {"P1_blaster_niche_ge_8%": sm["niche"]["pooled"]["blaster"] >= 0.08, "P2_scout_niche_ge_8%": sm["niche"]["pooled"]["scout"] >= 0.08, "P3_mender_bulwark_ban_loss_kept": bool(keep)}
    return req, pref


def main_select():
    arms = {k: json.loads((OUT / f"arm_{k}.json").read_text()) for k in ARMS if (OUT / f"arm_{k}.json").exists()}
    base_arm = json.loads(R2A_ARM.read_text()); base = B.summarize(base_arm["rows"], base_arm["rejection"])
    sums, dec = {}, {}
    for k, a in arms.items():
        st = json.loads((OUT / f"stats_{k}.json").read_text())
        sums[k] = evaluate(a, [st["units"][u]["cost"] for u in UNITS])
        dec[k] = decide(sums[k], base)
    ok = [k for k in sums if all(dec[k][0].values())]
    ranked = sorted(ok, key=lambda k: (-sum(dec[k][1].values()), ARMS[k]["n_changes"], k))
    chosen = ranked[0] if ranked else None
    labels = {k: f"{k}: {ARMS[k]['desc']}" for k in arms}
    md = "# B5-R3 exploration sweep (not a revision)\n" + B.fmt_table(sums, labels)
    lambda b: f"{b['mean']:.3f} [{b['lo95']:.3f}, {b['hi95']:.3f}]"
    hdr = "| arm | " + " | ".join(FAMILIES) + " |\n|---|" + "---|" * 4
    md += "\n\n**Niche share (G3 definition, margin >= 0.03; pooled over the 80-layout sample / per family)**\n\n| arm | unit | pooled | " + " | ".join(FAMILIES) + " |\n|---|---|---|" + "---|" * 4 + "\n"
    for k in sums:
        for u in UNITS:
            md += f"| {k} | {u} | {sums[k]['niche']['pooled'][u]:.1%} | " + " | ".join(f"{sums[k]['niche'][f][u]:.0%}" for f in FAMILIES) + " |\n"
    md += "\n**Gold share in Oracle armies (scout / sniper / bulwark / blaster / mender)**\n\n" + hdr + "\n"
    for k in sums:
        md += f"| {k} | " + " | ".join(" / ".join(f"{x:.2f}" for x in sums[k]["gold_share"][f]) for f in FAMILIES) + " |\n"
    md += "\n**Pooled familiar Sniper best-single share**: " + ", ".join(f"{k} {sums[k]['sniper_share_best_single_pooled_familiar']:.2f}" for k in sums) + "\n"
    md += "\n## Selection (SPEC 3.4 item 10)\n\n| arm | " + " | ".join(list(dec[next(iter(dec))][0]) + list(dec[next(iter(dec))][1])) + " |\n|---|" + "---|" * 6 + "\n"
    for k in sums:
        md += f"| {k} | " + " | ".join("yes" if v else "no" for v in (*dec[k][0].values(), *dec[k][1].values())) + " |\n"
    md += f"\nArms meeting all three required items: {ok}; ranked {ranked}; **chosen: {chosen}**\n"
    (OUT / "sweep_summary.md").write_text(md)
    (OUT / "sweep_summary.json").write_text(json.dumps({"summaries": sums, "required": {k: v[0] for k, v in dec.items()}, "preferences": {k: v[1] for k, v in dec.items()}, "chosen": chosen}, indent=1) + "\n")
    print(md)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--select" in sys.argv:
        return main_select()
    if "--arm" in sys.argv:
        name = sys.argv[sys.argv.index("--arm") + 1]
        res = run_arm(name); res["spec"] = ARMS[name]
        (OUT / f"arm_{name}.json").write_text(json.dumps(res) + "\n")
        print(f"arm {name}: {res['seconds']:.0f}s, rows {len(res['rows'])}")
        return
    for name, arm in ARMS.items():
        env = dict(os.environ)
        sp = OUT / f"stats_{name}.json"; sp.write_text(json.dumps(arm_stats(arm), indent=1) + "\n")
        env["BREACHLENS_STATS_PATH"] = str(sp); env.pop("BREACHLENS_GEN_PARAMS", None)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--arm", name], env=env, check=True, cwd=ROOT)
    main_select()


if __name__ == "__main__":
    main()
