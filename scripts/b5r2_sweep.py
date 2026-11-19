"""B5-R2 exploration sweep (NOT a revision; docs/SPEC_V2.md 3.4 item 9). Same sample/seeds as the R1b sweep.

  python scripts/b5r2_sweep.py              # every arm (own process each) then the pre-registered selection rule
  python scripts/b5r2_sweep.py --arm R2a    # one arm
  python scripts/b5r2_sweep.py --select     # selection from stored arm files

Base = R1b. R2a = Sniper cost 28; R2b = Sniper cost 30; R2c = R2a + archipelago island_gap 6; R2d = R2a + Mender heal 10, hp 50.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # the repo root is importable when run as `python scripts/<name>.py`
import json, os, subprocess, sys
from pathlib import Path

import numpy as np

from scripts import b5r1b_sweep as B

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2" / "b5r2_sweep"
FAMILIES, FAMILIAR, UNITS = B.FAMILIES, B.FAMILIAR, B.UNITS
ARMS = {
    "R2a": {"n_changes": 1, "sniper_cost": 28, "gap": 4, "mender": None, "desc": "R1b + Sniper cost 28"},
    "R2b": {"n_changes": 1, "sniper_cost": 30, "gap": 4, "mender": None, "desc": "R1b + Sniper cost 30"},
    "R2c": {"n_changes": 2, "sniper_cost": 28, "gap": 6, "mender": None, "desc": "R2a + island gap >= 6"},
    "R2d": {"n_changes": 2, "sniper_cost": 28, "gap": 4, "mender": (10, 50), "desc": "R2a + Mender heal 10, hp 50"},
}
REF = {"R1b": {"n_changes": 0, "desc": "R1b (base, stored sweep)"}}


def arm_stats(arm):
    s = B.arm_stats(B.ARMS["R1b"])
    s["units"]["sniper"]["cost"] = arm["sniper_cost"]
    if arm["mender"]:
        s["units"]["mender"]["heal"], s["units"]["mender"]["hp"] = arm["mender"]
    return s


def arm_gen(arm):
    g = B.arm_gen(B.ARMS["R1b"]); g["island_gap"] = arm["gap"]
    return g


def sniper_gold_share(rows, costs, fam):
    R = [r for r in rows if r["family"] == fam]
    gold = np.array([[r["oracle"]["all"]["counts"][u] * costs[u] for u in range(5)] for r in R], float)
    return float(gold[:, 1].sum() / max(gold.sum(), 1)), float(np.mean(gold[:, 1] / np.maximum(gold.sum(axis=1), 1)))


def criteria(sm, families=FAMILIES):
    bf = sm["by_family"]
    return {
        "i_sniper_best_single_share_le_0.60": sm["sniper_share_best_single_pooled_familiar"] <= 0.60,
        "ii_best_single_in_0.25_0.65": all(0.25 <= bf[f]["best_single_mean"] <= 0.65 for f in families),
        "iii_rejection_le_0.65": all(bf[f]["rejection_rate"] <= 0.65 for f in families),
        "iv_oracle_score_in_0.80_0.97": all(0.80 <= bf[f]["oracle_score"] <= 0.97 for f in families),
        "v_mender_monotone_ban_loss": any(bf[f]["ban_loss_monotone"]["mender"]["mean"] >= 0.02 and bf[f]["ban_loss_monotone"]["mender"]["lo95"] > 0 for f in families),
    }


def select(summaries):
    res = {k: {"criteria": (c := criteria(sm)), "n_met": int(sum(c.values()))} for k, sm in summaries.items()}
    order = sorted(res, key=lambda k: (-res[k]["n_met"], ARMS[k]["n_changes"], k))
    best = order[0]
    c = res[best]["criteria"]
    ok = c["ii_best_single_in_0.25_0.65"] and c["iv_oracle_score_in_0.80_0.97"] and res[best]["n_met"] >= 4
    return {"per_arm": res, "ranking": order, "best": best, "chosen": best if ok else None}


def main_select():
    arms = {k: json.loads((OUT / f"arm_{k}.json").read_text()) for k in (*ARMS,) if (OUT / f"arm_{k}.json").exists()}
    summaries = {k: B.summarize(a["rows"], a["rejection"]) for k, a in arms.items()}
    labels = {k: f"{k}: {ARMS[k]['desc']}" for k in arms}
    sel = select(summaries)
    gold = {}
    for k in arms:
        st = json.loads((OUT / f"stats_{k}.json").read_text())
        costs = [st["units"][u]["cost"] for u in UNITS]
        gold[k] = {f: sniper_gold_share(arms[k]["rows"], costs, f) for f in FAMILIES}
    (OUT / "sweep_summary.json").write_text(json.dumps({"summaries": summaries, "selection": sel, "sniper_gold_share": gold}, indent=1) + "\n")
    md = "# B5-R2 exploration sweep (not a revision)\n" + B.fmt_table(summaries, labels)
    md += "\n\n**Sniper share of gold in Oracle armies (pooled gold / mean per-layout)**\n\n| arm | " + " | ".join(FAMILIES) + " |\n|---|" + "---|" * 4 + "\n"
    for k in arms:
        md += f"| {labels[k]} | " + " | ".join(f"{gold[k][f][0]:.2f} / {gold[k][f][1]:.2f}" for f in FAMILIES) + " |\n"
    md += "\n**Pooled Sniper best-single share (familiar)**: " + ", ".join(f"{k} {summaries[k]['sniper_share_best_single_pooled_familiar']:.2f}" for k in arms) + "\n"
    md += "\n\n## Selection rule\n\n| arm | " + " | ".join(next(iter(sel["per_arm"].values()))["criteria"]) + " | met |\n|---|" + "---|" * 6 + "\n"
    for k in arms:
        c = sel["per_arm"][k]["criteria"]
        md += f"| {labels[k]} | " + " | ".join("yes" if v else "no" for v in c.values()) + f" | {sum(c.values())}/5 |\n"
    md += f"\nRanking: {sel['ranking']}; best: {sel['best']}; chosen (needs (ii), (iv), >= 4/5): **{sel['chosen']}**\n"
    (OUT / "sweep_summary.md").write_text(md)
    print(md)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--select" in sys.argv:
        return main_select()
    if "--arm" in sys.argv:
        name = sys.argv[sys.argv.index("--arm") + 1]
        res = B.run_arm(name)
        res["spec"] = ARMS[name]
        (OUT / f"arm_{name}.json").write_text(json.dumps(res) + "\n")
        print(f"arm {name}: {res['seconds']:.0f}s, rows {len(res['rows'])}")
        return
    for name, arm in ARMS.items():
        env = dict(os.environ)
        sp = OUT / f"stats_{name}.json"
        sp.write_text(json.dumps(arm_stats(arm), indent=1) + "\n")
        env.update(BREACHLENS_STATS_PATH=str(sp), BREACHLENS_GEN_PARAMS=json.dumps(arm_gen(arm)))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--arm", name], env=env, check=True, cwd=ROOT)
    main_select()


if __name__ == "__main__":
    main()
