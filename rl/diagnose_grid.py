"""STEP 9a: diagnostics on the existing grid results. No training, no model evaluation, no tuning.

Reads results/final_eval.csv, results/final_eval_tables/baselines_per_layout.csv, models/*/val_eval.csv.
Only fresh computation: fixed-single-troop scores on each group (pure simulator, for the hindsight reference).

    .venv/bin/python -m rl.diagnose_grid
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "diagnostics"
MODELS = ROOT / "models"
GROUPS = ["DEV-train", "validation", "FINAL-familiar", "transfer"]
TROOPS = ["swarm", "ranger", "tank", "splash"]
COST = {"swarm": 10, "ranger": 25, "tank": 30, "splash": 40}
RESUMED = {13, 14, 15}
SEEDS = list(range(10, 20))
EXPECT_TRAIN, EXPECT_EVAL, EXPECT_SHA = "6c2a800", "6fe8366", "e97cfbc7"


def load_inputs():
    fe = pd.read_csv(RES / "final_eval.csv")
    bl = pd.read_csv(RES / "final_eval_tables" / "baselines_per_layout.csv")
    return fe, bl


def per_model_group(fe: pd.DataFrame) -> pd.DataFrame:
    return fe.groupby(["run", "aug", "seed", "group"]).destruction.mean().reset_index()


def hindsight_best_single(bl: pd.DataFrame) -> pd.DataFrame:
    """NON-DEPLOYABLE: per (group, family) the best fixed single troop picked on that very group."""
    from rl.eval_all import load_group_records
    from siege.baselines import _layout_for_record, best_single_agent_score

    recs = load_group_records(tuple(GROUPS))
    rows = []
    for r in recs:
        lay = _layout_for_record(r)
        for t in TROOPS:
            rows.append({"layout_id": r["id"], "family": r["family"], "group": r["group"], "troop": t,
                         "score": best_single_agent_score(lay, t)})
    d = pd.DataFrame(rows)
    m = d.groupby(["group", "family", "troop"]).score.mean().reset_index()
    best = m.loc[m.groupby(["group", "family"]).score.idxmax()].rename(columns={"score": "mean_destruction"})
    return best, m


def table1(fe, bl, best):
    pm = per_model_group(fe)
    rows = []
    for g in GROUPS:
        b = bl[bl.group == g]
        n = len(b)
        hind = float(np.average(best[best.group == g].mean_destruction, weights=b.groupby("family").size().reindex(best[best.group == g].family).values))
        rows.append({"group": g, "n_layouts": n, "Random": b.Random.mean(), "Greedy": b.Greedy.mean(),
                     "BestSingle_DEVchosen": b.BestSingle.mean(), "PPO_plain": pm[(pm.group == g) & (pm.aug == "none")].destruction.mean(),
                     "PPO_D4": pm[(pm.group == g) & (pm.aug == "d4")].destruction.mean(),
                     "NONDEPLOYABLE_hindsight_best_single": hind,
                     "Oracle": b.Oracle.mean() if b.Oracle.notna().any() else np.nan})
    t = pd.DataFrame(rows)
    # per family version
    rf = []
    for g in GROUPS:
        for fam in sorted(bl[bl.group == g].family.unique()):
            b = bl[(bl.group == g) & (bl.family == fam)]
            f = fe[(fe.group == g) & (fe.family == fam)].groupby(["aug", "seed"]).destruction.mean().groupby("aug").mean()
            h = best[(best.group == g) & (best.family == fam)].iloc[0]
            rf.append({"group": g, "family": fam, "n_layouts": len(b), "Random": b.Random.mean(), "Greedy": b.Greedy.mean(),
                       "BestSingle_DEVchosen": b.BestSingle.mean(), "DEVchosen_troop": b.BestSingle_troop.iloc[0],
                       "PPO_plain": f["none"], "PPO_D4": f["d4"], "NONDEPLOYABLE_hindsight_best_single": h.mean_destruction,
                       "hindsight_troop": h.troop, "Oracle": b.Oracle.mean() if b.Oracle.notna().any() else np.nan})
    return t, pd.DataFrame(rf)


def learning_curves():
    rows = []
    for aug in ("none", "d4"):
        for s in SEEDS:
            v = pd.read_csv(MODELS / f"ppo_{aug}_s{s}" / "val_eval.csv")
            v = v[v["mode"] == "full"]
            rows.append(v[["timestep", "overall"]].assign(aug=aug, seed=s))
    lc = pd.concat(rows, ignore_index=True)
    lc.to_csv(OUT / "val_curves_long.csv", index=False)
    return lc


def curve_stats(lc: pd.DataFrame) -> pd.DataFrame:
    out = []
    for aug, g in lc.groupby("aug"):
        piv = g.pivot(index="timestep", columns="seed", values="overall")
        mean = piv.mean(axis=1)
        t_end = piv.index.max()
        t0 = piv.index[np.argmin(np.abs(piv.index - 750_000))]
        last = piv[piv.index >= t_end - 50_000].mean(axis=0)  # last ~3 evals per seed
        mid = piv[(piv.index >= t0 - 25_000) & (piv.index <= t0 + 25_000)].mean(axis=0)
        slope = np.polyfit(piv.index[piv.index >= 750_000] / 1e6, mean[piv.index >= 750_000], 1)[0]
        out.append({"aug": aug, "t_end": int(t_end), "mean_val_at_750k_window": mid.mean(), "mean_val_last_window": last.mean(),
                    "mean_change_last_250k": (last - mid).mean(), "std_of_per_seed_change": (last - mid).std(ddof=1),
                    "seeds_rising": int(((last - mid) > 0).sum()), "n_seeds": piv.shape[1],
                    "slope_per_1M_steps_over_last_250k": slope, "peak_of_mean_curve": mean.max(), "peak_t": int(mean.idxmax())})
    return pd.DataFrame(out)


def plot_curves(lc: pd.DataFrame, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"none": "#1f77b4", "d4": "#e07b00"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for aug, g in lc.groupby("aug"):
        piv = g.pivot(index="timestep", columns="seed", values="overall")
        ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), color=col[aug], alpha=.18)
        ax.plot(piv.index, piv.mean(axis=1), color=col[aug], lw=2, label={"none": "PPO (no aug)", "d4": "PPO + D4"}[aug] + " mean of 10, min-max band")
    ax.axvline(333_333, color="gray", ls=":", lw=1); ax.text(340_000, ax.get_ylim()[0] + .01, "curriculum phase B", fontsize=7, color="gray")
    ax.axvline(750_000, color="k", ls="--", lw=.8)
    ax.set_xlabel("timesteps"); ax.set_ylabel("validation destruction (20 layouts, full task)")
    ax.set_title("Validation curves (validation = seen via checkpoint selection logging)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def per_seed_table(fe, lc):
    pm = per_model_group(fe)
    head = fe[fe.group.isin(["FINAL-familiar", "transfer"])].groupby(["run", "aug", "seed"]).destruction.mean().rename("FINAL_headline")
    p = pm.pivot_table(index=["run", "aug", "seed"], columns="group", values="destruction").rename(
        columns={"validation": "validation_final_zip", "DEV-train": "DEV_train"})[["validation_final_zip", "DEV_train"]]
    last = lc.sort_values("timestep").groupby(["aug", "seed"]).overall.last().rename("validation_last_logged_eval")
    t = p.join(head).reset_index().merge(last.reset_index(), on=["aug", "seed"])
    t["resumed_twice"] = t.seed.isin(RESUMED)
    return t.sort_values(["aug", "seed"])


def army_table(fe, bl):
    from siege.baselines import _greedy_army

    gr = _greedy_army({})
    gcount = {t: gr.count(t) for t in TROOPS}
    ou = pd.read_csv(RES / "oracle_usage.csv")
    rows = []
    for g in GROUPS:
        for fam in sorted(fe[fe.group == g].family.unique()):
            sub = fe[(fe.group == g) & (fe.family == fam)]
            for aug, lab in (("none", "PPO_plain"), ("d4", "PPO_D4")):
                m = sub[sub.aug == aug][[f"army_{t}" for t in TROOPS]].mean()
                rows.append({"group": g, "family": fam, "agent": lab, **{t: m[f"army_{t}"] for t in TROOPS}})
            rows.append({"group": g, "family": fam, "agent": "Greedy", **gcount})
    t = pd.DataFrame(rows)
    t["gold_total"] = sum(t[k] * COST[k] for k in TROOPS)
    for k in TROOPS:
        t[f"goldshare_{k}"] = t[k] * COST[k] / t.gold_total
    # oracle: only gold shares exist (results/oracle_usage.csv, step 4: DEV familiar 80/family and transfer 41)
    o = ou.pivot(index="family", columns="troop", values="gold_share")
    orows = [{"group": "DEV(train+val, step 4)" if f != "ring_defence" else "transfer", "family": f, "agent": "Oracle (gold share only; no counts stored)",
              **{f"goldshare_{k}": o.loc[f, k] for k in TROOPS}} for f in o.index]
    return t, pd.DataFrame(orows)


def provenance(fe):
    s = json.loads((RES / "final_eval_summary.json").read_text())
    ids = pd.DataFrame(s["models"])
    out = {
        "summary_json_top_level_commit_(eval)": s["commit"],
        "summary_json_layouts_sha256": s["layouts_sha256"],
        "training_commits_in_summary_models": sorted(ids.train_git_commit.unique()),
        "training_dirty_flags": sorted(ids.train_git_dirty.unique().tolist()),
        "eval_commits_in_summary_models": sorted(ids.eval_git_commit.unique()),
        "layouts_sha_train_all_match": bool((ids.layouts_sha256_train == s["layouts_sha256"]).all()),
        "final_eval_csv_commits": sorted(fe.commit.unique()),
        "final_eval_csv_layouts_sha": sorted(fe.layouts_sha256.unique()),
        "n_models": len(ids),
    }
    out["training_commit_ok"] = all(c.startswith(EXPECT_TRAIN) for c in out["training_commits_in_summary_models"])
    out["eval_commit_ok"] = s["commit"].startswith(EXPECT_EVAL)
    out["sha_ok"] = s["layouts_sha256"].startswith(EXPECT_SHA)
    out["note"] = ("top-level 'commit' in final_eval_summary.json is the EVAL commit; the TRAINING commit is recorded per model "
                   "(models[].train_git_commit). final_eval files were not modified.")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fe, bl = load_inputs()
    best, allsingle = hindsight_best_single(bl)
    allsingle.to_csv(OUT / "single_troop_by_group_family.csv", index=False)
    t1, t1f = table1(fe, bl, best)
    t1.to_csv(OUT / "1_baselines_by_group.csv", index=False); t1f.to_csv(OUT / "1_baselines_by_group_family.csv", index=False)
    lc = learning_curves(); cs = curve_stats(lc)
    cs.to_csv(OUT / "2_curve_stats.csv", index=False); plot_curves(lc, OUT / "2_validation_curves.png")
    ps = per_seed_table(fe, lc); ps.to_csv(OUT / "3_per_seed.csv", index=False)
    mm = ps.groupby("aug")[["validation_final_zip", "DEV_train", "FINAL_headline"]].agg(["min", "max"])
    mm.to_csv(OUT / "3_per_seed_minmax.csv")
    worst_plain = ps[ps.aug == "none"].FINAL_headline.min(); best_d4 = ps[ps.aug == "d4"].FINAL_headline.max()
    sep = {m: bool(ps[ps.aug == "none"][m].min() > ps[ps.aug == "d4"][m].max()) for m in ("validation_final_zip", "DEV_train", "FINAL_headline")}
    at, orc = army_table(fe, bl)
    at.to_csv(OUT / "4_army_composition.csv", index=False); orc.to_csv(OUT / "4_army_oracle_goldshare.csv", index=False)
    pv = provenance(fe); pv["worst_plain_beats_best_d4"] = sep
    pv["worst_plain_FINAL_headline"], pv["best_d4_FINAL_headline"] = float(worst_plain), float(best_d4)
    (OUT / "5_provenance.json").write_text(json.dumps(pv, indent=2))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    for name, d in (("1", t1), ("1 by family", t1f), ("2 curve stats", cs), ("3 per seed", ps), ("3 min/max", mm), ("4 army", at), ("4 oracle", orc)):
        print(f"\n== {name} =="); print(d.round(3).to_string())
    print("\n== 5 provenance =="); print(json.dumps(pv, indent=2))


if __name__ == "__main__":
    main()
