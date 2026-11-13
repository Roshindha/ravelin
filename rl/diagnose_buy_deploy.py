"""STEP 9b: BUY-vs-DEPLOY decomposition (eval only, no training).

Armies (a own buying | b Greedy | c DEV-chosen best single troop, full budget | d Oracle per layout)
x deployers (PPO policy, scripted Greedy-style deployer, Oracle's own deployment for d only).

Layouts: the 20 validation layouts + a fixed, seeded, family-stratified 40-layout subset of DEV-train.
FINAL-familiar / transfer layouts are NEVER loaded (see ``select_layouts`` and tests).

    nohup .venv/bin/python -m rl.diagnose_buy_deploy > results/diagnostics/9b/run.log 2>&1 &
"""
from __future__ import annotations

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from rl import eval_all as EA

ROOT = EA.ROOT
OUT = ROOT / "results" / "diagnostics" / "9b"
TROOPS = EA.TROOPS
COST = {"swarm": 10, "ranger": 25, "tank": 30, "splash": 40}
ALLOWED_GROUPS = ("validation", "DEV-train")
SUBSET_SEED = 9090
N_TRAIN_PER_FAMILY = 20
WORKERS = 4
FAMILIES = ("clustered_core", "scattered_outposts")


# ---------------------------------------------------------------------------------------------
def select_layouts() -> list[dict]:
    """20 validation + 40 DEV-train (20 per family, seeded). Only DEV groups are ever loaded."""
    recs = EA.load_group_records(ALLOWED_GROUPS)
    assert all(r["group"] in ALLOWED_GROUPS and r["pool"] == "DEV" for r in recs)
    val = [r for r in recs if r["group"] == "validation"]
    out = list(val)
    for fam in FAMILIES:
        pool = sorted([r for r in recs if r["group"] == "DEV-train" and r["family"] == fam], key=lambda r: r["id"])
        rng = np.random.default_rng(SUBSET_SEED + len(fam))
        pick = sorted(rng.choice(len(pool), size=N_TRAIN_PER_FAMILY, replace=False).tolist())
        out += [pool[i] for i in pick]
    assert len(out) == 60 and len({r["id"] for r in out}) == 60
    return sorted(out, key=lambda r: r["id"])


def _rank_order() -> list[str]:
    from siege.baselines import lanchester_value
    from siege.troops import TROOP_TYPES

    return [s.name for s in sorted(TROOP_TYPES, key=lambda s: (lanchester_value(s), s.name), reverse=True)]


def counts_to_names(counts) -> list[str]:
    """Army list in the Greedy ranking order (the order the scripted deployer lays troops on the border)."""
    names: list[str] = []
    for t in _rank_order():
        names += [t] * int(counts[TROOPS.index(t)])
    return names


def names_to_counts(names) -> list[int]:
    return [list(names).count(t) for t in TROOPS]


def greedy_counts() -> list[int]:
    from siege.baselines import _greedy_army

    return names_to_counts(_greedy_army({}))


def single_counts(troop: str) -> list[int]:
    c = [0] * 4
    c[TROOPS.index(troop)] = 200 // COST[troop]
    return c


# ---------------------------------------------------------------------------------------------
def sim(placed: list[dict], layout: dict) -> float:
    from siege.baselines import _score_layout

    return _score_layout(placed, layout)[0] if placed else 0.0


def ranger_marginal(placed: list[dict], layout: dict, full: float) -> tuple[int, float]:
    """(ranger gold, destruction lost if the rangers are removed, same positions for everyone else)."""
    n_r = sum(p["type"] == "ranger" for p in placed)
    if n_r == 0:
        return 0, 0.0
    return n_r * COST["ranger"], full - sim([p for p in placed if p["type"] != "ranger"], layout)


def scripted_cell(counts, layout: dict, rec: dict) -> dict:
    from siege.armies import place_on_border
    from siege.baselines import _default_start_idx

    placed = place_on_border(counts_to_names(counts), start_idx=_default_start_idx(layout))
    d = sim(placed, layout)
    rg, rm = ranger_marginal(placed, layout, d)
    return {"destruction": d, "ranger_gold": rg, "ranger_marginal": rm, "bought": list(map(int, counts)), "deployed": list(map(int, counts))}


# --- oracle ------------------------------------------------------------------------------------
def _oracle_worker(rec: dict):
    from siege.baselines import _layout_for_record, oracle_search

    layout = _layout_for_record(rec)
    score, army, start = oracle_search(layout, np.random.default_rng([int(rec["id"]), 4242]))  # same seed/budget as step 4/8
    return int(rec["id"]), score, army, int(start)


def run_oracle(records: list[dict]) -> dict[int, dict]:
    with Pool(WORKERS) as p:
        res = p.map(_oracle_worker, records, chunksize=2)
    return {i: {"score": s, "army": a, "start": st} for i, s, a, st in res}


def oracle_cell(o: dict, layout: dict) -> dict:
    from siege.armies import place_on_border

    placed = place_on_border(o["army"], start_idx=o["start"])
    d = sim(placed, layout)
    rg, rm = ranger_marginal(placed, layout, d)
    c = names_to_counts(o["army"])
    return {"destruction": d, "ranger_gold": rg, "ranger_marginal": rm, "bought": c, "deployed": c}


# --- PPO ---------------------------------------------------------------------------------------
def ppo_episode(model, env, rec: dict, counts: list[int] | None) -> dict:
    """counts None -> the model buys; else forced buys (type order 0..3) then END_BUY, then the policy deploys."""
    from siege.env import END_BUY

    obs, _ = env.reset(seed=int(rec["id"]), options={"layout_record": rec, "prebought": False})
    if counts is not None:
        for k, n in enumerate(counts):
            for _ in range(int(n)):
                assert env.action_masks()[k]
                obs, *_ = env.step(k)
        assert env.action_masks()[END_BUY]
        obs, *_ = env.step(END_BUY)
    done = False
    while not done:
        a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, _, t, tr, _ = env.step(int(a))
        done = t or tr
    d = float(env.destruction)
    rg, rm = ranger_marginal(env.deployed_troops, env.layout, d)
    return {"destruction": d, "ranger_gold": rg, "ranger_marginal": rm,
            "bought": (env.owned + env.deployed).tolist(), "deployed": env.deployed.tolist()}


def _model_task(args):
    run, records, armies = args
    import torch

    torch.set_num_threads(1)
    from sb3_contrib import MaskablePPO

    from siege.env import SiegeEnv

    allowed = {int(r["id"]) for r in records}
    model = MaskablePPO.load(ROOT / "models" / run / "final.zip", device="cpu")
    env = SiegeEnv()  # canonical view, no D4Wrapper
    aug, seed = EA.parse_run(run)
    rows = []
    for rec in records:
        assert int(rec["id"]) in allowed and rec["group"] in ALLOWED_GROUPS
        for src, per_layout in armies.items():
            counts = per_layout[str(rec["id"])] if per_layout is not None else None
            r = ppo_episode(model, env, rec, counts)
            rows.append({"run": run, "aug": aug, "seed": seed, "army": src, "deployer": "ppo", "layout_id": int(rec["id"]),
                         "family": rec["family"], "set": rec["group"], **r})
    return rows


# ---------------------------------------------------------------------------------------------
def bootstrap_cell(vals_by_learner: np.ndarray, rng) -> tuple[float, float, float]:
    m = float(np.mean(vals_by_learner))
    if len(vals_by_learner) < 2:
        return m, np.nan, np.nan
    lo, hi = EA.boot_ci(vals_by_learner, rng)
    return m, lo, hi


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Matrix rows: (scope, family, army, deployer) -> mean destruction (+CI over learners where cells vary by learner)."""
    rng = np.random.default_rng(EA.BOOT_SEED + 9)
    out = []
    scopes = {"ALL60": df, "validation": df[df.set == "validation"], "DEV-train-subset": df[df.set == "DEV-train"]}
    for sname, sd in scopes.items():
        for fam in ["ALL", *FAMILIES]:
            f = sd if fam == "ALL" else sd[sd.family == fam]
            for (army, dep, aug), g in f.groupby(["army", "deployer", "aug"]):
                per = g.groupby("seed").destruction.mean()
                m, lo, hi = bootstrap_cell(per.to_numpy(), rng) if aug in ("none", "d4") else (g.destruction.mean(), np.nan, np.nan)
                out.append({"scope": sname, "family": fam, "army": army, "deployer": dep if aug == "-" else f"{dep}_{aug}",
                            "n_learners": per.shape[0] if aug in ("none", "d4") else 0, "n_layouts": g.layout_id.nunique(),
                            "mean_destruction": m, "ci95_lo": lo, "ci95_hi": hi})
    return pd.DataFrame(out)


def efficiency(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(EA.BOOT_SEED + 10)
    rows = []
    for fam in ["ALL", *FAMILIES]:
        f = df if fam == "ALL" else df[df.family == fam]
        # (1) ranger-only army, 8 rangers = 200 gold: destruction per 100 gold, PPO vs scripted
        for dep, aug in (("ppo", "none"), ("ppo", "d4"), ("scripted", "-")):
            g = f[(f.army == "single_ranger") & (f.deployer == dep) & (f.aug == aug)]
            per = g.groupby("seed").destruction.mean() * 100 / 200 if aug != "-" else pd.Series([g.destruction.mean() * 100 / 200])
            m, lo, hi = bootstrap_cell(per.to_numpy(), rng) if aug != "-" else (float(per.iloc[0]), np.nan, np.nan)
            rows.append({"family": fam, "measure": "ranger-only 8x army: destruction per 100 gold", "deployer": f"{dep}_{aug}" if aug != "-" else dep,
                         "value": m, "ci95_lo": lo, "ci95_hi": hi})
        # (2) marginal contribution of rangers inside mixed armies: destruction lost when rangers are removed, per 100 ranger gold
        for army in ("own", "oracle"):
            for dep, aug in (("ppo", "none"), ("ppo", "d4"), ("scripted", "none"), ("scripted", "d4"), ("oracle", "-")):
                g = f[(f.army == army) & (f.deployer == dep) & (f.aug == aug)]
                if g.empty or g.ranger_gold.sum() == 0:
                    continue
                per = g.groupby("seed").apply(lambda x: 100 * x.ranger_marginal.sum() / max(1, x.ranger_gold.sum()), include_groups=False)
                m, lo, hi = bootstrap_cell(per.to_numpy(), rng) if aug != "-" else (float(per.iloc[0]), np.nan, np.nan)
                rows.append({"family": fam, "measure": f"ranger marginal destruction per 100 ranger gold, army={army}",
                             "deployer": f"{dep}_{aug}" if aug != "-" else dep, "value": m, "ci95_lo": lo, "ci95_hi": hi})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    records = select_layouts()
    from siege.baselines import _layout_for_record

    layouts = {r["id"]: _layout_for_record(r) for r in records}
    pd.DataFrame([{"layout_id": r["id"], "family": r["family"], "set": r["group"]} for r in records]).to_csv(OUT / "layouts_used.csv", index=False)
    print(f"{len(records)} layouts: {pd.Series([r['group'] + '/' + r['family'] for r in records]).value_counts().to_dict()}", flush=True)

    # armies
    dev_all = EA.load_group_records(ALLOWED_GROUPS)
    best = EA.best_single_troops(dev_all)  # DEV-chosen, as in step 8
    print("DEV-chosen best single troop:", best, flush=True)
    orc = run_oracle(records)
    (OUT / "oracle_armies.json").write_text(json.dumps(
        {str(i): {"score": o["score"], "army": o["army"], "start_tile": o["start"], "counts": dict(zip(TROOPS, names_to_counts(o["army"])))}
         for i, o in sorted(orc.items())}, indent=2))
    print(f"oracle done {time.time() - t0:.0f}s, mean {np.mean([o['score'] for o in orc.values()]):.3f}", flush=True)
    gc = greedy_counts()
    armies = {
        "own": None,
        "greedy": {str(r["id"]): gc for r in records},
        "best_single": {str(r["id"]): single_counts(best[r["family"]]) for r in records},
        "oracle": {str(r["id"]): names_to_counts(orc[r["id"]]["army"]) for r in records},
        **{f"single_{t}": {str(r["id"]): single_counts(t) for r in records} for t in TROOPS},
    }
    runs = EA.run_names()
    tasks = [(run, records, armies) for run in runs]
    rows = []
    with Pool(WORKERS) as p:
        for n, r in enumerate(p.imap_unordered(_model_task, tasks), 1):
            rows += r
            print(f"models done {n}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)

    # scripted + oracle deployers
    srows = []
    for rec in records:
        lay = layouts[rec["id"]]
        base = {"layout_id": rec["id"], "family": rec["family"], "set": rec["group"]}
        for src in ("greedy", "best_single", "oracle", *[f"single_{t}" for t in TROOPS]):
            c = scripted_cell(armies[src][str(rec["id"])], lay, rec)
            srows.append({**base, "run": "scripted", "aug": "-", "seed": -1, "army": src, "deployer": "scripted", **c})
        c = oracle_cell(orc[rec["id"]], lay)
        srows.append({**base, "run": "oracle", "aug": "-", "seed": -1, "army": "oracle", "deployer": "oracle", **c})
    own = df[(df.army == "own")]
    for _, r in own.iterrows():  # scripted deployer applied to each learner's own bought army
        c = scripted_cell(r["bought"], layouts[r["layout_id"]], None)
        srows.append({"run": r["run"], "aug": r["aug"], "seed": r["seed"], "army": "own", "deployer": "scripted", "layout_id": r["layout_id"],
                      "family": r["family"], "set": r["set"], **c})
    df = pd.concat([df, pd.DataFrame(srows)], ignore_index=True)
    df["bought"] = df.bought.map(json.dumps); df["deployed"] = df.deployed.map(json.dumps)
    df.to_csv(OUT / "episodes.csv", index=False)
    mat = summarize(df); mat.to_csv(OUT / "matrix.csv", index=False)
    eff = efficiency(df); eff.to_csv(OUT / "ranger_efficiency.csv", index=False)
    # own-army counts per learner type (control)
    own_ct = df[(df.army == "own") & (df.deployer == "ppo")].copy()
    for i, t in enumerate(TROOPS):
        own_ct[f"n_{t}"] = own_ct.bought.map(lambda s: json.loads(s)[i])
    own_ct.groupby(["aug", "family"])[[f"n_{t}" for t in TROOPS]].mean().to_csv(OUT / "own_army_counts.csv")
    # deployed fraction (do policies leave troops undeployed with WAIT?)
    dep = df[df.deployer == "ppo"].copy()
    dep["undeployed"] = [sum(json.loads(b)) - sum(json.loads(d)) for b, d in zip(dep.bought, dep.deployed)]
    dep.groupby(["aug", "army"]).undeployed.mean().to_csv(OUT / "undeployed_troops.csv")
    print(f"ALL DONE {time.time() - t0:.0f}s", flush=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    show = mat[mat.scope == "ALL60"]
    for fam in ["ALL", *FAMILIES]:
        s = show[show.family == fam].copy()
        s["cell"] = s.apply(lambda r: f"{r.mean_destruction:.3f}" + (f" [{r.ci95_lo:.3f},{r.ci95_hi:.3f}]" if not np.isnan(r.ci95_lo) else ""), axis=1)
        print(f"\n== matrix, family={fam}, 60 layouts =="); print(s.pivot(index="army", columns="deployer", values="cell").to_string())
    print("\n== ranger efficiency =="); print(eff.round(3).to_string())


if __name__ == "__main__":
    main()
