"""B13c diagnostics after the G7 failure (docs/SPEC_V2.md 7.3; Gate changelog (y)). TUNE-val ONLY (role validation); no training, no tuning.

    .venv/bin/python scripts_b13_diag.py [--out results/b13] [--models-dir models_v2] [--seeds 9001,9002,9003]

Uses the final.zip (t = 300k) of the 3 plain pilot seeds, deterministic masked argmax on the canonical view (the G7 eval mode).
(a) 7.3 armies x deployers decomposition (every cell is a real env episode or the simulator through the scripted deployer),
(b) breach vs destruction: PPO vs BestSingle vs Greedy vs Oracle (baseline rows of B8) on the same 60 layouts,
(c) units bought / deployed and type mix vs BestSingle.
The Oracle-deploy column of 7.3 is NOT computed: the SPEC does not define a deployment-only search for a fixed army (see diagnostics.md).
"""
from __future__ import annotations

import argparse
import csv
import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from breachlens import evaluator as EV
from breachlens import oracle as O
from breachlens.armies import army_cost, default_start, greedy_types, place_on_border, random_types, single_unit_types, sort_by_rank
from breachlens.constants import BUY_BASE, END_BUY, N_UNIT_TYPES, PHASE_DEPLOY, UNIT_TYPES
from breachlens.env import BreachEnv
from breachlens.score import result_score
from breachlens.sim import Prepared, simulate

ROOT = Path(__file__).resolve().parent
SET, ROLE = "TUNE-val", "validation"
SEEDS = (9001, 9002, 9003)
FAMILIES = ("spine_pods", "archipelago", "bunker_nest")
ARMIES = ("own", "greedy", "best_single", "random", "oracle_army")
DEPLOYERS = ("ppo", "scripted")
RANDOM_ARMY_TAG = 7313                         # pinned here (SPEC 7.3 gives no seed for the random army): default_rng([crc32(layout id), tag])
B8_ROWS = ROOT / "data" / "v2" / "baselines" / "b8_layout_rows.csv"
B8_BEST = ROOT / "data" / "v2" / "baselines" / "b8_best_single.json"


def counts_of(types) -> list:
    return [int(sum(1 for t in types if t == k)) for k in range(N_UNIT_TYPES)]


def outcome(res, n_bought: int, n_deployed: int) -> dict:
    return {"score": result_score(res), "d": res.d_num / max(res.d_den, 1), "breach": float(bool(res.core_destroyed)),
            "n_bought": int(n_bought), "n_deployed": int(n_deployed)}


def scripted(layout, P, counts) -> dict:
    types = sort_by_rank([t for t, c in enumerate(counts) for _ in range(c)])          # scripted deployer: Greedy ranking order, default start tile
    return outcome(simulate(P, place_on_border(layout, types)), len(types), len(types))


def ppo_deploy_after_forced_buys(model, env, layout_idx, counts, seed) -> dict:
    """Forced army through real buy actions + END_BUY, then the policy deploys (deterministic masked argmax)."""
    env.reset(seed=seed, options={"layout_index": layout_idx})
    for t, c in enumerate(counts):
        for _ in range(c):
            env.step(BUY_BASE + t)
    env.step(END_BUY)
    return _finish(model, env)


def _finish(model, env) -> dict:
    obs = env._obs()
    done = env.done
    while not done:
        a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, _, done, _, _ = env.step(int(a))
    return outcome(env.result, env.bought, len(env.army))


def own_buy_then_ppo(model, env, layout_idx, seed) -> tuple[list, dict]:
    """The policy's own game; returns (army counts at END_BUY, outcome). Identical in steps to rl.eval_ppo_v2.ModelPolicy."""
    obs, _ = env.reset(seed=seed, options={"layout_index": layout_idx})
    counts, done = None, False
    while not done:
        a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, _, done, _, _ = env.step(int(a))
        if counts is None and env.phase == PHASE_DEPLOY:
            counts = [int(c) for c in env.owned]
    return counts, outcome(env.result, env.bought, len(env.army))


def load_b8(set_name: str = SET) -> dict:
    out = defaultdict(dict)
    with open(B8_ROWS) as f:
        for r in csv.DictReader(f):
            if r["set"] == set_name:
                out[r["policy"]][r["id"]] = {k: float(r[k]) for k in ("score", "d", "breach", "n_bought", "n_deployed")} | {"family": r["family"]}
    return out


def fixed_armies(rec, layout, P, best_single: dict) -> dict:
    """The seed-independent forced armies of one layout (counts)."""
    rng = np.random.default_rng([zlib.crc32(rec["id"].encode()), RANDOM_ARMY_TAG])
    army_search = O.army_search(layout, O.oracle_rng(rec["seed"], O.ARMY_SEARCH_TAG), prepared=P)         # G8(a): army search, scripted deployment
    return {"greedy": counts_of(greedy_types()), "best_single": counts_of(single_unit_types(best_single[rec["family"]])),
            "random": counts_of(random_types(rng)), "oracle_army": counts_of(army_search.types)}


def run(models_dir: Path, seeds=SEEDS, items=None) -> dict:
    from sb3_contrib import MaskablePPO
    items = items if items is not None else EV.load_set(SET, ROLE, entry_point="scripts_b13_diag")       # TUNE-val only, role validation
    best_single = {k: int(v) for k, v in json.loads(B8_BEST.read_text())["train_chosen_for_set"][SET].items()}
    models = {s: MaskablePPO.load(models_dir / f"pilot_plain_s{s}" / "final.zip", device="cpu") for s in seeds}
    cells = []                                        # one row per (seed, layout, army, deployer)
    for i, (rec, layout) in enumerate(items):
        P = Prepared(layout)
        fixed = fixed_armies(rec, layout, P, best_single)
        env = BreachEnv(layouts=[layout], role="train")            # the evaluator already loaded the layout through the loaders (same path as ModelPolicy)
        scripted_fixed = {a: scripted(layout, P, c) for a, c in fixed.items()}
        for s, model in models.items():
            seed = zlib.crc32(rec["id"].encode())
            own_counts, own_ppo = own_buy_then_ppo(model, env, 0, seed)
            armies = {"own": own_counts, **fixed}
            res = {("own", "ppo"): own_ppo, ("own", "scripted"): scripted(layout, P, own_counts)}
            for a, c in fixed.items():
                res[(a, "ppo")] = ppo_deploy_after_forced_buys(model, env, 0, c, seed)
                res[(a, "scripted")] = scripted_fixed[a]
            for (a, d), o in res.items():
                cells.append({"seed": s, "id": rec["id"], "family": rec["family"], "army": a, "deployer": d, "army_counts": json.dumps(armies[a]),
                              "army_gold": army_cost([t for t, c in enumerate(armies[a]) for _ in range(c)]), **o})
    return {"cells": cells, "items": items, "best_single": best_single}


# ---------------------------------------------------------------------------------------------------- tables
def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def decomposition_table(cells, key=("army", "deployer")) -> list:
    """Mean over layouts per (seed, family, army, deployer), plus pooled over seeds ('ALL') and over families ('ALL')."""
    rows = []
    for seed in [*sorted({c["seed"] for c in cells}), "ALL"]:
        for fam in [*FAMILIES, "ALL"]:
            for a in ARMIES:
                for d in DEPLOYERS:
                    sel = [c for c in cells if c["army"] == a and c["deployer"] == d and (seed == "ALL" or c["seed"] == seed) and (fam == "ALL" or c["family"] == fam)]
                    rows.append({"seed": seed, "family": fam, "army": a, "deployer": d, "n": len(sel), "score": mean(c["score"] for c in sel),
                                 "d": mean(c["d"] for c in sel), "breach": mean(c["breach"] for c in sel),
                                 "n_bought": mean(c["n_bought"] for c in sel), "n_deployed": mean(c["n_deployed"] for c in sel)})
    return rows


def controls(cells, b8, val_log_dir: Path, seeds) -> dict:
    """(i) own x PPO equals the logged t = 300k validation eval layout by layout (max |diff| = 0); (ii) greedy x scripted equals the B8 Greedy rows;
    (iii) best_single x scripted equals the B8 BestSingle rows."""
    out = {}
    for s in seeds:
        logged = {}
        for ln in (val_log_dir / f"pilot_plain_s{s}" / "val_layouts.jsonl").read_text().splitlines():
            r = json.loads(ln)
            if r["timestep"] == 300_000:
                logged[r["id"]] = r
        mine = {c["id"]: c for c in cells if c["seed"] == s and c["army"] == "own" and c["deployer"] == "ppo"}
        out[f"own_x_ppo_vs_val_eval_s{s}"] = {"n": len(mine), "max_abs_diff_score": max(abs(mine[i]["score"] - logged[i]["score"]) for i in mine),
                                              "max_abs_diff_d": max(abs(mine[i]["d"] - logged[i]["d"]) for i in mine),
                                              "max_abs_diff_breach": max(abs(mine[i]["breach"] - logged[i]["breach"]) for i in mine)}
    for name, key, pol in (("greedy_x_scripted_vs_b8_Greedy", "greedy", "Greedy"), ("best_single_x_scripted_vs_b8_BestSingle", "best_single", "BestSingle")):
        mine = {c["id"]: c for c in cells if c["seed"] == seeds[0] and c["army"] == key and c["deployer"] == "scripted"}
        out[name] = {"n": len(mine), "max_abs_diff_score": max(abs(mine[i]["score"] - b8[pol][i]["score"]) for i in mine),
                     "max_abs_diff_d": max(abs(mine[i]["d"] - b8[pol][i]["d"]) for i in mine)}
    return out


def breach_vs_destruction(cells, b8, items, bs_fallback=None) -> tuple[list, list]:
    """Per (seed, family): breach rate, D, score for PPO / BestSingle / Greedy / Oracle on the same layouts, and the 2x2 breach table + the
    'PPO higher D, no breach, BestSingle breaches' fraction."""
    fam_of = {r["id"]: r["family"] for r, _ in items}
    ppo = {(c["seed"], c["id"]): c for c in cells if c["army"] == "own" and c["deployer"] == "ppo"}
    seeds = sorted({c["seed"] for c in cells})
    table, pair = [], []
    for seed in [*seeds, "ALL"]:
        for fam in [*FAMILIES, "ALL"]:
            ids = [i for i in fam_of if fam == "ALL" or fam_of[i] == fam]
            ss = seeds if seed == "ALL" else [seed]
            for name, get in (("PPO", lambda s, i: ppo[(s, i)]), ("BestSingle", lambda s, i: b8["BestSingle"][i]), ("Greedy", lambda s, i: b8["Greedy"][i]),
                              ("Oracle", lambda s, i: b8["Oracle"][i])):
                v = [get(s, i) for s in ss for i in ids]
                table.append({"seed": seed, "family": fam, "policy": name, "n": len(v), "breach": mean(x["breach"] for x in v), "d": mean(x["d"] for x in v),
                              "score": mean(x["score"] for x in v)})
            tot = len(ss) * len(ids)
            pr = [(ppo[(s, i)], b8["BestSingle"][i]) for s in ss for i in ids]
            pair.append({"seed": seed, "family": fam, "n": tot,
                         "both_breach": sum(p["breach"] == 1 and b["breach"] == 1 for p, b in pr) / tot,
                         "only_ppo_breaches": sum(p["breach"] == 1 and b["breach"] == 0 for p, b in pr) / tot,
                         "only_bestsingle_breaches": sum(p["breach"] == 0 and b["breach"] == 1 for p, b in pr) / tot,
                         "neither_breaches": sum(p["breach"] == 0 and b["breach"] == 0 for p, b in pr) / tot,
                         "ppo_higher_d_no_breach_while_bs_breaches": sum(p["d"] > b["d"] and p["breach"] == 0 and b["breach"] == 1 for p, b in pr) / tot,
                         "ppo_higher_d": sum(p["d"] > b["d"] for p, b in pr) / tot,
                         "ppo_higher_score": sum(p["score"] > b["score"] for p, b in pr) / tot})
    return table, pair


def composition(cells, b8, items, best_single) -> list:
    """Mean units bought / deployed and type mix (mean count per type, share of gold per type) per (seed, family): PPO own buying, BestSingle, Greedy."""
    fam_of = {r["id"]: r["family"] for r, _ in items}
    unit_cost = [army_cost([t]) for t in range(N_UNIT_TYPES)]
    seeds = sorted({c["seed"] for c in cells})
    rows = []

    def add(seed, fam, policy, count_list, n_bought, n_deployed):
        arr = np.array(count_list, dtype=float).reshape(-1, N_UNIT_TYPES)
        gold = arr * np.array(unit_cost)
        row = {"seed": seed, "family": fam, "policy": policy, "n": len(arr), "mean_bought": mean(n_bought), "mean_deployed": mean(n_deployed),
               "mean_gold": float(gold.sum(1).mean())}
        for k, u in enumerate(UNIT_TYPES):
            row[f"count_{u}"] = float(arr[:, k].mean())
        for k, u in enumerate(UNIT_TYPES):
            row[f"goldshare_{u}"] = float(gold[:, k].sum() / max(gold.sum(), 1))
        rows.append(row)

    for seed in [*seeds, "ALL"]:
        for fam in [*FAMILIES, "ALL"]:
            ids = [i for i in fam_of if fam == "ALL" or fam_of[i] == fam]
            ss = seeds if seed == "ALL" else [seed]
            own = {(c["seed"], c["id"]): c for c in cells if c["army"] == "own" and c["deployer"] == "ppo"}
            # the bought army (counts at END_BUY) is stored on the own x scripted row (same army); n_deployed comes from the own x ppo row
            own_counts = {(c["seed"], c["id"]): json.loads(c["army_counts"]) for c in cells if c["army"] == "own" and c["deployer"] == "scripted"}
            add(seed, fam, "PPO", [own_counts[(s, i)] for s in ss for i in ids], [own[(s, i)]["n_bought"] for s in ss for i in ids],
                [own[(s, i)]["n_deployed"] for s in ss for i in ids])
            bs = [counts_of(single_unit_types(best_single[fam_of[i]])) for _ in ss for i in ids]
            add(seed, fam, "BestSingle", bs, [sum(c) for c in bs], [sum(c) for c in bs])
            gr = [counts_of(greedy_types()) for _ in ss for i in ids]
            add(seed, fam, "Greedy", gr, [sum(c) for c in gr], [sum(c) for c in gr])
    return rows


def write_csv(rows, path: Path) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def f3(x) -> str:
    return f"{x:.3f}"


def render_md(decomp, ctrl, bvd, pair, comp, best_single, quote: str) -> str:
    L = ["# B13c diagnostics (TUNE-val only, role validation; final.zip at t = 300k of pilot_plain_s9001 / 9002 / 9003; no training, no FINAL access)", ""]
    L += ["## 1a. armies x deployers decomposition (SPEC 7.3, quoted)", "", quote, ""]
    L += ["Pinned choices (SPEC silent, none depends on a result): forced random army = `random_types(default_rng([crc32(layout id), 7313]))` (one army per layout, "
          "the same for all seeds and deployers); forced best-single = per-family type chosen on TUNE-train (B8, Gate changelog (k)): "
          + ", ".join(f"{k}={UNIT_TYPES[v]}" for k, v in sorted(best_single.items())) + "; forced Oracle army = `army_search` result "
          "(128 random + 128 hill-climb, scripted deployment, rng [layout seed, 4343], SPEC 3.4 G8(a)); scripted deployer = `place_on_border` in Greedy ranking order from the "
          "default start tile; PPO deployer = deterministic masked argmax after the forced buys + END_BUY (a unit left undeployed when the policy plays WAIT is forfeited).", ""]
    L += ["**Oracle-deploy column NOT computed (SPEC ambiguity, owner decision needed):** the table row says `forced Oracle army (army-search result)` with an Oracle-deploy "
          "reference cell, but the SPEC defines no deployment-only search for a fixed army. See the options in the report message.", ""]
    L += ["Reading caveats: the forced Oracle army is SELECTED by searching armies under the scripted deployer, so the scripted column is optimistic for that row by construction "
          "(winner's curse) and the PPO column is not; the forced armies other than `own` are not armies the pilot policy chose, so its deployment on them is partly off-distribution. "
          "Differences between rows are armies (buying), differences inside a row are deployers.", ""]
    L += ["### Controls", "", "| control | max abs diff |", "|---|---|"]
    for k, v in ctrl.items():
        L.append(f"| {k} (n={v['n']}) | score {v['max_abs_diff_score']:.3g}, D {v['max_abs_diff_d']:.3g}" + (f", breach {v['max_abs_diff_breach']:.3g}" if "max_abs_diff_breach" in v else "") + " |")
    L.append("")
    for fam in [*FAMILIES, "ALL"]:
        L += [f"### Mean score / D / breach, {fam}, pooled over the 3 seeds (per-seed numbers in decomposition.csv)", "",
              "| army \\ deployer | PPO deploy | scripted deploy | PPO - scripted |", "|---|---|---|---|"]
        for a in ARMIES:
            g = {d: next(r for r in decomp if r["seed"] == "ALL" and r["family"] == fam and r["army"] == a and r["deployer"] == d) for d in DEPLOYERS}
            cell = lambda r: f"{f3(r['score'])} / {f3(r['d'])} / {f3(r['breach'])}"
            L.append(f"| {a} | {cell(g['ppo'])} | {cell(g['scripted'])} | {g['ppo']['score'] - g['scripted']['score']:+.3f} |")
        L.append("")
    L += ["### Per seed, all families (score)", "", "| army \\ deployer | " + " | ".join(f"{s} {d}" for s in (9001, 9002, 9003) for d in DEPLOYERS) + " |", "|---|" + "---|" * 6]
    for a in ARMIES:
        L.append(f"| {a} | " + " | ".join(f3(next(r for r in decomp if r["seed"] == s and r["family"] == "ALL" and r["army"] == a and r["deployer"] == d)["score"])
                                         for s in (9001, 9002, 9003) for d in DEPLOYERS) + " |")
    L += ["", "## 1b. breach vs destruction (same 60 layouts; PPO = own buying + PPO deployment = the G7 eval)", "",
          "| seed | family | policy | breach | D | score |", "|---|---|---|---|---|---|"]
    for r in bvd:
        L.append(f"| {r['seed']} | {r['family']} | {r['policy']} | {f3(r['breach'])} | {f3(r['d'])} | {f3(r['score'])} |")
    L += ["", "PPO vs BestSingle layout by layout (fractions of layouts; BestSingle = train-chosen type, B8 rows):", "",
          "| seed | family | n | both breach | only PPO | only BestSingle | neither | PPO higher D, no breach, BS breaches | PPO higher D | PPO higher score |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in pair:
        L.append(f"| {r['seed']} | {r['family']} | {r['n']} | {f3(r['both_breach'])} | {f3(r['only_ppo_breaches'])} | {f3(r['only_bestsingle_breaches'])} | "
                 f"{f3(r['neither_breaches'])} | {f3(r['ppo_higher_d_no_breach_while_bs_breaches'])} | {f3(r['ppo_higher_d'])} | {f3(r['ppo_higher_score'])} |")
    L += ["", "## 1c. units bought / deployed and type mix (mean per layout)", "",
          "| seed | family | policy | bought | deployed | gold | " + " | ".join(f"n {u}" for u in UNIT_TYPES) + " | " + " | ".join(f"gold% {u}" for u in UNIT_TYPES) + " |",
          "|---|---|---|---|---|---|" + "---|" * (2 * N_UNIT_TYPES)]
    for r in comp:
        L.append(f"| {r['seed']} | {r['family']} | {r['policy']} | {r['mean_bought']:.2f} | {r['mean_deployed']:.2f} | {r['mean_gold']:.0f} | "
                 + " | ".join(f"{r['count_' + u]:.2f}" for u in UNIT_TYPES) + " | " + " | ".join(f"{100 * r['goldshare_' + u]:.0f}" for u in UNIT_TYPES) + " |")
    return "\n".join(L) + "\n"


SPEC_73_QUOTE = (
    "> ### 7.3 Pilot diagnostic: armies × deployers decomposition (DEV only)\n>\n"
    "> Part of B13, reusing the prototype's `diagnose_buy_deploy` design. Layouts: **TUNE-val (60) only** — FINAL and LAB-DESIGN are never loaded (test). "
    "Applied to the 3 pilot PPO plain seeds:\n>\n"
    "> | army \\ deployer | pilot PPO deploy | scripted deploy | Oracle deploy (reference) |\n> |---|---|---|---|\n"
    "> | own buying (policy buys) | ✓ | ✓ | – |\n> | forced Greedy army | ✓ | ✓ | – |\n> | forced best-single army | ✓ | ✓ | – |\n"
    "> | forced random army (phase-A generator) | ✓ | ✓ | – |\n> | forced Oracle army (army-search result) | ✓ | ✓ | ✓ |\n>\n"
    "> Armies are forced through real buy actions + END_BUY; control checks as in step 9b (own-buy × PPO reproduces the validation eval exactly; "
    "scripted × Greedy reproduces the baseline file). Purpose: when G7 fails, say whether buying or deployment is the bottleneck, and give the redesign "
    "(or shaped-reward ablation) something specific to target. It is a **diagnostic, not a tuning loop**: it can trigger a revision only through the iteration rule above; "
    "no hyperparameter is changed because of it.")


def main(argv=None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "results" / "b13"))
    p.add_argument("--models-dir", default=str(ROOT / "models_v2"))
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    a = p.parse_args(argv)
    seeds = tuple(int(x) for x in a.seeds.split(","))
    out, models = Path(a.out), Path(a.models_dir)
    out.mkdir(parents=True, exist_ok=True)
    r = run(models, seeds)
    cells, items, bs = r["cells"], r["items"], r["best_single"]
    b8 = load_b8()
    decomp = decomposition_table(cells)
    ctrl = controls(cells, b8, models, list(seeds))
    bvd, pair = breach_vs_destruction(cells, b8, items)
    comp = composition(cells, b8, items, bs)
    write_csv(cells, out / "decomposition_cells.csv")
    write_csv(decomp, out / "decomposition.csv")
    write_csv(bvd, out / "breach_vs_destruction.csv")
    write_csv(pair, out / "ppo_vs_bestsingle_breach_pairs.csv")
    write_csv(comp, out / "composition.csv")
    (out / "controls.json").write_text(json.dumps(ctrl, indent=1))
    (out / "diagnostics.md").write_text(render_md(decomp, ctrl, bvd, pair, comp, bs, SPEC_73_QUOTE))
    print(json.dumps(ctrl, indent=1))
    return {"controls": ctrl}


if __name__ == "__main__":
    main()
