"""B14c item 3: planner cost re-measured with REAL trained grid models (B12 note). DEV-validation only (role validation; FINAL refused), no training.

2 grid PPO models (grid_ppo_plain_s0, grid_ppo_d4_s0, final.zip) x 20 DEV-validation layouts (7 spine_pods, 7 archipelago, 6 bunker_nest: the first ids of
each family in loader order). Per-plan time of the planner, RAS, RPS, RPS-full at N = 8 / 32 / 128 (nested prefixes of one 128-plan batch per layout, single
process), 7-worker contention factor, deterministic-eval and TTA episode times with real models, and the B15 session projection (including RPS-full-N, which
the B12 projection omitted). Writes results/b14/planner_cost.json + .md.   caffeinate -dimsu .venv/bin/python scripts/b14_cost.py
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # the repo root is importable when run as `python scripts/<name>.py`

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO

from scripts import b12_cost as C
from breachlens import evaluator as E
from breachlens import pool as P
from rl import planner as PL
from rl.ddqn import QNet, default_obs_space, q_values_fn
from rl.eval_ppo_v2 import masked_probs
from rl.tta import tta_action

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "b14"
MODELS = ROOT / "models_v2"
RUNS = ("grid_ppo_plain_s0", "grid_ppo_d4_s0")
SET, ROLE = "DEV-validation", "validation"
PER_FAM = {"spine_pods": 7, "archipelago": 7, "bunker_nest": 6}


def pick_items():
    items, cnt = [], {k: 0 for k in PER_FAM}
    for rec, lay in E.load_set(SET, ROLE):
        if cnt[rec["family"]] < PER_FAM[rec["family"]]:
            cnt[rec["family"]] += 1
            items.append((rec, lay))
    return items


def _par(args):
    th.set_num_threads(1)
    run, idxs = args
    items = pick_items()
    items = [items[i] for i in idxs]
    m = MaskablePPO.load(MODELS / run / "final.zip", device="cpu")
    rows = C.time_method("planner", items, 0, PL.ModelProbs(m), ph="x")
    return float(np.mean([s for r in rows for s in r["sec"]]))


def main() -> int:
    th.set_num_threads(1)
    before = P.LEDGER_PATH.read_bytes()
    items = pick_items()
    assert len(items) == 20 and all(not P.is_final(r["family"], r["seed"]) for r, _ in items)
    models = {r: MaskablePPO.load(MODELS / r / "final.zip", device="cpu") for r in RUNS}
    res = {"set": SET, "role": ROLE, "n_layouts": 20, "models": list(RUNS), "git_commit": P._git("rev-parse", "HEAD"), "cpu": C.T.cpu_model()}
    cells = {}
    for name, run in zip(("planner (PPO plain s0)", "planner (PPO+D4 s0)"), RUNS):
        cells[name] = C.cell_stats(C.time_method("planner", items, 0, PL.ModelProbs(models[run]), ph=PL.state_dict_hash(models[run].policy)))
    for m in ("RAS", "RPS", "RPSfull"):
        cells[m] = C.cell_stats([r for s in range(2) for r in C.time_method(m, items, s)])
    res["single_process"] = cells
    t_plan = {"planner": float(np.mean([cells[k]["per_plan_mean_s"] for k in cells if k.startswith("planner")])), "RAS": cells["RAS"]["per_plan_mean_s"],
              "RPS": cells["RPS"]["per_plan_mean_s"], "RPSfull": cells["RPSfull"]["per_plan_mean_s"]}
    t_plan_max = dict(t_plan, planner=max(cells[k]["per_plan_mean_s"] for k in cells if k.startswith("planner")))
    # 7 workers at once (the planner of the plain model; contention = per-plan time under load / alone)
    chunks = [list(range(i, 20, 7)) for i in range(7)]
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(7) as pool:
        par = pool.map(_par, [(RUNS[0], c) for c in chunks], chunksize=1)
    f_cont = float(np.mean(par) / cells["planner (PPO plain s0)"]["per_plan_mean_s"])
    res["parallel_7"] = {"wall_s": time.perf_counter() - t0, "per_plan_mean_s_under_load": float(np.mean(par)), "contention_factor": f_cont,
                         "note": "7 processes each plan a 2-3 layout slice with the plain model; the other 3 methods are CPU-light and are scaled by the same factor"}
    # deterministic eval and TTA with the real models (PPO plain / D4: real; DDQN: real final.pt)
    import torch as t_
    ck = t_.load(MODELS / "grid_ddqn_s0" / "final.pt", map_location="cpu", weights_only=False)
    qnet = QNet(default_obs_space(), ck["n_actions"], ck["hyperparameters"]["net_arch"])
    qnet.load_state_dict(ck["online"])
    qfn = q_values_fn(qnet)
    pp = {r: masked_probs(models[r]) for r in RUNS}
    ev = {}
    for r in RUNS:
        ev[f"{r} deterministic"] = C.time_eval(items, lambda o, m, r=r: int(np.argmax(np.where(m, pp[r](o, m), -1.0))))
        ev[f"{r} + TTA"] = C.time_eval(items, lambda o, m, r=r: tta_action(pp[r], o, m))
    ev["grid_ddqn_s0 deterministic"] = C.time_eval(items, lambda o, m: int(np.argmax(np.where(m, qfn(o, m), -np.inf))))
    ev["grid_ddqn_s0 + TTA"] = C.time_eval(items, lambda o, m: tta_action(qfn, o, m, aggregate="mean"))
    res["eval_episode_s"] = ev
    # B15 projection. Plans: planner plain + D4 (10 learners x 120 layouts x 128 each), RAS / RPS / RPS-full 10 seeds x 120 x 128 each.
    n128 = 10 * 120 * 128
    plans = {"planner_plain": n128, "planner_d4": n128, "RAS": n128, "RPS": n128, "RPSfull": n128}
    key = {"planner_plain": "planner", "planner_d4": "planner", "RAS": "RAS", "RPS": "RPS", "RPSfull": "RPSfull"}

    def proj(t, f):
        cpu = {k: plans[k] * t[key[k]] * f / 3600 for k in plans}
        return {"cpu_hours": cpu, "total_cpu_hours": sum(cpu.values()), "wall_hours_7_workers": sum(cpu.values()) / 7}
    nplan = sum(plans.values())
    n_eps = 10 * 120
    ev_cpu_s = n_eps * (ev[f"{RUNS[0]} deterministic"] + ev[f"{RUNS[0]} + TTA"] + ev[f"{RUNS[1]} deterministic"] + ev[f"{RUNS[1]} + TTA"]
                        + ev["grid_ddqn_s0 deterministic"] + ev["grid_ddqn_s0 + TTA"])
    res["session_projection"] = {"plans_total": nplan, "mean_plan_s": t_plan, "planner_worst_model_plan_s": t_plan_max["planner"],
                                 "ideal": proj(t_plan, 1.0), "with_contention": proj(t_plan, f_cont), "worst_model_with_contention": proj(t_plan_max, f_cont),
                                 "deterministic_and_TTA_eval": {"cpu_hours": ev_cpu_s / 3600, "wall_hours_7_contention": ev_cpu_s / 3600 / 7 * f_cont,
                                                                "note": "30 runs x 120 FINAL layouts: PPO plain, PPO+D4, DDQN, each deterministic and +TTA"}}
    res["fallback_selector_on_measured_times"] = PL.select_fallback({k: t_plan[k] * f_cont for k in ("planner", "RAS", "RPS")})
    assert P.LEDGER_PATH.read_bytes() == before, "ledger changed"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "planner_cost.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    L = ["# B14c item 3: planner cost with real grid models (DEV-validation, 20 layouts, 2 PPO models)", "",
         "Per layout, seconds (single process; N = 8 / 32 / 128 are nested prefixes of one 128-plan batch):", "",
         "| cell | N=8 | N=32 | N=128 | per plan (mean) | per plan p95 | mean units | mean env steps |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in cells.items():
        L.append(f"| {k} | {v['8']['mean_s']:.3f} | {v['32']['mean_s']:.3f} | {v['128']['mean_s']:.3f} | {v['per_plan_mean_s'] * 1000:.2f} ms | {v['per_plan_p95_s'] * 1000:.2f} ms | {v['mean_units']:.1f} | {v['mean_env_steps']:.1f} |")
    sp = res["session_projection"]
    L += ["", f"7-worker contention factor (per-plan under load / alone): {f_cont:.2f}.", "",
          "Deterministic / TTA episode seconds (mean over the 20 layouts): " + "; ".join(f"{k} {v * 1000:.1f} ms" for k, v in ev.items()), "",
          f"B15 planner session = {nplan:,} plans (planner plain + D4, RAS, RPS, RPS-full; 10 learners x 120 layouts x 128 each). Wall at 7 workers: ideal {sp['ideal']['wall_hours_7_workers']:.3f} h; "
          f"with contention {sp['with_contention']['wall_hours_7_workers']:.3f} h; worst model ({sp['planner_worst_model_plan_s'] * 1000:.2f} ms/plan) with contention {sp['worst_model_with_contention']['wall_hours_7_workers']:.3f} h. "
          f"Deterministic + TTA evaluation of the 30 models: {sp['deterministic_and_TTA_eval']['wall_hours_7_contention']:.3f} h wall.",
          f"Fallback selector on the measured times: step {res['fallback_selector_on_measured_times']['step']} ({res['fallback_selector_on_measured_times']['label']}), fits 3 h: {res['fallback_selector_on_measured_times']['fits']}."]
    (OUT / "planner_cost.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
