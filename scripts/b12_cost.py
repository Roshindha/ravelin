"""B12 cost measurement (TUNE-val, role 'tuning'; no FINAL access, no policy is trained).

Learners are RANDOM-INIT seeded PPO policies (and a random-init QNet for the DDQN rows): they only supply realistic network cost. Their
plans are not trained-policy plans, so the army size (and hence simulator cost) can differ; heavy stubs (network forward kept; full army / maximum
number of cheap units) are timed as a heavy reference. Run:  caffeinate -dimsu .venv/bin/python scripts/b12_cost.py
Writes data/v2/b12_cost.json.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # the repo root is importable when run as `python scripts/<name>.py`

import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch as th

from breachlens import evaluator as E
from breachlens import pool as P
from breachlens.env import END_BUY, N_ACTIONS, WAIT, BreachEnv
from breachlens.sim import Prepared
from rl import planner as PL
from rl import train_ppo_v2 as T
from rl.ddqn import QNet, default_obs_space, q_values_fn
from rl.tta import tta_action

OUT = Path("data/v2/b12_cost.json")
N_LEARNERS_MEASURED = 2           # 7.2 protocol: 2 learners
SET, ROLE = "TUNE-val", "tuning"


class HeavyStub:
    """Heavy reference policy WITH the network forward pass (its output is computed, then replaced): keeps buying until the budget / cap runs
    out, then deploys; never picks END_BUY / WAIT unless forced. `cheapest_only` buys only the cheapest unit type (largest army, up to the cap of 20)."""

    def __init__(self, net_probs, cheapest_only: bool):
        self.net, self.cheapest_only = net_probs, cheapest_only

    def __call__(self, obs, mask):
        self.net(obs, mask)
        p = np.ones(N_ACTIONS)
        if self.cheapest_only:
            from breachlens.env import UNIT_COST
            p[:len(UNIT_COST)] = 0.0
            p[int(np.argmin(UNIT_COST))] = 1.0
        p[END_BUY] = p[WAIT] = 1e-9
        return p


def random_init(seed: int):
    th.manual_seed(seed)
    return T.build_model(T.build_venv("TUNE-train", "none", seed, 1, None), seed, T.PPO_CONFIG)


def time_method(method, items, seed, probs_fn=None, ph="x", n=PL.N_MAX):
    """Per layout: seconds of each of the n plans (in plan order), plus army sizes. One sampled batch per layout; prefixes give N = 8 / 32 / 128."""
    out = []
    for rec, layout in items:
        cache = PL.PlanCache()
        plans = cache.get(method, rec, layout, Prepared(layout), seed, probs_fn, n, ph)
        out.append({"id": rec["id"], "family": rec["family"], "sec": [p["seconds"] for p in plans], "units": [p["n_deployed"] for p in plans],
                    "steps": [len(p["actions"]) for p in plans]})
    return out


def cell_stats(rows):
    res = {}
    for N in PL.N_VALUES:
        per_layout = np.array([sum(r["sec"][:N]) for r in rows])
        res[str(N)] = {"mean_s": float(per_layout.mean()), "median_s": float(np.median(per_layout)), "max_s": float(per_layout.max())}
    allp = np.array([s for r in rows for s in r["sec"]])
    res["per_plan_mean_s"] = float(allp.mean())
    res["per_plan_p95_s"] = float(np.percentile(allp, 95))
    res["mean_units"] = float(np.mean([u for r in rows for u in r["units"]]))
    res["mean_env_steps"] = float(np.mean([s for r in rows for s in r["steps"]]))
    fam = defaultdict(list)
    for r in rows:
        fam[r["family"]].append(sum(r["sec"][:PL.N_MAX]))
    res["per_family_N128_mean_s"] = {k: float(np.mean(v)) for k, v in fam.items()}
    return res


def _par_worker(args):
    th.set_num_threads(1)
    seed, idxs = args
    items = E.load_set(SET, ROLE)
    items = [items[i] for i in idxs]
    t0 = time.perf_counter()
    rows = time_method("planner", items, seed, PL.ModelProbs(random_init(seed)), ph="x")
    return time.perf_counter() - t0, float(np.mean([s for r in rows for s in r["sec"]]))


def episode(layout, rec, pick):
    env = BreachEnv(layouts=[layout], role="train")
    obs, _ = env.reset(seed=0, options={"layout_index": 0})
    done = False
    while not done:
        obs, _, done, _, _ = env.step(pick(obs, env.action_masks()))


def time_eval(items, pick, reps=1):
    t = []
    for rec, layout in items:
        t0 = time.perf_counter()
        episode(layout, rec, pick)
        t.append(time.perf_counter() - t0)
    return float(np.mean(t))


def main():
    th.set_num_threads(1)
    before = P.LEDGER_PATH.read_bytes()
    items = E.load_set(SET, ROLE)
    assert len(items) == 60 and all(not P.is_final(r["family"], r["seed"]) for r, _ in items)
    models = [random_init(s) for s in range(N_LEARNERS_MEASURED)]
    res = {"set": SET, "role": ROLE, "n_layouts": len(items), "learners_measured": N_LEARNERS_MEASURED, "learner_kind": "random-init seeded PPO (not trained)",
           "git_commit": P._git("rev-parse", "HEAD"), "cpu": T.cpu_model(), "torch_threads": th.get_num_threads()}

    # --- single-process timings (the 60 layouts, 2 learners for the planner, 2 seeds for the controls)
    cells = {}
    pl = [time_method("planner", items, s, PL.ModelProbs(models[s]), ph=PL.state_dict_hash(models[s].policy)) for s in range(N_LEARNERS_MEASURED)]
    cells["planner"] = cell_stats([r for rows in pl for r in rows])
    for m in ("RAS", "RPS"):
        cells[m] = cell_stats([r for s in range(N_LEARNERS_MEASURED) for r in time_method(m, items, s)])
    cells["planner_heavy_fullarmy_stub (20 layouts, with network)"] = cell_stats(time_method("planner", items[:20], 0, HeavyStub(PL.ModelProbs(models[0]), False), ph="stub"))
    cells["planner_heavy_maxunits_stub (20 layouts, with network)"] = cell_stats(time_method("planner", items[:20], 0, HeavyStub(PL.ModelProbs(models[0]), True), ph="stub"))
    res["single_process"] = cells
    t_plan = {"planner": cells["planner"]["per_plan_mean_s"], "RAS": cells["RAS"]["per_plan_mean_s"], "RPS": cells["RPS"]["per_plan_mean_s"]}
    t_heavy = {"planner": cells["planner_heavy_maxunits_stub (20 layouts, with network)"]["per_plan_mean_s"], "RAS": t_plan["RAS"], "RPS": t_plan["RPS"]}

    # --- 7 parallel workers: the contention factor on this machine (per-plan time under load / alone)
    chunks = [list(range(i, 60, 7)) for i in range(7)]
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(7) as pool:
        par = pool.map(_par_worker, [(0, c) for c in chunks], chunksize=1)
    par_wall = time.perf_counter() - t0
    alone = cells["planner"]["per_plan_mean_s"]
    f_cont = float(np.mean([p[1] for p in par]) / alone)
    res["parallel_7"] = {"wall_s": par_wall, "per_plan_mean_s_under_load": float(np.mean([p[1] for p in par])), "contention_factor": f_cont}

    # --- deterministic evaluation and TTA (per episode, mean over the 60 layouts)
    ppo_probs = PL.ModelProbs(models[0])
    qnet = QNet(default_obs_space(), N_ACTIONS)
    qfn = q_values_fn(qnet)

    def greedy_ppo(obs, mask):
        p = np.where(mask, ppo_probs(obs, mask), -1.0)
        return int(np.argmax(p))

    def greedy_q(obs, mask):
        return int(np.argmax(np.where(mask, qfn(obs, mask), -np.inf)))
    ev = {"PPO deterministic": time_eval(items, greedy_ppo), "PPO + TTA(8 views, logprob)": time_eval(items, lambda o, m: tta_action(ppo_probs, o, m)),
          "DDQN deterministic": time_eval(items, greedy_q), "DDQN + TTA(8 views, mean)": time_eval(items, lambda o, m: tta_action(qfn, o, m, aggregate="mean"))}
    res["eval_episode_s"] = ev

    # --- B15 session projection (120 FINAL layouts x 10 seeds), ideal /7 and contention-scaled
    def proj(t, f):
        tt = {k: v * f for k, v in t.items()}
        full = PL.project_wall_hours(tt, PL.cost_cells())
        return {"full": full, "fallback": PL.select_fallback(tt)}
    res["session_projection"] = {
        "measured_random_init_ideal": proj(t_plan, 1.0), "measured_random_init_contention": proj(t_plan, f_cont),
        "heavy_maxunits_stub_ideal": proj(t_heavy, 1.0), "heavy_maxunits_stub_contention": proj(t_heavy, f_cont)}
    n_eps = 10 * 120
    other_cpu_s = n_eps * (ev["PPO deterministic"] * 2 + ev["PPO + TTA(8 views, logprob)"] * 2 + ev["DDQN deterministic"] + ev["DDQN + TTA(8 views, mean)"])
    res["other_eval"] = {"cpu_hours": other_cpu_s / 3600, "wall_hours_7_ideal": other_cpu_s / 3600 / 7, "wall_hours_7_contention": other_cpu_s / 3600 / 7 * f_cont,
                         "note": "PPO plain + PPO+D4 deterministic and +TTA, DDQN deterministic and +TTA, 10 seeds x 120 layouts; Oracle / baselines excluded (spec 12.2 row)"}
    assert P.LEDGER_PATH.read_bytes() == before, "ledger changed"
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps({k: res[k] for k in ("single_process", "parallel_7", "eval_episode_s", "other_eval")}, indent=1)[:6000])
    for k, v in res["session_projection"].items():
        print(k, "full wall h:", round(v["full"]["total_wall_hours"], 3), "| fallback step:", v["fallback"]["step"], v["fallback"]["label"], round(v["fallback"]["projection"]["total_wall_hours"], 3))


if __name__ == "__main__":
    sys.exit(main())
