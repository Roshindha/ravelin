"""B2 simulator speed gate (docs/SPEC_V2.md section 11, 'B2 simulator speed gate').

(a) battle time: 200 random legal armies (>= 10 units) across the 4 provisional families, single core.
(b) random-masked-agent env loop (simulator included, no network, no observation building) in 1 process,
    then 6 parallel processes. Writes data/v2/speed.json. Layouts: the B4 real design layouts.
"""
from __future__ import annotations

import cProfile
import json
import multiprocessing as mp
import platform
import pstats
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from breachlens import config
from breachlens import layouts as Y
from breachlens.armies import free_tiles, random_army
from breachlens.actions import decode_action, encode_buy, encode_deploy
from breachlens.constants import BUDGET, END_BUY, MAX_UNITS, N_UNIT_TYPES
from breachlens.sim import Prepared, simulate

OUT = Path(__file__).parent / "data" / "v2" / "speed.json"
GATE = {"mean_ms_max": 60.0, "p95_ms_max": 150.0, "steps_per_s_min": 300.0}
N_TYPES = N_UNIT_TYPES  # B3: the Mender is in the random armies
N_LAYOUTS_PER_FAMILY = 25


def build_layouts():
    """B4: the REAL design layouts (first 25 of TUNE-train per familiar family + first 25 of LAB-DESIGN)."""
    out = []
    for f in Y.FAMILIES:
        pool = Y.load_design_set("LAB-DESIGN" if f == "labyrinth" else "TUNE-train", None if f == "labyrinth" else f)
        out += [(f, rec["seed"], L) for rec, L in pool[:N_LAYOUTS_PER_FAMILY]]
    return out


def stress_cfg(hp_mult=20):
    """Hand-built worst case: same stats but unit hp x20 so battles run long (up to the 240-tick cap)."""
    raw = json.loads(config.STATS_PATH.read_text())
    for u in raw["units"].values():
        u["hp"] *= hp_mult
    return config._build(raw)


def battle_times(warm: bool, n=200, seed=0, cfg=None, n_types=N_TYPES):
    rng = np.random.default_rng(seed)
    lays = build_layouts()
    prepared = {(f, s): Prepared(L, cfg or config.CFG) for f, s, L in lays} if (warm or cfg) else None
    # one warm-up pass on the same armies so warm means "layout caches populated", as in training
    jobs = []
    for i in range(n):
        f, s, L = lays[int(rng.integers(len(lays)))]
        jobs.append((f, s, L, random_army(rng, L, min_units=10, n_types=n_types)))
    if warm:
        for f, s, L, a in jobs:
            simulate(prepared[(f, s)], a)
    ts, ticks = [], []
    for f, s, L, a in jobs:
        t0 = time.perf_counter()
        r = simulate(prepared[(f, s)] if prepared else L, a)
        ts.append((time.perf_counter() - t0) * 1000)
        ticks.append(r.ticks)
    ts = np.array(ts)
    return {"mean_ms": float(ts.mean()), "p95_ms": float(np.percentile(ts, 95)), "max_ms": float(ts.max()),
            "mean_ticks": float(np.mean(ticks)), "mean_units": float(np.mean([len(a) for *_, a in jobs])), "n": n}


def episode(rng, prepared_layout, layout):
    """One random legal BUY -> DEPLOY -> BATTLE episode via the action codec. Returns (steps, result)."""
    gold, owned, steps, army = BUDGET, [], 0, []
    costs = [config.CFG.units[t].cost for t in range(N_TYPES)]
    while True:  # BUY phase: uniform random legal buy, END_BUY with p=0.1 once >= 1 unit
        legal = [encode_buy(t) for t in range(N_TYPES) if costs[t] <= gold] if len(owned) < MAX_UNITS else []
        if owned and (not legal or rng.random() < 0.1):
            a = END_BUY
        else:
            a = legal[int(rng.integers(len(legal)))]
        steps += 1
        kind, t, _ = decode_action(a)
        if kind == "end_buy":
            break
        owned.append(t)
        gold -= costs[t]
    free = set(free_tiles(layout))
    remaining = list(owned)
    while remaining:  # DEPLOY phase: random legal (type, free tile)
        t = remaining.pop(int(rng.integers(len(remaining))))
        tile = sorted(free)[int(rng.integers(len(free)))]
        free.discard(tile)
        a = encode_deploy(t, tile)
        _, tt, tl = decode_action(a)
        army.append((tt, tl))
        steps += 1
    steps += 1  # the final deploy step triggers the battle
    return steps, simulate(prepared_layout, army)


def env_loop(seed: int, seconds: float):
    rng = np.random.default_rng(seed)
    lays = build_layouts()
    prepared = {(f, s): Prepared(L) for f, s, L in lays}
    for f, s, L in lays[:20]:  # warm-up
        episode(rng, prepared[(f, s)], L)
    steps = eps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        f, s, L = lays[int(rng.integers(len(lays)))]
        st, _ = episode(rng, prepared[(f, s)], L)
        steps += st
        eps += 1
    dt = time.perf_counter() - t0
    return steps, eps, dt


def _worker(args):
    return env_loop(*args)


def parallel(n_proc: int, seconds: float):
    with mp.get_context("spawn").Pool(n_proc) as pool:
        res = pool.map(_worker, [(100 + i, seconds) for i in range(n_proc)])
    steps = sum(r[0] for r in res)
    eps = sum(r[1] for r in res)
    dt = max(r[2] for r in res)
    return {"procs": n_proc, "steps_per_s": steps / dt, "episodes_per_s": eps / dt,
            "per_proc_steps_per_s": [r[0] / r[2] for r in res]}


def profile_top(n=3, cfg=None):
    rng = np.random.default_rng(5)
    lays = build_layouts()
    prepared = {(f, s): Prepared(L, cfg or config.CFG) for f, s, L in lays}
    jobs = [(prepared[(f, s)], random_army(rng, L, min_units=10, n_types=N_TYPES)) for f, s, L in
            [lays[int(rng.integers(len(lays)))] for _ in range(200)]]
    for P, a in jobs:
        simulate(P, a)
    pr = cProfile.Profile()
    pr.enable()
    for P, a in jobs:
        simulate(P, a)
    pr.disable()
    st = pstats.Stats(pr)
    rows = []
    total = st.total_tt
    for (fn, line, name), (cc, nc, tt, ct, _) in st.stats.items():
        rows.append({"func": f"{Path(fn).name}:{line}:{name}", "tottime_s": tt, "cumtime_s": ct, "calls": nc})
    rows.sort(key=lambda r: -r["tottime_s"])
    for r in rows:
        r["pct_of_total"] = 100 * r["tottime_s"] / total
    return rows[:n]


def main(seconds=20.0):
    out = {"sim_version": config.sim_version(), "gate": GATE, "python": sys.version.split()[0],
           "machine": platform.platform(), "cpu_count": mp.cpu_count(),
           "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "layouts": f"B4 real design layouts ({N_LAYOUTS_PER_FAMILY}/family: TUNE-train x3 + LAB-DESIGN)", "mender": "included (5 unit types in random armies)"}
    out["battle_warm"] = battle_times(True)
    out["battle_warm_4types_no_mender_B2_workload"] = battle_times(True, n_types=4)   # same workload as the B2 numbers
    out["battle_cold"] = battle_times(False)
    out["battle_stress_hp20x_warm"] = battle_times(True, cfg=stress_cfg())
    steps, eps, dt = env_loop(1, seconds)
    out["env_1proc"] = {"steps_per_s": steps / dt, "episodes_per_s": eps / dt, "steps_per_episode": steps / eps}
    out["env_6proc"] = parallel(6, seconds)
    out["top3_hotspots"] = profile_top(3)
    out["top3_hotspots_stress"] = profile_top(3, cfg=stress_cfg())
    w = out["battle_warm"]
    out["passed"] = bool(w["mean_ms"] <= GATE["mean_ms_max"] and w["p95_ms"] <= GATE["p95_ms_max"]
                         and out["env_1proc"]["steps_per_s"] >= GATE["steps_per_s_min"])
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 20.0)
