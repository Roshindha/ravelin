"""B7: PRELIMINARY env-only throughput (masked-random agent, one process, no network). The full estimate is B10.

Usage: python scripts_b7_throughput.py [seconds_per_config=30]. Writes data/v2/env_throughput.json. Layouts are DEV-train / TUNE-train only.
"""
import json
import platform
import resource
import sys
import time

import numpy as np

from breachlens.env import BreachEnv
from breachlens.sim import FIELD_CACHE_SIZE, SMALL_CACHE_SIZE


def run(seconds, policy="uniform", **kw):
    """policy "uniform": uniform over legal actions (stops buying early: ~3-5 units). "buy_all": uniform over legal non-END_BUY actions
    while any exists (buys until broke / cap, i.e. 12-20 unit armies = the slow battles)."""
    env = BreachEnv(**kw)
    rng = np.random.default_rng(0)
    steps = eps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        env.reset(seed=int(rng.integers(1 << 30)))
        done = False
        while not done:
            legal = np.flatnonzero(env.action_masks())
            if policy == "buy_all" and len(legal) > 1:
                legal = legal[legal != 265]
            a = int(rng.choice(legal))
            _, _, done, _, _ = env.step(a)
            steps += 1
        eps += 1
    dt = time.perf_counter() - t0
    fields = [len(s.P._fields) for s in env._static.values()]
    return {"steps_per_s": round(steps / dt, 1), "episodes_per_s": round(eps / dt, 2), "steps_per_episode": round(steps / eps, 2),
            "seconds": round(dt, 1), "cached_layouts": len(env._static), "field_cache_entries_max": max(fields) if fields else 0,
            "field_cache_entries_mean": round(float(np.mean(fields)), 1) if fields else 0}


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    out = {"python": platform.python_version(), "machine": platform.machine(), "seconds_per_config": secs,
           "field_cache_size": FIELD_CACHE_SIZE, "small_cache_size": SMALL_CACHE_SIZE, "runs": {}}
    cfgs = {
        "DEV-train, full game (BUY->DEPLOY), layout_cache=64": dict(source="DEV-train", role="train", layout_cache=64),
        "DEV-train, prebought=random (phase A), layout_cache=64": dict(source="DEV-train", role="train", prebought="random", layout_cache=64),
        "DEV-train, full game, layout_cache=0 (Prepared rebuilt every episode)": dict(source="DEV-train", role="train", layout_cache=0),
        "TUNE-train, full game, layout_cache=64": dict(source="TUNE-train", role="train", layout_cache=64),
        "DEV-train, buy-until-broke random (large armies), layout_cache=64": dict(source="DEV-train", role="train", layout_cache=64, policy="buy_all"),
    }
    for name, kw in cfgs.items():
        out["runs"][name] = run(secs, **kw)
        print(f"{name}: {out['runs'][name]}", flush=True)
    out["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    with open("data/v2/env_throughput.json", "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print("max RSS MB", out["max_rss_mb"])
