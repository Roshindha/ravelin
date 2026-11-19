"""B13d: set the grid step budget T (SPEC 12.1) from the SHAPED pilot curve (7.4) and write data/v2/step_budget.json. No training, no simulation.
    .venv/bin/python scripts/b13d_budget.py
Inputs: plain slope = shaped pilot (models_v2/shaped_plain_s9201-9203, condition-mean curve, 200k..300k); D4 slope = the only D4 pilot that exists (unshaped, B13b),
r = min per-job steps/s at P = 6 of the shaped throughput measurement (data/v2/b13d_shaped_throughput.json)."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # the repo root is importable when run as `python scripts/<name>.py`
import json
from pathlib import Path

import numpy as np

from breachlens import g7 as G
from rl.budget import step_budget

ROOT = Path(__file__).resolve().parents[1]


def slope(prefix: str, seeds) -> float:
    runs = [G.read_run(ROOT / "models_v2" / f"{prefix}{s}") for s in seeds]
    ts = G.slope_ts(300_000, 25_000)
    return G.slope_per_100k({"score": {t: float(np.mean([r["score"][t] for r in runs])) for t in ts}}, 300_000, 25_000)


def main():
    sp, sd = slope("shaped_plain_s", (9201, 9202, 9203)), slope("pilot_d4_s", (9101, 9102, 9103))
    thr = json.loads((ROOT / "data/v2/b13d_shaped_throughput.json").read_text())
    r = thr["per_job_min"]
    out = step_budget(sp, sd, r)
    out["inputs"] = {"plain": "shaped pilot seeds 9201-9203, condition-mean curve, t=200k..300k (5 points), LS slope per 100k",
                     "d4": "unshaped D4 pilot seeds 9101-9103 (no shaped D4 pilot exists), same window",
                     "r": "min per-job steps/s at P=6, shaped plain, phase B, 60 s warm-up + 180 s window (data/v2/b13d_shaped_throughput.json)"}
    out["sensitivity"] = {"T_if_r_is_unshaped_B10_169.4": step_budget(sp, sd, 169.4)["T"], "T_if_D4_ignored": step_budget(sp, -1.0, r)["T"]}
    (ROOT / "data/v2/step_budget.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
