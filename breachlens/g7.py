"""B13a: the G7 decision, computed from run logs exactly as pinned in Gate changelog (s), (t), (u).

Inputs per run directory (written by rl.train_ppo_v2): val_eval.csv (group ALL rows: timestep, score_mean, d_mean, breach_mean, mean_n_bought),
val_layouts.jsonl (per layout and eval: family, army = 5-vector of unit counts bought), manifest.json, model_identity.json.

  metric per seed        = mean of the TUNE-val mean scores at t = H - 50k, H - 25k, H   (H = 300k, or 600k after the extension)
  pass (all of)          = mean over plain seeds >= bar  AND  >= 2 of 3 seeds > bar  AND  no collapse
  collapse (any seed)    = < 10 distinct armies (5-vectors) over the 60 layouts at t = H, OR one unit type above 90 % of the units bought in EVERY familiar family at once at t = H (changelog (x))
  extension (once)       = G7 failed at 300k AND mean over seeds of the LS slope over the evals t = 200k..300k (5 points), in score per 100k steps, >= +0.02
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

BAR_PATH = Path(__file__).resolve().parent.parent / "data" / "v2" / "g7_bar.json"
MIN_DISTINCT = 10
MAX_SHARE = 0.90
SLOPE_THRESHOLD = 0.02
N_PLAIN = 3


def window(horizon: int, freq: int) -> list:
    return [horizon - 2 * freq, horizon - freq, horizon]


def slope_ts(horizon: int, freq: int) -> list:
    return [horizon - k * freq for k in range(4, -1, -1)]


def read_run(run_dir) -> dict:
    run_dir = Path(run_dir)
    evals, d, b, nb = {}, {}, {}, {}
    with (run_dir / "val_eval.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["group"] == "ALL":
                t = int(r["timestep"])
                evals[t], d[t], b[t], nb[t] = float(r["score_mean"]), float(r["d_mean"]), float(r["breach_mean"]), float(r["mean_n_bought"])
    layouts: dict = {}
    for ln in (run_dir / "val_layouts.jsonl").read_text().splitlines():
        if ln.strip():
            x = json.loads(ln)
            layouts.setdefault(int(x["timestep"]), []).append(x)
    return {"score": evals, "d": d, "breach": b, "n_bought": nb, "layouts": layouts}


def seed_metric(run: dict, horizon: int, freq: int) -> dict:
    ts = window(horizon, freq)
    missing = [t for t in ts if t not in run["score"]]
    if missing:
        raise ValueError(f"missing validation evals at t={missing}")
    f = lambda k: float(np.mean([run[k][t] for t in ts]))
    return {"score": f("score"), "d": f("d"), "breach": f("breach"), "n_bought": f("n_bought"), "evals": ts}


def slope_per_100k(run: dict, horizon: int, freq: int) -> float:
    ts = slope_ts(horizon, freq)
    missing = [t for t in ts if t not in run["score"]]
    if missing:
        raise ValueError(f"missing validation evals at t={missing}")
    return float(np.polyfit(np.array(ts, dtype=float), np.array([run["score"][t] for t in ts]), 1)[0] * 100_000)


def collapse(run: dict, t: int) -> dict:
    rows = run["layouts"].get(t)
    if not rows:
        raise ValueError(f"no per-layout rows at t={t}")
    distinct = len({tuple(int(c) for c in r["army"]) for r in rows})
    fam_share, fam_type, shares = {}, {}, []
    for fam in sorted({r["family"] for r in rows}):
        tot = np.sum([r["army"] for r in rows if r["family"] == fam], axis=0).astype(float)
        sh = tot / tot.sum() if tot.sum() > 0 else np.ones_like(tot)
        shares.append(sh)
        fam_share[fam], fam_type[fam] = float(sh.max()), int(sh.argmax())
    mimic = [int(k) for k in range(len(shares[0])) if all(sh[k] > MAX_SHARE for sh in shares)]       # one type above 90 % in every family at once
    return {"t": t, "distinct_armies": distinct, "distinct_ok": distinct >= MIN_DISTINCT, "max_type_share_per_family": fam_share,
            "max_type_per_family": fam_type, "single_type_in_every_family_above_90pct": mimic, "share_ok": not mimic}


def judge(plain: dict, bar: float, horizon: int, freq: int, expect_seeds: int | None = N_PLAIN) -> dict:
    """G7 criteria at one horizon. `plain` = {seed: run dict}. Pass/fail per criterion plus the numbers."""
    per = {s: seed_metric(r, horizon, freq) for s, r in plain.items()}
    col = {s: collapse(r, horizon) for s, r in plain.items()}
    mean = float(np.mean([m["score"] for m in per.values()]))
    above = [s for s, m in per.items() if m["score"] > bar]
    crit = {"mean_ge_bar": mean >= bar, "two_of_three_above_bar": len(above) >= 2,
            "distinct_armies_ge_10_every_seed": all(c["distinct_ok"] for c in col.values()),
            "no_single_type_above_90pct_in_every_family_every_seed": all(c["share_ok"] for c in col.values())}
    return {"horizon": horizon, "window": window(horizon, freq), "bar": bar, "mean_score": mean, "per_seed": per, "seeds_above_bar": above, "collapse": col,
            "criteria": crit, "pass": all(crit.values()), "valid_g7": expect_seeds is None or len(per) == expect_seeds}


def decide(plain: dict, bar: float, horizon: int, freq: int, expect_seeds: int | None = N_PLAIN, allow_extension: bool = True) -> dict:
    """Full decision. horizon = 300k (first look; may return EXTEND) or 600k (after the extension; the 300k look must have licensed it).
    `allow_extension=False` (the SPEC 7.4 shaped pilot, Gate changelog (z)): the 300k look is final, EXTEND is never returned, the slope is reported only."""
    if not allow_extension and horizon > 300_000:
        raise ValueError("the shaped pilot gets no extension (Gate changelog (z))")
    first_h = horizon if horizon <= 300_000 else 300_000
    first = judge(plain, bar, first_h, freq, expect_seeds)
    slopes = {s: slope_per_100k(r, first_h, freq) for s, r in plain.items()}
    mean_slope = float(np.mean(list(slopes.values())))
    out = {"first_look": first, "slopes_per_100k": slopes, "mean_slope_per_100k": mean_slope, "slope_threshold": SLOPE_THRESHOLD,
           "slope_ge_threshold": mean_slope >= SLOPE_THRESHOLD}
    if horizon == first_h:
        out["verdict"] = "PASS" if first["pass"] else ("EXTEND" if (out["slope_ge_threshold"] and allow_extension) else "FAIL")
        return out
    if first["pass"] or not out["slope_ge_threshold"]:
        out["verdict"] = "INVALID: the extension was not licensed by the 300k look (G7 passed, or mean slope < +0.02)"
        return out
    second = judge(plain, bar, horizon, freq, expect_seeds)
    out["second_look"] = second
    out["verdict"] = "PASS" if second["pass"] else "FAIL"
    return out
