"""B13e report for the DDQN train_freq pilot (row (p), shaped per row (cc)). NON-GATING: prints the row-(p) selection and diagnostics, never PASS / FAIL.

    .venv/bin/python scripts_b13_ddqn_report.py [--models-dir models_v2] [--horizon 300000] [--freq 25000] [--seeds 9301,9302]

Selection (row p): per arm, mean TUNE-val true score ('ALL' row of val_eval.csv) over the last 3 evals (t = H-50k, H-25k, H) and both seeds; higher arm wins;
if |difference| < 0.01 choose train_every (train_freq) 8. Reads logs only; writes nothing.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TIE = 0.01
ARMS = (4, 8)


def read_run(run_dir: Path) -> dict:
    ev = {}
    with (run_dir / "val_eval.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["group"] == "ALL":
                ev[int(r["timestep"])] = {k: float(r[k]) for k in ("score_mean", "breach_mean", "d_mean", "mean_n_bought")}
    armies = {}
    with (run_dir / "val_layouts.jsonl").open() as fh:
        for ln in fh:
            r = json.loads(ln)
            armies.setdefault(int(r["timestep"]), set()).add(tuple(int(c) for c in r["army"]))
    return {"eval": ev, "distinct": {t: len(s) for t, s in armies.items()}}


def last3(horizon: int, freq: int) -> list:
    return [horizon - 2 * freq, horizon - freq, horizon]


def arm_stats(runs: dict, horizon: int, freq: int) -> dict:
    """runs = {seed: read_run(...)}; raises KeyError if a needed eval is missing (incomplete pilot)."""
    ts = last3(horizon, freq)
    per_seed = {s: float(np.mean([r["eval"][t]["score_mean"] for t in ts])) for s, r in runs.items()}
    return {"per_seed": per_seed, "mean": float(np.mean(list(per_seed.values())))}


def choose(mean4: float, mean8: float, tie: float = TIE) -> dict:
    diff = round(mean4 - mean8, 12)                                  # exact-boundary differences must not depend on float noise
    if abs(diff) < tie:
        return {"train_freq": 8, "rule": f"|difference| {abs(diff):.4f} < {tie}: tie -> 8"}
    return {"train_freq": 4 if diff > 0 else 8, "rule": f"higher mean score: |difference| {abs(diff):.4f} >= {tie}"}


def report(models: Path, seeds: list, horizon: int, freq: int, tmpl: str = "ddqn_pilot_f{freq}_s{seed}") -> str:
    runs = {a: {s: read_run(models / tmpl.format(freq=a, seed=s)) for s in seeds} for a in ARMS}
    stats = {a: arm_stats(runs[a], horizon, freq) for a in ARMS}
    ts = last3(horizon, freq)
    out = [f"DDQN train_freq pilot (rows p / cc; shaped training, TUNE-val true score) -- NON-GATING, no bar applies; H = {horizon}, evals {ts}"]
    for a in ARMS:
        out.append(f"\narm train_freq {a}:")
        out.append("  seed  mean_last3  | at H: score   breach     D   bought  distinct_armies")
        for s in seeds:
            r = runs[a][s]
            e = r["eval"][horizon]
            out.append(f"  {s}  {stats[a]['per_seed'][s]:.4f}      | {e['score_mean']:.4f}  {e['breach_mean']:.3f}  {e['d_mean']:.3f}  {e['mean_n_bought']:5.2f}  {r['distinct'].get(horizon, 0)}")
        out.append(f"  arm mean (last 3 evals, both seeds) = {stats[a]['mean']:.4f}")
    c = choose(stats[4]["mean"], stats[8]["mean"])
    out.append(f"\ndifference (4 - 8) = {stats[4]['mean'] - stats[8]['mean']:+.4f}   rule: {c['rule']}")
    out.append(f"CHOSEN train_freq = {c['train_freq']}   (row p selection; frozen before the training grid)")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default=str(ROOT / "models_v2"))
    p.add_argument("--horizon", type=int, default=300_000)
    p.add_argument("--freq", type=int, default=25_000)
    p.add_argument("--seeds", default="9301,9302")
    a = p.parse_args(argv)
    try:
        print(report(Path(a.models_dir), [int(x) for x in a.seeds.split(",")], a.horizon, a.freq))
    except (FileNotFoundError, KeyError) as e:
        print(f"INCOMPLETE: missing run files / evals ({e!r}); no selection printed")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
