"""One writer for every trainer's validation rows (B14a item 1). PPO and DDQN both call `write_validation`, so `val_eval.csv` has the same columns,
including the run provenance, whatever the algorithm; `val_layouts.jsonl` is written here too.

Provenance columns (taken from the evaluator's `run_meta`, which is already on every summary row): score_version, sim_version, generator_hash,
pool_sha256, git_commit, git_dirty, seed, role, set. A row that lacks one of them (missing, None, empty or "unknown") is refused before anything is
written, so a trainer cannot silently drop provenance.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

METRIC_FIELDS = ["timestep", "group", "n_layouts", "breach_mean", "breach_lo95", "breach_hi95", "score_mean", "score_lo95", "score_hi95", "d_mean",
                 "mean_n_bought", "mean_n_deployed", "eval_seconds"]
PROVENANCE_FIELDS = ["score_version", "sim_version", "generator_hash", "pool_sha256", "git_commit", "git_dirty", "seed", "role", "set"]
FIELDS = METRIC_FIELDS + PROVENANCE_FIELDS


def missing_fields(path) -> list:
    """Names of FIELDS absent from the header of a validation CSV, plus provenance columns that are empty in any row."""
    with Path(path).open(newline="") as fh:
        rd = csv.DictReader(fh)
        miss = [f for f in FIELDS if f not in (rd.fieldnames or [])]
        for row in rd:
            miss += [f"{f} (empty)" for f in PROVENANCE_FIELDS if f in row and (row[f] in ("", None, "unknown"))]
    return sorted(set(miss))


def write_validation(csv_path: Path, jsonl_path: Path, t: int, res: dict, seconds: float) -> dict:
    """Append one evaluation (`res` = breachlens.evaluator.evaluate output) at timestep `t`; returns the ALL-group summary row."""
    for r in res["summary"]:
        bad = [k for k in PROVENANCE_FIELDS if k not in r or r[k] in (None, "", "unknown")]
        if bad:
            raise ValueError(f"validation summary row {r.get('group')!r} lacks provenance fields {bad}")
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in res["summary"]:
            w.writerow({"timestep": t, **{k: r[k] for k in FIELDS if k in r}, "eval_seconds": round(seconds, 2)})
    with jsonl_path.open("a") as fh:
        for r in res["rows"]:
            ep = res["episodes"][r["id"]][0]
            fh.write(json.dumps({"timestep": t, "id": r["id"], "family": r["family"], "score": r["score"], "d": r["d"], "breach": r["breach"],
                                 "army": ep["army"], "n_deployed": ep["n_deployed"]}) + "\n")
    return next(r for r in res["summary"] if r["group"] == "ALL")
