"""B8 evaluator: runs a policy on a set under a role and reports breach rate (headline) and mean score with bootstrap 95 % CIs, by group.

* Scores come ONLY from `breachlens.score` (outcomes are integer d_num / d_den / core flags; one int -> float conversion).
* Bootstrap: over layouts, 10 000 resamples, seed 8080 (`gates.bootstrap_mean`, thresholds in data/v2/gate_thresholds.json).
* Every layout row and every summary row carries REQUIRED_META (score_version, sim_version, generator_hash, pool_sha256, git_commit,
  seed, role, set); `check_meta` raises if any is missing, and the writers re-check.
* Layouts come from the pool loaders (role guards, ledger). role='eval' needs unlock=True, writes one 'eval' ledger entry per evaluation and
  loads FINAL / transfer only. `items=` supplies already-loaded (record, layout) pairs (synthetic layouts in tests); a non-eval role
  still refuses FINAL ids.
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import pool as P
from .config import sim_version
from .gates import bootstrap_mean
from .score import SCORE_VERSION, score_value
from .sim import Prepared

REQUIRED_META = ("score_version", "sim_version", "generator_hash", "pool_sha256", "git_commit", "seed", "role", "set")
EVAL_SEED = 8080                                   # evaluator seed = bootstrap seed (spec 3.4); Random derives its own per-layout seeds from layout ids
DEV_SETS = ("DEV-train", "DEV-validation")
DESIGN = ("TUNE-train", "TUNE-val", "LAB-DESIGN")
FINAL = P.FINAL_SPLITS


def run_meta(seed: int, role: str, set_name: str) -> dict:
    gh = json.loads(P.GENERATOR_HASH_PATH.read_text())["generator_hash"]
    return {"score_version": SCORE_VERSION, "sim_version": sim_version(), "generator_hash": gh, "pool_sha256": P.pool_sha256(),
            "git_commit": P._git("rev-parse", "HEAD"), "git_dirty": P.git_dirty(), "seed": int(seed), "role": role, "set": set_name}


def is_final_layout(rec: dict) -> bool:
    """True for a FINAL / transfer layout record (family + seed). Lets rl/ modules guard themselves without importing the pool loaders."""
    return P.is_final(rec["family"], rec["seed"])


def check_meta(row: dict) -> None:
    missing = [k for k in REQUIRED_META if k not in row or row[k] is None or row[k] == "" or row[k] == "unknown"]
    if missing:
        raise ValueError(f"row is missing required metadata: {missing}")


def load_set(set_name: str, role: str, unlock: bool = False, entry_point: str = "breachlens.evaluator") -> list:
    """[(record, layout dict)] through the pool loaders (role guards and ledger apply)."""
    if set_name in FINAL:
        if role != "eval":
            raise PermissionError(f"role {role!r} may not load {set_name}")
        recs = P.load_pool_records(set_name, "eval", purpose="eval", unlock=unlock, entry_point=entry_point)
        return [(r, P.load_layout(r, "eval", purpose="eval", unlock=unlock, entry_point=entry_point)) for r in recs]
    if role == "eval":
        raise PermissionError("role 'eval' loads FINAL splits only")
    recs = P.load_pool_records(set_name, role, entry_point=entry_point) if set_name in DEV_SETS else P.design_records(set_name, role)
    return [(r, P.load_layout(r, role, entry_point=entry_point)) for r in recs]


def load_dev_layout(split: str, index: int | None = None, layout_id: str | None = None, entry_point: str = "breachlens.evaluator") -> tuple:
    """(record, layout) of ONE DEV-train / DEV-validation layout (role train / validation; B16a explainability). Every other split raises
    PermissionError before anything is loaded; a FINAL / transfer / design-set id is refused by the pool role guard."""
    roles = {"DEV-train": "train", "DEV-validation": "validation"}
    if split not in roles:
        raise PermissionError(f"only DEV familiar layouts (DEV-train, DEV-validation) may be loaded here; {split!r} is refused")
    role = roles[split]
    recs = P.load_pool_records(split, role, entry_point=entry_point)
    if layout_id is not None:
        P.assert_dev_only([layout_id], role)
        hit = [r for r in recs if r["id"] == layout_id]
        if not hit:
            raise PermissionError(f"layout {layout_id!r} is not in {split}")
        rec = hit[0]
    else:
        if index is None or not 0 <= index < len(recs):
            raise ValueError(f"index must be in 0..{len(recs) - 1} for {split}")
        rec = recs[index]
    return rec, P.load_layout(rec, role, entry_point=entry_point)


def dev_layout_ids(split: str, entry_point: str = "breachlens.evaluator") -> list:
    """(B17) Ids of one DEV-train / DEV-validation split in pool order, through the role-guarded loader (ids only; every other split raises PermissionError)."""
    roles = {"DEV-train": "train", "DEV-validation": "validation"}
    if split not in roles:
        raise PermissionError(f"only DEV familiar splits (DEV-train, DEV-validation) may be listed here; {split!r} is refused")
    return [r["id"] for r in P.load_pool_records(split, roles[split], entry_point=entry_point)]


def _work(args):
    policy, rec, layout = args
    t0 = time.perf_counter()
    eps = policy.episodes(rec, layout, Prepared(layout))
    sc = [score_value(e["d_num"], e["d_den"], e["core"]) for e in eps]
    return {"id": rec["id"], "family": rec["family"], "layout_seed": int(rec["seed"]), "policy": policy.name, "n_episodes": len(eps),
            "score": float(np.mean(sc)), "d": float(np.mean([e["d_num"] / max(e["d_den"], 1) for e in eps])),
            "breach": float(np.mean([e["core"] for e in eps])), "n_bought": float(np.mean([e["n_bought"] for e in eps])),
            "n_deployed": float(np.mean([e["n_deployed"] for e in eps])), "policy_seeds": json.dumps(policy.seeds(rec)),
            "extra": json.dumps({k: v for k, v in eps[0].items() if k in ("chosen_type", "oracle_sims", "oracle_start", "oracle_gold", "reason", "cached",
                                                                          "chosen_index", "n_sims", "plan_seconds")}),
            **(policy.provenance() if hasattr(policy, "provenance") else {}),             # B12: N, method, policy hash of the planner / controls
            "episodes": eps, "seconds": time.perf_counter() - t0}


def _groups(rows: list) -> OrderedDict:
    g = OrderedDict()
    for fam in OrderedDict.fromkeys(r["family"] for r in rows):
        g[fam] = [r for r in rows if r["family"] == fam]
    g["ALL"] = list(rows)
    return g


def summarize(rows: list, meta: dict, policy_name: str) -> list:
    out = []
    for grp, rs in _groups(rows).items():
        row = {"policy": policy_name, "group": grp, "n_layouts": len(rs), **meta}
        for key in ("breach", "score", "d"):
            b = bootstrap_mean([r[key] for r in rs])
            row[f"{key}_mean"], row[f"{key}_lo95"], row[f"{key}_hi95"] = b["mean"], b["lo95"], b["hi95"]
        row["mean_n_bought"] = float(np.mean([r["n_bought"] for r in rs]))
        row["mean_n_deployed"] = float(np.mean([r["n_deployed"] for r in rs]))
        out.append(row)
    return out


def evaluate(policy, set_name: str, role: str, *, unlock: bool = False, items: list | None = None, procs: int = 1, seed: int = EVAL_SEED,
             entry_point: str = "breachlens.evaluator") -> dict:
    """Returns {'rows': per-layout rows, 'summary': per-group rows, 'meta': ..., 'seconds': wall time}. Every row passes `check_meta`."""
    if role not in P.ROLES or role == "build":
        raise ValueError(f"evaluator role must be one of train / validation / tuning / design / eval, got {role!r}")
    if role == "eval" and not unlock:
        raise PermissionError("role 'eval' needs unlock=True")
    meta = run_meta(seed, role, set_name)
    check_meta(meta)
    with P.eval_session():                                                             # loader entries + the summary entry share one session_id
        return _evaluate(policy, set_name, role, unlock, items, procs, meta, entry_point)


def _evaluate(policy, set_name, role, unlock, items, procs, meta, entry_point) -> dict:
    if items is None:
        items = load_set(set_name, role, unlock=unlock, entry_point=entry_point)        # FINAL loads write the ledger inside the loaders
    elif role != "eval":
        P.assert_dev_only([r["id"] for r, _ in items], role)                           # a non-eval role never scores FINAL / transfer / LAB-DESIGN ids
    if hasattr(policy, "precheck"):
        policy.precheck([r for r, _ in items])                                          # B12: a policy that would refuse a layout does so BEFORE any ledger entry
    if role == "eval":
        P.log_final_access("eval", entry_point, {"action": "evaluate", "set": set_name, "policy": policy.name, "n": len(items)})
    t0 = time.perf_counter()
    tasks = [(policy, r, L) for r, L in items]
    if procs > 1 and len(tasks) > 1:
        with mp.get_context("spawn").Pool(procs) as pool:
            res = pool.map(_work, tasks, chunksize=1)
    else:
        res = [_work(t) for t in tasks]
    rows, episodes = [], {}
    for r in res:
        episodes[r["id"]] = r.pop("episodes")
        rows.append({**r, **meta})
    for r in rows:
        check_meta(r)
    summary = summarize(rows, {**meta, **(policy.provenance() if hasattr(policy, "provenance") else {})}, policy.name)
    for r in summary:
        check_meta(r)
    return {"rows": rows, "summary": summary, "meta": meta, "episodes": episodes, "seconds": time.perf_counter() - t0}


def write_csv(rows: list, path) -> None:
    for r in rows:
        check_meta(r)
    keys = list(OrderedDict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_json(rows: list, path) -> None:
    for r in rows:
        check_meta(r)
    Path(path).write_text(json.dumps(rows, indent=1, sort_keys=True))
