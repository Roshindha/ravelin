"""Stat-dependent acceptance rules (spec 5.1 / B5): `too_hard` and `single_type_trivial`, and the TUNE / LAB-DESIGN builder that applies
them on top of the structural rules of `layouts.py` by drawing further seeds from the same streams.

Only streams T and L are walked. Stream F (pool / FINAL) is never generated or loaded here (`layouts.stream_of` guard).
Rules (thresholds in data/v2/gate_thresholds.json, "acceptance"):
  * single_type_trivial : the best single-unit army (scripted deployment) scores > `single_type_trivial_best_single_gt` (0.99).
  * too_hard            : an acceptance Oracle (oracle.py, rng [seed, 5151], warm starts) finds no breach.
A layout is rejected if EITHER rule fires (both are always evaluated so the per-rule rates are exact).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

from . import layouts as Y
from . import oracle as O
from .armies import greedy_types, place_on_border, single_unit_types
from .config import sim_version
from .sim import Prepared, simulate

THRESHOLDS_PATH = Path(__file__).resolve().parents[1] / "data" / "v2" / "gate_thresholds.json"


def thresholds(path: Path = THRESHOLDS_PATH) -> dict:
    return json.loads(Path(path).read_text())


BULWARK, MENDER_T = 2, 4
TEMPLATE_BULWARK_MENDER = [BULWARK] * 6 + [MENDER_T] * 2


def warm_starts() -> tuple:
    """Pre-registered Oracle warm starts: 5 single-unit armies, the Bulwark+Mender template, the Greedy army (default start tile)."""
    arms = [single_unit_types(t) for t in range(5)] + [list(TEMPLATE_BULWARK_MENDER), greedy_types()]
    return tuple((tuple(a), None) for a in arms)


def stat_check(L: dict, seed: int, prepared=None, n_random: int | None = None, n_refine: int | None = None) -> dict:
    """Stat-dependent acceptance data for one structurally accepted layout."""
    th = thresholds()["acceptance"]
    P = prepared or Prepared(L)
    singles = [simulate(P, place_on_border(L, single_unit_types(t))).score for t in range(5)]
    orc = O.oracle_search(L, O.oracle_rng(seed, O.ACCEPT_TAG), n_random=th["oracle_n_random"] if n_random is None else n_random,
                          n_refine=th["oracle_n_refine"] if n_refine is None else n_refine, warm_starts=warm_starts(), prepared=P)
    best = max(singles)
    trivial = best > th["single_type_trivial_best_single_gt"]
    too_hard = not orc.breach
    return {"singles": singles, "best_single": best, "trivial": trivial, "oracle_score": orc.score, "oracle_breach": orc.breach,
            "too_hard": too_hard, "reasons": (["single_type_trivial"] if trivial else []) + (["too_hard"] if too_hard else [])}


def _eval_candidate(args):
    family, seed = args
    L = Y.candidate(family, seed)
    bad = Y.check(L, family)
    if bad:
        return {"seed": seed, "bad": bad}
    return {"seed": seed, "bad": [], "hash": Y.content_hash(L), "stat": stat_check(L, seed)}


def _walk(stream: str, family: str, n: int, taken: set, pool, block: int = 32):
    """Sequential semantics (a pure function of the stream and the rules), evaluated in parallel blocks."""
    lo, hi = Y.stream_range(stream, family)
    assert stream in ("T", "L"), "stream F is never walked here"
    out, seed = [], lo
    rej = {"structural": {}, "structural_rejected": 0, "duplicates": 0, "stat_evaluated": 0, "single_type_trivial": 0, "too_hard": 0, "stat_rejected": 0,
           "stat_rejected_seeds": []}
    while len(out) < n:
        seeds = list(range(seed, min(seed + block, hi)))
        if not seeds:
            raise RuntimeError("stream exhausted")
        res = pool.map(_eval_candidate, [(family, s) for s in seeds]) if pool else [_eval_candidate((family, s)) for s in seeds]
        for r in res:
            if len(out) >= n:
                break
            seed = r["seed"] + 1
            if r["bad"]:
                rej["structural_rejected"] += 1
                for b in r["bad"]:
                    rej["structural"][b] = rej["structural"].get(b, 0) + 1
                continue
            if r["hash"] in taken:
                rej["duplicates"] += 1
                continue
            st = r["stat"]
            rej["stat_evaluated"] += 1
            rej["single_type_trivial"] += st["trivial"]
            rej["too_hard"] += st["too_hard"]
            if st["reasons"]:
                rej["stat_rejected"] += 1
                rej["stat_rejected_seeds"].append(r["seed"])
                continue
            taken.add(r["hash"])
            out.append({"id": f"{family}-{r['seed']}", "family": family, "seed": r["seed"], "content_hash": r["hash"]})
    rej["stat_rejection_rate"] = rej["stat_rejected"] / max(rej["stat_evaluated"], 1)
    rej["flag_above_50pct"] = rej["stat_rejection_rate"] > thresholds()["acceptance"]["max_stat_rejection_rate_flag"]
    return out, seed - lo, rej


def build_accepted_design_sets(sizes: dict | None = None, lab_size: int | None = None, procs: int | None = None) -> dict:
    """TUNE-train / TUNE-val (per familiar family) and LAB-DESIGN under structural + stat-dependent rules. Same schema as
    `layouts.build_design_sets()` plus `stat_acceptance`. The walk is sequential in seed order, so a smaller build is a prefix of the full one."""
    sizes = sizes or Y.SET_SIZES
    lab_size = Y.LAB_DESIGN_SIZE if lab_size is None else lab_size
    procs = procs or min(8, os.cpu_count() or 1)
    taken: set = set()
    sets = {"TUNE-train": [], "TUNE-val": [], "LAB-DESIGN": []}
    used, stat = {}, {}
    pool = mp.get_context("spawn").Pool(procs) if procs > 1 else None
    try:
        for fam in Y.FAMILIAR:
            recs, n_used, rej = _walk("T", fam, sizes["TUNE-train"] + sizes["TUNE-val"], taken, pool)
            sets["TUNE-train"] += [dict(x, set="TUNE-train") for x in recs[:sizes["TUNE-train"]]]
            sets["TUNE-val"] += [dict(x, set="TUNE-val") for x in recs[sizes["TUNE-train"]:]]
            used[fam] = {"stream": "T", "seed_lo": Y.stream_range("T", fam)[0], "seeds_consumed": n_used}
            stat[fam] = rej
        recs, n_used, rej = _walk("L", "labyrinth", lab_size, taken, pool)
        sets["LAB-DESIGN"] = [dict(x, set="LAB-DESIGN") for x in recs]
        used["labyrinth"] = {"stream": "L", "seed_lo": Y.stream_range("L", "labyrinth")[0], "seeds_consumed": n_used}
        stat["labyrinth"] = rej
    finally:
        if pool:
            pool.close(); pool.join()
    structural = {f: {"structural_rejected": s["structural_rejected"], **s["structural"], "duplicates": s["duplicates"]} for f, s in stat.items()}
    return {
        "streams": {s: {"base": b, "stride_per_family": Y.STREAM_STRIDE, "ranges": {f: list(Y.stream_range(s, f)) for f in Y.FAMILIES}}
                    for s, b in Y.STREAM_BASE.items()},
        "stream_use": used, "rejections": structural, "stat_acceptance": stat, "sim_version": sim_version(),
        "note": ("B5: structural rules (layouts.py) followed by the stat-dependent rules too_hard / single_type_trivial (acceptance.py), "
                 "further seeds drawn from the same streams T / L. Stream F is reserved for the pool (B6); no F seed was generated or loaded."),
        "counts": {k: len(v) for k, v in sets.items()}, "sets": sets,
    }
