"""B6: the 300-layout pool (DEV 180 + FINAL 120) from seed stream F, the FINAL-access ledger and the role-guarded loaders.

Spec: docs/SPEC_V2.md 3.5 (streams, FINAL-access ledger), 5.2 (pool of 300), 11 (B6).

* Stream F of every family is split into two disjoint sub-ranges so that DEV and FINAL rejection counts are cleanly separable:
  DEV = [lo, lo + FINAL_OFFSET), FINAL = [lo + FINAL_OFFSET, hi). Labyrinth is FINAL-transfer only (its DEV sub-range is never walked).
* The walk is sequential in seed order (a pure function of the stream and the rules, evaluated in parallel blocks): structural rules
  (layouts.check), D4-canonical duplicate check against EVERY other set (TUNE-train, TUNE-val, LAB-DESIGN, DEV, FINAL; a collision is a
  redraw), then the stat-dependent rules (too_hard, single_type_trivial; acceptance.stat_check). Per family the first 50 / 10 accepted
  DEV layouts are DEV-train / DEV-validation; the first 20 (familiar) or 60 (labyrinth) accepted FINAL layouts are FINAL.
* Workers return only the rule ids that fired. Scores, baselines and Oracle results never leave the worker, so for FINAL only an
  accept/reject bit and the rule id can be stored or printed.
* Every FINAL load goes through this module, appends to data/v2/final_access.log (purpose build | eval) and is role-guarded:
  train / validation / tuning roles raise PermissionError on any FINAL or transfer id.
* Generators (breachlens/layouts.py) and stats are frozen at B5: `check_frozen()` compares them to data/v2/generator_hash.json.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import layouts as Y
from .acceptance import stat_check
from .config import sim_version
from .gates import generator_hash

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v2"
POOL_PATH = DATA / "pool_v1.json"
POOL_SHA_PATH = DATA / "pool_v1.sha256"
GENERATOR_HASH_PATH = DATA / "generator_hash.json"
LEDGER_PATH = DATA / "final_access.log"
POOL_VERSION = "pool_v1"

FINAL_OFFSET = 500_000                       # DEV: [lo, lo + 500k); FINAL: [lo + 500k, lo + 1M) inside each family's F range
SPLITS = ("DEV-train", "DEV-validation", "FINAL-familiar", "FINAL-transfer")
PER_FAMILY = {"DEV-train": 50, "DEV-validation": 10, "FINAL-familiar": 20}   # per familiar family (spec 5.2)
TRANSFER_N = 60                                                              # labyrinth, FINAL-transfer
FINAL_SPLITS = ("FINAL-familiar", "FINAL-transfer")
ROLES = ("train", "validation", "tuning", "build", "eval")
ROLE_SPLITS = {"train": ("DEV-train",), "validation": ("DEV-validation",), "tuning": ("DEV-train", "DEV-validation")}


# ---------------------------------------------------------------- stream geometry
def dev_range(family: str) -> tuple[int, int]:
    lo, _ = Y.stream_range("F", family)
    return lo, lo + FINAL_OFFSET


def final_range(family: str) -> tuple[int, int]:
    lo, hi = Y.stream_range("F", family)
    return lo + FINAL_OFFSET, hi


def is_final(family: str, seed: int) -> bool:
    """True for every FINAL / transfer layout: labyrinth anywhere in stream F, a familiar family in the FINAL sub-range."""
    if Y.stream_of(seed) != "F":
        return False
    return family == "labyrinth" or seed >= Y.stream_range("F", family)[0] + FINAL_OFFSET


def parse_id(layout_id: str) -> tuple[str, int]:
    fam, _, seed = layout_id.rpartition("-")
    return fam, int(seed)


def is_final_id(layout_id: str) -> bool:
    return is_final(*parse_id(layout_id))


# ---------------------------------------------------------------- frozen generators / stats
def check_frozen() -> dict:
    """Generator source and stats must equal what B5 recorded; raises otherwise. Returns the provenance block."""
    rec = json.loads(GENERATOR_HASH_PATH.read_text())
    gh = generator_hash()
    if gh != rec["generator_hash"]:
        raise RuntimeError(f"generator hash {gh} != recorded {rec['generator_hash']} (B5 freeze)")
    ds = json.loads(Y.DESIGN_SETS_PATH.read_text())["sim_version"]
    sv = sim_version()
    if sv != ds:
        raise RuntimeError(f"stats version {sv} != stats version of the B5 design sets {ds}")
    if Y.gen_override_active() or os.environ.get("BREACHLENS_STATS_PATH"):
        raise RuntimeError("generator / stats override hooks must be unset for the pool build")
    return {"generator_hash": gh, "generator_version": Y.GENERATOR_VERSION, "stats_version": sv, "generator_hash_record": rec["commit"]}


# ---------------------------------------------------------------- FINAL-access ledger
def ledger_path() -> Path:
    return Path(os.environ.get("BREACHLENS_FINAL_LEDGER") or LEDGER_PATH)


def _git(*a) -> str:
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def log_final_access(purpose: str, entry_point: str, detail: dict | None = None) -> dict:
    """Append one ledger line (JSON). purpose is 'build' or 'eval'; this step only ever writes 'build'."""
    if purpose not in ("build", "eval"):
        raise ValueError(purpose)
    entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": _git("rev-parse", "HEAD"),
             "dirty": bool(_git("status", "--porcelain")), "entry_point": entry_point, "purpose": purpose, **(detail or {})}
    p = ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def read_ledger(path: Path | None = None) -> list[dict]:
    p = Path(path) if path else ledger_path()
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


# ---------------------------------------------------------------- role-guarded loaders
def _check_role(role: str) -> None:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")


def assert_dev_only(layout_ids, role: str) -> None:
    """Training / validation / tuning code paths call this (or the loaders below): any FINAL or transfer id raises."""
    _check_role(role)
    if role in ROLE_SPLITS:
        bad = [i for i in layout_ids if is_final_id(i)]
        if bad:
            raise PermissionError(f"role {role!r} may not load FINAL / transfer layouts: {bad[:3]}")


def load_pool_records(split: str, role: str, purpose: str | None = None, unlock: bool = False, entry_point: str = "unknown",
                      path: Path = POOL_PATH) -> list[dict]:
    """Records of one split. DEV splits: roles train (DEV-train), validation (DEV-validation), tuning (both), build, eval.
    FINAL splits: role 'build' with purpose 'build', or role 'eval' with purpose 'eval' AND unlock=True; every such load is ledgered."""
    _check_role(role)
    if split not in SPLITS:
        raise ValueError(split)
    recs = [r for r in json.loads(Path(path).read_text())["records"] if r["split"] == split]
    if split in FINAL_SPLITS:
        ok = (role == "build" and purpose == "build") or (role == "eval" and purpose == "eval" and unlock)
        if not ok:
            raise PermissionError(f"role {role!r} / purpose {purpose!r} may not load {split}")
        log_final_access(purpose, entry_point, {"split": split, "n": len(recs)})
    elif role in ROLE_SPLITS and split not in ROLE_SPLITS[role]:
        raise PermissionError(f"role {role!r} may not load {split}")
    return recs


def load_layout(rec: dict, role: str, purpose: str | None = None, unlock: bool = False, entry_point: str = "unknown") -> dict:
    """Regenerate one pool layout from (family, seed), content-hash verified. Same guards and ledger as load_pool_records."""
    _check_role(role)
    final = is_final(rec["family"], rec["seed"])
    if final:
        ok = (role == "build" and purpose == "build") or (role == "eval" and purpose == "eval" and unlock)
        if not ok:
            raise PermissionError(f"role {role!r} / purpose {purpose!r} may not load FINAL layout {rec['id']}")
        log_final_access(purpose, entry_point, {"id": rec["id"]})
    elif role in ROLE_SPLITS and rec["split"] not in ROLE_SPLITS[role]:
        raise PermissionError(f"role {role!r} may not load {rec['split']}")
    L = Y.generate(rec["family"], rec["seed"])
    assert Y.content_hash(L) == rec["content_hash"], f"layout content changed for {rec['id']}"
    return L


def dev_records(role: str, path: Path = POOL_PATH) -> list[dict]:
    """The only record source a training / validation / tuning sampler may use: DEV splits permitted for `role`, never FINAL."""
    _check_role(role)
    if role not in ROLE_SPLITS:
        raise PermissionError(f"dev_records is for train / validation / tuning, not {role!r}")
    out = [r for s in ROLE_SPLITS[role] for r in load_pool_records(s, role, path=path)]
    assert_dev_only([r["id"] for r in out], role)
    return out


# ---------------------------------------------------------------- the walk
def _eval_candidate(args):
    """Worker. Returns ONLY the rule ids that fired (and the content hash): no score, baseline or Oracle value leaves this function."""
    family, seed = args
    L = Y.candidate(family, seed)
    bad = Y.check(L, family)
    if bad:
        return {"seed": seed, "structural": bad, "hash": None, "stat_rules": None}
    h = Y.content_hash(L)
    st = stat_check(L, seed)
    return {"seed": seed, "structural": [], "hash": h, "stat_rules": list(st["reasons"]), "_too_hard": st["too_hard"], "_trivial": st["trivial"]}


def walk(group: str, family: str, n: int, taken: set, pool=None, block: int = 32) -> tuple[list, dict]:
    """Walk the DEV or FINAL sub-range of stream F of `family` until `n` layouts are accepted. `taken` holds the D4-canonical content hashes
    of every layout already in some set; accepted hashes are added. Returns (records [id, family, seed, content_hash], report)."""
    assert group in ("DEV", "FINAL")
    lo, hi = dev_range(family) if group == "DEV" else final_range(family)
    assert Y.stream_of(lo) == "F" and Y.stream_of(hi - 1) == "F"
    assert group == "FINAL" or family != "labyrinth", "labyrinth is FINAL-transfer only"
    out, seed = [], lo
    rep = {"group": group, "family": family, "candidates": 0, "accepted": 0, "structural": {}, "structural_rejected": 0, "redraws_d4_duplicate": 0,
           "stat_evaluated": 0, "single_type_trivial": 0, "too_hard": 0, "stat_rejected": 0, "bits": []}
    while len(out) < n:
        seeds = list(range(seed, min(seed + block, hi)))
        if not seeds:
            raise RuntimeError("stream exhausted")
        res = pool.map(_eval_candidate, [(family, s) for s in seeds]) if pool else [_eval_candidate((family, s)) for s in seeds]
        for r in res:
            if len(out) >= n:
                break
            seed = r["seed"] + 1
            rep["candidates"] += 1
            if r["structural"]:
                rep["structural_rejected"] += 1
                for b in r["structural"]:
                    rep["structural"][b] = rep["structural"].get(b, 0) + 1
                rep["bits"].append([r["seed"], 0, list(r["structural"])])
                continue
            if r["hash"] in taken:
                rep["redraws_d4_duplicate"] += 1
                rep["bits"].append([r["seed"], 0, ["d4_duplicate"]])
                continue
            rep["stat_evaluated"] += 1
            rep["single_type_trivial"] += r["_trivial"]
            rep["too_hard"] += r["_too_hard"]
            if r["stat_rules"]:
                rep["stat_rejected"] += 1
                rep["bits"].append([r["seed"], 0, list(r["stat_rules"])])
                continue
            taken.add(r["hash"])
            out.append({"id": f"{family}-{r['seed']}", "family": family, "seed": r["seed"], "content_hash": r["hash"]})
            rep["accepted"] += 1
            rep["bits"].append([r["seed"], 1, []])
    rep["seeds_consumed"] = seed - lo
    rep["seed_lo"], rep["seed_hi_exclusive_consumed"] = lo, seed
    return out, rep


def design_set_hashes() -> dict:
    """{content_hash: set name} for TUNE-train, TUNE-val and LAB-DESIGN (stored hashes of data/v2/design_sets.json)."""
    d = Y.load_design_sets_meta()
    return {r["content_hash"]: name for name, recs in d["sets"].items() for r in recs}


def build_pool(procs: int | None = None, entry_point: str = "scripts_b6_build_pool.py") -> dict:
    """Build the whole pool document (not yet written). One 'build' ledger entry per FINAL walk."""
    prov = check_frozen()
    procs = procs or min(8, os.cpu_count() or 1)
    taken = set(design_set_hashes())
    n_design = len(taken)
    records, reports = [], {}
    p = mp.get_context("spawn").Pool(procs) if procs > 1 else None
    try:
        for fam in Y.FAMILIAR:                                   # DEV first, so a FINAL candidate colliding with DEV is a FINAL redraw
            n = PER_FAMILY["DEV-train"] + PER_FAMILY["DEV-validation"]
            recs, rep = walk("DEV", fam, n, taken, p)
            for i, r in enumerate(recs):
                records.append(dict(r, split="DEV-train" if i < PER_FAMILY["DEV-train"] else "DEV-validation"))
            reports[f"DEV/{fam}"] = rep
        for fam in Y.FAMILIAR:
            log_final_access("build", entry_point, {"action": "walk", "group": "FINAL-familiar", "family": fam})
            recs, rep = walk("FINAL", fam, PER_FAMILY["FINAL-familiar"], taken, p)
            records += [dict(r, split="FINAL-familiar") for r in recs]
            reports[f"FINAL/{fam}"] = rep
        log_final_access("build", entry_point, {"action": "walk", "group": "FINAL-transfer", "family": "labyrinth"})
        recs, rep = walk("FINAL", "labyrinth", TRANSFER_N, taken, p)
        records += [dict(r, split="FINAL-transfer") for r in recs]
        reports["FINAL/labyrinth"] = rep
    finally:
        if p:
            p.close(); p.join()
    for r in records:
        r["pool"] = "DEV" if r["split"].startswith("DEV") else "FINAL"
    # FINAL candidate log: ONLY seed, accept bit and rule ids. DEV keeps the same shape (seeds consumed, counts per rule).
    summary = {}
    for k, rep in reports.items():
        summary[k] = {x: rep[x] for x in ("candidates", "accepted", "structural", "structural_rejected", "redraws_d4_duplicate", "stat_evaluated",
                                          "single_type_trivial", "too_hard", "stat_rejected", "seeds_consumed", "seed_lo")}
        summary[k]["candidate_bits"] = rep["bits"]
    return {
        "pool_version": POOL_VERSION, **prov, "design_sets_sha256": hashlib.sha256(Y.DESIGN_SETS_PATH.read_bytes()).hexdigest(),
        "streams": {"F": {"base": Y.STREAM_BASE["F"], "stride_per_family": Y.STREAM_STRIDE, "final_offset": FINAL_OFFSET,
                          "dev_range": {f: list(dev_range(f)) for f in Y.FAMILIES}, "final_range": {f: list(final_range(f)) for f in Y.FAMILIES}}},
        "split_sizes": {"DEV-train": 150, "DEV-validation": 30, "FINAL-familiar": 60, "FINAL-transfer": 60},
        "design_set_hashes_checked": n_design,
        "records": records, "walk": summary,
        "note": ("B6: 300-layout pool from stream F. FINAL records carry id / family / split / seed / content hash only; walk.candidate_bits holds "
                 "[seed, accept bit, rule ids] per candidate. No score, baseline or Oracle value is stored for any FINAL layout."),
    }


def pool_sha256(path: Path = POOL_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
