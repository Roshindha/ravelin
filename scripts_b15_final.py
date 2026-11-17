"""B15 session runner: the single FINAL evaluation session of docs/SPEC_V2.md row (gg), plus its rehearsal.

REAL RUN (once; never from a dirty tree; never a second time):
    .venv/bin/python scripts_b15_final.py --unlock-final
    .venv/bin/python scripts_b15_final.py --unlock-final --restart --cause machine      # only after a crash, row (gg) (e)(5)
REHEARSAL (DEV-validation stands in for FINAL-familiar, TUNE-val for a second group; role validation; temp dir; temp ledger; no FINAL layout is loaded):
    .venv/bin/python scripts_b15_final.py --rehearse

What the session does (row (gg) (e)):
  1. preconditions, checked and logged to <out>/preconditions.json (any failure = refusal, nothing is written but that log);
  2. <out>/session.json (exclusive create) BEFORE any FINAL layout is loaded; its session_id is the ONE eval session (`pool.eval_session`) of every
     FINAL load and ledger entry;
  3. the two FINAL groups are loaded once through the loaders (role eval, unlock), their ids asserted against the declared groups;
  4. the cells of `scripts_b15_cells.all_cells()` in the pinned order; each writes <out>/raw/<cell>.csv in exclusive-create mode and then appends one
     line (cell, path, rows, sha256, commit) to <out>/manifest.jsonl; a finished cell is never recomputed;
  5. crash rule: exception -> <out>/crash_log.md at once; a restart reuses session.json, keeps verified cells, discards incomplete cell files.
The statistics are NOT here: `scripts_b15_analysis.py` reads the raw files only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import zlib
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import scripts_b15_cells as C

ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "docs" / "SPEC_V2.md"
AUDIT_JSON = ROOT / "results" / "b14" / "grid_audit.json"
SEEN_ROWS = ROOT / "results" / "b14" / "seen_layout_rows.csv"
BEST_SINGLE_JSON = ROOT / "data" / "v2" / "baselines" / "b8_best_single.json"
MODELS = ROOT / "models_v2"
REAL_OUT = ROOT / "results" / "b15"
ENTRY_POINT = "scripts_b15_final.py"
FINAL_FILE = {"ppo": "final.zip", "ddqn": "final.pt"}


class Refusal(Exception):
    """A precondition or guard failed: nothing was loaded or written beyond the log."""


class CorruptCell(Exception):
    """A manifest line whose raw file is missing or does not hash: corruption, not a crash. The session stops."""


class SimulatedKill(BaseException):
    """Rehearsal only: stands for a killed process (not caught by the crash handler, like a SIGKILL)."""


# ---------------------------------------------------------------------------------------------------------------- context
GROUPS_REAL = OrderedDict([
    ("FINAL-familiar", {"set": "FINAL-familiar", "n": 60, "families": {"spine_pods": 20, "archipelago": 20, "bunker_nest": 20}}),
    ("FINAL-transfer", {"set": "FINAL-transfer", "n": 60, "families": {"labyrinth": 60}})])
GROUPS_REHEARSAL = OrderedDict([
    ("FINAL-familiar", {"set": "DEV-validation", "n": 30, "families": {"spine_pods": 10, "archipelago": 10, "bunker_nest": 10}}),
    ("FINAL-transfer", {"set": "TUNE-val", "n": 60, "families": {"spine_pods": 20, "archipelago": 20, "bunker_nest": 20}})])


class Ctx:
    """Everything that differs between the real run, the rehearsal and the tests. Nothing here changes a protocol rule."""

    def __init__(self, *, mode: str, out: Path, unlock: bool = False, groups=None, cells=None, procs: int = 7, models_dir: Path = MODELS,
                 audit_path: Path = AUDIT_JSON, seen_path: Path = SEEN_ROWS, best_single_path: Path = BEST_SINGLE_JSON, spec_path: Path = SPEC,
                 role: str | None = None, loader=None, run_suite: bool | None = None, require_clean: bool | None = None,
                 suite_cmd: list | None = None, git_dirty=None, crash_hook=None, log=lambda *a: print(*a, flush=True), require_final_ids: bool | None = None):
        assert mode in ("real", "rehearse")
        self.mode, self.out, self.unlock, self.procs, self.log = mode, Path(out), unlock, procs, log
        self.groups = groups or (GROUPS_REAL if mode == "real" else GROUPS_REHEARSAL)
        self.cells = cells or C.all_cells()
        self.models_dir, self.audit_path, self.seen_path, self.best_single_path, self.spec_path = (Path(models_dir), Path(audit_path), Path(seen_path),
                                                                                                   Path(best_single_path), Path(spec_path))
        self.role = role or ("eval" if mode == "real" else "validation")
        self.loader = loader
        self.run_suite = (mode == "real") if run_suite is None else run_suite
        self.require_clean = (mode == "real") if require_clean is None else require_clean
        self.suite_cmd = suite_cmd or [sys.executable, "-m", "pytest"]
        self.git_dirty = git_dirty
        self.crash_hook = crash_hook
        self.require_final_ids = (mode == "real") if require_final_ids is None else require_final_ids      # tests of the real code path use DEV layouts

    @property
    def raw(self) -> Path:
        return self.out / "raw"

    @property
    def manifest(self) -> Path:
        return self.out / "manifest.jsonl"

    @property
    def session_path(self) -> Path:
        return self.out / "session.json"

    def dirty(self) -> bool:
        from breachlens import pool as P
        return (self.git_dirty or P.git_dirty)()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------------------------- pinned values of row (gg)
def row_gg(spec_path: Path = SPEC) -> str:
    rows = [ln for ln in Path(spec_path).read_text().splitlines() if ln.startswith("| 2026-11-17 | (gg)")]
    if len(rows) != 1:
        raise Refusal(f"{spec_path}: expected exactly one row (gg), found {len(rows)}")
    return rows[0]


def row_hh(spec_path: Path = SPEC) -> str:
    rows = [ln for ln in Path(spec_path).read_text().splitlines() if ln.startswith("| 2026-11-17 | (hh)")]
    if len(rows) != 1:
        raise Refusal(f"{spec_path}: expected exactly one row (hh), found {len(rows)}")
    return rows[0]


def pinned(spec_path: Path = SPEC) -> dict:
    r = row_gg(spec_path)

    def one(rx):
        m = re.search(rx, r)
        if not m:
            raise Refusal(f"row (gg) does not contain the pattern {rx!r}")
        return m.groups()
    pool_pre, pool_suf = one(r"the pool sha256 is `([0-9a-f]{8})\.\.\.([0-9a-f]{6})`")
    gen = re.search(r"the correct value is ([0-9a-f]{64})", row_hh(spec_path))
    if not gen:
        raise Refusal("row (hh) does not contain the full generator hash")
    return {"audit_sha256": one(r"sha256 of that file ([0-9a-f]{64})")[0], "seen_rows_sha256": one(r"seen_layout_rows\.csv` sha256 ([0-9a-f]{64})")[0],
            "pool": (pool_pre, pool_suf), "generator": gen.group(1)}


# ---------------------------------------------------------------------------------------------------------------- preconditions (row (gg) (e)(2))
def check_head_has_row(ctx: Ctx) -> str:
    row_gg(ctx.spec_path)
    row_hh(ctx.spec_path)
    if ctx.mode == "real":
        from breachlens import pool as P
        txt = P._git("show", "HEAD:docs/SPEC_V2.md")
        if txt == "unknown" or not all(any(ln.startswith(f"| 2026-11-17 | ({r})") for ln in txt.splitlines()) for r in ("gg", "hh")):
            raise Refusal("HEAD does not contain rows (gg) and (hh) in docs/SPEC_V2.md")
    return "row (gg) present"


def check_clean_tree(ctx: Ctx) -> str:
    if ctx.require_clean and ctx.dirty():
        raise Refusal("the tree is dirty (git_dirty True): the FINAL session needs a clean commit")
    return f"git_dirty {ctx.dirty()}" + ("" if ctx.require_clean else " (not required in this mode)")


def check_pool_and_generator(ctx: Ctx) -> str:
    from breachlens import pool as P
    pn = pinned(ctx.spec_path)
    sha = P.pool_sha256()
    if not (sha.startswith(pn["pool"][0]) and sha.endswith(pn["pool"][1])):
        raise Refusal(f"pool sha256 {sha} does not match row (gg) {pn['pool'][0]}...{pn['pool'][1]}")
    gh = json.loads(P.GENERATOR_HASH_PATH.read_text())["generator_hash"]
    if gh != pn["generator"]:                                      # the FULL hash: row (hh) and the B5 record must agree
        raise Refusal(f"generator hash {gh} (B5 record) != row (hh) {pn['generator']}")
    P.check_frozen()                                               # and the generator source must still hash to the B5 record
    return f"pool {sha[:8]}...{sha[-6:]}, generator {gh} (row (hh) = B5 record = source)"


def check_ledger(ctx: Ctx, restart: bool = False, session_id: str | None = None) -> str:
    from breachlens import pool as P
    led = P.read_ledger()
    n_build = sum(e.get("purpose") == "build" for e in led)
    ev = [e for e in led if e.get("purpose") == "eval"]
    if n_build != 5:
        raise Refusal(f"ledger shows {n_build} build entries (expected 5)")
    if not restart:
        if ev:
            raise Refusal(f"ledger shows {len(ev)} eval entries (expected 0): FINAL has been used")
        return "ledger 5 build / 0 eval"
    other = [e for e in ev if e.get("session_id") != session_id]
    if other:
        raise Refusal(f"ledger holds {len(other)} eval entries outside session {session_id}")
    return f"ledger 5 build / {len(ev)} eval, all in session {session_id}, {P.count_eval_sessions(led)} session(s)"


def check_session_json(ctx: Ctx, restart: bool = False) -> str:
    exists = ctx.session_path.exists()
    if restart and not exists:
        raise Refusal("--restart needs an existing session.json")
    if not restart and exists:
        raise Refusal(f"{ctx.session_path} exists: the real run refuses a second start (use --restart after a crash)")
    return "session.json " + ("present (restart)" if exists else "absent (fresh)")


def audit_runs(ctx: Ctx) -> dict:
    d = json.loads(ctx.audit_path.read_text())
    return {r["run"]: r for r in d["runs"]}


def check_audit(ctx: Ctx, rerun: bool = True, restart: bool = False) -> str:
    pn = pinned(ctx.spec_path)
    sha = sha256_file(ctx.audit_path)
    if sha != pn["audit_sha256"]:
        raise Refusal(f"{ctx.audit_path.name} sha256 {sha} != row (gg) {pn['audit_sha256']}")
    if not rerun:
        return "audit file sha256 ok"
    import scripts_b14_audit as A
    out, _, dev = A.run(write=False)                                # read-only: the pinned audit file is not rewritten
    if restart:                                                     # the audit's own "ledger is 5 build / 0 eval" is a fresh-start condition; a restart is checked by check_ledger
        dev = [d for d in dev if not d.startswith("ledger is ")]
    ok = sum(1 for r in out["runs"] if not r["deviations"])
    if dev or out["n_runs"] != 30 or ok != 30:
        raise Refusal(f"grid audit re-run: {ok}/{out['n_runs']} runs clean, deviations: {dev[:3]}")
    return f"grid audit re-run {ok}/30 pass; audit file sha256 ok"


def check_models(ctx: Ctx) -> str:
    runs = audit_runs(ctx)
    want = {c["run"] for c in ctx.cells if "run" in c}
    miss = sorted(r for r in want if r not in runs)
    if miss:
        raise Refusal(f"runs not in the audit file: {miss[:3]}")
    for r in sorted(want):
        algo = "ddqn" if "ddqn" in r else "ppo"
        p = ctx.models_dir / r / FINAL_FILE[algo]
        if not p.exists():
            raise Refusal(f"missing model {p}")
        if sha256_file(p) != runs[r]["final_sha256"]:
            raise Refusal(f"model {p} sha256 differs from the grid audit file")
    return f"{len(want)} model files present, sha256 equal to the audit file"


def check_seen_rows(ctx: Ctx) -> str:
    sha = sha256_file(ctx.seen_path)
    if sha != pinned(ctx.spec_path)["seen_rows_sha256"]:
        raise Refusal(f"{ctx.seen_path.name} sha256 {sha} != row (gg)")
    return "seen_layout_rows.csv sha256 ok"


def best_single_choice(ctx: Ctx) -> dict:
    d = json.loads(ctx.best_single_path.read_text())
    by = d["train_chosen_for_set"]["DEV-train"]
    return {"by_family": {k: int(v) for k, v in by.items()}, "pooled": int(d["pooled_dev_train_type_for_labyrinth_transfer_B15"])}


def check_best_single(ctx: Ctx) -> str:
    ch = best_single_choice(ctx)
    sniper = C.UNIT_TYPES.index("sniper")
    if set(ch["by_family"].values()) != {sniper} or ch["pooled"] != sniper:
        raise Refusal(f"BestSingle types {ch} are not Sniper everywhere, as row (gg) states")
    return "BestSingle = Sniper in every family and pooled"


def check_suite(ctx: Ctx) -> str:
    if not ctx.run_suite:
        return "not run in this mode"
    p = subprocess.run(ctx.suite_cmd, cwd=ROOT, capture_output=True, text=True)
    tail = (p.stdout.strip().splitlines() or [""])[-1]
    if p.returncode != 0:
        raise Refusal(f"full suite exit code {p.returncode}: {tail}")
    return f"full suite pytest exit code 0 ({tail})"


def run_preconditions(ctx: Ctx, restart: bool = False, session_id: str | None = None) -> list:
    steps = [("row (gg) in HEAD", lambda: check_head_has_row(ctx)), ("clean tree", lambda: check_clean_tree(ctx)),
             ("pool and generator hashes", lambda: check_pool_and_generator(ctx)),
             ("ledger", lambda: check_ledger(ctx, restart, session_id)), ("session.json", lambda: check_session_json(ctx, restart)),
             ("grid audit", lambda: check_audit(ctx, True, restart)), ("model files", lambda: check_models(ctx)), ("seen rows", lambda: check_seen_rows(ctx)),
             ("BestSingle types", lambda: check_best_single(ctx))]
    log, failed = [], False
    for name, fn in steps:
        try:
            log.append({"check": name, "ok": True, "detail": fn()})
        except Refusal as e:
            log.append({"check": name, "ok": False, "detail": str(e)})
            failed = True
    if not failed:                                                    # the suite only runs when everything cheap passed; the tree is re-checked after it
        for name, fn in (("full suite", lambda: check_suite(ctx)), ("clean tree after the suite", lambda: check_clean_tree(ctx))):
            try:
                log.append({"check": name, "ok": True, "detail": fn()})
            except Refusal as e:
                log.append({"check": name, "ok": False, "detail": str(e)})
                failed = True
                break
    ctx.out.mkdir(parents=True, exist_ok=True)
    (ctx.out / "preconditions.json").write_text(json.dumps({"time": now(), "mode": ctx.mode, "restart": restart, "checks": log}, indent=1))
    if failed:
        raise Refusal("preconditions failed: " + "; ".join(f"{x['check']}: {x['detail']}" for x in log if not x["ok"]))
    return log


# ---------------------------------------------------------------------------------------------------------------- raw files and manifest
def read_manifest(ctx: Ctx) -> OrderedDict:
    out = OrderedDict()
    if ctx.manifest.exists():
        for ln in ctx.manifest.read_text().splitlines():
            if ln.strip():
                e = json.loads(ln)
                if e["cell"] in out:
                    raise CorruptCell(f"duplicate manifest line for cell {e['cell']}")
                out[e["cell"]] = e
    return out


def verify_entry(ctx: Ctx, e: dict) -> None:
    p = ctx.out / e["path"]
    if not p.exists():
        raise CorruptCell(f"manifest line for {e['cell']} but {p} is missing")
    if sha256_file(p) != e["sha256"]:
        raise CorruptCell(f"{p} does not hash to its manifest line")
    with open(p, newline="") as f:
        n = sum(1 for _ in csv.DictReader(f))
    if n != e["rows"]:
        raise CorruptCell(f"{p} has {n} rows, manifest says {e['rows']}")


def write_cell(ctx: Ctx, name: str, rows: list, commit: str, session_id: str) -> dict:
    """Exclusive create: a second write of the same cell raises FileExistsError. The manifest line is appended only after the file is complete."""
    ctx.raw.mkdir(parents=True, exist_ok=True)
    path = ctx.raw / f"{name}.csv"
    with open(path, "x", newline="") as f:                                       # raises FileExistsError if the cell file exists
        w = csv.DictWriter(f, fieldnames=list(C.RAW_COLUMNS), extrasaction="raise")
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    e = {"cell": name, "path": str(path.relative_to(ctx.out)), "rows": len(rows), "sha256": sha256_file(path), "commit": commit, "session_id": session_id, "time": now()}
    with open(ctx.manifest, "a") as f:
        f.write(json.dumps(e, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return e


def log_crash(ctx: Ctx, last_cell: str | None, text: str, how: str) -> None:
    ctx.out.mkdir(parents=True, exist_ok=True)
    with open(ctx.out / "crash_log.md", "a") as f:
        f.write(f"## Crash {now()}\n\n- detected: {how}\n- last completed cell: {last_cell}\n- error text:\n\n```\n{text.strip()}\n```\n\n"
                "TODO (owner): record this crash in docs/LIMITATIONS.md before restarting; if the cause is a code defect, STOP and report to the owner.\n\n")


# ---------------------------------------------------------------------------------------------------------------- worker side (models, policies)
_MODELS: dict = {}


def _init_worker() -> None:
    import torch as th
    th.set_num_threads(1)


def _model(models_dir: str, run: str, algo: str):
    key = (models_dir, run)
    if key not in _MODELS:
        d = Path(models_dir) / run
        if algo == "ppo":
            from sb3_contrib import MaskablePPO
            _MODELS[key] = MaskablePPO.load(d / "final.zip", device="cpu")
        else:
            import torch as th
            from rl import ddqn as D
            ck = th.load(d / "final.pt", map_location="cpu", weights_only=False)
            net = D.QNet(D.default_obs_space(), ck["n_actions"], ck["hyperparameters"]["net_arch"])
            net.load_state_dict(ck["online"])
            net.eval()
            _MODELS[key] = net
    return _MODELS[key]


def tta_policy(act_fn, name: str):
    from breachlens.baselines import Policy, _ep
    from breachlens.constants import N_UNIT_TYPES
    from breachlens.env import BreachEnv
    import numpy as np

    class TTAPolicy(Policy):
        """Episode with an action chooser (here: the D4 test-time ensemble of rl.tta); same env, reset seed and army counting as ModelPolicy."""

        def episodes(self, rec, layout, P):
            env = BreachEnv(layouts=[layout], role="train")
            obs, _ = env.reset(seed=zlib.crc32(rec["id"].encode()), options={"layout_index": 0})
            done = False
            while not done:
                obs, _, done, _, _ = env.step(int(act_fn(obs, env.action_masks())))
            army = (np.bincount([t for t, _ in env.army], minlength=N_UNIT_TYPES) + np.asarray(env.owned)).tolist()
            return [_ep(env.result, env.bought, len(env.army), army=army)]
    p = TTAPolicy()
    p.name = name
    return p


KEEP_EXTRA = ("chosen_type", "oracle_sims", "oracle_start", "oracle_gold", "reason", "chosen_index", "n_sims")        # no wall times: raw files stay reproducible


def _row(r: dict, N=None, policy_hash: str = "") -> dict:
    eps = r["episodes"]
    armies = [e["army"] for e in eps if e.get("army") is not None]
    army = None
    if armies:
        import numpy as np
        army = np.mean(np.asarray(armies, dtype=float), axis=0).tolist()
    return {"id": r["id"], "N": N, "n_episodes": r["n_episodes"], "score": r["score"], "d": r["d"], "breach": r["breach"], "n_bought": r["n_bought"],
            "n_deployed": r["n_deployed"], "army": army, "chosen_type": eps[0].get("chosen_type"), "policy_hash": policy_hash,
            "extra": json.dumps({k: v for k, v in eps[0].items() if k in KEEP_EXTRA}, sort_keys=True)}


def _task(args) -> list:
    """One (cell, layout) unit of work -> list of raw measures (one per N for best-of-N cells). Runs in a worker or in the main process."""
    import numpy as np
    from breachlens import baselines as B
    from breachlens import evaluator as EV
    from breachlens.armies import greedy_types, single_unit_types
    cell, rec, layout, w = args
    kind = cell["kind"]
    work = lambda pol: EV._work((pol, rec, layout))
    if kind == "baseline":
        name = cell["cond"]
        if name == "bestsingle_hindsight":                                                   # the five single-unit rows; the main process picks per group and family
            out = []
            for t in range(len(C.UNIT_TYPES)):
                x = _row(work(B.SingleUnit(t)))
                x.update(single_type=t, chosen_type=t, army=np.bincount(single_unit_types(t), minlength=len(C.UNIT_TYPES)).astype(float).tolist())
                out.append(x)
            return out
        if name == "bestsingle":
            pol = B.BestSingle(w["best_single"]["by_family"], pooled=w["best_single"]["pooled"])
        else:
            pol = {"random": B.RandomPolicy, "greedy": B.Greedy, "bestsingle_perlayout": B.BestSinglePerLayout, "oracle": B.Oracle}[name]()
        row = _row(work(pol))
        types = greedy_types() if name == "greedy" else (single_unit_types(row["chosen_type"]) if row["chosen_type"] is not None else None)
        if types is not None:
            row["army"] = np.bincount(types, minlength=len(C.UNIT_TYPES)).astype(float).tolist()
        return [row]
    if kind == "learner":
        m = _model(w["models_dir"], cell["run"], cell["algo"])
        if not cell["tta"]:
            if cell["algo"] == "ppo":
                from rl.eval_ppo_v2 import ModelPolicy
                pol = ModelPolicy(m, name=cell["name"])
            else:
                from rl.ddqn import DDQNPolicy
                pol = DDQNPolicy(m, name=cell["name"])
        else:
            from rl.tta import tta_action
            if cell["algo"] == "ppo":
                from rl.eval_ppo_v2 import masked_probs
                fn = masked_probs(m)
                pol = tta_policy(lambda o, mk: tta_action(fn, o, mk, aggregate="logprob"), cell["name"])
            else:
                from rl.ddqn import q_values_fn
                fn = q_values_fn(m)
                pol = tta_policy(lambda o, mk: tta_action(fn, o, mk, aggregate="mean"), cell["name"])
        return [_row(work(pol))]
    from rl import planner as PL
    if kind == "control":
        pols = PL.make_cells({"ras": "RAS", "rps": "RPS", "rpsfull": "RPSfull"}[cell["method"]], cell["seed"], tuple(cell["ns"]), role=w["role"], unlock=w["unlock"])
        return [_row(work(p), N=p.N, policy_hash=p.policy_hash) for p in pols]
    assert kind == "planner"
    m = _model(w["models_dir"], cell["run"], "ppo")
    probs = PL.ModelProbs(m)
    pols = PL.make_cells("planner", cell["seed"], tuple(cell["ns"]), probs_fn=probs, policy_hash=probs.checkpoint_hash(), role=w["role"], unlock=w["unlock"])
    return [_row(work(p), N=p.N, policy_hash=p.policy_hash) for p in pols]


# ---------------------------------------------------------------------------------------------------------------- main-process side
def select_hindsight(rows_by_group: dict, w_rows: list) -> list:
    """BestSingle-hindsight: per group and family the single-unit type with the highest mean score on that group's layouts (ties -> lowest type index)."""
    import numpy as np
    out = []
    for g, items in rows_by_group.items():
        fam = {}
        for (rec, _), five in zip(items, w_rows[g]):
            fam.setdefault(rec["family"], []).append([x["score"] for x in sorted(five, key=lambda x: x["single_type"])])
        best = {f: int(np.argmax(np.mean(v, axis=0))) for f, v in fam.items()}      # np.argmax: lowest index on ties
        for (rec, _), five in zip(items, w_rows[g]):
            out.append((g, [x for x in five if x["single_type"] == best[rec["family"]]][0]))
    return out


def verify_group(ctx: Ctx, g: str, items: list) -> None:
    """The evaluator asserts that the layout ids are the declared group (row (gg) / SPEC section 11 B15 test)."""
    from breachlens import pool as P
    spec = ctx.groups[g]
    ids = [r["id"] for r, _ in items]
    if len(ids) != spec["n"] or len(set(ids)) != len(ids):
        raise Refusal(f"group {g}: {len(ids)} layouts / {len(set(ids))} distinct, declared {spec['n']}")
    fam = {}
    for r, _ in items:
        fam[r["family"]] = fam.get(r["family"], 0) + 1
    if fam != spec["families"]:
        raise Refusal(f"group {g}: families {fam} != declared {spec['families']}")
    if any(r.get("split", spec["set"]) != spec["set"] for r, _ in items):
        raise Refusal(f"group {g}: a layout record carries another split than {spec['set']}")
    if ctx.require_final_ids:
        if not all(P.is_final(r["family"], r["seed"]) for r, _ in items):
            raise Refusal(f"group {g}: a layout is not a FINAL / transfer layout")
    else:
        P.assert_dev_only(ids, "validation")                                              # the rehearsal never touches FINAL


def load_groups(ctx: Ctx) -> OrderedDict:
    from breachlens import evaluator as EV
    out = OrderedDict()
    for g, spec in ctx.groups.items():
        if ctx.loader is not None:
            items = ctx.loader(spec["set"])
        elif ctx.mode == "real":
            items = EV.load_set(spec["set"], "eval", unlock=ctx.unlock, entry_point=ENTRY_POINT)       # one load per group, through the loaders (ledgered)
        else:
            items = EV.load_set(spec["set"], "validation", entry_point=ENTRY_POINT)
        verify_group(ctx, g, items)
        out[g] = items
    return out


def compute_cell(ctx: Ctx, cell: dict, items: OrderedDict, pool, session_id: str, model_sha: dict) -> list:
    """All raw rows of one cell, groups in declared order, layouts in loader order."""
    from breachlens import evaluator as EV
    w = {"models_dir": str(ctx.models_dir), "role": ctx.role, "unlock": ctx.mode == "real" and ctx.unlock, "best_single": best_single_choice(ctx)}
    tasks, owner = [], []
    for g, its in items.items():
        for rec, lay in its:
            tasks.append((cell, rec, lay, w))
            owner.append((g, rec))
    res = pool.map(_task, tasks, chunksize=1) if pool is not None else [_task(t) for t in tasks]
    if cell["name"] == "bestsingle_hindsight":
        per_group = {g: [] for g in items}
        for (g, _), r in zip(owner, res):
            per_group[g].append(r)
        picked = select_hindsight(items, per_group)
        res = [[x] for _, x in picked]
        owner = [(g, rec) for g, its in items.items() for rec, _ in its]
    commit = _git_head()
    metas = {g: EV.run_meta(C.EVAL_SEED, ctx.role, ctx.groups[g]["set"]) for g in items}
    rows = []
    for (g, rec), rs in zip(owner, res):
        meta = metas[g]
        for x in rs:
            army = x["army"] or [""] * len(C.UNIT_TYPES)
            row = {"cell": cell["name"], "condition": cell["cond"], "N": "" if x["N"] is None else x["N"],
                   "learner_seed": "" if cell["seed"] is None else cell["seed"], "tta": int(bool(cell.get("tta"))), "group": g,
                   "set": ctx.groups[g]["set"], "family": rec["family"], "id": rec["id"], "layout_seed": int(rec["seed"]), "n_episodes": x["n_episodes"],
                   "score": x["score"], "d": x["d"], "breach": x["breach"], "n_bought": x["n_bought"], "n_deployed": x["n_deployed"],
                   **dict(zip(C.ARMY_COLS, army)), "chosen_type": "" if x["chosen_type"] is None else x["chosen_type"],
                   "model_sha256": model_sha.get(cell.get("run"), ""), "policy_hash": x["policy_hash"], "extra": x["extra"], "session_id": session_id,
                   "score_version": meta["score_version"], "sim_version": meta["sim_version"], "generator_hash": meta["generator_hash"],
                   "pool_sha256": meta["pool_sha256"], "git_commit": commit, "git_dirty": meta["git_dirty"], "seed": meta["seed"], "role": meta["role"]}
            EV.check_meta(row)
            rows.append(row)
    n_lay = sum(len(v) for v in items.values())
    assert len(rows) == C.expected_rows(cell, n_lay), (cell["name"], len(rows), n_lay)
    return rows


def _git_head() -> str:
    from breachlens import pool as P
    return P._git("rev-parse", "HEAD")


def plan_cells(ctx: Ctx) -> tuple:
    """(kept, pending) under the crash rule: verified cells are kept; a cell file without a verified manifest line is moved to discarded/ and logged."""
    man = read_manifest(ctx)
    names = {c["name"] for c in ctx.cells}
    stray = [k for k in man if k not in names]
    if stray:
        raise CorruptCell(f"manifest lists cells that are not in the cell list: {stray[:3]}")
    kept, pending, discarded = [], [], []
    for c in ctx.cells:
        if c["name"] in man:
            verify_entry(ctx, man[c["name"]])
            kept.append(c)
            continue
        p = ctx.raw / f"{c['name']}.csv"
        if p.exists():
            d = ctx.out / "discarded"
            d.mkdir(parents=True, exist_ok=True)
            dest = d / f"{c['name']}.{int(time.time())}.csv"
            shutil.move(str(p), str(dest))
            discarded.append(c["name"])
        pending.append(c)
    if discarded:
        with open(ctx.out / "crash_log.md", "a") as f:
            f.write(f"## Restart {now()}\n\n- incomplete raw files discarded and recomputed from scratch: {', '.join(discarded)}\n\n")
    return kept, pending


def _state(ctx: Ctx) -> dict:
    p = ctx.out / "run_state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _set_state(ctx: Ctx, **kw) -> None:
    (ctx.out / "run_state.json").write_text(json.dumps(kw, indent=1))


def run_session(ctx: Ctx, restart: bool = False, cause: str | None = None, defect_agreed: bool = False) -> dict:
    """The whole session. Returns a summary. Refusals raise `Refusal` before anything is loaded."""
    from breachlens import pool as P
    if ctx.mode == "real" and not ctx.unlock:
        raise Refusal("real mode needs --unlock-final")
    ctx.out.mkdir(parents=True, exist_ok=True)
    session_id = None
    if restart:
        if cause not in ("machine", "code-fix"):
            raise Refusal("--restart needs --cause machine|code-fix")
        if not ctx.session_path.exists():
            raise Refusal("--restart needs an existing session.json")
        sess = json.loads(ctx.session_path.read_text())
        session_id = sess["session_id"]
        st = _state(ctx)
        if st.get("status") == "complete":
            raise Refusal("the session is complete: there is nothing to restart and no second session exists")
        if st.get("status") == "running":                                                  # the previous process died without writing its own crash entry
            last = list(read_manifest(ctx))[-1:] or [None]
            log_crash(ctx, last[0], "process died without an exception record (kill / power / reboot)", "restart detection")
        head, last_commit = _git_head(), (list(read_manifest(ctx).values())[-1:] or [{"commit": sess["commit"]}])[0]["commit"]
        if cause == "machine" and head != last_commit:
            raise Refusal(f"--cause machine but HEAD {head[:7]} differs from the commit of the last completed cell {last_commit[:7]}")
        if cause == "code-fix" and not defect_agreed:
            raise Refusal("--cause code-fix only after the defect is reported to the user and agreed (--defect-agreed)")
        if cause == "code-fix" and head == last_commit:
            raise Refusal("--cause code-fix but HEAD equals the commit of the last completed cell: a fix is a new commit")
    log = run_preconditions(ctx, restart, session_id)
    if not restart:
        session_id = uuid.uuid4().hex
        sess = {"session_id": session_id, "created": now(), "mode": ctx.mode, "commit": _git_head(), "git_dirty": ctx.dirty(),
                "groups": {g: s["set"] for g, s in ctx.groups.items()}, "n_cells": len(ctx.cells), "cells": [c["name"] for c in ctx.cells],
                "preconditions": [x["check"] for x in log]}
        with open(ctx.session_path, "x") as f:                                             # exclusive create, BEFORE any FINAL layout is loaded
            json.dump(sess, f, indent=1)
    runs = audit_runs(ctx)
    model_sha = {r: runs[r]["final_sha256"] for r in {c["run"] for c in ctx.cells if "run" in c}}
    kept, pending = plan_cells(ctx)
    ctx.log(f"session {session_id} ({'restart' if restart else 'fresh'}): {len(kept)} cells kept, {len(pending)} to compute")
    _set_state(ctx, status="running", pid=os.getpid(), started=now(), restart=restart, cause=cause, commit=_git_head())
    done_now, last = [], (list(read_manifest(ctx))[-1:] or [None])[0]
    t0 = time.perf_counter()
    try:
        with P.eval_session(session_id):                                                   # every FINAL load and ledger entry of the session carries this id
            if pending:
                items = load_groups(ctx)
                pool = mp.get_context("spawn").Pool(ctx.procs, initializer=_init_worker) if ctx.procs > 1 else None
                try:
                    for cell in pending:
                        t1 = time.perf_counter()
                        rows = compute_cell(ctx, cell, items, pool, session_id, model_sha)
                        if ctx.crash_hook:
                            ctx.crash_hook("before_write", cell["name"], ctx)
                        e = write_cell(ctx, cell["name"], rows, _git_head(), session_id)
                        P.log_final_access("eval", ENTRY_POINT, {"action": "cell", "cell": cell["name"], "rows": e["rows"], "sha256": e["sha256"]})
                        done_now.append(cell["name"])
                        last = cell["name"]
                        ctx.log(f"[{len(kept) + len(done_now):3d}/{len(ctx.cells)}] {cell['name']:32s} {e['rows']:4d} rows  {time.perf_counter() - t1:6.1f}s")
                        if ctx.crash_hook:
                            ctx.crash_hook("after_cell", cell["name"], ctx)
                finally:
                    if pool is not None:
                        pool.terminate()
                        pool.join()
        man = read_manifest(ctx)
        assert [c["name"] for c in ctx.cells] == list(man), "manifest does not list every cell exactly once in the pinned order"
        for e in man.values():
            verify_entry(ctx, e)
        n_sess = P.count_eval_sessions()
        if n_sess != 1:
            raise RuntimeError(f"count_eval_sessions() = {n_sess}, expected 1")
        summary = {"session_id": session_id, "cells": len(man), "computed_this_run": len(done_now), "kept": len(kept), "wall_seconds": time.perf_counter() - t0,
                   "eval_sessions": n_sess, "finished": now()}
        (ctx.out / "session_complete.json").write_text(json.dumps(summary, indent=1))
        _set_state(ctx, status="complete", finished=now())
        return summary
    except Exception:
        log_crash(ctx, last, traceback.format_exc(), "exception handler")
        _set_state(ctx, status="crashed", time=now(), last_cell=last)
        raise


# ---------------------------------------------------------------------------------------------------------------- rehearsal
def rehearse(procs: int, keep: bool, crash_at: int | None, tmp_root: str | None) -> int:
    import scripts_b15_analysis as A
    from breachlens import pool as P
    t0 = time.perf_counter()
    tmp = Path(tempfile.mkdtemp(prefix="b15_rehearsal_", dir=tmp_root))
    real_ledger, real_before = P.LEDGER_PATH, P.LEDGER_PATH.read_bytes()
    P.LEDGER_PATH = tmp / "final_access.log"                                              # the real ledger is never written
    shutil.copy(real_ledger, P.LEDGER_PATH)
    out = tmp / "b15"
    print(f"REHEARSAL (not a result). temp dir {tmp}", flush=True)
    ok = False
    try:
        # ---- refusals
        refusals = rehearse_refusals(tmp, procs)
        # ---- the session with a simulated kill in the middle of the cell list
        cells = C.all_cells()
        k = len(cells) // 2 if crash_at is None else crash_at
        state = {"killed": False}

        def hook(event, name, ctx):
            if event == "before_write" and not state["killed"] and name == cells[k]["name"]:
                state["killed"] = True
                state["sha_before"] = {e["cell"]: e["sha256"] for e in read_manifest(ctx).values()}
                (ctx.raw).mkdir(parents=True, exist_ok=True)
                with open(ctx.raw / f"{name}.csv", "x") as f:                              # a half-written cell file, as a kill in the middle of a write leaves
                    f.write("cell,condition\n" + name + ",partial")
                raise SimulatedKill(name)

        ctx = Ctx(mode="rehearse", out=out, procs=procs, crash_hook=hook, run_suite=False, require_clean=False)
        try:
            run_session(ctx)
            raise AssertionError("the simulated kill did not happen")
        except SimulatedKill as e:
            print(f"-- simulated kill at cell {e} (index {k}); a half-written file was left behind")
        sid1 = json.loads(ctx.session_path.read_text())["session_id"]
        ctx2 = Ctx(mode="rehearse", out=out, procs=procs, run_suite=False, require_clean=False)
        summary = run_session(ctx2, restart=True, cause="machine") if _head_matches(out) else run_session(ctx2, restart=True, cause="code-fix")
        sid2 = json.loads(ctx2.session_path.read_text())["session_id"]
        man = read_manifest(ctx2)
        assert sid1 == sid2 and all(e["session_id"] == sid1 for e in man.values()), "restart changed the session id"
        for c, sha in state["sha_before"].items():
            assert man[c]["sha256"] == sha, f"kept cell {c} was recomputed"
        assert summary["kept"] == k and summary["computed_this_run"] == len(cells) - k, summary
        assert (out / "discarded").exists() and any((out / "discarded").glob(f"{cells[k]['name']}.*")), "partial file was not discarded"
        assert P.count_eval_sessions() == 1
        try:                                                                              # a finished session cannot be restarted, a second fresh start is refused
            run_session(Ctx(mode="rehearse", out=out, procs=procs, run_suite=False, require_clean=False))
            raise AssertionError("second fresh start was not refused")
        except Refusal:
            pass
        # ---- analysis (raw files only), twice, determinism
        tables = out / "tables"
        A.analyze(out, tables, label="REHEARSAL")
        h1 = _tree_hash(tables)
        A.analyze(out, tmp / "tables2", label="REHEARSAL")
        assert h1 == _tree_hash(tmp / "tables2"), "analysis is not deterministic"
        assert P.LEDGER_PATH != real_ledger and real_ledger.read_bytes() == real_before, "the real ledger changed"
        wall = time.perf_counter() - t0
        print(f"\nREHEARSAL OK: {len(cells)} cells, kill + restart (same session_id {sid1[:8]}..., {summary['kept']} kept, {summary['computed_this_run']} recomputed), "
              f"refusals {refusals}, analysis deterministic, real ledger untouched, temp ledger {P.count_eval_sessions()} eval session; wall {wall:.0f} s ({wall / 60:.1f} min)")
        print(f"session wall (kill run + restart run) {wall:.0f} s total incl. refusals and analysis; tables: {tables}")
        dest = ROOT / "results" / "b15_rehearsal"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(tables, dest / "tables")                                           # results/* is gitignored; REHEARSAL tables, never results
        shutil.copy(out / "manifest.jsonl", dest / "manifest.jsonl")
        print(f"rehearsal tables copied to {dest}")
        ok = True
        return 0
    finally:
        P.LEDGER_PATH = real_ledger
        if keep or not ok:
            print(f"kept: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def _head_matches(out: Path) -> bool:
    man = list(read_manifest(Ctx(mode="rehearse", out=out)).values())
    return (not man) or man[-1]["commit"] == _git_head()


def _tree_hash(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(d).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(d)).encode() + p.read_bytes())
    return h.hexdigest()


def rehearse_refusals(tmp: Path, procs: int) -> list:
    """Each refusal the real run has, provoked on purpose; every one must raise Refusal. Returns the list of refusals that fired."""
    from breachlens import pool as P
    fired = []

    def expect(name, fn):
        try:
            fn()
        except (Refusal, PermissionError, CorruptCell) as e:
            fired.append(name)
            return
        raise AssertionError(f"refusal {name!r} did not fire")
    base = lambda **kw: Ctx(mode="real", unlock=True, out=tmp / "refusal_out", **kw)
    expect("no --unlock-final", lambda: run_session(Ctx(mode="real", unlock=False, out=tmp / "refusal_out")))
    expect("dirty tree", lambda: check_clean_tree(base(git_dirty=lambda: True)))
    # ledger with an eval entry
    led_backup = P.LEDGER_PATH.read_bytes()
    try:
        with open(P.LEDGER_PATH, "a") as f:
            f.write(json.dumps({"purpose": "eval", "session_id": "other", "action": "x"}) + "\n")
        expect("ledger eval entries", lambda: check_ledger(base()))
        expect("restart with a foreign eval session", lambda: check_ledger(base(), restart=True, session_id="mine"))
    finally:
        P.LEDGER_PATH.write_bytes(led_backup)
    d = tmp / "refusal_out"
    d.mkdir(exist_ok=True)
    (d / "session.json").write_text("{}")
    expect("existing session.json", lambda: check_session_json(base()))
    (d / "session.json").unlink()
    expect("restart without session.json", lambda: check_session_json(base(), restart=True))
    # models: a copy of the models dir with one model missing, and one with a flipped byte
    md = tmp / "models_copy"
    for r in ("grid_ppo_plain_s0", "grid_ddqn_s0"):
        (md / r).mkdir(parents=True)
        for fn in (MODELS / r).glob("final.*"):
            shutil.copy(fn, md / r / fn.name)
    sub = [c for c in C.all_cells() if c.get("run") in ("grid_ppo_plain_s0", "grid_ddqn_s0")]
    audit = tmp / "audit_copy.json"
    shutil.copy(AUDIT_JSON, audit)
    (md / "grid_ddqn_s0" / "final.pt").rename(md / "grid_ddqn_s0" / "final.pt.moved")
    expect("missing model", lambda: check_models(base(cells=sub, models_dir=md, audit_path=audit)))
    (md / "grid_ddqn_s0" / "final.pt.moved").rename(md / "grid_ddqn_s0" / "final.pt")
    b = bytearray((md / "grid_ppo_plain_s0" / "final.zip").read_bytes())
    b[len(b) // 2] ^= 0xFF
    (md / "grid_ppo_plain_s0" / "final.zip").write_bytes(bytes(b))
    expect("model hash mismatch", lambda: check_models(base(cells=sub, models_dir=md, audit_path=audit)))
    j = json.loads(audit.read_text())
    j["runs"][0]["final_sha256"] = "0" * 64
    audit.write_text(json.dumps(j))
    expect("audit file hash mismatch", lambda: check_audit(base(audit_path=audit), rerun=False))
    expect("full suite not green", lambda: check_suite(base(run_suite=True, suite_cmd=[sys.executable, "-c", "import sys; sys.exit(3)"])))
    # restart rules
    expect("restart without cause", lambda: run_session(Ctx(mode="rehearse", out=tmp / "nosession"), restart=True, cause=None))
    return fired


# ---------------------------------------------------------------------------------------------------------------- CLI
def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rehearse", action="store_true", help="whole session on DEV-validation + TUNE-val, temp dir, temp ledger (no FINAL)")
    ap.add_argument("--unlock-final", action="store_true", help="REAL run: load FINAL through the loaders (needs a clean commit and 0 eval entries in the ledger)")
    ap.add_argument("--restart", action="store_true", help="continue a crashed real session (same session_id)")
    ap.add_argument("--cause", choices=("machine", "code-fix"), help="with --restart: what killed the session (row (gg) (e)(5))")
    ap.add_argument("--defect-agreed", action="store_true", help="with --cause code-fix: the defect was reported to the user and the fix agreed (row (hh))")
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--keep", action="store_true", help="rehearsal: keep the temp dir")
    ap.add_argument("--crash-at", type=int, default=None, help="rehearsal: index of the cell whose write is 'killed' (default: the middle)")
    ap.add_argument("--tmp-root", default=None)
    a = ap.parse_args(argv)
    if a.rehearse:
        if a.unlock_final or a.restart:
            print("REFUSED: --rehearse never runs with --unlock-final / --restart", file=sys.stderr)
            return 2
        return rehearse(a.procs, a.keep, a.crash_at, a.tmp_root)
    if not a.unlock_final:
        print("REFUSED: the real run needs --unlock-final (and a clean commit, 0 eval entries in the ledger, no session.json)", file=sys.stderr)
        return 2
    ctx = Ctx(mode="real", out=REAL_OUT, unlock=True, procs=a.procs)
    try:
        s = run_session(ctx, restart=a.restart, cause=a.cause, defect_agreed=a.defect_agreed)
    except (Refusal, CorruptCell) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
