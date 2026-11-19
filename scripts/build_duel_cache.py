"""B17: build the duel opponent cache (docs/SPEC_V2.md, Gate changelog row (kk)).

    .venv/bin/python scripts/build_duel_cache.py --estimate          # time ONE layout, project the full build, write nothing
    .venv/bin/python scripts/build_duel_cache.py --procs 6           # full build of data/v2/duel_cache.json (all 30 DEV-validation layouts)

Per layout and contestant: score, D, breach, units bought per type, deploy tiles. Contestants: the three seed-0 arms WITH TTA (B15 aggregation, same
chooser code as ``scripts_b15_final.tta_policy``), BestSingle (B8 per-family type), Oracle (B8 ``oracle_search``, 200 + 200, B8 warm starts, rng
[layout seed, 4242]). Layouts only through ``breachlens.evaluator.load_dev_layout`` (DEV-validation); nothing else can be loaded. No timestamps:
the file is a deterministic function of (code, models, layouts). Not a result: demo data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from breachlens import duel as D  # noqa: E402
from breachlens import evaluator as EV  # noqa: E402
from breachlens import oracle as O  # noqa: E402
from breachlens import pool as P  # noqa: E402
from breachlens.acceptance import warm_starts  # noqa: E402
from breachlens.armies import place_on_border, single_unit_types  # noqa: E402
from breachlens.baselines import BestSingle  # noqa: E402
from breachlens.config import sim_version  # noqa: E402
from breachlens.constants import UNIT_TYPES  # noqa: E402
from breachlens.env import BreachEnv  # noqa: E402
from breachlens.score import SCORE_VERSION  # noqa: E402
from breachlens.sim import Prepared, simulate  # noqa: E402

BEST_SINGLE_JSON = ROOT / "data" / "v2" / "baselines" / "b8_best_single.json"
ARM_FOR = {"ppo_plain_tta": "ppo_plain", "ppo_d4_tta": "ppo_d4", "ddqn_tta": "ddqn"}
MAX_UNATTENDED_SECONDS = 600


def best_single_map() -> dict:
    d = json.loads(BEST_SINGLE_JSON.read_text())
    return {"by_family": {k: int(v) for k, v in d["train_chosen_for_set"]["DEV-train"].items()}, "pooled": int(d["pooled_dev_train_type_for_labyrinth_transfer_B15"])}


def tta_episode(model, algo: str, rec: dict, layout: dict) -> tuple[dict, list]:
    """The learner's B15 TTA episode (reset seed crc32(id)); returns (row, actions). Chooser = rl.explain._choose(tta=True), i.e. rl.tta.tta_action
    with the row (gg) aggregation; the test compares its actions with `scripts_b15_final._task`."""
    from rl import explain as X
    choose = X._choose(model, algo, True)
    env = BreachEnv(layouts=[layout], role="train")
    obs, _ = env.reset(seed=D.reset_seed(rec["id"]), options={"layout_index": 0})
    taken, done = [], False
    while not done:
        a = int(choose(obs, env.action_masks()))
        taken.append(a)
        obs, _, done, _, _ = env.step(a)
    return D.outcome_row("", env.result, env.army, env.owned, env.bought), taken


def best_single_row(rec: dict, layout: dict, bs: dict) -> dict:
    t = BestSingle(bs["by_family"], pooled=bs["pooled"]).type_for(rec["family"])
    types = single_unit_types(t)
    army = place_on_border(layout, types)
    return D.outcome_row("best_single", simulate(Prepared(layout), army), army, [0] * len(UNIT_TYPES), len(types))


def oracle_row(rec: dict, layout: dict) -> dict:
    res = O.oracle_search(layout, O.oracle_rng(rec["seed"]), n_random=O.N_RANDOM, n_refine=O.N_REFINE, warm_starts=warm_starts(), prepared=Prepared(layout))
    return D.outcome_row("oracle", res.result, list(res.army), [0] * len(UNIT_TYPES), res.units)


def load_models(models_dir: Path | None = None) -> dict:
    from rl import explain as X
    return {arm: X.load_display_model(arm, models_dir or X.MODELS) for arm in ("ppo_plain", "ppo_d4", "ddqn")}


def contestants_for(rec: dict, layout: dict, models: dict, bs: dict) -> dict:
    out = {}
    for key, arm in ARM_FOR.items():
        m, algo, _ = models[arm]
        row, _ = tta_episode(m, algo, rec, layout)
        out[key] = dict(row, contestant=key)
    out["best_single"] = best_single_row(rec, layout, bs)
    out["oracle"] = oracle_row(rec, layout)
    return {k: out[k] for k in D.CONTESTANTS}


def _work(args):
    layout_id = args
    models = _work.models if hasattr(_work, "models") else None
    if models is None:
        import torch as th
        th.set_num_threads(1)
        _work.models = models = load_models()
    rec, layout = EV.load_dev_layout(D.DUEL_SPLIT, None, layout_id, entry_point="scripts.build_duel_cache")
    t0 = time.perf_counter()
    res = contestants_for(rec, layout, models, best_single_map())
    return layout_id, {"family": rec["family"], "contestants": res}, time.perf_counter() - t0


def _git(*a) -> str:
    import subprocess
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def provenance() -> dict:
    from rl import explain as X
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    models = {}
    for arm, (run, algo) in X.DISPLAY_MODELS.items():
        f = X.MODELS / run / ("final.zip" if algo == "ppo" else "final.pt")
        models[arm + "_tta"] = {"run": run, "algo": algo, "file": f.name, "sha256": sha(f)}
    return {"split": D.DUEL_SPLIT, "code_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")), "models": models,
            "sim_version": sim_version(), "score_version": SCORE_VERSION,
            "generator_hash": json.loads(P.GENERATOR_HASH_PATH.read_text())["generator_hash"], "pool_sha256": P.pool_sha256(),
            "tta": "B15 row (gg): PPO mean log-prob over the 8 D4 views, DDQN mean of the 8 mapped-back Q-values, then masked argmax",
            "best_single": {"source": "data/v2/baselines/b8_best_single.json (DEV-train choice)", **best_single_map(), "unit_names": list(UNIT_TYPES)},
            "oracle": {"label": "search baseline, not the optimum", "n_random": O.N_RANDOM, "n_refine": O.N_REFINE, "approx_simulations": O.N_RANDOM + O.N_REFINE,
                       "warm_starts": "B8 (breachlens.acceptance.warm_starts)", "rng": "default_rng([layout seed, 4242])"},
            "reset_seed": "crc32(layout id), shared with the human",
            "note": "demo data on DEV-validation layouts; not a result; no timestamps"}


def layout_ids() -> list[str]:
    return EV.dev_layout_ids(D.DUEL_SPLIT, entry_point="scripts.build_duel_cache")


def build(out: Path, procs: int = 1, ids: list[str] | None = None, allow_dirty: bool = False) -> dict:
    ids = ids or layout_ids()
    for i in ids:
        D._check_duel_id(i)                                   # FINAL / transfer / non-DEV-validation ids refuse here, before any work
    prov = provenance()
    if prov["git_dirty"] and not allow_dirty:
        raise SystemExit("refusing to build from a dirty tree (provenance would be wrong); commit first or pass --allow-dirty for a scratch build")
    if procs > 1:
        with mp.get_context("spawn").Pool(procs) as pool:
            res = pool.map(_work, ids, chunksize=1)
    else:
        res = [_work(i) for i in ids]
    cache = {"version": D.CACHE_VERSION, "meta": prov, "layouts": {i: v for i, v, _ in res}}
    Path(out).write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    return cache


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--estimate", action="store_true", help="time one layout, project the full build, write nothing")
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--out", default=str(D.CACHE_PATH))
    p.add_argument("--allow-dirty", action="store_true")
    a = p.parse_args(argv)
    ids = layout_ids()
    if a.estimate:
        t0 = time.perf_counter()
        _, v, secs = _work(ids[0])
        load_s = time.perf_counter() - t0 - secs
        total = secs * len(ids)
        print(f"one layout ({ids[0]}): {secs:.1f} s (+ {load_s:.1f} s model load); {len(ids)} layouts serial = {total / 60:.1f} min; "
              f"with 6 procs about {total / 6 / 60 + load_s / 60:.1f} min")
        for k, r in v["contestants"].items():
            print(f"  {k:<14s} score {r['score']:.3f} breach {r['breach']} bought {r['n_bought']}")
        print("PROJECTION: " + ("over 10 min serial -> do NOT run unattended from Claude Code" if total > MAX_UNATTENDED_SECONDS else "under 10 min"))
        return 0
    build(Path(a.out), a.procs, allow_dirty=a.allow_dirty)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
