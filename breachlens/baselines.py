"""B8 baseline policies (docs/SPEC_V2.md 3.4, 11; Gate changelog (k)).

A policy turns one layout into a list of episodes; each episode carries the integer outcome (d_num, d_den, core_destroyed) and army counts.
The score is NEVER computed here: the evaluator converts outcomes through `breachlens.score`.
Policies: Random (uniform over legal env actions through BreachEnv, 5 seeds derived from the layout id), Greedy (B5 definition, no Mender pairing),
SingleUnit(t), BestSingle (type chosen per family on the matching training split; selection helper below), BestSingle-hindsight and
BestSingle-perlayout (diagnostics, never gate baselines), Oracle (oracle.py, 200 + 200 with warm starts, optional cache).
"""
from __future__ import annotations

import zlib

import numpy as np

from . import oracle as O
from .acceptance import warm_starts
from .armies import greedy_types, place_on_border, single_unit_types
from .constants import UNIT_TYPES
from .score import result_key
from .sim import Prepared, simulate

N_RANDOM_SEEDS = 5


def _ep(r, n_bought: int, n_deployed: int, **extra) -> dict:
    return {"d_num": int(r.d_num), "d_den": int(r.d_den), "core": bool(r.core_destroyed), "n_bought": int(n_bought), "n_deployed": int(n_deployed),
            "reason": r.reason, "ticks": int(r.ticks), **extra}


def random_seeds(layout_id: str, n: int = N_RANDOM_SEEDS) -> list[int]:
    """Fixed per-layout seeds derived deterministically from the layout id (crc32 of 'id:k')."""
    return [zlib.crc32(f"{layout_id}:{k}".encode()) for k in range(n)]


class Policy:
    name = "policy"

    def episodes(self, rec: dict, layout: dict, P: Prepared) -> list[dict]:     # pragma: no cover - interface
        raise NotImplementedError

    def seeds(self, rec: dict) -> list:
        return []


class SingleUnit(Policy):
    def __init__(self, t: int):
        self.t, self.name = int(t), f"Single-{UNIT_TYPES[int(t)]}"

    def episodes(self, rec, layout, P):
        types = single_unit_types(self.t)
        return [_ep(simulate(P, place_on_border(layout, types)), len(types), len(types))]


class Greedy(Policy):
    name = "Greedy"

    def episodes(self, rec, layout, P):
        types = greedy_types()
        return [_ep(simulate(P, place_on_border(layout, types)), len(types), len(types))]


class BestSingle(Policy):
    """Single-unit army whose type is fixed per family (`by_family`) or, for a family without a training split, ONE pooled type (`pooled`)."""

    def __init__(self, by_family: dict | None = None, pooled: int | None = None, name: str = "BestSingle"):
        self.by_family, self.pooled, self.name = dict(by_family or {}), pooled, name

    def type_for(self, family: str) -> int:
        if family in self.by_family:
            return int(self.by_family[family])
        if self.pooled is not None:
            return int(self.pooled)
        raise KeyError(f"no BestSingle type for family {family!r}")

    def episodes(self, rec, layout, P):
        t = self.type_for(rec["family"])
        types = single_unit_types(t)
        return [_ep(simulate(P, place_on_border(layout, types)), len(types), len(types), chosen_type=t)]


class BestSinglePerLayout(Policy):
    """DIAGNOSTIC upper reference, not a baseline: the best of the 5 single-unit armies on each layout (exact key; ties -> lowest type index)."""
    name = "BestSingle-perlayout"

    def episodes(self, rec, layout, P):
        best = None
        for t in range(len(UNIT_TYPES)):
            types = single_unit_types(t)
            r = simulate(P, place_on_border(layout, types))
            num, den = result_key(r)
            if best is None or num * best[1] > best[0] * den:      # exact cross-multiplied comparison (denominators can differ)
                best = (num, den, t, r, types)
        _, _, t, r, types = best
        return [_ep(r, len(types), len(types), chosen_type=t)]


class Oracle(Policy):
    """oracle.oracle_search with the pre-registered warm starts, rng [layout seed, 4242]. `cache` = {layout id: episode dict} (read only here)."""

    def __init__(self, n_random: int = O.N_RANDOM, n_refine: int = O.N_REFINE, cache: dict | None = None):
        self.n_random, self.n_refine, self.cache, self.name = n_random, n_refine, cache, "Oracle"

    def seeds(self, rec):
        return [int(rec["seed"]), O.ORACLE_TAG]

    def episodes(self, rec, layout, P):
        if self.cache and rec["id"] in self.cache:
            return [dict(self.cache[rec["id"]], cached=True)]
        res = O.oracle_search(layout, O.oracle_rng(rec["seed"]), n_random=self.n_random, n_refine=self.n_refine, warm_starts=warm_starts(), prepared=P)
        return [_ep(res.result, res.units, res.units, oracle_sims=res.sims, oracle_start=res.start, oracle_gold=res.gold)]


class RandomPolicy(Policy):
    """Uniform over the legal actions of the real env (BUY, END_BUY, DEPLOY, WAIT), 5 seeds derived from the layout id. A floor, not a competitor."""
    name = "Random"

    def seeds(self, rec):
        return random_seeds(rec["id"])

    def episodes(self, rec, layout, P):
        from .env import BreachEnv                      # local import: env pulls gymnasium
        env = BreachEnv(layouts=[layout], role="train")  # synthetic layout path: the evaluator already loaded the layout through the pool loaders
        out = []
        for s in self.seeds(rec):
            env.reset(seed=s, options={"layout_index": 0})
            rng = np.random.default_rng(s)
            info = {}
            done = False
            while not done:
                legal = np.flatnonzero(env.action_masks())
                _, _, done, _, info = env.step(int(legal[int(rng.integers(len(legal)))]))
            r = env.result
            out.append(_ep(r, env.bought, len(env.army)))
        return out


def select_best_single(single_rows: list[dict]) -> dict:
    """Pick the BestSingle type per family from per-layout rows of Single-<type> policies on the TRAINING split.
    `single_rows`: evaluator layout rows with fields family, policy ('Single-<Unit>'), score. Highest mean score per family; ties -> lowest type index.
    Also returns the pooled choice (mean over the pooled familiar layouts) for a family without a training split (labyrinth transfer)."""
    names = {f"Single-{u}": i for i, u in enumerate(UNIT_TYPES)}
    fams = sorted({r["family"] for r in single_rows})
    out = {"by_family": {}, "means": {}, "pooled": None, "pooled_means": None}
    pooled = np.zeros(len(UNIT_TYPES))
    cnt = np.zeros(len(UNIT_TYPES))
    for f in fams:
        m = np.zeros(len(UNIT_TYPES))
        n = np.zeros(len(UNIT_TYPES))
        for r in single_rows:
            if r["family"] == f and r["policy"] in names:
                m[names[r["policy"]]] += r["score"]
                n[names[r["policy"]]] += 1
        assert (n > 0).all(), f"missing single-unit rows for family {f}"
        m = m / n
        out["by_family"][f] = int(np.argmax(m))                  # np.argmax returns the lowest index on ties
        out["means"][f] = m.tolist()
        pooled += m * n
        cnt += n
    out["pooled"] = int(np.argmax(pooled / cnt))
    out["pooled_means"] = (pooled / cnt).tolist()
    return out
