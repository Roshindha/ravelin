"""Oracle search (spec 3.4 G8, 7; B5). One implementation, reused by B8 (baselines) and the gate script: no second copy.

* `oracle_search`  -- full Oracle: army AND deployment search (random armies with random start tiles, then hill-climbing
  with replace / add / remove / start-shift / order-swap moves). A lower bound on the true optimum, not exact.
* `army_search`    -- G8(a): army search only, scripted deployment (Greedy ranking order, default start tile), moves
  replace / add / remove.
Both are pure functions of (layout, rng, budget, allowed set, warm starts). Pairing rule (spec 3.4): runs that are compared
use the SAME rng stream (`oracle_rng(layout_seed)`); `allowed` only restricts which unit types may be drawn.
Comparisons inside a search use an exact integer key (no float decides anything).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .armies import army_cost, place_on_border, random_types, sort_by_rank, default_start
from .config import CFG
from .constants import BUDGET, N_BORDER, N_UNIT_TYPES
from .sim import Prepared, simulate

ORACLE_TAG = 4242          # spec 3.4: per-layout seed stream [layout_id, 4242]
ARMY_SEARCH_TAG = 4343
ACCEPT_TAG = 5151          # acceptance-time Oracle (too_hard rule): a different stream than the gate Oracle
N_RANDOM, N_REFINE = 200, 200            # full Oracle: ~400 sims
AS_RANDOM, AS_REFINE = 128, 128          # army search: 256 sims


def oracle_rng(layout_seed: int, tag: int = ORACLE_TAG) -> np.random.Generator:
    return np.random.default_rng([int(layout_seed), int(tag)])


@dataclass
class SearchResult:
    score: float
    key: int
    d: float
    breach: bool
    types: tuple          # unit types in deployment order
    start: int            # border start index
    army: tuple           # ((type, tile), ...)
    sims: int
    units: int
    gold: int
    result: object = None  # BattleResult of the best plan
    free: object = None    # SearchResult of the best evaluated army containing no unit of type `track_free` (monotone ban-loss estimator)


def score_key(r) -> tuple:
    """Exact integer ranking key within one layout: score = key / (10 * den)."""
    den = max(r.d_den, 1)
    return 6 * r.d_num + 4 * den * (1 if r.core_destroyed else 0), 10 * den


class _Eval:
    def __init__(self, layout, prepared, track_free=None):
        self.L = layout
        self.P = prepared or Prepared(layout)
        self.sims = 0
        self.best = None   # (key, types, start, battle)
        self.track = track_free
        self.best_free = None

    def run(self, types, start):
        if not types:
            return None
        army = place_on_border(self.L, types, start)
        r = simulate(self.P, army)
        self.sims += 1
        k = score_key(r)[0]
        if self.best is None or k > self.best[0]:
            self.best = (k, tuple(types), int(start), r)
        if self.track is not None and self.track not in types and (self.best_free is None or k > self.best_free[0]):
            self.best_free = (k, tuple(types), int(start), r)
        return k

    def _result(self, best) -> SearchResult:
        k, types, start, r = best
        key, den10 = score_key(r)
        return SearchResult(score=key / den10, key=key, d=r.d, breach=r.core_destroyed, types=types, start=start,
                            army=tuple(place_on_border(self.L, types, start)), sims=self.sims, units=len(types),
                            gold=army_cost(types), result=r)

    def final(self) -> SearchResult:
        if self.best is None:
            return SearchResult(0.0, 0, 0.0, False, (), 0, (), self.sims, 0, 0, None)
        res = self._result(self.best)
        res.free = self._result(self.best_free) if self.best_free is not None else None
        return res


def _legal_add(types, t, budget, cap):
    return len(types) < cap and army_cost(types) + army_cost([t]) <= budget


def _mutate(types, start, rng, allowed, budget, cap, moves):
    """One hill-climbing proposal; returns (types, start) or None if the drawn move is not applicable (a wasted step)."""
    types = list(types)
    mv = moves[int(rng.integers(len(moves)))]
    spent = army_cost(types)
    if mv == "replace" and types:
        i = int(rng.integers(len(types)))
        new = allowed[int(rng.integers(len(allowed)))]
        if spent - army_cost([types[i]]) + army_cost([new]) > budget:
            return None
        types[i] = new
    elif mv == "add":
        new = allowed[int(rng.integers(len(allowed)))]
        if not _legal_add(types, new, budget, cap):
            return None
        types.insert(int(rng.integers(len(types) + 1)), new)
    elif mv == "remove" and len(types) > 1:
        types.pop(int(rng.integers(len(types))))
    elif mv == "shift":
        start = (start + int(rng.integers(-7, 8))) % N_BORDER
    elif mv == "swap" and len(types) > 1:
        i, j = (int(x) for x in rng.choice(len(types), size=2, replace=False))
        types[i], types[j] = types[j], types[i]
    else:
        return None
    return types, start


def oracle_search(layout: dict, rng, n_random: int = N_RANDOM, n_refine: int = N_REFINE, allowed=None,
                  warm_starts=(), prepared=None, budget: int = BUDGET, cap: int | None = None, track_free=None) -> SearchResult:
    """Full Oracle. `warm_starts`: ((types, start|None), ...) evaluated first (they consume no rng, so a banned-type Oracle
    that skips some of them keeps the same random stream); any warm start containing a type outside `allowed` is skipped."""
    allowed = tuple(range(N_UNIT_TYPES)) if allowed is None else tuple(allowed)
    cap = CFG.unit_cap if cap is None else cap
    ev = _Eval(layout, prepared, track_free)
    for types, start in warm_starts:
        if set(types) <= set(allowed) and army_cost(types) <= budget and len(types) <= cap:
            ev.run(list(types), default_start(layout) if start is None else start)
    for _ in range(n_random):
        types = random_types(rng, allowed, budget=budget, cap=cap)
        start = int(rng.integers(N_BORDER))
        ev.run(types, start)
    for _ in range(n_refine):
        if ev.best is None:
            break
        _, bt, bs, _ = ev.best
        prop = _mutate(bt, bs, rng, allowed, budget, cap, ("replace", "add", "remove", "shift", "swap"))
        if prop is not None:
            ev.run(*prop)
    return ev.final()


def monotone_ban_pair(layout: dict, seed: int, unit: int, ablated: SearchResult, warm_starts=(), prepared=None, n_random: int = N_RANDOM,
                      n_refine: int = N_REFINE, budget: int = BUDGET, cap: int | None = None) -> dict:
    """Monotone ban-loss estimator (gate changelog (d)). `ablated` = the Oracle search without `unit` (same rng stream `oracle_rng(seed)`).
    The full Oracle is re-run on the same rng stream and the same warm starts PLUS the ablated best army (with its start tile), so
    full >= ablated by construction. Every unit-free army the full search evaluates is also a candidate for the ablated result (the ablated
    side gets everything the full side saw): ablated' = best of (ablated, best unit-free army inside the full search). loss = full - ablated' >= 0."""
    ws = tuple(warm_starts) + (((tuple(ablated.types), ablated.start),) if ablated.types else ())
    full = oracle_search(layout, oracle_rng(seed), n_random=n_random, n_refine=n_refine, warm_starts=ws, prepared=prepared, budget=budget, cap=cap,
                         track_free=unit)
    abl = ablated
    if full.free is not None and full.free.key > abl.key:
        abl = full.free
    assert full.key >= abl.key
    return {"full": full, "ablated": abl, "loss": full.score - abl.score}


def army_search(layout: dict, rng, n_random: int = AS_RANDOM, n_refine: int = AS_REFINE, allowed=None, prepared=None,
                budget: int = BUDGET, cap: int | None = None) -> SearchResult:
    """G8(a): search over armies only; every candidate is deployed by the scripted deployer (Greedy ranking order, default start)."""
    allowed = tuple(range(N_UNIT_TYPES)) if allowed is None else tuple(allowed)
    cap = CFG.unit_cap if cap is None else cap
    ev = _Eval(layout, prepared)
    start = default_start(layout)
    for _ in range(n_random):
        ev.run(sort_by_rank(random_types(rng, allowed, budget=budget, cap=cap)), start)
    for _ in range(n_refine):
        if ev.best is None:
            break
        prop = _mutate(ev.best[1], start, rng, allowed, budget, cap, ("replace", "add", "remove"))
        if prop is not None:
            ev.run(sort_by_rank(prop[0]), start)
    return ev.final()
