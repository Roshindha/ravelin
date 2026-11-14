"""Army helpers shared by tests, scripts and (later) baselines: random legal armies and a simple scripted army.

Stats come from breachlens.config only. B4 moved these out of the provisional bench layouts.
"""
from __future__ import annotations

import numpy as np

from .actions import BORDER


def free_tiles(layout: dict) -> list[int]:
    walls = set(map(tuple, layout["walls"]))
    return [i for i, rc in enumerate(BORDER) if rc not in walls]


def random_army(rng, layout: dict, min_units: int = 10, max_units: int = 20, budget: int = 240,
                n_types: int = 4) -> list:
    """Random legal army (no Menders in B2): random buys within budget/cap, each on a distinct free border tile."""
    from .config import CFG
    costs = [CFG.units[t].cost for t in range(n_types)]
    for _ in range(100):
        types, gold = [], budget
        while len(types) < max_units:
            opts = [t for t in range(n_types) if costs[t] <= gold]
            if not opts or (len(types) >= min_units and rng.random() < 0.1):
                break
            t = int(rng.choice(opts))
            types.append(t)
            gold -= costs[t]
        if len(types) >= min_units:
            break
    tiles = free_tiles(layout)
    pick = rng.choice(len(tiles), size=len(types), replace=False)
    return [(t, tiles[int(k)]) for t, k in zip(types, pick)]


# Simple scripted army (a fixed mixed composition, clustered on consecutive free border tiles starting at the
# tile nearest the core; all choices are D4-covariant). Counts are fixed by this table; the spend is derived from
# config (2 Bulwark + 2 Sniper + 2 Blaster + 1 Mender + 4 Scout = 240 gold at the current stats).
SCRIPTED_COMPOSITION = (("bulwark", 2), ("sniper", 2), ("blaster", 2), ("mender", 1), ("scout", 4))


def scripted_army(layout: dict) -> list:
    from .config import CFG
    from .constants import UNIT_TYPES
    types = [UNIT_TYPES.index(n) for n, k in SCRIPTED_COMPOSITION for _ in range(k)]
    assert sum(CFG.units[t].cost for t in types) <= 240
    core = layout["core"]["cell"]
    tiles = free_tiles(layout)
    d2 = lambda i: (BORDER[i][0] - core[0]) ** 2 + (BORDER[i][1] - core[1]) ** 2
    start = min(tiles, key=lambda i: (d2(i), i))  # exact-distance ties broken by index (scripted; not part of any D4 claim)
    k0 = tiles.index(start)
    return [(t, tiles[(k0 + j) % len(tiles)]) for j, t in enumerate(types)]


# ------------------------------------------------------------------ B5: scripted deployer, single-unit armies, Greedy, random buys
def _cfg():
    from .config import CFG
    return CFG


def default_start(layout: dict) -> int:
    """Default start tile of the scripted deployer: the border tile nearest the core (squared Euclidean; exact ties -> lowest
    index; scripted, not part of any D4 claim)."""
    core = layout["core"]["cell"]
    tiles = free_tiles(layout)
    return min(tiles, key=lambda i: ((BORDER[i][0] - core[0]) ** 2 + (BORDER[i][1] - core[1]) ** 2, i))


def place_on_border(layout: dict, types, start: int | None = None) -> list:
    """Scripted deployment: units in the given order on consecutive free border tiles (clockwise) from `start`.

    `start` is a border index; if that tile is not free the next free tile clockwise is used. Default: `default_start`."""
    tiles = free_tiles(layout)
    s = default_start(layout) if start is None else int(start) % len(BORDER)
    k0 = next((j for j, t in enumerate(tiles) if t >= s), 0)
    return [(int(t), tiles[(k0 + j) % len(tiles)]) for j, t in enumerate(types)]


def unit_cost(t: int) -> int:
    return _cfg().units[t].cost


def army_cost(types) -> int:
    return sum(unit_cost(t) for t in types)


def single_unit_types(t: int, budget: int = 240, cap: int | None = None) -> list:
    """'Best fixed single-unit army' composition (spec 3.4): as many units of one type as fit in min(budget, cap)."""
    cap = _cfg().unit_cap if cap is None else cap
    return [t] * min(cap, budget // unit_cost(t))


def greedy_rank() -> tuple:
    """Unit types ordered by Lanchester value hp*damage/cost^2 (descending; ties by type index). A Mender (damage 0) ranks last.
    This is the 'Greedy ranking order' used for scripted deployment and by `greedy_types`."""
    c = _cfg()
    return tuple(sorted(range(len(c.units)), key=lambda t: (-(c.units[t].hp * c.units[t].damage) / c.units[t].cost ** 2, t)))


def sort_by_rank(types) -> list:
    rank = {t: i for i, t in enumerate(greedy_rank())}
    return sorted(types, key=lambda t: (rank[t], t))


def greedy_types(budget: int = 240, cap: int | None = None) -> list:
    """Greedy army: walk the Lanchester ranking, buy as many of each as fit (cap respected), no Mender pairing in B5
    (B8 may refine the rule; any change is a G9 input change)."""
    cap = _cfg().unit_cap if cap is None else cap
    types, gold = [], budget
    for t in greedy_rank():
        k = min(gold // unit_cost(t), cap - len(types))
        types += [t] * k
        gold -= k * unit_cost(t)
    return types


def random_types(rng, allowed=None, stop_p: float = 0.1, budget: int = 240, cap: int | None = None) -> list:
    """Phase-A style random legal army (spec section 6): random affordable (allowed) types, stop w.p. `stop_p` after each buy, <= cap units."""
    cap = _cfg().unit_cap if cap is None else cap
    pool = list(range(len(_cfg().units))) if allowed is None else list(allowed)
    types, gold = [], budget
    while len(types) < cap:
        opts = [t for t in pool if unit_cost(t) <= gold]
        if not opts:
            break
        t = int(opts[int(rng.integers(len(opts)))])
        types.append(t)
        gold -= unit_cost(t)
        if rng.random() < stop_p:
            break
    return types
