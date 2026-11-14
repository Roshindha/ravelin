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
