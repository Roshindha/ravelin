"""PROVISIONAL layout builders for the B2 speed gate and simulator tests only.

The real generators, acceptance rules and design sets are B4. These are rough sketches of the four
SPEC_V2 section 5.1 families, sized so that sim cost is representative (6-12 buildings, 4-10 defenders,
up to ~40 walls). They are NOT balanced and must not be used for any gate (G1-G9). Replaced in B4.
"""
from __future__ import annotations

import numpy as np

from .actions import BORDER
from .constants import GRID

FAMILIES = ("spine_pods", "archipelago", "bunker_nest", "labyrinth")


def _free(occ, r, c):
    return 1 <= r < GRID - 1 and 1 <= c < GRID - 1 and (r, c) not in occ


def _put(layout, occ, kind, cell, **kw):
    if not _free(occ, *cell):
        return False
    occ.add(cell)
    if kind == "building":
        layout["buildings"].append({"cell": cell, "value": int(kw["value"])})
    elif kind == "def":
        layout["defenders"].append({"cell": cell, "type": kw["type"]})
    elif kind == "wall":
        layout["walls"].append(cell)
    elif kind == "core":
        layout["core"] = {"cell": cell}
    return True


def _defender_type(rng, lancer_p, flak_p):
    x = rng.random()
    return "lancer" if x < lancer_p else ("flak" if x < lancer_p + flak_p else "gun")


def _scatter(layout, occ, rng, kind, n, region, **kw):
    placed, tries = 0, 0
    while placed < n and tries < 200:
        tries += 1
        cell = (int(rng.integers(region[0], region[1])), int(rng.integers(region[2], region[3])))
        if kind == "building":
            ok = _put(layout, occ, "building", cell, value=int(rng.integers(1, 4)))
        else:
            ok = _put(layout, occ, "def", cell, type=_defender_type(rng, kw.get("lp", 0.2), kw.get("fp", 0.15)))
        placed += ok


def make_layout(family: str, seed: int) -> dict:
    rng = np.random.default_rng([FAMILIES.index(family), seed])
    L = {"buildings": [], "core": None, "walls": [], "defenders": []}
    occ: set = set()
    if family == "spine_pods":
        col = int(rng.integers(6, 8))
        for r in range(3, 11):
            _put(L, occ, "wall" if r % 2 else "def", (r, col), type=_defender_type(rng, 0.25, 0.1))
        _put(L, occ, "core", (11, col))
        for k in range(int(rng.integers(3, 5))):
            side = -1 if k % 2 else 1
            r0 = 3 + 2 * k
            for j in range(int(rng.integers(3, 5))):
                _put(L, occ, "building", (r0 + (j % 2), col + side * (2 + j // 2)), value=int(rng.integers(1, 4)))
            _put(L, occ, "def", (r0 + 1, col + side * 3), type=_defender_type(rng, 0.25, 0.1))
    elif family == "archipelago":
        centers = []
        for _ in range(int(rng.integers(4, 7))):
            for _t in range(30):
                c = (int(rng.integers(2, GRID - 2)), int(rng.integers(2, GRID - 2)))
                if all(abs(c[0] - o[0]) + abs(c[1] - o[1]) >= 5 for o in centers):
                    centers.append(c)
                    break
        for k, (r, c) in enumerate(centers):
            if k == 0:
                _put(L, occ, "core", (r, c))
            else:
                _put(L, occ, "building", (r, c), value=int(rng.integers(1, 4)))
            for j in range(int(rng.integers(0, 3))):
                _put(L, occ, "building", (r + int(rng.integers(-1, 2)), c + int(rng.integers(-1, 2))),
                     value=int(rng.integers(1, 4)))
            for j in range(int(rng.integers(1, 3))):
                _put(L, occ, "def", (r + int(rng.integers(-2, 3)), c + int(rng.integers(-2, 3))),
                     type=_defender_type(rng, 0.1, 0.25))
    elif family == "bunker_nest":
        for lo, hi, gap in ((3, 10, (3, 6)), (5, 8, (8, 6))):
            for k in range(lo, hi + 1):
                for cell in ((lo, k), (hi, k), (k, lo), (k, hi)):
                    if cell != gap:
                        _put(L, occ, "wall", cell)
        _put(L, occ, "core", (6, 6))
        for cell in ((4, 4), (4, 9), (9, 4), (9, 9), (4, 6), (9, 7), (6, 4), (7, 9)):
            _put(L, occ, "def", cell, type=_defender_type(rng, 0.2, 0.1))
        _put(L, occ, "building", (7, 6), value=3)
        _scatter(L, occ, rng, "building", int(rng.integers(4, 7)), (1, GRID - 1, 1, GRID - 1))
    else:  # labyrinth
        for k in range(2, 12):
            for cell in ((2, k), (11, k), (k, 2), (k, 11)):
                if cell != (2, 7):
                    _put(L, occ, "wall", cell)
        for r, gap in ((4, 10), (6, 3), (8, 10)):
            for c in range(3, 11):
                if c != gap:
                    _put(L, occ, "wall", (r, c))
        _put(L, occ, "core", (9, 4))
        for cell in ((3, 9), (5, 5), (7, 8), (9, 7), (10, 5)):
            _put(L, occ, "def", cell, type=_defender_type(rng, 0.3, 0.2))
        for cell in ((3, 4), (5, 8), (7, 4), (10, 9)):
            _put(L, occ, "building", cell, value=int(rng.integers(1, 4)))
    if not L["buildings"]:
        _put(L, occ, "building", (7, 7), value=1)
    return L


from .armies import free_tiles, random_army  # noqa: E402,F401  (moved in B4; kept importable for the B2/B3 fixtures)
