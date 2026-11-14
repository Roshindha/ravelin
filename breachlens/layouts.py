"""BreachLens (v2) layout generators, structural acceptance rules and the TUNE / LAB-DESIGN design sets.

Spec: docs/SPEC_V2.md 3.1b (defender mix), 3.5 (design sets, streams), 5.1 (families), 11 (B4).

* A layout is a pure function of `(family, seed)`: `candidate(family, seed)` builds one candidate, `check(layout, family)`
  returns the list of acceptance-rule violations. A seed whose candidate violates a rule is skipped (rejected); the next
  seed of the stream is tried. Nothing in this module reads unit stats or runs the simulator, so every rule here is
  stat-independent. The stat-dependent rules of spec 5.1 (too_hard, single-type-trivial) need the Oracle / baselines
  (B8) and are applied when the pool is built (B6); they are NOT part of B4.
* Seed streams T (TUNE), L (LAB-DESIGN) and F (FINAL / pool, reserved, never generated here) are disjoint ranges.
* The design-set builder and loader refuse any seed that lies in stream F.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import deque
from pathlib import Path

import numpy as np

from .constants import GRID

FAMILIES = ("spine_pods", "archipelago", "bunker_nest", "labyrinth")
FAMILIAR = FAMILIES[:3]
LO, HI = 1, GRID - 2          # structures live on the interior ring 1..12; the 52 border cells stay free for deployment

# ---------------------------------------------------------------- seed streams (spec 3.5)
STREAM_STRIDE = 1_000_000     # candidate seeds per family inside a stream
STREAM_BASE = {"T": 100_000_000, "L": 200_000_000, "F": 300_000_000}
PROTOTYPE_SEED_MAX = 25_000_516   # highest seed in the prototype pools (results/layouts_v*.json)
SET_SIZES = {"TUNE-train": 60, "TUNE-val": 20}   # per familiar family; LAB-DESIGN is 40 labyrinth layouts
LAB_DESIGN_SIZE = 40
DESIGN_SETS_PATH = Path(__file__).resolve().parents[1] / "data" / "v2" / "design_sets.json"


def stream_range(stream: str, family: str) -> tuple[int, int]:
    """Half-open seed range [lo, hi) of `family` inside `stream`."""
    lo = STREAM_BASE[stream] + FAMILIES.index(family) * STREAM_STRIDE
    return lo, lo + STREAM_STRIDE


def stream_of(seed: int) -> str | None:
    for s, b in STREAM_BASE.items():
        if b <= seed < b + len(FAMILIES) * STREAM_STRIDE:
            return s
    return None


# ---------------------------------------------------------------- acceptance constants (all structural)
COUNT_RANGES = {   # (min, max) inclusive
    "spine_pods":  {"buildings": (9, 16), "defenders": (6, 12), "walls": (0, 30), "value_total": (14, 36)},
    "archipelago": {"buildings": (5, 12), "defenders": (6, 12), "walls": (0, 12), "value_total": (8, 26)},
    "bunker_nest": {"buildings": (4, 9),  "defenders": (7, 13), "walls": (36, 60), "value_total": (7, 22)},
    "labyrinth":   {"buildings": (3, 6),  "defenders": (4, 8),  "walls": (30, 90), "value_total": (5, 16)},
}
# Generator parameters that revisions may change (spec 3.5 iteration rule). The defaults ARE the current revision's values; with
# lancer_share None and defender_count_mult 1.0 the generators are bit-identical to revision 0 (no extra rng draw).
GEN_PARAMS = {"lancer_share": None,        # None = revision-0 rule (Lancer 0-2 per layout w.p. LANCER_P); a number = target share of defenders that are Lancers
              "defender_count_mult": 1.0,  # multiplies every family's defender count (and the accepted defender range)
              "island_gap": 4,             # B5-R2: archipelago minimum island-center Chebyshev spacing (4 = revision 0, bit-identical)
              "sealed_fraction": 0.0}      # B5-R1b: fraction of layouts whose core is enclosed by a ring of 8 walls (separate rng stream: 0.0 draws nothing)
if os.environ.get("BREACHLENS_GEN_PARAMS"):     # exploration hook (B5-R1 sweep only); the gate run asserts it is unset
    GEN_PARAMS.update(json.loads(os.environ["BREACHLENS_GEN_PARAMS"]))


def gen_override_active() -> bool:
    return bool(os.environ.get("BREACHLENS_GEN_PARAMS"))


def _sc(n: int) -> int:
    """Defender count after `defender_count_mult` (round half up; identity at 1.0)."""
    m = GEN_PARAMS["defender_count_mult"]
    return n if m == 1.0 else int(math.floor(n * m + 0.5))


LANCER_P = {"spine_pods": 0.88, "bunker_nest": 0.88}   # generator probabilities behind the spec's >= 70 % set-level rule
FLAK_P = {"archipelago": 0.80}                          # ... and the >= 50 % rule
MIN_GUN_FRAC = 0.5                                      # familiar families: Gun turrets are at least half of the defenders


# ---------------------------------------------------------------- small geometry helpers
def _tr(k: int, rc):
    """D4 symmetry k (k<4: rotate k times; k>=4: flip columns then rotate k-4 times)."""
    r, c = rc
    if k >= 4:
        c = GRID - 1 - c
    for _ in range(k % 4):
        r, c = c, GRID - 1 - r
    return (r, c)


def transform(L: dict, k: int) -> dict:
    return {"buildings": [{"cell": _tr(k, b["cell"]), "value": b["value"]} for b in L["buildings"]],
            "core": {"cell": _tr(k, L["core"]["cell"])},
            "walls": [_tr(k, w) for w in L["walls"]],
            "defenders": [{"cell": _tr(k, d["cell"]), "type": d["type"]} for d in L["defenders"]]}


class _Builder:
    def __init__(self):
        self.occ: set = set()
        self.L = {"buildings": [], "core": None, "walls": [], "defenders": []}

    def free(self, rc) -> bool:
        return LO <= rc[0] <= HI and LO <= rc[1] <= HI and tuple(rc) not in self.occ

    def _take(self, rc) -> bool:
        rc = (int(rc[0]), int(rc[1]))
        if not self.free(rc):
            return False
        self.occ.add(rc)
        return True

    def building(self, rc, value) -> bool:
        ok = self._take(rc)
        if ok:
            self.L["buildings"].append({"cell": (int(rc[0]), int(rc[1])), "value": int(value)})
        return ok

    def core(self, rc) -> bool:
        ok = self._take(rc)
        if ok:
            self.L["core"] = {"cell": (int(rc[0]), int(rc[1]))}
        return ok

    def wall(self, rc) -> bool:
        ok = self._take(rc)
        if ok:
            self.L["walls"].append((int(rc[0]), int(rc[1])))
        return ok

    def defender(self, rc, typ) -> bool:
        ok = self._take(rc)
        if ok:
            self.L["defenders"].append({"cell": (int(rc[0]), int(rc[1])), "type": typ})
        return ok


def _defender_types(rng, n: int, lancer_p: float, flak_p: float) -> list:
    """Types for n defenders: Gun majority (>= 50 %), >= 1 Lancer w.p. lancer_p, >= 1 Flak w.p. flak_p.
    With GEN_PARAMS['lancer_share'] set, Flaks are drawn as before and the Lancer count is the stochastic rounding of share * n
    (capped so that Guns stay >= ceil(n / 2))."""
    budget = n // 2                 # non-gun slots so that guns >= ceil(n/2)
    lancers = flaks = 0
    share = GEN_PARAMS["lancer_share"]
    if share is None:
        if rng.random() < lancer_p and budget:
            lancers = 1 + int(rng.random() < 0.35 and budget > 1)
        if rng.random() < flak_p and budget - lancers > 0:
            flaks = 1 + int(rng.random() < 0.3 and budget - lancers > 1)
    else:
        if rng.random() < flak_p and budget:
            flaks = 1 + int(rng.random() < 0.3 and budget > 1)
        lancers = min(int(share * n + rng.random()), budget - flaks)
    types = ["lancer"] * lancers + ["flak"] * flaks + ["gun"] * (n - lancers - flaks)
    return [types[i] for i in rng.permutation(n)]


def _value(rng) -> int:
    return int(rng.integers(1, 4))


def _scatter(rng, bld, n, cells, make):
    cells = list(cells)
    placed = 0
    for i in rng.permutation(len(cells)):
        if placed >= n:
            break
        placed += bool(make(cells[int(i)]))
    return placed


# ---------------------------------------------------------------- family generators (build in a canonical frame, then a random D4 frame)
def _spine_pods(rng) -> dict:
    b = _Builder()
    c = int(rng.integers(5, 9))
    r_top, r_bot = 2, int(rng.integers(9, 12))
    b.core((r_bot, c))
    spine_def = [(r, c) for r in range(r_top, r_bot) if (r - r_top) % 2 == 0]
    types = _defender_types(rng, _sc(len(spine_def) + 4), LANCER_P["spine_pods"], 0.12)   # spine + pod guards share one type draw
    ti = 0
    for r in range(r_top, r_bot):
        if (r - r_top) % 2 == 0:
            b.defender((r, c), types[ti]); ti += 1
        else:
            b.wall((r, c))
    n_pods = int(rng.integers(3, 5))
    anchors = [int(round(a)) for a in np.linspace(r_top + 1, r_bot - 4, n_pods)]
    side0 = 1 if rng.random() < 0.5 else -1
    for i, ar in enumerate(anchors):
        s = side0 * (1 if i % 2 == 0 else -1)
        cl = [(ar + di, c + s * (3 + dj)) for di in (0, 1) for dj in (0, 1)]
        nb = int(rng.integers(3, 5))
        for j in rng.permutation(4)[:nb]:
            b.building(cl[int(j)], _value(rng))
        guard = [(ar + 2, c + s * 3), (ar - 1, c + s * 3), (ar + 2, c + s * 4)]
        for g in guard:
            if ti < len(types) and b.defender(g, types[ti]):
                ti += 1
                break
        if rng.random() < 0.5:
            b.wall((ar, c + s))            # a short stub of wall on the branch, never closing it
    while ti < len(types):                 # unplaced guards: more turrets along the spine's flanks
        r = int(rng.integers(r_top, r_bot))
        if b.defender((r, c + int(rng.choice([-1, 1]))), types[ti]):
            ti += 1
        else:
            if rng.random() < 0.2:
                break
    return b.L


def _archipelago(rng) -> dict:
    b = _Builder()
    n_isl = int(rng.integers(4, 7))
    centers = []
    for _ in range(120):
        if len(centers) == n_isl:
            break
        cc = (int(rng.integers(2, HI)), int(rng.integers(2, HI)))
        if all(max(abs(cc[0] - o[0]), abs(cc[1] - o[1])) >= GEN_PARAMS["island_gap"] for o in centers):
            centers.append(cc)
    n_def_per = [int(rng.integers(1, 3)) for _ in centers]
    for k in range(_sc(sum(n_def_per)) - sum(n_def_per)):     # defender_count_mult: extra defenders spread round-robin over the islands (no rng)
        n_def_per[k % len(n_def_per)] += 1
    types = _defender_types(rng, sum(n_def_per), 0.2, FLAK_P["archipelago"])
    ti = 0
    core_i = int(rng.integers(len(centers)))
    for i, (r, c) in enumerate(centers):
        ring = [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
        if i == core_i:
            b.core((r, c))
            _scatter(rng, b, int(rng.integers(0, 3)), ring, lambda x: b.building(x, _value(rng)))
        else:
            b.building((r, c), _value(rng))
            _scatter(rng, b, int(rng.integers(0, 3)), ring, lambda x: b.building(x, _value(rng)))
        out = [(r + dr, c + dc) for dr in range(-2, 3) for dc in range(-2, 3) if max(abs(dr), abs(dc)) == 2]
        for _ in range(n_def_per[i]):
            if ti < len(types) and _scatter(rng, b, 1, out, lambda x, t=types[ti]: b.defender(x, t)):
                ti += 1
        if rng.random() < 0.3:                             # an occasional stub of wall on an island
            _scatter(rng, b, int(rng.integers(1, 4)), out, b.wall)
    return b.L


def _door(side: int, off: int, h: int, cr: int, cc: int, width: int) -> set:
    """Cells of a door of `width` on `side` (0 N,1 E,2 S,3 W) of the ring of half-size h around (cr, cc), at offset `off`."""
    cells = set()
    for w in range(width):
        o = off + w
        cells.add([(cr - h, cc + o), (cr + o, cc + h), (cr + h, cc + o), (cr + o, cc - h)][side])
    return cells


def _bunker_nest(rng) -> dict:
    b = _Builder()
    cr, cc = int(rng.integers(5, 9)), int(rng.integers(5, 9))
    b.core((cr, cc))
    s_in = int(rng.integers(4))
    s_out = int((s_in + rng.integers(1, 4)) % 4)         # outer door never on the inner door's side: the way in winds
    door_in = _door(s_in, int(rng.integers(-1, 1)), 2, cr, cc, int(rng.integers(1, 3)))
    door_out = _door(s_out, int(rng.integers(-2, 2)), 4, cr, cc, int(rng.integers(1, 3)))
    for h, door in ((2, door_in), (4, door_out)):
        for r in range(cr - h, cr + h + 1):
            for c in range(cc - h, cc + h + 1):
                if max(abs(r - cr), abs(c - cc)) == h and (r, c) not in door:
                    b.wall((r, c))
    n_corr = _sc(int(rng.integers(6, 10)))
    n_out = _sc(int(rng.integers(0, 3)))
    types = _defender_types(rng, n_corr + n_out, LANCER_P["bunker_nest"], 0.1)
    corr = [(r, c) for r in range(cr - 3, cr + 4) for c in range(cc - 3, cc + 4) if max(abs(r - cr), abs(c - cc)) == 3]
    _scatter(rng, b, n_corr, corr, lambda x: b.defender(x, types[len(b.L["defenders"])]))
    inner = [(cr + dr, cc + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    _scatter(rng, b, int(rng.integers(1, 3)), inner, lambda x: b.building(x, int(rng.integers(2, 4))))
    outside = [(r, c) for r in range(LO, HI + 1) for c in range(LO, HI + 1) if max(abs(r - cr), abs(c - cc)) >= 6]
    _scatter(rng, b, int(rng.integers(3, 6)), outside, lambda x: b.building(x, _value(rng)))
    n_more = len(types) - len(b.L["defenders"])
    if n_more > 0:
        _scatter(rng, b, n_more, [(r, c) for r, c in outside if max(abs(r - cr), abs(c - cc)) <= 7],
                 lambda x: b.defender(x, types[len(b.L["defenders"])]))
    return b.L


def _labyrinth(rng) -> dict:
    b = _Builder()
    for k in range(LO, HI + 1):                              # perimeter ring on layer 1 (cells 1..12)
        for cell in ((LO, k), (HI, k), (k, LO), (k, HI)):
            b.wall(cell)
    n = int(rng.integers(2, 5))
    for _ in range(100):                                     # corridor widths: n+1 corridors + n lines fill rows 2..11
        w = [int(x) for x in rng.integers(1, 5, size=n + 1)]
        if sum(w) == 10 - n:
            break
    else:
        w = [1] * (n + 1)
        w[0] += 10 - n - sum(w)
    rows, r = [], 2
    for wi in w:
        rows.append((r, r + wi - 1))
        r += wi + 1
    side = int(rng.integers(2))                              # door side of line 0: 0 = left end, 1 = right end
    # entrance: a door in the top ring row
    ent_w = int(rng.integers(1, 3))
    ent_c = int(rng.integers(2, 12 - ent_w))
    for cc_ in range(ent_c, ent_c + ent_w):
        b.L["walls"].remove((LO, cc_)); b.occ.discard((LO, cc_))
    junction = []
    last_door_side = side
    for i in range(n):
        lr = rows[i][1] + 1
        dw = int(rng.integers(1, 3))
        s = (side + i) % 2
        door = set(range(2, 2 + dw)) if s == 0 else set(range(12 - dw, 12))
        for cc_ in range(2, 12):
            if cc_ not in door:
                b.wall((lr, cc_))
        last_door_side = s
        jc = (max(door) + 1) if s == 0 else (min(door) - 1)     # first corridor cell past the door, along the line
        junction.append((lr + 1, jc if s == 0 else jc))
    # core at the far end of the last corridor
    cr0, cr1 = rows[-1]
    core_c = 11 if last_door_side == 0 else 2
    b.core((int(rng.integers(cr0, cr1 + 1)), core_c))
    corr_cells = [(r, c) for (r0, r1) in rows for r in range(r0, r1 + 1) for c in range(2, 12)]
    _scatter(rng, b, int(rng.integers(3, 6)), corr_cells, lambda x: b.building(x, _value(rng)))
    sites = junction + [(2, ent_c + 1 if ent_c + 1 <= 11 else ent_c)]
    n_def = _sc(max(4, len(sites)) + int(rng.integers(0, 2)))
    types = ["gun", "lancer", "flak"]
    if GEN_PARAMS["lancer_share"] is None:
        types += ["gun" if rng.random() < 0.55 else str(rng.choice(["lancer", "flak"])) for _ in range(n_def - 3)]
    else:   # extras: Lancer w.p. `lancer_share`, 10 % Flak, the rest Gun (each layout keeps its mandatory one of each type)
        sh = GEN_PARAMS["lancer_share"]
        types += ["lancer" if u < sh else "flak" if u < sh + 0.1 else "gun" for u in (rng.random() for _ in range(n_def - 3))]
    types = [types[i] for i in rng.permutation(n_def)]
    placed = 0
    for site in sites:
        if placed < n_def and b.defender(site, types[placed]):
            placed += 1
    while placed < n_def:
        if _scatter(rng, b, 1, corr_cells, lambda x, t=types[placed]: b.defender(x, t)):
            placed += 1
        else:
            break
    return b.L


_GEN = {"spine_pods": _spine_pods, "archipelago": _archipelago, "bunker_nest": _bunker_nest, "labyrinth": _labyrinth}


SEAL_TAG = 7373


def core_ring(L: dict) -> list:
    r, c = L["core"]["cell"]
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc)]


def is_sealed(L: dict) -> bool:
    """The core's 8 neighbours are all walls (so it cannot be reached without breaching a wall)."""
    walls = set(map(tuple, L["walls"]))
    return all(x in walls for x in core_ring(L))


def seal_core(L: dict) -> dict:
    """Enclose the core by walls on its 8 neighbours. A building / defender standing on a ring cell is displaced to the nearest free interior
    cell outside the ring (scan order: squared distance, row, column), so all counts are preserved. Pure and deterministic. If a ring cell
    lies outside the interior the layout is returned unchanged."""
    ring = core_ring(L)
    if any(not (LO <= r <= HI and LO <= c <= HI) for r, c in ring):
        return L
    walls = set(map(tuple, L["walls"]))
    blockers = {tuple(b["cell"]) for b in L["buildings"]} | {tuple(d["cell"]) for d in L["defenders"]}
    occ = walls | blockers | {tuple(L["core"]["cell"])}
    ringset = set(ring)
    move, new_walls = {}, []
    for x in ring:
        if x in walls:
            continue
        new_walls.append(x)
        if x in blockers:
            best = min(((r - x[0]) ** 2 + (c - x[1]) ** 2, r, c) for r in range(LO, HI + 1) for c in range(LO, HI + 1)
                       if (r, c) not in occ and (r, c) not in ringset and (r, c) not in move.values())
            move[x] = (best[1], best[2])
    out = {"buildings": [dict(b, cell=move.get(tuple(b["cell"]), tuple(b["cell"]))) for b in L["buildings"]], "core": dict(L["core"]),
           "walls": [tuple(w) for w in L["walls"]] + new_walls,
           "defenders": [dict(d, cell=move.get(tuple(d["cell"]), tuple(d["cell"]))) for d in L["defenders"]]}
    return out


def candidate(family: str, seed: int) -> dict:
    """One candidate layout, a pure function of (family, seed). Not yet checked against the acceptance rules."""
    rng = np.random.default_rng([FAMILIES.index(family), int(seed)])
    L = _GEN[family](rng)
    k = int(rng.integers(8))                                 # random D4 frame for variety (shape rules are D4-invariant)
    L = transform(L, k)
    if L["core"] is None:
        L["core"] = {"cell": (-1, -1)}                       # surfaces as a rule violation, never a crash
    elif GEN_PARAMS["sealed_fraction"] > 0 and np.random.default_rng([FAMILIES.index(family), int(seed), SEAL_TAG]).random() < GEN_PARAMS["sealed_fraction"]:
        L = seal_core(L)
    return L


# ---------------------------------------------------------------- acceptance rules (structural, stat-independent)
def passable_reach(L: dict) -> set:
    """Cells reachable from any non-wall border cell by 8-connected steps with no corner cutting (walls impassable)."""
    walls = set(map(tuple, L["walls"]))
    ok = lambda r, c: 0 <= r < GRID and 0 <= c < GRID and (r, c) not in walls
    seen, dq = set(), deque()
    for r in range(GRID):
        for c in range(GRID):
            if (r in (0, GRID - 1) or c in (0, GRID - 1)) and ok(r, c):
                seen.add((r, c)); dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n = (r + dr, c + dc)
                if (dr or dc) and ok(*n) and n not in seen and (dr == 0 or dc == 0 or (ok(r + dr, c) and ok(r, c + dc))):
                    seen.add(n); dq.append(n)
    return seen


def check(L: dict, family: str) -> list:
    """List of violated acceptance rules (empty = accepted). Every rule is stat-independent."""
    bad = []
    cells = ([tuple(b["cell"]) for b in L["buildings"]] + [tuple(L["core"]["cell"])]
             + [tuple(d["cell"]) for d in L["defenders"]] + [tuple(w) for w in L["walls"]])
    if not L["buildings"] and not L["defenders"]:
        bad.append("empty")
    if not L["buildings"]:
        bad.append("no_value_building")
    if L["core"]["cell"] == (-1, -1):
        return bad + ["no_core"]
    if len(set(cells)) != len(cells):
        bad.append("overlap")
    if any(not (LO <= r <= HI and LO <= c <= HI) for r, c in cells):
        bad.append("out_of_interior")
    rc = dict(COUNT_RANGES[family])
    if GEN_PARAMS["defender_count_mult"] != 1.0:
        lo_d, hi_d = rc["defenders"]
        rc["defenders"] = (_sc(lo_d), int(math.ceil(hi_d * GEN_PARAMS["defender_count_mult"])))
    sealed = GEN_PARAMS["sealed_fraction"] > 0 and is_sealed(L)   # relaxation for sealed layouts only: +8 walls allowed, core need not be reachable
    if sealed:
        rc["walls"] = (rc["walls"][0], rc["walls"][1] + 8)
    n_b, n_d, n_w = len(L["buildings"]), len(L["defenders"]), len(L["walls"])
    vt = sum(b["value"] for b in L["buildings"])
    for name, v in (("buildings", n_b), ("defenders", n_d), ("walls", n_w), ("value_total", vt)):
        if not rc[name][0] <= v <= rc[name][1]:
            bad.append(f"count_{name}")
    if any(b["value"] not in (1, 2, 3) for b in L["buildings"]):
        bad.append("value_weight")
    types = [d["type"] for d in L["defenders"]]
    if family in FAMILIAR and types.count("gun") < MIN_GUN_FRAC * len(types):
        bad.append("gun_majority")
    if family == "labyrinth" and not {"gun", "lancer", "flak"} <= set(types):
        bad.append("labyrinth_defender_types")
    reach = passable_reach(L)
    if tuple(L["core"]["cell"]) not in reach and not sealed:
        bad.append("unreachable_core")
    if any(tuple(b["cell"]) not in reach for b in L["buildings"]):
        bad.append("unreachable_building")
    return bad


def generate(family: str, seed: int) -> dict:
    """The accepted layout for (family, seed); raises if the seed is rejected (callers walk a stream and skip rejects)."""
    L = candidate(family, seed)
    bad = check(L, family)
    if bad:
        raise ValueError(f"({family}, {seed}) rejected: {bad}")
    return L


# ---------------------------------------------------------------- content hash (D4-canonical, so a mirrored copy is the same layout)
def _serialize(L: dict) -> str:
    return json.dumps({"b": sorted((list(b["cell"]), b["value"]) for b in L["buildings"]), "c": list(L["core"]["cell"]),
                       "w": sorted(map(list, L["walls"])), "d": sorted((list(d["cell"]), d["type"]) for d in L["defenders"])},
                      sort_keys=True, separators=(",", ":"))


def content_hash(L: dict) -> str:
    return hashlib.sha256(min(_serialize(transform(L, k)) for k in range(8)).encode()).hexdigest()


# ---------------------------------------------------------------- design sets
def _walk_stream(stream: str, family: str, n: int, taken_hashes: set, rejections: dict):
    lo, hi = stream_range(stream, family)
    out, seed = [], lo
    while len(out) < n:
        if seed >= hi:
            raise RuntimeError("stream exhausted")
        L = candidate(family, seed)
        bad = check(L, family)
        h = content_hash(L) if not bad else None
        if not bad and h in taken_hashes:
            bad = ["duplicate_content"]
        if bad:
            for r in bad:
                rejections[r] = rejections.get(r, 0) + 1
            rejections["_seeds_rejected"] = rejections.get("_seeds_rejected", 0) + 1
        else:
            taken_hashes.add(h)
            out.append({"id": f"{family}-{seed}", "family": family, "seed": seed, "content_hash": h})
        seed += 1
    return out, seed - lo


def build_design_sets() -> dict:
    """TUNE-train (60/family), TUNE-val (20/family) from stream T and LAB-DESIGN (40) from stream L. Never touches F."""
    taken: set = set()
    sets = {"TUNE-train": [], "TUNE-val": [], "LAB-DESIGN": []}
    rej, used = {}, {}
    for fam in FAMILIAR:
        r = rej.setdefault(fam, {})
        recs, n_used = _walk_stream("T", fam, SET_SIZES["TUNE-train"] + SET_SIZES["TUNE-val"], taken, r)
        sets["TUNE-train"] += [dict(x, set="TUNE-train") for x in recs[:SET_SIZES["TUNE-train"]]]
        sets["TUNE-val"] += [dict(x, set="TUNE-val") for x in recs[SET_SIZES["TUNE-train"]:]]
        used[fam] = {"stream": "T", "seed_lo": stream_range("T", fam)[0], "seeds_consumed": n_used}
    r = rej.setdefault("labyrinth", {})
    recs, n_used = _walk_stream("L", "labyrinth", LAB_DESIGN_SIZE, taken, r)
    sets["LAB-DESIGN"] = [dict(x, set="LAB-DESIGN") for x in recs]
    used["labyrinth"] = {"stream": "L", "seed_lo": stream_range("L", "labyrinth")[0], "seeds_consumed": n_used}
    return {
        "streams": {s: {"base": b, "stride_per_family": STREAM_STRIDE,
                        "ranges": {f: list(stream_range(s, f)) for f in FAMILIES}} for s, b in STREAM_BASE.items()},
        "stream_use": used, "rejections": rej,
        "note": ("Stream F is reserved for the pool (B6); no F seed was generated or loaded in B4. Acceptance in B4 is "
                 "structural only (stat-independent); too_hard / single-type-trivial are applied at pool build (B6)."),
        "counts": {k: len(v) for k, v in sets.items()}, "sets": sets,
    }


def load_design_sets_meta(path: Path = DESIGN_SETS_PATH) -> dict:
    return json.loads(Path(path).read_text())


def load_design_set(name: str, family: str | None = None, path: Path = DESIGN_SETS_PATH) -> list:
    """[(record, layout)] for 'TUNE-train' | 'TUNE-val' | 'LAB-DESIGN', regenerated from (family, seed) and hash-verified."""
    out = []
    for rec in load_design_sets_meta(path)["sets"][name]:
        if family and rec["family"] != family:
            continue
        if stream_of(rec["seed"]) not in ("T", "L"):
            raise PermissionError(f"seed {rec['seed']} is not in a design stream (T/L)")
        L = generate(rec["family"], rec["seed"])
        assert content_hash(L) == rec["content_hash"], f"layout content changed for {rec['id']}"
        out.append((rec, L))
    return out
