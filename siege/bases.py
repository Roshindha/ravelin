from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Building:
    x: int
    y: int
    hp: int
    value: int
    name: str = "building"

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "hp": self.hp, "value": self.value, "name": self.name}


@dataclass(frozen=True, slots=True)
class Turret:
    x: int
    y: int
    hp: int
    damage: int
    range: int
    name: str = "turret"

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "hp": self.hp, "damage": self.damage, "range": self.range, "name": self.name}


def default_wall_tile(x: int, y: int, hp: int = 80) -> dict:
    return {"x": x, "y": y, "hp": hp, "name": "wall"}


# Tunable balance knobs, shared by all three families. See docs/troops.md for the rationale behind
# the current values -- they were raised from an initial pass that let a rangers-only army clear
# almost every layout (ranged units outrange these turrets and ignore walls entirely when targeting,
# since `combat._select_target` has no line-of-sight check).
BUILDING_HP_RANGE = (50, 91)
BUILDING_VALUE_RANGE = (10, 31)
TURRET_HP_RANGE = (70, 111)
TURRET_DAMAGE_RANGE = (10, 19)
TURRET_RANGE_RANGE = (4, 7)
WALL_HP = 70

N_BUILDINGS_RANGE = (6, 13)  # [6, 12] inclusive
N_TURRETS_RANGE = (2, 6)  # [2, 5] inclusive

FAMILIES = ("ring_defence", "clustered_core", "scattered_outposts")


def _rand_building(rng, x: int, y: int) -> dict:
    return {
        "x": int(x),
        "y": int(y),
        "hp": int(rng.integers(*BUILDING_HP_RANGE)),
        "value": int(rng.integers(*BUILDING_VALUE_RANGE)),
    }


def _rand_turret(rng, x: int, y: int) -> dict:
    return {
        "x": int(x),
        "y": int(y),
        "hp": int(rng.integers(*TURRET_HP_RANGE)),
        "damage": int(rng.integers(*TURRET_DAMAGE_RANGE)),
        "range": int(rng.integers(*TURRET_RANGE_RANGE)),
    }


def _box_ring(lo: int, hi: int) -> dict[tuple[int, int], int]:
    """A hollow square wall ring: the perimeter of [lo, hi] x [lo, hi]."""
    walls: dict[tuple[int, int], int] = {}
    for x in range(lo, hi + 1):
        walls[(x, lo)] = WALL_HP
        walls[(x, hi)] = WALL_HP
    for y in range(lo, hi + 1):
        walls[(lo, y)] = WALL_HP
        walls[(hi, y)] = WALL_HP
    return walls


# -- ring_defence: one walled core, buildings/turrets inside --------------------------------------

CORE_LO, CORE_HI = 5, 10  # interior 6x6 core (inclusive)
RING_LO, RING_HI = 4, 11  # wall ring bounding box one tile outside the core


def generate_ring_defence(rng) -> dict:
    """A single walled core: every building/turret sits behind one continuous wall ring."""
    core_cells = [(x, y) for x in range(CORE_LO, CORE_HI + 1) for y in range(CORE_LO, CORE_HI + 1)]
    order = rng.permutation(len(core_cells))

    n_buildings = int(rng.integers(*N_BUILDINGS_RANGE))
    n_turrets = int(rng.integers(*N_TURRETS_RANGE))

    buildings = [_rand_building(rng, *core_cells[order[i]]) for i in range(n_buildings)]
    turrets = [_rand_turret(rng, *core_cells[order[n_buildings + i]]) for i in range(n_turrets)]

    return {"buildings": buildings, "turrets": turrets, "walls": _box_ring(RING_LO, RING_HI)}


# -- clustered_core: two walled clusters, each with a gate gap -------------------------------------

_CLUSTER_A = (2, 6)  # (lo, hi) bounding box for cluster A's interior cells
_CLUSTER_B = (9, 13)
_CLUSTER_RING_A = (1, 7)
_CLUSTER_RING_B = (8, 14)


def _cluster_cells(lo: int, hi: int) -> list[tuple[int, int]]:
    return [(x, y) for x in range(lo, hi + 1) for y in range(lo, hi + 1)]


def _open_gate(walls: dict[tuple[int, int], int], ring: dict[tuple[int, int], int], rng) -> None:
    """Remove two adjacent ring tiles to punch a gate, mutating `walls` in place."""
    ring_tiles = sorted(ring)
    gate_start = int(rng.integers(0, len(ring_tiles)))
    for offset in (0, 1):
        tile = ring_tiles[(gate_start + offset) % len(ring_tiles)]
        walls.pop(tile, None)


def generate_clustered_core(rng) -> dict:
    """Two separately-walled clusters, each with a punched gate -- splits defenses and budget."""
    n_buildings = int(rng.integers(*N_BUILDINGS_RANGE))
    n_turrets = int(rng.integers(*N_TURRETS_RANGE))
    n_buildings_a = n_buildings // 2
    n_turrets_a = max(1, n_turrets // 2)

    cells_a = _cluster_cells(*_CLUSTER_A)
    order_a = rng.permutation(len(cells_a))
    cells_b = _cluster_cells(*_CLUSTER_B)
    order_b = rng.permutation(len(cells_b))

    buildings = [_rand_building(rng, *cells_a[order_a[i]]) for i in range(n_buildings_a)]
    buildings += [_rand_building(rng, *cells_b[order_b[i]]) for i in range(n_buildings - n_buildings_a)]

    turrets = [_rand_turret(rng, *cells_a[order_a[n_buildings_a + i]]) for i in range(n_turrets_a)]
    turrets += [_rand_turret(rng, *cells_b[order_b[n_buildings - n_buildings_a + i]]) for i in range(n_turrets - n_turrets_a)]

    ring_a = _box_ring(*_CLUSTER_RING_A)
    ring_b = _box_ring(*_CLUSTER_RING_B)
    _open_gate(ring_a, ring_a, rng)
    _open_gate(ring_b, ring_b, rng)
    walls = {**ring_a, **ring_b}

    return {"buildings": buildings, "turrets": turrets, "walls": walls}


# -- scattered_outposts: several small, independently-defended outposts ---------------------------

_OUTPOST_ANCHORS = [(2, 2), (2, 13), (13, 2), (13, 13), (7, 7)]  # 4 corners + center ("the core")


def generate_scattered_outposts(rng) -> dict:
    """Several small outposts spread across the board, each with its own mini wall nook.

    The center anchor (7, 7) is treated as "the core": it always gets a building and, budget
    permitting, the first turret, since it is the outpost every troop type has to pass closest to.
    """
    n_outposts = int(rng.integers(3, 6))
    anchor_order = rng.permutation(len(_OUTPOST_ANCHORS))
    # Always include the center anchor (index 4) among the chosen outposts.
    center_idx = _OUTPOST_ANCHORS.index((7, 7))
    chosen = [center_idx] + [int(i) for i in anchor_order if i != center_idx][: n_outposts - 1]
    anchors = [_OUTPOST_ANCHORS[i] for i in chosen]

    n_buildings = int(rng.integers(*N_BUILDINGS_RANGE))
    n_turrets = int(rng.integers(*N_TURRETS_RANGE))

    buildings: list[dict] = []
    turrets: list[dict] = []
    walls: dict[tuple[int, int], int] = {}
    for i, (ax, ay) in enumerate(anchors):
        # Round-robin the building/turret budget across outposts so each gets at least one building.
        n_b_here = n_buildings // n_outposts + (1 if i < n_buildings % n_outposts else 0)
        n_t_here = n_turrets // n_outposts + (1 if i < n_turrets % n_outposts else 0)
        # A two-tile wall nook guarding the outpost's "back corner" (away from the board center).
        # Computed first and excluded from the building/turret cell pool so a wall never lands on
        # top of a building/turret tile.
        corner_dx = -1 if ax >= 8 else 1
        corner_dy = -1 if ay >= 8 else 1
        wall_cells = {(ax + corner_dx, ay), (ax, ay + corner_dy)}
        local_cells = [(ax + dx, ay + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (ax + dx, ay + dy) not in wall_cells]
        n_b_here = min(n_b_here, len(local_cells))
        n_t_here = min(n_t_here, len(local_cells) - n_b_here)
        order = rng.permutation(len(local_cells))
        for k in range(n_b_here):
            buildings.append(_rand_building(rng, *local_cells[order[k]]))
        for k in range(n_t_here):
            turrets.append(_rand_turret(rng, *local_cells[order[n_b_here + k]]))
        for wx, wy in wall_cells:
            walls[(wx, wy)] = WALL_HP

    return {"buildings": buildings, "turrets": turrets, "walls": walls}


_GENERATORS = {
    "ring_defence": generate_ring_defence,
    "clustered_core": generate_clustered_core,
    "scattered_outposts": generate_scattered_outposts,
}


def generate_base_layout(rng, family: str | None = None) -> dict:
    """Generate a layout from one of `FAMILIES` using `rng` (a numpy Generator).

    Deterministic given `rng`'s state, so callers that seed `rng` get reproducible layouts.
    Picks a random family when `family` is omitted.
    """
    if family is None:
        family = FAMILIES[int(rng.integers(0, len(FAMILIES)))]
    layout = _GENERATORS[family](rng)
    layout["family"] = family
    return layout
