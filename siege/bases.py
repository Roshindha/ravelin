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


CORE_LO, CORE_HI = 5, 10  # interior 6x6 core (inclusive) where buildings/turrets sit
RING_LO, RING_HI = 4, 11  # wall ring bounding box one tile outside the core


def _wall_ring() -> dict[tuple[int, int], int]:
    walls: dict[tuple[int, int], int] = {}
    for x in range(RING_LO, RING_HI + 1):
        walls[(x, RING_LO)] = 80
        walls[(x, RING_HI)] = 80
    for y in range(RING_LO, RING_HI + 1):
        walls[(RING_LO, y)] = 80
        walls[(RING_HI, y)] = 80
    return walls


def generate_base_layout(rng) -> dict:
    """Randomly place buildings/turrets inside a walled core using `rng` (a numpy Generator).

    Deterministic given `rng`'s state, so callers that seed `rng` get reproducible layouts.
    """
    core_cells = [(x, y) for x in range(CORE_LO, CORE_HI + 1) for y in range(CORE_LO, CORE_HI + 1)]
    order = rng.permutation(len(core_cells))

    n_buildings = int(rng.integers(3, 6))
    n_turrets = int(rng.integers(1, 3))

    buildings = []
    for i in range(n_buildings):
        x, y = core_cells[order[i]]
        buildings.append({"x": int(x), "y": int(y), "hp": int(rng.integers(40, 81)), "value": int(rng.integers(10, 31))})

    turrets = []
    for i in range(n_turrets):
        x, y = core_cells[order[n_buildings + i]]
        turrets.append(
            {
                "x": int(x),
                "y": int(y),
                "hp": int(rng.integers(30, 51)),
                "damage": int(rng.integers(8, 16)),
                "range": int(rng.integers(3, 7)),
            }
        )

    return {"buildings": buildings, "turrets": turrets, "walls": _wall_ring()}
