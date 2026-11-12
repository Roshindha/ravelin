"""Troop definitions for Base Siege.

See docs/troops.md for the full balance rationale and iteration history. Summary of the
current (v2) stats:

+----------+-------+------+--------+-------+-------+---------------------+-----------------------------+
| name     | cost  | hp   | damage | speed | range | role                | target preference           |
+==========+=======+======+========+=======+=======+=====================+=============================+
| swarm    | 10    | 18   | 8      | 1.5   | 1.2   | cheap skirmisher    | building > turret > wall    |
| ranger   | 25    | 30   | 12     | 1.0   | 4.0   | ranged attacker     | turret > building > wall    |
| tank     | 30    | 150  | 24     | 0.7   | 1.4   | frontliner          | wall > building > turret    |
| splash   | 40    | 55   | 34     | 0.6   | 2.5   | wall-breaker / nuke | wall > building > turret    |
+----------+-------+------+--------+-------+-------+---------------------+-----------------------------+

The 200 gold budget creates trade-offs: a player can field a dense swarm, a few ranged
units, a tank line, or a mixed rush that includes a splash unit for walls. The budget is
high enough to support multiple unit types while still forcing a difficult choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TroopSpec:
    name: str
    cost: int
    hp: int
    damage: int
    speed: float
    range: float
    role: str
    target_preference: tuple[str, ...]
    # Area damage: every building/turret/wall within `splash_radius` (Euclidean, cells) of the struck
    # tile also takes `int(damage * splash_falloff)`. 0 radius = single-target. Radius-based, so
    # it is D4-invariant.
    splash_radius: float = 0.0
    splash_falloff: float = 0.0

    @property
    def key(self) -> str:
        return self.name.lower().replace(" ", "_")


TROOP_TYPES = (
    TroopSpec("swarm", 10, 18, 8, 1.5, 1.2, "cheap skirmisher", ("building", "turret", "wall")),
    TroopSpec("ranger", 25, 30, 12, 1.0, 4.0, "ranged attacker", ("turret", "building", "wall")),
    TroopSpec("tank", 30, 150, 24, 0.7, 1.4, "frontliner", ("wall", "building", "turret")),
    TroopSpec("splash", 40, 75, 34, 0.6, 2.5, "wall-breaker", ("wall", "building", "turret"), splash_radius=1.5, splash_falloff=0.3),
)

TROOP_TYPE_BY_NAME = {spec.key: spec for spec in TROOP_TYPES}
TROOP_COST = np.array([spec.cost for spec in TROOP_TYPES], dtype=np.int32)


def get_troop_spec(name: str | int) -> TroopSpec:
    if isinstance(name, int):
        return TROOP_TYPES[name]
    key = name.lower().replace(" ", "_")
    return TROOP_TYPE_BY_NAME[key]
