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
