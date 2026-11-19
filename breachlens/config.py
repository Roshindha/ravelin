"""Single config module for unit / defender / structure stats (docs/SPEC_V2.md 3.1, 3.1b).

All stats live in data/v2/unit_stats.json. This module loads them once, converts hp-like values to
fixed point (scale S, spec 3.3 rule 1) and ranges to integer tenths, and exposes `sim_version()`
(sha256 of the stats file bytes plus SIM_CODE_VERSION) for model / eval identity records.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .constants import UNIT_TYPES

SIM_CODE_VERSION = "breachlens-sim-b5.1-rev1"  # bump on ANY change to simulator behaviour
STATS_PATH = Path(__file__).resolve().parents[1] / "data" / "v2" / "unit_stats.json"
# Exploration hook (B5-R1 sweep only): scripts/b5r1_sweep.py points this at a per-arm stats file in a subprocess. The real gate run and the
# tests assert it is unset (`stats_override_active()`), so no gate result can silently come from an override.
if os.environ.get("BREACHLENS_STATS_PATH"):
    STATS_PATH = Path(os.environ["BREACHLENS_STATS_PATH"])


def stats_override_active() -> bool:
    return bool(os.environ.get("BREACHLENS_STATS_PATH"))

CLASS_BUILDING, CLASS_TURRET, CLASS_WALL = 0, 1, 2
_CLASS_IDS = {"building": CLASS_BUILDING, "turret": CLASS_TURRET, "wall": CLASS_WALL}


@dataclass(frozen=True)
class UnitStats:
    name: str
    idx: int
    cost: int
    hp: int            # fixed point
    damage: int        # fixed point per tick
    armor: int         # fixed point, flat per hit
    range_x10: int
    speed_x10: int
    pref: tuple        # class ids in preference order
    class_rank: tuple  # class_rank[class_id] = position in pref (3 if absent)
    splash_r_x10: int
    splash_pct: int
    heal: int          # fixed point per tick
    heal_range_x10: int
    class_pct: tuple = (100, 100, 100)   # damage percent vs structure class (CLASS_BUILDING incl. the core, CLASS_TURRET, CLASS_WALL); B5-R1b sweep parameter


@dataclass(frozen=True)
class DefenderStats:
    name: str
    idx: int
    hp: int
    damage: int
    damage_neighbour: int
    min_range_x10: int
    max_range_x10: int
    period: int
    area: bool
    splash_r_x10: int


@dataclass(frozen=True)
class Config:
    S: int
    units: tuple       # UnitStats in UNIT_TYPES order
    defenders: tuple   # DefenderStats
    defender_index: dict
    wall_hp: int
    core_hp: int
    building_hp_default: int
    stalemate_ticks: int
    min_hit: int       # fixed point
    unit_cap: int = 20 # max units per army (rules.unit_cap, default 20; B5-R1b sweep parameter)


def _build(raw: dict) -> Config:
    S = int(raw["fixed_point_scale"])
    if tuple(raw["units"]) != UNIT_TYPES:
        raise ValueError(f"unit_stats.json unit order {tuple(raw['units'])} != constants.UNIT_TYPES {UNIT_TYPES}")
    units = []
    for i, name in enumerate(UNIT_TYPES):
        u = raw["units"][name]
        pref = tuple(_CLASS_IDS[c] for c in u["pref"])
        rank = [len(pref)] * 3
        for r, c in enumerate(pref):
            rank[c] = r
        units.append(UnitStats(
            name=name, idx=i, cost=int(u["cost"]), hp=int(u["hp"]) * S, damage=int(u["damage"]) * S,
            armor=int(u["armor"]) * S, range_x10=int(u["range_x10"]), speed_x10=int(u["speed_x10"]),
            pref=pref, class_rank=tuple(rank), splash_r_x10=int(u["splash_r_x10"]),
            splash_pct=int(u["splash_pct"]), heal=int(u["heal"]) * S, heal_range_x10=int(u["heal_range_x10"]),
            class_pct=tuple(int(u.get("class_damage_pct", {}).get(c, 100)) for c in ("building", "turret", "wall"))))
    defenders = []
    for i, (name, d) in enumerate(raw["defenders"].items()):
        defenders.append(DefenderStats(
            name=name, idx=i, hp=int(d["hp"]) * S, damage=int(d["damage"]) * S,
            damage_neighbour=int(d["damage_neighbour"]) * S, min_range_x10=int(d["min_range_x10"]),
            max_range_x10=int(d["max_range_x10"]), period=int(d["period"]), area=bool(d["area"]),
            splash_r_x10=int(d["splash_r_x10"])))
    st, rl = raw["structures"], raw["rules"]
    return Config(S=S, units=tuple(units), defenders=tuple(defenders),
                  defender_index={d.name: d.idx for d in defenders},
                  wall_hp=int(st["wall_hp"]) * S, core_hp=int(st["core_hp"]) * S,
                  building_hp_default=int(st["building_hp_default"]) * S,
                  stalemate_ticks=int(rl["stalemate_ticks"]), min_hit=int(rl["min_hit_hp"]) * S, unit_cap=int(rl.get("unit_cap", 20)))


def load_config(path: Path = STATS_PATH) -> Config:
    return _build(json.loads(Path(path).read_text()))


def sim_version(path: Path = STATS_PATH) -> str:
    """sha256 over the stats file bytes and the simulator code version string."""
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    h.update(b"\x00")
    h.update(SIM_CODE_VERSION.encode())
    return h.hexdigest()


CFG = load_config()
