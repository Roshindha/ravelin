"""Helpers for building test armies and placing them on the border, used by balance checks
and acceptance rules -- not by SiegeEnv itself (the env builds its own army from buy/deploy actions).
"""
from __future__ import annotations

import numpy as np

from siege.env import BORDER, N_BORDER
from siege.troops import get_troop_spec

BUDGET = 200

# A few fixed mixed compositions, each spending the full 200-gold budget exactly.
MIXED_ARMIES = {
    "mixed_balanced": {"swarm": 5, "ranger": 2, "tank": 3},  # 50 + 50 + 90 = 190 (+ change, under budget)
    "mixed_breach": {"splash": 2, "tank": 2, "ranger": 2},  # 80 + 60 + 50 = 190
}


def best_affordable_army(troop_name: str, budget: int = BUDGET) -> list[str]:
    """As many of a single troop type as the budget allows."""
    spec = get_troop_spec(troop_name)
    return [spec.name] * (budget // spec.cost)


def mixed_army(composition: dict[str, int]) -> list[str]:
    army: list[str] = []
    for name, count in composition.items():
        army.extend([name] * count)
    return army


def place_on_border(troop_names: list[str], start_idx: int = 0) -> list[dict]:
    """Deploy troops as a single concentrated push: consecutive border tiles starting at
    `start_idx` (wrapping around). This is what a combined-arms mixed army actually needs to
    show any synergy -- spreading every unit to its own isolated gap (one army per access
    point) means units never fight alongside each other regardless of composition.
    """
    n = len(troop_names)
    if n == 0:
        return []
    placed = []
    for i, name in enumerate(troop_names):
        r, c = BORDER[(start_idx + i) % N_BORDER]
        placed.append({"type": name, "x": r, "y": c})
    return placed
