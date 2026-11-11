import math

from siege.combat import BattleResult, simulate_battle
from siege.troops import TROOP_COST, TROOP_TYPES


def _hand_army():
    return [
        {"type": "swarm", "x": 1, "y": 2},
        {"type": "ranger", "x": 1, "y": 4},
        {"type": "tank", "x": 2, "y": 3},
    ]


def test_troop_costs_are_reasonable():
    assert len(TROOP_COST) == len(TROOP_TYPES) == 4
    assert all(c > 0 for c in TROOP_COST)
    assert sum(int(c) for c in TROOP_COST) >= 100
    assert sum(int(c) for c in TROOP_COST) < 200


def test_simulate_battle_is_deterministic():
    walls = {(3, 3), (3, 4), (3, 5)}
    buildings = [
        {"x": 9, "y": 3, "hp": 60, "value": 20},
        {"x": 10, "y": 3, "hp": 80, "value": 30},
    ]
    turrets = [{"x": 8, "y": 7, "hp": 40, "damage": 10, "range": 5}]

    army = _hand_army()
    states = []
    for _ in range(50):
        result = simulate_battle(army, buildings, walls, turrets, max_ticks=24)
        states.append((result.destruction, result.total_value, result.destroyed_value, result.frames[-1]["troops"]))
    assert all(s == states[0] for s in states[1:])


def test_no_negative_hp_or_values_and_bounds():
    result = simulate_battle(_hand_army(), [{"x": 12, "y": 12, "hp": 30, "value": 25}], {(6, 6)}, [], max_ticks=20)
    assert 0.0 <= result.destruction <= 1.0
    assert result.total_value >= 0
    assert result.destroyed_value >= 0
    assert all(t["hp"] >= 0 for t in result.final_troops)
    assert all(b["hp"] >= 0 for b in result.final_buildings)


def test_troop_cannot_walk_through_walls():
    army = [{"type": "swarm", "x": 2, "y": 2}]
    walls = {(2, 3), (3, 2)}
    buildings = []
    turrets = []
    result = simulate_battle(army, buildings, walls, turrets, max_ticks=3)
    for troop in result.final_troops:
        assert (troop["x"], troop["y"]) not in walls
