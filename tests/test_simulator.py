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


def test_target_preference_order_is_respected():
    # A tank prefers wall > building > turret: with an adjacent wall and a
    # farther building both reachable, it must attack the wall it is next to,
    # not skip past it to the building just because buildings are listed first.
    from siege.combat import _select_target, _normalize_buildings, _normalize_turrets, _normalize_walls

    troop = {"x": 4, "y": 3, "range": 1.4, "target_preference": ("wall", "building", "turret")}
    walls = _normalize_walls({(4, 4)})
    buildings = _normalize_buildings([{"x": 5, "y": 5, "hp": 50, "value": 10}])
    turrets = _normalize_turrets([])
    kind, pos, dist = _select_target(troop, buildings, turrets, walls)
    assert kind == "wall" and pos == (4, 4)


def test_tank_breaches_adjacent_wall_before_farther_building():
    army = [{"type": "tank", "x": 4, "y": 3}]
    walls = [{"x": 4, "y": 4, "hp": 24}]  # exactly one tank hit (damage 24) to destroy
    buildings = [{"x": 5, "y": 5, "hp": 50, "value": 10}]
    turrets = []
    result = simulate_battle(army, buildings, walls, turrets, max_ticks=2)
    assert (4, 4) not in result.frames[-1]["walls"] or all(w["x"] != 4 or w["y"] != 4 for w in result.frames[-1]["walls"])
    assert result.destroyed_value == 0  # the building was never touched


# -- STEP 5c: splash area damage, simultaneous turrets -------------------------------------------
def test_splash_hits_neighbours_but_single_target_does_not():
    from siege.troops import get_troop_spec

    sp = get_troop_spec("splash")
    assert sp.splash_radius > 0 and sp.splash_falloff > 0
    layout_b = [
        {"x": 5, "y": 5, "hp": 80, "value": 10},  # struck target
        {"x": 5, "y": 6, "hp": 80, "value": 10},  # adjacent -> splashed
        {"x": 5, "y": 8, "hp": 80, "value": 10},  # 3 away -> untouched
    ]
    splash_hit = simulate_battle([{"type": "splash", "x": 5, "y": 4}], layout_b, {}, [], max_ticks=1)
    hp = {(b["x"], b["y"]): b["hp"] for b in splash_hit.frames[-1]["buildings"]}
    assert hp[(5, 5)] == 80 - 34
    assert hp[(5, 6)] == 80 - int(34 * sp.splash_falloff)
    assert hp[(5, 8)] == 80
    swarm_hit = simulate_battle([{"type": "swarm", "x": 5, "y": 4}], layout_b, {}, [], max_ticks=1)
    assert {(b["x"], b["y"]): b["hp"] for b in swarm_hit.frames[-1]["buildings"]}[(5, 6)] == 80


def test_splash_damages_walls_and_counts_splash_kills_toward_destruction():
    # list-of-dicts form carries explicit hp (a {pos: hp} dict currently falls back to the 80 default)
    walls = [{"x": 5, "y": 5, "hp": 70}, {"x": 5, "y": 6, "hp": 70}]
    r = simulate_battle([{"type": "splash", "x": 5, "y": 4}], [], walls, [], max_ticks=1)
    wall_hp = {(w["x"], w["y"]): w["hp"] for w in r.frames[-1]["walls"]}
    assert wall_hp[(5, 5)] == 70 - 34 and wall_hp[(5, 6)] < 70
    weak = [{"x": 5, "y": 5, "hp": 80, "value": 10}, {"x": 5, "y": 6, "hp": 5, "value": 30}]
    r2 = simulate_battle([{"type": "splash", "x": 5, "y": 4}], weak, {}, [], max_ticks=1)
    assert r2.destroyed_value == 30  # the splashed building died: its value counts


def test_turret_fire_is_simultaneous_so_turret_order_does_not_matter():
    army = [{"type": "swarm", "x": 5, "y": 5}, {"type": "swarm", "x": 5, "y": 6}]
    turrets = [
        {"x": 5, "y": 3, "hp": 99, "damage": 18, "range": 6},
        {"x": 5, "y": 2, "hp": 99, "damage": 18, "range": 6},
    ]
    buildings = [{"x": 15, "y": 15, "hp": 50, "value": 10}]
    a = simulate_battle(army, buildings, {}, turrets, max_ticks=3)
    b = simulate_battle(army, buildings, {}, list(reversed(turrets)), max_ticks=3)
    assert [t["hp"] for t in a.final_troops] == [t["hp"] for t in b.final_troops]


def test_walls_have_80_hp():
    """Walls are 80 hp everywhere: generator constant, generated layouts, and the simulator's battle state."""
    import numpy as np

    from siege.bases import FAMILIES, WALL_HP, generate_base_layout

    assert WALL_HP == 80
    for family in FAMILIES:
        layout = generate_base_layout(np.random.default_rng(3), family=family)
        assert layout["walls"] and set(layout["walls"].values()) == {80}
        r = simulate_battle([], layout["buildings"], layout["walls"], layout["turrets"], max_ticks=1)
        assert {w["hp"] for w in r.frames[0]["walls"]} == {80}
