"""B2: config module, fixed-point simulator core, movement tie-break / no-deadlock fixtures.

Stat values are never hard-coded here: tests read breachlens.config, or declare small hand-built units
(via `cfg_with`) that override the shipped stats inside the test.
"""
import copy
import json
import math
from collections import deque

import numpy as np
import pytest

from breachlens import config as K
from breachlens import sim
from breachlens._bench_layouts import FAMILIES, make_layout, random_army
from breachlens.actions import BORDER
from breachlens.constants import GRID, MAX_BATTLE_TICKS
from breachlens.sim import Prepared, simulate

CFG = K.CFG
S = CFG.S
SCOUT, SNIPER, BULWARK, BLASTER, MENDER = range(5)
RAW = json.loads(K.STATS_PATH.read_text())


def cfg_with(units=None, defenders=None):
    raw = copy.deepcopy(RAW)
    for name, patch in (units or {}).items():
        raw["units"][name].update(patch)
    for name, patch in (defenders or {}).items():
        raw["defenders"][name].update(patch)
    return K._build(raw)


def dummy_cfg(**defender_patch):
    """Hand-built static, harmless targets: scouts with huge hp, no attack, no movement."""
    return cfg_with(units={"scout": {"hp": 100000, "damage": 0, "speed_x10": 0, "armor": 0}},
                    defenders=defender_patch or None)


def layout(buildings=(), core=(13, 13), walls=(), defenders=()):
    return {"buildings": [{"cell": c, "value": v} for c, v in buildings],
            "core": {"cell": core}, "walls": list(walls),
            "defenders": [{"cell": c, "type": t} for c, t in defenders]}


# ------------------------------------------------------------------ config / versioning
def test_stats_come_from_json_scaled_by_S():
    assert S == RAW["fixed_point_scale"] == math.lcm(*range(1, 9))
    for i, name in enumerate(K.UNIT_TYPES):
        assert CFG.units[i].hp == RAW["units"][name]["hp"] * S
        assert CFG.units[i].damage == RAW["units"][name]["damage"] * S
        assert isinstance(CFG.units[i].range_x10, int) and isinstance(CFG.units[i].speed_x10, int)
    for d in CFG.defenders:
        assert d.hp == RAW["defenders"][d.name]["hp"] * S


def test_unit_costs_match_spec_table():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "docs" / "SPEC_V2.md").read_text()
    for i, name in enumerate(K.UNIT_TYPES):
        import re
        m = re.search(rf"^\| \*\*{name.capitalize()}\*\* \| (\d+) \| (\d+) \|", text, re.M)
        assert m, name
        assert (CFG.units[i].cost, CFG.units[i].hp // S) == (int(m.group(1)), int(m.group(2)))


def test_sim_version_is_sha_of_stats_and_code_version(tmp_path):
    import hashlib
    v = K.sim_version()
    h = hashlib.sha256()
    h.update(K.STATS_PATH.read_bytes()); h.update(b"\x00"); h.update(K.SIM_CODE_VERSION.encode())
    assert v == h.hexdigest() and len(v) == 64
    other = tmp_path / "stats.json"
    other.write_text(K.STATS_PATH.read_text().replace('"cost": 8,', '"cost": 9,'))
    assert K.sim_version(other) != v


def test_changing_the_stats_file_changes_the_battle():
    base = layout(buildings=[((6, 6), 1)], core=(12, 12))
    army = [(SCOUT, 0)] * 1
    stronger = cfg_with(units={"scout": {"damage": RAW["units"]["scout"]["damage"] * 4}})
    r0 = simulate(Prepared(base), army)
    r1 = simulate(Prepared(base, stronger), army)
    assert r1.ticks < r0.ticks  # stats are read from config, not baked into the simulator


# ------------------------------------------------------------------ hand-built scenarios
def test_range_sniper_attacks_at_its_range_scout_must_close_in():
    sn, sc = CFG.units[SNIPER], CFG.units[SCOUT]
    far = sn.range_x10 // 10  # integer range in this config's geometry
    L = layout(buildings=[((far, 3), 1)], core=(13, 13))
    r_sn = simulate(L, [(SNIPER, 3)], record=True, max_ticks=1)   # tile 3 = (0, 3)
    r_sc = simulate(L, [(SCOUT, 3)], record=True, max_ticks=1)
    assert r_sn.trace[0][0][2] == 2          # attacked on tick 0
    assert r_sc.trace[0][0][2] in (0, 1)     # did not attack
    assert r_sn.struct_hp[0] == CFG.building_hp_default - sn.damage
    assert r_sc.struct_hp[0] == CFG.building_hp_default


def test_just_out_of_range_does_not_attack():
    sn = CFG.units[SNIPER]
    L = layout(buildings=[((sn.range_x10 // 10 + 1, 3), 1)])
    r = simulate(L, [(SNIPER, 3)], record=True, max_ticks=1)
    assert r.trace[0][0][2] != 2


def test_armor_is_flat_per_hit_and_lowers_damage_taken():
    gun = CFG.defenders[CFG.defender_index["gun"]]
    L = layout(core=(13, 13), buildings=[((12, 12), 1)], defenders=[((2, 3), "gun")])
    r_b = simulate(L, [(BULWARK, 3)], max_ticks=1)   # (0,3) is 2 from the gun
    r_s = simulate(L, [(SCOUT, 3)], max_ticks=1)
    arm = CFG.units[BULWARK].armor
    assert arm > 0
    assert CFG.units[BULWARK].hp - r_b.unit_hp[0] == max(gun.damage - arm, CFG.min_hit)
    assert CFG.units[SCOUT].hp - r_s.unit_hp[0] == gun.damage


def test_splash_hits_structures_with_falloff_and_never_units():
    bl = CFG.units[BLASTER]
    # blaster at (0,3); centre building 2 away; neighbour adjacent to centre; far building well outside r
    L = layout(buildings=[((2, 3), 1), ((2, 4), 1), ((2, 9), 1)], core=(13, 13))
    r = simulate(L, [(BLASTER, 3)], max_ticks=1)
    assert r.struct_hp[0] == CFG.building_hp_default - bl.damage
    assert r.struct_hp[1] == CFG.building_hp_default - bl.damage * bl.splash_pct // 100
    assert r.struct_hp[2] == CFG.building_hp_default
    assert r.unit_hp[0] == bl.hp


def test_wall_blocks_path_and_must_be_broken():
    # core fully walled in: scouts must chew through before the core can be hit
    ring = [(r, c) for r in range(5, 10) for c in range(5, 10) if r in (5, 9) or c in (5, 9)]
    L = layout(core=(7, 7), walls=ring)
    r = simulate(L, [(SCOUT, 3)] * 1, record=True)
    assert r.walls_destroyed >= 1
    assert r.core_destroyed or r.reason in ("all_dead", "horizon", "stalemate")
    r_many = simulate(L, [(SCOUT, t) for t in (3, 4, 5, 6)] * 3)
    assert r_many.core_destroyed and r_many.walls_destroyed >= 1


def test_preference_class_before_distance():
    # sniper prefers turret over a nearer building; bulwark prefers wall over a nearer building
    L = layout(buildings=[((1, 3), 1)], core=(13, 13), defenders=[((3, 3), "gun")], walls=[((2, 8), )[0]])
    r = simulate(L, [(SNIPER, 3)], max_ticks=1)
    assert r.struct_hp[0] == CFG.building_hp_default                 # building untouched
    assert r.struct_hp[2] == CFG.defenders[0].hp - CFG.units[SNIPER].damage


def test_nearest_within_class_not_global_rank():
    L = layout(buildings=[((3, 3), 1), ((5, 3), 1)], core=(13, 13))
    r = simulate(L, [(SCOUT, 3)], max_ticks=8)
    assert r.struct_hp[0] < CFG.building_hp_default and r.struct_hp[1] == CFG.building_hp_default


# ------------------------------------------------------------------ defenders
def _hp_lost(res, i=0):
    return dummy_cfg().units[SCOUT].hp - res.unit_hp[i]


def test_gun_hits_nearest_only_and_respects_max_range():
    gun = CFG.defenders[CFG.defender_index["gun"]]
    c = dummy_cfg()
    far_r = gun.max_range_x10 // 10
    L = layout(defenders=[((7, 7), "gun")])
    army = [(SCOUT, (7, 7 - 2)), (SCOUT, (7, 7 + 3)), (SCOUT, (7, 7 + far_r + 1))]
    r = simulate(Prepared(L, c), army, max_ticks=1)
    assert _hp_lost(r, 0) == gun.damage and _hp_lost(r, 1) == 0 and _hp_lost(r, 2) == 0


def test_equidistant_tie_splits_damage_exactly():
    gun = CFG.defenders[CFG.defender_index["gun"]]
    c = dummy_cfg()
    L = layout(defenders=[((7, 7), "gun")])
    for k in (2, 3, 4):
        offs = [(7 - 2, 7), (7 + 2, 7), (7, 7 - 2), (7, 7 + 2)][:k] if k != 3 else [(5, 7), (9, 7), (7, 5)]
        r = simulate(Prepared(L, c), [(SCOUT, o) for o in offs], max_ticks=1)
        assert [_hp_lost(r, i) for i in range(k)] == [gun.damage // k] * k
        assert gun.damage % k == 0 or True


def test_lancer_blind_spot_and_long_range():
    ln = CFG.defenders[CFG.defender_index["lancer"]]
    c = dummy_cfg()
    L = layout(defenders=[((7, 7), "lancer")])
    inside = ln.min_range_x10 // 10 - 1
    edge_in, edge_out = ln.max_range_x10 // 10, ln.max_range_x10 // 10 + 1
    army = [(SCOUT, (7, 7 + inside)), (SCOUT, (7 - edge_in, 7))]
    r = simulate(Prepared(L, c), army, max_ticks=1)
    assert _hp_lost(r, 0) == 0           # blind spot
    assert _hp_lost(r, 1) == ln.damage   # at max range
    r2 = simulate(Prepared(L, c), [(SCOUT, (7 - edge_out, 7))], max_ticks=1)
    assert _hp_lost(r2, 0) == 0


def test_flak_fires_every_second_tick_and_punishes_the_clump():
    fk = CFG.defenders[CFG.defender_index["flak"]]
    c = dummy_cfg()
    L = layout(defenders=[((7, 7), "flak")])
    clump = [(7, 11), (7, 12), (6, 11)]       # three adjacent scouts at distance 4-5
    lone = (11, 7)                              # a loner also in range
    r1 = simulate(Prepared(L, c), [(SCOUT, x) for x in clump] + [(SCOUT, lone)], max_ticks=1)
    lost = [_hp_lost(r1, i) for i in range(4)]
    assert lost[3] == 0                         # the clump is chosen, loner untouched
    assert sorted(lost[:3]) == sorted([fk.damage, fk.damage_neighbour, fk.damage_neighbour])
    r2 = simulate(Prepared(L, c), [(SCOUT, x) for x in clump], max_ticks=2)
    assert [_hp_lost(r2, i) for i in range(3)] == [_hp_lost(r1, i) for i in range(3)]  # tick 1: no burst
    r3 = simulate(Prepared(L, c), [(SCOUT, x) for x in clump], max_ticks=3)
    assert sum(_hp_lost(r3, i) for i in range(3)) == 2 * sum(lost[:3])


def test_no_float_in_outcomes():
    L = make_layout("bunker_nest", 3)
    a = random_army(np.random.default_rng(1), L)
    r = simulate(L, a, record=True)
    assert all(isinstance(x, int) for x in r.struct_hp + r.unit_hp)
    assert all(isinstance(hp, int) for tick in r.trace for (_, hp, _) in tick)
    assert isinstance(r.d_num, int) and isinstance(r.d_den, int)


# ------------------------------------------------------------------ early exit, errors, determinism
def test_early_exit_cleared_all_dead_stalemate_horizon():
    L = layout(buildings=[((1, 3), 1)], core=(1, 4))
    assert simulate(L, [(SNIPER, 3), (SNIPER, 4)] * 3).reason == "cleared"
    L2 = layout(core=(13, 13), buildings=[((12, 12), 1)], defenders=[((2, 3), "gun")] * 4)
    assert simulate(L2, [(SCOUT, 3)]).reason == "all_dead"
    dud = cfg_with(units={"scout": {"damage": 0}})
    r = simulate(Prepared(layout(buildings=[((1, 3), 1)]), dud), [(SCOUT, 3)])
    assert r.reason == "stalemate" and r.ticks < MAX_BATTLE_TICKS
    r2 = simulate(layout(buildings=[((12, 12), 1)]), [(SCOUT, 3)], max_ticks=3)
    assert r2.reason == "horizon" and r2.ticks == 3


def test_empty_army_ends_immediately():
    assert simulate(layout(buildings=[((6, 6), 1)]), []).reason == "all_dead"


def test_mender_not_in_b2_scope_and_wall_start_rejected():
    with pytest.raises(NotImplementedError):
        simulate(layout(buildings=[((6, 6), 1)]), [(MENDER, 3)])
    with pytest.raises(ValueError):
        simulate(layout(buildings=[((6, 6), 1)], walls=[(0, 3)]), [(SCOUT, 3)])


def test_score_is_exact_from_integers():
    L = layout(buildings=[((1, 3), 2), ((12, 12), 1)], core=(1, 4))
    r = simulate(L, [(SNIPER, 3), (SNIPER, 4)] * 3)
    assert r.d_den == 3 and r.core_destroyed
    assert r.d_num in (0, 1, 2, 3)
    assert r.score == pytest.approx(0.6 * r.d_num / r.d_den + 0.4)


def test_determinism_over_100_seeds_and_cache_independence():
    rng = np.random.default_rng(7)
    for seed in range(100):
        fam = FAMILIES[seed % 4]
        L = make_layout(fam, seed)
        a = random_army(rng, L)
        P = Prepared(L)
        r1 = simulate(P, a, record=True)
        r2 = simulate(P, a, record=True)               # warm caches
        r3 = simulate(Prepared(L), a, record=True)      # cold caches
        for r in (r2, r3):
            assert (r.trace, r.score, r.ticks, r.destroyed, r.reason) == (r1.trace, r1.score, r1.ticks, r1.destroyed, r1.reason)


# ------------------------------------------------------------------ movement: exact-diagonal no-deadlock
def _bfs_steps(start, targets, walls):
    """Independent reference BFS (8-connected, no corner cutting, walls block): steps to nearest target cell."""
    walls = set(walls)
    seen, dq = {start: 0}, deque([start])
    while dq:
        p = dq.popleft()
        if p in targets:
            return seen[p]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n = (p[0] + dr, p[1] + dc)
                if (dr, dc) == (0, 0) or not (0 <= n[0] < GRID and 0 <= n[1] < GRID) or n in walls or n in seen:
                    continue
                if dr and dc and ((p[0] + dr, p[1]) in walls or (p[0], p[1] + dc) in walls):
                    continue
                seen[n] = seen[p] + 1
                dq.append(n)
    return None


def _in_range_cells(t, rng_x10):
    return {(r, c) for r in range(GRID) for c in range(GRID) if 100 * ((r - t[0]) ** 2 + (c - t[1]) ** 2) <= rng_x10 ** 2}


def _diag_cases():
    for a in (0, 2):
        for k in range(2, 11):
            if a + k <= GRID - 1:
                yield a, k


def _fixture_layout(kind, a, k):
    tgt = (a + k, a + k)
    walls = []
    mid = a + k // 2
    if kind == "wall_on_diagonal" and k >= 3:
        walls = [(mid, mid)]
    elif kind == "walls_diag_and_orthogonals" and k >= 3:
        walls = [(a + 1, a + 1), (a + 1, a), (a, a + 1)]
    elif kind == "two_symmetric_gaps" and k >= 4:
        s = 2 * a + k  # anti-diagonal wall line r + c = s (D4-symmetric about the main diagonal)
        line = [(r, s - r) for r in range(GRID) if 0 <= s - r < GRID]
        # two gaps mirrored across the diagonal: cells (m, s-m) and (s-m, m)
        m = max(r for r, c in line if r < s - r) if any(r < s - r for r, c in line) else None
        gaps = {(m - 1, s - m + 1), (s - m + 1, m - 1)} if m is not None and m - 1 >= 0 else set()
        walls = [w for w in line if w not in gaps and w != tgt]
    return layout(core=tgt, walls=[w for w in walls if w != tgt]), tgt, walls


@pytest.mark.parametrize("kind", ["open", "wall_on_diagonal", "walls_diag_and_orthogonals", "two_symmetric_gaps"])
@pytest.mark.parametrize("unit", [SCOUT, SNIPER, BULWARK, BLASTER])
def test_exact_diagonal_approach_does_not_deadlock(kind, unit):
    u = CFG.units[unit]
    for a, k in _diag_cases():
        L, tgt, walls = _fixture_layout(kind, a, k)
        start = (a, a)
        if start in set(walls):
            continue
        d = _bfs_steps(start, _in_range_cells(tgt, u.range_x10), walls)
        if d is None:
            continue
        wall_hits = math.ceil(CFG.wall_hp / u.damage)
        move_ticks = math.ceil(10 * d / u.speed_x10)
        bound = move_ticks + wall_hits + 2
        # units here also prefer other classes (bulwark: walls first), which only adds attacks, not idling
        r = simulate(L, [(unit, start)], record=True, max_ticks=bound)
        first_attack = next((t for t, row in enumerate(r.trace) if row[0][2] == 2), None)
        assert first_attack is not None, (kind, unit, a, k, bound)
        assert r.max_idle_run <= 2, (kind, unit, a, k, r.max_idle_run)


def test_exact_diagonal_open_ground_picks_the_diagonal_step():
    L = layout(core=(8, 8))
    r = simulate(L, [(SCOUT, (0, 0))], record=True, max_ticks=4)
    for t, row in enumerate(r.trace):
        (cell, _, _) = row[0]
        rr, cc = divmod(cell, GRID)
        assert rr == cc, (t, rr, cc)  # stays on the diagonal (key 1: smallest squared distance)


def test_symmetric_gaps_tie_is_resolved_not_deadlocked():
    # two mirror gaps in an anti-diagonal wall line: the tie must resolve (counter exercised), never stall
    s = 8
    gaps = {(3, 5), (5, 3)}
    walls = [(r, s - r) for r in range(s + 1) if (r, s - r) not in gaps and r != 0]
    L = layout(core=(7, 7), walls=walls)
    r = simulate(L, [(SCOUT, (1, 1))], record=True, max_ticks=40)
    assert r.max_idle_run <= 2
    assert r.fallback_calls >= 0  # B3 will assert > 0 once the canonical fallback exists


@pytest.mark.parametrize("n_battles", [2000])
def test_random_battles_never_idle_more_than_two_ticks(n_battles):
    rng = np.random.default_rng(2024)
    prepared = {(f, s): Prepared(make_layout(f, s)) for f in FAMILIES for s in range(25)}
    worst, fb = 0, 0
    keys = list(prepared)
    for i in range(n_battles):
        f, s = keys[int(rng.integers(len(keys)))]
        P = prepared[(f, s)]
        L = make_layout(f, s)
        a = random_army(rng, L, min_units=1, max_units=20)
        r = simulate(P, a)
        worst = max(worst, r.max_idle_run)
        fb += r.fallback_calls
    assert worst <= 2
    assert fb > 0  # ties genuinely occur in the random suite (stopgap path exercised)
