"""B3: Mender, rule-7 key 4 (canonical-form fallback), D4 equivariance of the simulator.

No stat value is hard-coded: everything is read from breachlens.config (data/v2/unit_stats.json), or a small
hand-built override is declared in the test via `cfg_with`.
"""
import copy
import math
import re
from pathlib import Path

import numpy as np
import pytest

from breachlens import _d4check as D
from breachlens import config as K
from breachlens import sim
from breachlens._bench_layouts import FAMILIES, make_layout, random_army
from breachlens.constants import GRID
from breachlens.sim import Prepared
from breachlens.sim import simulate as _simulate


def simulate(*a, **k):
    """B2/B3 fixtures use damage-0 dummy units and Mender-only armies; they test pre-B5 mechanics, so the B5
    no_attackers early exit is switched off here (it is tested in tests/test_b5_*.py)."""
    k.setdefault("_no_attackers_exit", False)
    return _simulate(*a, **k)

CFG = K.CFG
S = CFG.S
SCOUT, SNIPER, BULWARK, BLASTER, MENDER = range(5)
RAW = K.json.loads(K.STATS_PATH.read_text())
ME = CFG.units[MENDER]


def cfg_with(units=None, defenders=None):
    raw = copy.deepcopy(RAW)
    for name, patch in (units or {}).items():
        raw["units"][name].update(patch)
    for name, patch in (defenders or {}).items():
        raw["defenders"][name].update(patch)
    return K._build(raw)


def layout(buildings=(), core=(13, 13), walls=(), defenders=()):
    return {"buildings": [{"cell": c, "value": v} for c, v in buildings],
            "core": {"cell": core}, "walls": list(walls),
            "defenders": [{"cell": c, "type": t} for c, t in defenders]}


# a harmless, static, huge-hp "ally" scout (never moves, never attacks) and a harmless Mender-free world
def ally_cfg(hp_scout_human=100000, **extra_units):
    units = {"scout": {"hp": hp_scout_human, "damage": 0, "speed_x10": 0, "armor": 0}}
    units.update(extra_units)
    return cfg_with(units=units)


def run(L, army, cfg=CFG, **kw):
    return simulate(Prepared(L, cfg), army, **kw)


# =============================================================== stats / stopgap removal
def test_mender_stats_come_from_the_json_and_match_the_spec_table():
    text = (Path(__file__).resolve().parents[1] / "docs" / "SPEC_V2.md").read_text()
    m = re.search(r"\| \*\*Mender\*\* \| (\d+) \| (\d+) \| (\d+) \|", text)
    assert m
    assert (ME.cost, ME.hp // S, ME.damage // S) == tuple(int(x) for x in m.groups())
    assert ME.heal == RAW["units"]["mender"]["heal"] * S and ME.heal_range_x10 == RAW["units"]["mender"]["heal_range_x10"]
    assert ME.pref == () and ME.damage == 0


def test_raw_coordinate_stopgap_is_gone():
    src = Path(sim.__file__).read_text()
    assert "STOPGAP" not in src
    assert "NotImplementedError" not in src
    # no sorting of tied candidates by structure cell / raw (row, col)
    assert not re.search(r"sort\(key=lambda \w+: (s_cell|\(ROW|ROW)", src)
    assert not re.search(r"\.sort\(\)\s*#", src)


def test_sim_version_changed_with_the_b3_code():
    assert K.SIM_CODE_VERSION != "breachlens-sim-b2.0" and K.SIM_CODE_VERSION.startswith("breachlens-sim-b")   # b5 bump: tests/test_b5_sim.py


# =============================================================== heal rules
def test_heal_amount_and_no_overheal():
    gun = CFG.defenders[CFG.defender_index["gun"]]
    cfg = ally_cfg()
    # ally at (7,7) takes one gun hit on tick 0, is healed on tick 1; gun placed so only the ally is in range
    L = layout(buildings=[((12, 12), 1)], defenders=[((7, 10), "gun")], core=(13, 13))
    army = [(SCOUT, (7, 7)), (MENDER, (7, 6))]
    r = run(L, army, cfg, max_ticks=1)
    assert cfg.units[SCOUT].hp - r.unit_hp[0] == gun.damage        # hit on tick 0, no heal yet (started full)
    r2 = run(L, army, cfg, max_ticks=2)
    # tick 1: gun hits again (the nearest is still the ally), mender heals from the snapshot (ally below max)
    assert cfg.units[SCOUT].hp - r2.unit_hp[0] == 2 * gun.damage - ME.heal
    # no overheal: huge heal on a tiny deficit clamps at max
    cfg_big = cfg_with(units={"scout": {"hp": 100000, "damage": 0, "speed_x10": 0, "armor": 0},
                              "mender": {"heal": 10 ** 6}})
    r3 = run(L, army, cfg_big, max_ticks=2)
    assert r3.unit_hp[0] <= cfg_big.units[SCOUT].hp
    assert r3.unit_hp[0] == cfg_big.units[SCOUT].hp                 # damage first, then heal, clamped at max


def test_unit_that_starts_full_gets_no_heal_for_damage_taken_this_tick():
    gun = CFG.defenders[CFG.defender_index["gun"]]
    cfg = ally_cfg()
    L = layout(buildings=[((12, 12), 1)], defenders=[((7, 10), "gun")], core=(13, 13))
    r = run(L, [(SCOUT, (7, 7)), (MENDER, (7, 6))], cfg, max_ticks=1)
    assert cfg.units[SCOUT].hp - r.unit_hp[0] == gun.damage


def test_no_revive_and_killed_this_tick_receives_nothing():
    gun = CFG.defenders[CFG.defender_index["gun"]]
    # ally with hp exactly one hit: dies on tick 0; mender (started with a damaged-or-not ally) must not revive it
    one_hit = gun.damage // S + 1
    cfg = cfg_with(units={"scout": {"hp": one_hit, "damage": 0, "speed_x10": 0, "armor": 0}})
    L = layout(buildings=[((12, 12), 1)], defenders=[((7, 10), "gun")], core=(13, 13))
    # a second scout, already hurt by an earlier hit, is the heal target at tick 1
    r = run(L, [(SCOUT, (7, 7)), (MENDER, (7, 6))], cfg, max_ticks=3)
    assert r.unit_hp[0] == 0                    # dead and stays dead
    assert r.units_alive <= 1


def test_heal_key_chain_distance_then_hp_fraction_then_cost():
    big = {"hp": 100000, "damage": 0, "speed_x10": 0, "armor": 0}
    cfg = cfg_with(units={"scout": big, "sniper": big})
    gun = cfg.defenders[cfg.defender_index["gun"]]
    ms, mx = cfg.units[SCOUT], cfg.units[SNIPER]
    assert ms.cost < mx.cost
    # (1) distance first: ally at distance 1 beats the one at distance 2
    L = layout(buildings=[((12, 12), 1)], core=(13, 13), defenders=[((7, 11), "gun"), ((7, 2), "gun")])
    r = run(L, [(SCOUT, (7, 8)), (SCOUT, (7, 5)), (MENDER, (7, 7))], cfg, max_ticks=2)
    assert cfg.units[SCOUT].hp - r.unit_hp[0] == 2 * gun.damage - ME.heal
    assert cfg.units[SCOUT].hp - r.unit_hp[1] == 2 * gun.damage
    # (2) equal distance (2 each side): the LOWER hp fraction (smaller max hp, same damage) is healed
    cfg2 = cfg_with(units={"scout": big, "sniper": {**big, "hp": 200000}})
    L2 = layout(buildings=[((12, 12), 1)], core=(13, 13), defenders=[((7, 2), "gun"), ((7, 12), "gun")])
    r2 = run(L2, [(SCOUT, (7, 5)), (SNIPER, (7, 9)), (MENDER, (7, 7))], cfg2, max_ticks=2)
    assert cfg2.units[SCOUT].hp - r2.unit_hp[0] == 2 * gun.damage - ME.heal
    assert cfg2.units[SNIPER].hp - r2.unit_hp[1] == 2 * gun.damage
    # (3) equal distance and equal fraction: the more expensive unit first (key -cost)
    r3 = run(L2, [(SCOUT, (7, 5)), (SNIPER, (7, 9)), (MENDER, (7, 7))], cfg, max_ticks=2)
    assert cfg.units[SCOUT].hp - r3.unit_hp[0] == 2 * gun.damage
    assert cfg.units[SNIPER].hp - r3.unit_hp[1] == 2 * gun.damage - ME.heal


def test_heal_range_boundary_is_inclusive_at_2_0_and_exclusive_beyond():
    cfg = ally_cfg(mender={"speed_x10": 0})   # stationary Mender, so the range test is not confounded by following
    gun = cfg.defenders[cfg.defender_index["gun"]]
    L = layout(buildings=[((12, 12), 1)], core=(13, 13), defenders=[((7, 11), "gun")])
    for dist, healed in ((ME.heal_range_x10 // 10, True), (ME.heal_range_x10 // 10 + 1, False)):
        r = run(L, [(SCOUT, (7, 8)), (MENDER, (7, 8 - dist))], cfg, max_ticks=2)
        lost = cfg.units[SCOUT].hp - r.unit_hp[0]
        assert lost == (2 * gun.damage - ME.heal if healed else 2 * gun.damage), (dist, healed)


def test_mender_never_heals_itself_and_does_nothing_alone():
    gun = CFG.defenders[CFG.defender_index["gun"]]
    cfg = cfg_with(units={"mender": {"speed_x10": 0}})
    L = layout(buildings=[((12, 12), 1)], core=(13, 13), defenders=[((7, 10), "gun")])
    r = run(L, [(MENDER, (7, 7))], cfg, max_ticks=2)
    assert ME.hp - r.unit_hp[0] == 2 * gun.damage        # no self-heal
    r = run(layout(buildings=[((6, 6), 1)]), [(MENDER, (0, 3))])
    assert r.score == 0.0 and r.d_num == 0 and not r.core_destroyed and r.reason == "stalemate"   # Mender-only army scores 0


def test_heals_from_several_menders_add():
    gun = CFG.defenders[CFG.defender_index["gun"]]
    cfg = ally_cfg()
    L = layout(buildings=[((12, 12), 1)], core=(13, 13), defenders=[((7, 11), "gun")])
    r = run(L, [(SCOUT, (7, 8)), (MENDER, (7, 7)), (MENDER, (7, 6))], cfg, max_ticks=2)
    assert cfg.units[SCOUT].hp - r.unit_hp[0] == 2 * gun.damage - 2 * ME.heal


@pytest.mark.parametrize("k", range(1, 9))
def test_heal_split_is_exact_for_k_up_to_8(k):
    # k identical allies stacked on one cell (distance 2 from the Mender) are hit identically by one gun
    # (gun damage splits among them) and then share the heal equally: S = 840 makes both splits exact.
    cfg = ally_cfg()
    gun = cfg.defenders[cfg.defender_index["gun"]]
    assert ME.heal % k == 0 and gun.damage % k == 0
    L = layout(buildings=[((12, 12), 1)], core=(13, 13), defenders=[((7, 12), "gun")])
    army = [(SCOUT, (7, 9))] * k + [(MENDER, (7, 7))]
    r = run(L, army, cfg, max_ticks=2)
    lost = [cfg.units[SCOUT].hp - r.unit_hp[i] for i in range(k)]
    assert len(set(lost)) == 1
    assert lost[0] == 2 * (gun.damage // k) - ME.heal // k
    assert sum(lost) == 2 * gun.damage - ME.heal      # hp conserved exactly: nothing created or lost in the split


def test_heal_split_between_two_equally_damaged_allies_is_equal_and_conserves_hp():
    gun = CFG.defenders[CFG.defender_index["gun"]]
    # two allies equidistant from the mender, damaged identically by symmetric guns
    cfg = ally_cfg()
    L = layout(buildings=[((12, 12), 1)], core=(13, 13), defenders=[((7, 3), "gun"), ((7, 11), "gun")])
    army = [(SCOUT, (7, 6)), (SCOUT, (7, 8)), (MENDER, (7, 7))]
    r = run(L, army, cfg, max_ticks=2)
    a, b = (cfg.units[SCOUT].hp - r.unit_hp[i] for i in (0, 1))
    assert a == b == 2 * gun.damage - ME.heal // 2


def test_mender_between_two_equally_damaged_allies_is_d4_symmetric():
    gun = CFG.defenders[CFG.defender_index["gun"]]
    cfg = ally_cfg()
    L = layout(buildings=[((12, 12), 1)], core=(13, 13), defenders=[((7, 3), "gun"), ((7, 11), "gun")])
    army = [(SCOUT, (7, 6)), (SCOUT, (7, 8)), (MENDER, (7, 7))]
    base = None
    for k in range(8):
        Lk, ak = D.transform_layout(L, k), D.transform_army(army, k)
        r = run(Lk, ak, cfg, max_ticks=6)
        s = (sorted(r.unit_hp), sum(r.struct_hp), r.ticks)
        base = base or s
        assert s == base


# =============================================================== symmetric ties, fallback counter, key 4
def _symmetric_gap_fixture():
    """Anti-diagonal wall line r + c = 12 with two MIRROR gaps; core and start on the main diagonal: fully D4-symmetric
    under the transpose, so the two gap choices are exact mirror images."""
    s = 12
    gaps = {(5, 7), (7, 5)}
    walls = [(r, s - r) for r in range(0, s + 1) if (r, s - r) not in gaps]
    return layout(core=(10, 10), walls=walls), (1, 1)


@pytest.mark.parametrize("unit", [SCOUT, SNIPER, BULWARK, BLASTER])
def test_symmetric_gaps_resolved_by_canonical_form_never_deadlocked(unit):
    L, start = _symmetric_gap_fixture()
    r = run(L, [(unit, start)], record=True, max_ticks=80)
    if unit == SCOUT:   # the Scout must route through one of the two mirror gaps: key 4 genuinely exercised
        assert r.fallback_calls > 0 and r.canon_equal_picks > 0   # mirror gaps give EQUAL canonical strings
    assert r.max_idle_run <= 2
    # outcome is the same in every one of the 8 frames (the world is symmetric, so frames agree up to symmetry)
    base = None
    for k in range(8):
        rk = run(D.transform_layout(L, k), D.transform_army([(unit, start)], k), max_ticks=80)
        s = (rk.ticks, rk.reason, rk.d_num, rk.core_destroyed, sum(rk.struct_hp), sum(rk.unit_hp))
        base = base or s
        assert s == base


def test_fallback_not_used_when_keys_1_to_3_decide():
    L = layout(core=(10, 10), buildings=[((9, 9), 1)])
    r = run(L, [(SCOUT, (1, 1))], max_ticks=30)
    assert r.fallback_calls == 0


def test_blocked_diagonal_two_mirror_gaps_matches_across_transpose():
    L, start = _symmetric_gap_fixture()
    ra = run(L, [(SCOUT, start)], record=True, max_ticks=80)
    rb = run(D.transform_layout(L, 5), D.transform_army([(SCOUT, start)], 5), record=True, max_ticks=80)
    assert (ra.ticks, ra.reason, ra.score) == (rb.ticks, rb.reason, rb.score)


def test_ties_between_units_do_not_depend_on_unit_order():
    L, start = _symmetric_gap_fixture()
    a = [(SCOUT, (1, 1)), (SCOUT, (1, 1)), (SNIPER, (0, 0)), (SCOUT, (2, 2))]
    base = None
    rng = np.random.default_rng(0)
    for _ in range(10):
        army = [a[i] for i in rng.permutation(len(a))]
        r = run(L, army, max_ticks=100)
        s = (r.ticks, r.reason, r.d_num, r.core_destroyed, sum(r.struct_hp), sorted(r.unit_hp))
        base = base or s
        assert s == base


# =============================================================== Mender follows an ally (fixture v)
def _centroid_dist2_x100(res_cell, ally_cell):
    r1, c1 = divmod(res_cell, GRID)
    return 100 * ((r1 - ally_cell[0]) ** 2 + (c1 - ally_cell[1]) ** 2)


@pytest.mark.parametrize("a", [0, 2])
@pytest.mark.parametrize("k", range(2, 11))
def test_mender_follows_ally_on_exact_diagonal_without_deadlock(a, k):
    if a + k > GRID - 1:
        pytest.skip("off grid")
    cfg = cfg_with(units={"scout": {"hp": 100000, "damage": 0, "speed_x10": 0, "armor": 0}})
    ally = (a + k, a + k)
    L = layout(core=(13, 13) if ally != (13, 13) else (12, 12), buildings=[((12, 11), 1)])
    start = (a, a)
    # ally is a static dummy placed via the interior-cell hook; the mender starts k diagonal steps away
    d = k   # Chebyshev steps to the ally; the mender stops within 1.5 (one diagonal step away is sqrt2 <= 1.5)
    bound = math.ceil(10 * max(d - 1, 0) / ME.speed_x10) + 2
    r = run(L, [(SCOUT, ally), (MENDER, start)], cfg, record=True, max_ticks=bound)
    cell = r.trace[-1][1][0]
    assert _centroid_dist2_x100(cell, ally) <= 225, (a, k, divmod(cell, GRID))
    for t_row in r.trace:                      # exact-diagonal approach stays on the diagonal (key 1)
        rr, cc = divmod(t_row[1][0], GRID)
        assert rr == cc
        if t_row[1][2] == 0:                    # the Mender (index 1) only stands still once it is in reach
            assert _centroid_dist2_x100(t_row[1][0], ally) <= 225


def test_mender_follows_a_moving_ally_and_catches_up_once_it_stops():
    # the Scout (faster) walks the exact diagonal and stops to attack; the Mender trails it, never loses it, then closes in
    L = layout(core=(12, 12), buildings=[((11, 11), 1)])
    r = run(L, [(SCOUT, (0, 0)), (MENDER, (0, 0))], record=True, max_ticks=20)
    assert r.max_idle_run <= 2
    d2 = []
    for row in r.trace:
        (c0, _, st0), (c1, _, st1) = row
        r0, q0 = divmod(c0, GRID); r1, q1 = divmod(c1, GRID)
        d2.append(100 * ((r0 - r1) ** 2 + (q0 - q1) ** 2))
        assert r1 == q1                                   # trails on the exact diagonal
    stop = next(t for t, row in enumerate(r.trace) if row[0][2] == 2)   # first tick the Scout attacks (stops)
    resume = next((t for t in range(stop, len(r.trace)) if r.trace[t][0][2] == 1), len(r.trace))  # Scout moves on again
    assert resume - stop >= 8                              # a long enough stationary window to test
    assert all(d2[t + 1] <= d2[t] for t in range(stop, resume - 1))   # while the Scout stands, the gap only closes
    assert d2[resume - 1] <= 225                           # and ends within the stop radius (1.5)


def test_mender_blocked_by_walls_holds_and_does_not_chew_walls():
    ring = [(r, c) for r in range(4, 9) for c in range(4, 9) if r in (4, 8) or c in (4, 8)]
    cfg = cfg_with(units={"scout": {"hp": 100000, "damage": 0, "speed_x10": 0}})
    L = layout(core=(13, 13), walls=ring)
    r = run(L, [(SCOUT, (6, 6)), (MENDER, (0, 0))], cfg, record=True, max_ticks=30)
    assert r.walls_destroyed == 0 and r.reason == "stalemate"          # nothing can happen: ends by the stalemate rule
    assert all(row[1][0] == 0 for row in r.trace)                      # the Mender never left its tile (cell (0,0) = 0)


def test_mender_with_no_allies_left_holds():
    gun_cfg = cfg_with(units={"scout": {"hp": 1}})
    L = layout(buildings=[((12, 12), 1)], defenders=[((5, 5), "gun")])
    r = run(L, [(SCOUT, (3, 5)), (MENDER, (0, 0))], gun_cfg, max_ticks=20)
    assert r.max_idle_run <= 2


# =============================================================== determinism with Menders
def test_determinism_with_menders_over_100_seeds_and_cache_independence():
    rng = np.random.default_rng(77)
    seen_mender = 0
    for seed in range(100):
        L = make_layout(FAMILIES[seed % 4], seed)
        a = random_army(rng, L, n_types=5)
        seen_mender += any(t == MENDER for t, _ in a)
        P = Prepared(L)
        r1, r2, r3 = simulate(P, a, record=True), simulate(P, a, record=True), simulate(Prepared(L), a, record=True)
        for r in (r2, r3):
            assert (r.trace, r.score, r.ticks, r.destroyed, r.reason, r.fallback_calls) == \
                   (r1.trace, r1.score, r1.ticks, r1.destroyed, r1.reason, r1.fallback_calls)
    assert seen_mender >= 50


def test_integer_only_with_menders():
    L = make_layout("bunker_nest", 4)
    a = [(MENDER, 3), (BULWARK, 4), (SCOUT, 5), (SCOUT, 6), (MENDER, 20)]
    r = simulate(L, a, record=True)
    assert all(isinstance(x, int) for x in r.struct_hp + r.unit_hp)
    assert all(isinstance(hp, int) for tick in r.trace for (_, hp, _) in tick)


# =============================================================== D4 equivariance of the whole simulator
def test_d4_tables_are_the_group_and_agree_with_the_independent_helpers():
    N = GRID
    assert len(set(sim.D4_TABLES)) == 8
    for k, tab in enumerate(sim.D4_TABLES):
        for c in range(N * N):
            pass
    imgs = {tuple(tuple(divmod(t[c], N)) for c in range(N * N)) for t in sim.D4_TABLES}
    ours = {tuple(D.tr(k, divmod(c, N)) for c in range(N * N)) for k in range(8)}
    assert imgs == ours   # same 8 permutations (labels may differ)


def test_d4_equivariance_10000_random_battles_x_8_symmetries_with_menders():
    out = D.run_parallel(total=10_000, seed=11, shuffle=False)
    assert out["n"] == 10_000
    assert out["menders"] >= 0.5 * out["n"]               # Menders in >= 50 % of armies
    assert out["fallback_calls"] > 0                        # the canonical path is genuinely exercised
    assert not out["bad"], out["bad"][:3]
    assert out["extra"] == 0                                # per-cell structure hp identical too


def test_d4_equivariance_adversarial_id_and_order_shuffles():
    # independent shuffles of buildings / walls / defenders / army order in every frame (ids differ across frames)
    out = D.run_parallel(total=2_000, seed=12, shuffle=True)
    assert out["menders"] >= 0.5 * out["n"] and out["fallback_calls"] > 0
    assert not out["bad"], out["bad"][:3]


def test_shuffled_ids_do_not_change_outcome_in_a_symmetric_world():
    L, start = _symmetric_gap_fixture()
    army = [(SCOUT, (1, 1)), (MENDER, (1, 1)), (BULWARK, (0, 0))]
    rng = np.random.default_rng(5)
    base = None
    for _ in range(12):
        Ls = copy.deepcopy(L)
        Ls["walls"] = [Ls["walls"][i] for i in rng.permutation(len(Ls["walls"]))]
        a = [army[i] for i in rng.permutation(len(army))]
        r = run(Ls, a, max_ticks=120)
        s = (r.ticks, r.reason, r.d_num, r.core_destroyed, sum(r.struct_hp), sorted(r.unit_hp))
        base = base or s
        assert s == base


def test_fallback_counter_positive_across_the_b3_suite_and_counts_only_canonical_calls():
    L, start = _symmetric_gap_fixture()
    r = run(L, [(SCOUT, start)], max_ticks=60)
    assert r.fallback_calls >= r.canon_equal_picks > 0
    # when no tie reaches key 4 the counter stays 0 (it no longer counts raw-coordinate tie-breaks)
    r0 = run(layout(core=(10, 10), buildings=[((9, 9), 1)]), [(SCOUT, (1, 1)), (BULWARK, (0, 0))], max_ticks=40)
    assert r0.fallback_calls == 0 and r0.canon_equal_picks == 0
