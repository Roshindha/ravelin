"""B5: `no_attackers` early exit (spec 3.3 rule 8), wall counters, sim version bump."""
import numpy as np

from breachlens import config as K
from breachlens import layouts as Y
from breachlens.armies import random_army
from breachlens.sim import Prepared, simulate

CFG = K.CFG
SCOUT, SNIPER, BULWARK, BLASTER, MENDER = range(5)


def layout(buildings=(), core=(13, 13), walls=(), defenders=()):
    return {"buildings": [{"cell": c, "value": v} for c, v in buildings], "core": {"cell": core}, "walls": list(walls),
            "defenders": [{"cell": c, "type": t} for c, t in defenders]}


def test_sim_code_version_bumped_and_changes_sim_version():
    assert K.SIM_CODE_VERSION == "breachlens-sim-b5.1-rev1"
    old = "breachlens-sim-b3.0"
    import hashlib
    h = hashlib.sha256(); h.update(K.STATS_PATH.read_bytes()); h.update(b"\x00"); h.update(old.encode())
    assert K.sim_version() != h.hexdigest()


def test_mender_only_army_ends_immediately_as_no_attackers_not_stalemate():
    r = simulate(layout(buildings=[((6, 6), 1)]), [(MENDER, 3), (MENDER, 20)])
    assert r.reason == "no_attackers" and r.ticks == 0 and r.score == 0.0 and r.d_num == 0 and not r.core_destroyed
    r2 = simulate(layout(buildings=[((6, 6), 1)]), [(MENDER, 3)], _no_attackers_exit=False)
    assert r2.reason == "stalemate" and r2.score == r.score   # old behaviour: same score, different (stalemate) label


def test_exit_fires_when_the_last_attacker_dies_and_only_a_mender_remains():
    # a Scout dies to two Gun hits in front of a gun; the Mender starts far away and never enters the gun's range in time
    L = layout(buildings=[((12, 12), 1)], core=(12, 11), defenders=[((2, 3), "gun")])
    army = [(SCOUT, 3), (MENDER, 26)]
    r = simulate(L, army)
    assert r.reason == "no_attackers" and r.units_alive == 1 and r.ticks < 20
    assert r.unit_hp[0] == 0 and r.unit_hp[1] > 0
    assert simulate(L, army, _no_attackers_exit=False).reason in ("stalemate", "all_dead", "horizon")


def test_all_dead_and_cleared_keep_priority_over_no_attackers():
    L2 = layout(core=(13, 13), buildings=[((12, 12), 1)], defenders=[((2, 3), "gun")] * 4)
    assert simulate(L2, [(SCOUT, 3)]).reason == "all_dead"
    L = layout(buildings=[((1, 3), 1)], core=(1, 4))
    assert simulate(L, [(SNIPER, 3), (SNIPER, 4), (MENDER, 20)] * 3).reason == "cleared"
    assert simulate(layout(buildings=[((6, 6), 1)]), []).reason == "all_dead"


def test_exit_never_changes_score_or_structure_state_on_real_layouts():
    rng = np.random.default_rng(5050)
    pool = [(rec, L, Prepared(L)) for name in ("TUNE-train", "LAB-DESIGN") for rec, L in Y.load_design_set(name)]
    seen = 0
    for _ in range(300):
        rec, L, P = pool[int(rng.integers(len(pool)))]
        army = random_army(rng, L, min_units=1, max_units=14, n_types=5)
        if rng.random() < 0.5:   # many armies end with only Menders left: dilute with fragile Scouts + Menders
            army = [(t, tile) for t, tile in army if t in (SCOUT, MENDER)][:8] or army
        a, b = simulate(P, army), simulate(P, army, _no_attackers_exit=False)
        seen += a.reason == "no_attackers"
        assert (a.score, a.d_num, a.core_destroyed, a.destroyed, a.struct_hp) == (b.score, b.d_num, b.core_destroyed, b.destroyed, b.struct_hp)
        assert a.ticks <= b.ticks
        if a.reason == "no_attackers":
            assert not any(t != MENDER and hp > 0 for (t, _), hp in zip(army, a.unit_hp)) or a.ticks == 0
    assert seen > 0, "the no_attackers path was never exercised"


def test_no_attackers_only_when_no_live_unit_has_damage():
    rng = np.random.default_rng(5051)
    for _ in range(150):
        rec, L = Y.load_design_set("TUNE-train")[int(rng.integers(180))]
        army = random_army(rng, L, min_units=1, max_units=20, n_types=5)
        r = simulate(L, army)
        alive_attacker = any(t != MENDER and hp > 0 for (t, _), hp in zip(army, r.unit_hp))
        if r.reason == "no_attackers":
            assert not alive_attacker or r.ticks == 0
        if alive_attacker and r.reason == "stalemate":
            assert r.reason != "no_attackers"


def test_wall_counters_bulwark_attacks_walls_scout_does_not():
    ring = [(r, c) for r in range(5, 10) for c in range(5, 10) if r in (5, 9) or c in (5, 9)]
    L = layout(buildings=[((7, 7), 1)], core=(7, 8), walls=ring)
    rb = simulate(L, [(BULWARK, 3)] * 1 + [(BULWARK, 4)])
    assert rb.wall_attack_ticks[BULWARK] > 0 and rb.walls_hit[BULWARK] > 0 and rb.wall_attack_any_ticks > 0
    assert rb.attack_ticks[BULWARK] >= rb.wall_attack_ticks[BULWARK] and rb.alive_ticks[BULWARK] >= rb.attack_ticks[BULWARK]
    rs = simulate(layout(buildings=[((1, 3), 1)], core=(1, 4)), [(SCOUT, 3)])
    assert rs.wall_attack_ticks == (0, 0, 0, 0, 0) and rs.walls_hit == (0, 0, 0, 0, 0) and rs.wall_attack_any_ticks == 0
    assert rs.attack_ticks[SCOUT] > 0


def test_counters_do_not_influence_outcomes_determinism():
    rng = np.random.default_rng(5052)
    for _ in range(20):
        rec, L = Y.load_design_set("TUNE-val")[int(rng.integers(60))]
        army = random_army(rng, L, min_units=2, max_units=18, n_types=5)
        a, b = simulate(L, army), simulate(Prepared(L), army)
        assert a == b
