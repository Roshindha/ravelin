"""B7: canonical score, BreachEnv (API, masks, determinism, D4, observation channels, shaping), bounded caches, layout access guards.

No test here loads FINAL / transfer layout content: FINAL ids appear only as plain record dicts to check that the guards refuse them.
"""
import copy
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from breachlens import layouts as Y
from breachlens import pool as P
from breachlens import score as SC
from breachlens.actions import BORDER
from breachlens.armies import place_on_border
from breachlens.config import CFG
from breachlens.constants import (DEPLOY_BASE, END_BUY, GRID, N_ACTIONS, N_BORDER, PHASE_BATTLE, PHASE_BUY, PHASE_DEPLOY, WAIT)
from breachlens.env import (BORDER_CELL, MAX_STEPS, BreachEnv, IllegalActionError, d4_action_perm, d4_border_perm, d4_cell_perm,
                            d4_transform_grid, d4_transform_mask)
from breachlens.oracle import oracle_rng, oracle_search, score_key
from breachlens.sim import LRU, Prepared, simulate

ROOT = Path(__file__).resolve().parents[1]
DEV_RECS = P.dev_records("train")                       # DEV-train records (ids only)
TUNE_RECS = P.design_records("TUNE-train", "train")
_LAYOUTS: dict = {}


def layout_of(rec):
    if rec["id"] not in _LAYOUTS:
        _LAYOUTS[rec["id"]] = P.load_layout(rec, "train")           # role-guarded loader; DEV / TUNE only in this module
    return _LAYOUTS[rec["id"]]


def spread(recs, n):
    return [recs[int(i)] for i in np.linspace(0, len(recs) - 1, n)]


def play(env, rng, options=None, seed=0):
    """One masked-random episode; returns (list of actions, list of rewards, final info)."""
    env.reset(seed=seed, options=options)
    acts, rews, info, done = [], [], {}, False
    while not done:
        a = int(rng.choice(np.flatnonzero(env.action_masks())))
        _, r, done, trunc, info = env.step(a)
        assert not trunc
        acts.append(a)
        rews.append(r)
    return acts, rews, info


# ------------------------------------------------------------------ canonical score
def test_score_ratio_on_constructed_outcomes():
    assert SC.score_ratio(0, 10, False) == (0, 100)
    assert SC.score_ratio(10, 10, False) == (60, 100) and SC.score_value(10, 10, False) == 0.6
    assert SC.score_ratio(0, 10, True) == (40, 100) and SC.score_value(0, 10, True) == 0.4
    assert SC.score_value(10, 10, True) == 1.0
    assert SC.score_value(7, 14, True) == float(Fraction(6 * 7 + 4 * 14, 140)) == 0.7
    assert SC.score_value(0, 0, True) == 0.4 and SC.score_value(0, 0, False) == 0.0     # no value buildings: D = 0
    for dn, dd, c in [(1, 3, False), (2, 7, True), (5, 11, True), (9, 13, False)]:
        n, d = SC.score_ratio(dn, dd, c)
        assert SC.score_value(dn, dd, c) == float(Fraction(n, d))                        # one correctly rounded conversion


def test_score_is_monotone_in_both_terms():
    for dd in (3, 7, 13):
        for c in (False, True):
            vals = [SC.score_value(k, dd, c) for k in range(dd + 1)]
            assert vals == sorted(vals) and len(set(vals)) == dd + 1
        assert all(SC.score_value(k, dd, True) > SC.score_value(k, dd, False) for k in range(dd + 1))


def test_no_second_score_formula_outside_score_module():
    pat = re.compile(r"0\.6 ?\*|0\.4 ?\*|\b6 ?\* ?\w*d_num|\b4 ?\* ?den\b|\bd_num ?/ ?\(?10")
    files = [p for p in ROOT.rglob("*.py") if "tests" not in p.relative_to(ROOT).parts and ".venv" not in p.parts
             and p.name != "score.py"]
    assert files
    for p in files:
        assert not pat.search(p.read_text()), f"{p}: score formula outside breachlens/score.py"
    sim_src = (ROOT / "breachlens" / "sim.py").read_text()
    assert "score_value" in sim_src and "result_key" in (ROOT / "breachlens" / "oracle.py").read_text()


def test_env_oracle_and_simulator_scores_are_bit_identical_on_200_layouts():
    """env terminal reward == Oracle score == simulator score, 0 ulp (assert ==), on 100 DEV-train + 100 TUNE-train layouts with a random
    masked-agent army and an Oracle army (small Oracle budget: the property does not depend on how good the army is)."""
    recs = spread(DEV_RECS, 100) + spread(TUNE_RECS, 100)
    env = BreachEnv(records=recs, role="train")
    rng = np.random.default_rng(7)
    for i, rec in enumerate(recs):
        L = layout_of(rec)
        # (1) random army through the env from the BUY phase
        acts, rews, info = play(env, rng, options={"layout_index": i}, seed=i)
        army = [(a - DEPLOY_BASE) // N_BORDER for a in acts if DEPLOY_BASE <= a < END_BUY]
        tiles = [(a - DEPLOY_BASE) % N_BORDER for a in acts if DEPLOY_BASE <= a < END_BUY]
        r = simulate(Prepared(L), list(zip(army, tiles)))
        num, den = SC.result_key(r)
        assert rews[-1] == info["score"] == r.score == SC.result_score(r) == num / den
        assert all(x == 0.0 for x in rews[:-1])
        # (2) Oracle army replayed through the env (prebought counts + the Oracle's deployment order)
        res = oracle_search(L, oracle_rng(rec["seed"]), n_random=6, n_refine=6)
        counts = [res.types.count(t) for t in range(5)]
        env.reset(seed=i, options={"layout_index": i, "prebought": counts})
        for t, tile in res.army:
            _, rew, done, _, info = env.step(DEPLOY_BASE + t * N_BORDER + tile)
        assert done
        key, k_den = score_key(res.result)
        assert rew == info["score"] == res.score == res.result.score == key / k_den == SC.result_score(res.result)
        assert info["d_num"] == res.result.d_num and info["core_destroyed"] == bool(res.breach)


# ------------------------------------------------------------------ Gymnasium API
def test_check_env_passes_on_pool_and_on_synthetic_layouts():
    check_env(BreachEnv(source="DEV-train", role="train"), skip_render_check=True)
    # (with prebought="random" the checker's pre-reset action_space.sample() is drawn from the previous episode's mask, so it can be
    # illegal after the reseeded reset; the default BUY-start game is what check_env covers)
    check_env(BreachEnv(source="TUNE-val", role="validation"), skip_render_check=True)
    check_env(BreachEnv(layouts=[layout_of(DEV_RECS[0])]), skip_render_check=True)


def test_spaces_and_shapes():
    env = BreachEnv(source="DEV-train", role="train")
    o, info = env.reset(seed=0)
    assert env.action_space.n == N_ACTIONS == 267
    assert o["grid"].shape == (9, 14, 14) and o["grid"].dtype == np.float32
    assert o["state"].shape == (11,) and o["state"].dtype == np.float32
    assert env.observation_space.contains(o) and info["layout_id"]
    assert o["grid"].min() >= 0 and o["grid"].max() <= 1
    assert env.action_masks().shape == (267,) and env.action_masks().dtype == bool
    for _ in range(50):
        assert env.action_masks()[env.action_space.sample()]            # sample() draws from the current mask


def test_documented_max_steps_is_reached_but_never_exceeded():
    assert MAX_STEPS == 2 * CFG.unit_cap + 1 == 41
    env = BreachEnv(source="DEV-train", role="train")
    env.reset(seed=1, options={"layout_index": 0})
    for _ in range(20):
        env.step(0)                                                     # 20 Scouts (160 gold)
    assert not env.action_masks()[0] and not env.action_masks()[1:5].any() or env.gold < 8   # cap 20 reached: no buy is legal
    env.step(END_BUY)
    steps = 21
    while True:
        a = int(np.flatnonzero(env.action_masks()[:END_BUY])[0])
        _, _, done, _, _ = env.step(a)
        steps += 1
        if done:
            break
    assert steps == 41 == MAX_STEPS


# ------------------------------------------------------------------ masks
def expected_mask(env):
    """Independent re-statement of spec 4.1 from the env's public state."""
    m = np.zeros(N_ACTIONS, dtype=bool)
    if env.done:
        m[WAIT] = True
        return m
    walls = {tuple(w) for w in env._st.layout["walls"]}
    if env.phase == PHASE_BUY:
        for t in range(5):
            m[t] = CFG.units[t].cost <= env.gold and env.bought < 20
        m[END_BUY] = env.bought >= 1
    elif env.phase == PHASE_DEPLOY:
        for t in range(5):
            for i, rc in enumerate(BORDER):
                m[DEPLOY_BASE + t * N_BORDER + i] = env.owned[t] > 0 and rc not in walls and not env.occupied[i]
        m[WAIT] = True
    return m


def test_mask_fuzz_1000_masked_random_episodes():
    recs = spread(DEV_RECS, 25) + spread(TUNE_RECS, 25)
    env = BreachEnv(records=recs, role="train")
    rng = np.random.default_rng(11)
    bad_tried = 0
    for ep in range(1000):
        pre = "random" if ep % 3 == 0 else None
        env.reset(seed=ep, options={"layout_index": ep % len(recs), "prebought": pre})
        done, steps, gold_before = False, 0, env.gold
        while not done:
            m = env.action_masks()
            assert np.array_equal(m, expected_mask(env))
            assert m.any() and 0 <= env.gold <= 240 and env.bought <= 20 and sum(env.owned) <= env.bought
            ill = np.flatnonzero(~m)
            for a in rng.choice(ill, size=min(3, len(ill)), replace=False):      # every illegal action raises and changes nothing
                before = (env.phase, env.gold, tuple(env.owned), len(env.army))
                with pytest.raises(IllegalActionError):
                    env.step(int(a))
                assert before == (env.phase, env.gold, tuple(env.owned), len(env.army))
                bad_tried += 1
            _, r, done, trunc, info = env.step(int(rng.choice(np.flatnonzero(m))))
            steps += 1
            assert not trunc and steps <= MAX_STEPS
            assert (r == 0.0) or done
        assert info["score"] == r and 0.0 <= r <= 1.0
        with pytest.raises(RuntimeError):
            env.step(WAIT)
    assert bad_tried > 5000


def test_budget_cap_end_buy_and_wait_semantics():
    L = layout_of(DEV_RECS[0])
    env = BreachEnv(layouts=[L])
    env.reset(seed=0)
    m = env.action_masks()
    assert m[:5].all() and not m[END_BUY] and not m[WAIT] and not m[DEPLOY_BASE:END_BUY].any()      # nothing bought: END_BUY illegal
    for _ in range(7):
        env.step(4)                                                                                  # 7 Menders = 196 gold, 44 left
    m = env.action_masks()
    assert env.gold == 240 - 7 * CFG.units[4].cost
    assert list(m[:5]) == [CFG.units[t].cost <= env.gold for t in range(5)]
    with pytest.raises(IllegalActionError):
        env.step(WAIT)                                                                               # WAIT is not legal while buying
    env.step(END_BUY)
    m = env.action_masks()
    assert not m[:5].any() and not m[END_BUY] and m[WAIT]
    assert m[DEPLOY_BASE + 4 * N_BORDER:DEPLOY_BASE + 5 * N_BORDER].any() and not m[DEPLOY_BASE:DEPLOY_BASE + 4 * N_BORDER].any()
    walls = {tuple(w) for w in L["walls"]}
    assert all((not m[DEPLOY_BASE + 4 * N_BORDER + i]) == (rc in walls) for i, rc in enumerate(BORDER))   # wall border tiles are never legal
    env.step(DEPLOY_BASE + 4 * N_BORDER + int(np.flatnonzero(m[DEPLOY_BASE + 4 * N_BORDER:])[0]))
    with pytest.raises(IllegalActionError):
        env.step(DEPLOY_BASE + 0)                                                                    # type with no owned units
    _, r, done, _, info = env.step(WAIT)                                                             # WAIT ends deployment, forfeits 6 Menders
    assert done and info["n_units"] == 1 and r == info["score"]


def test_wait_with_nothing_deployed_is_a_zero_score_battle():
    env = BreachEnv(layouts=[layout_of(DEV_RECS[1])], prebought=[0, 0, 1, 0, 0])
    env.reset(seed=0)
    _, r, done, _, info = env.step(WAIT)
    assert done and r == 0.0 and info["n_units"] == 0


def test_prebought_random_is_always_legal():
    env = BreachEnv(layouts=[layout_of(DEV_RECS[0])], prebought="random")
    seen = np.zeros(5, dtype=int)
    for ep in range(500):
        env.reset(seed=ep)
        assert env.phase == PHASE_DEPLOY and env.bought == sum(env.owned) >= 1
        cost = sum(c * CFG.units[t].cost for t, c in enumerate(env.owned))
        assert cost <= 240 and env.gold == 240 - cost and env.bought <= 20
        assert env.action_masks()[DEPLOY_BASE:END_BUY].any() and not env.action_masks()[:5].any()
        seen += np.array(env.owned) > 0
    assert (seen > 0).all()                                                                         # all five types incl. Menders occur
    with pytest.raises(ValueError):
        BreachEnv(layouts=[layout_of(DEV_RECS[0])], prebought=[10, 10, 10, 10, 10])               # over budget / cap


# ------------------------------------------------------------------ determinism
def test_bit_identical_replay_and_reset_seed_is_honoured():
    recs = spread(DEV_RECS, 10) + spread(TUNE_RECS, 10)
    def run(env, seed, acts=None):
        o, info = env.reset(seed=seed)
        trace = [(o["grid"].tobytes(), o["state"].tobytes(), repr(sorted(info.items())))]
        out, rng, done = [], np.random.default_rng(seed), False
        while not done:
            a = acts[len(out)] if acts is not None else int(rng.choice(np.flatnonzero(env.action_masks())))
            o, r, done, t, info = env.step(a)
            out.append(a)
            trace.append((o["grid"].tobytes(), o["state"].tobytes(), r, done, t, repr(sorted(info.items()))))
        return out, trace
    for pre in (None, "random"):
        for seed in range(30):
            e1, e2 = BreachEnv(records=recs, role="train", prebought=pre), BreachEnv(records=recs, role="tuning", prebought=pre)
            a1, t1 = run(e1, seed)
            a2, t2 = run(e2, seed, acts=a1)                      # same layout draw + same action sequence in a second env instance
            assert a1 == a2 and t1 == t2
            a3, t3 = run(e1, seed, acts=a1)                      # the same instance again after other episodes
            assert t1 == t3
    env = BreachEnv(records=recs, role="train")
    layouts_seen = {env.reset(seed=s)[1]["layout_id"] for s in range(40)}
    assert len(layouts_seen) > 3                                  # different seeds draw different layouts
    assert env.reset(seed=5)[1] == env.reset(seed=5)[1]


# ------------------------------------------------------------------ D4 equivariance of the env
def test_d4_permutations_are_consistent():
    tile_cells = np.array(BORDER_CELL)
    for k in range(8):
        bp, cp, ap = d4_border_perm(k), d4_cell_perm(k), d4_action_perm(k)
        assert sorted(bp) == list(range(52)) and sorted(cp) == list(range(196)) and sorted(ap) == list(range(267))
        assert (cp[tile_cells] == tile_cells[bp]).all()
        assert (ap[:5] == np.arange(5)).all() and ap[END_BUY] == END_BUY and ap[WAIT] == WAIT
    assert (d4_border_perm(0) == np.arange(52)).all()


def test_env_is_d4_equivariant_on_100_layouts_x_8_symmetries():
    """For every symmetry k: transformed layout -> transformed obs grid, permuted mask, identical state vector and identical rewards /
    scores for the transformed action sequence (random armies include Menders; half the episodes buy, half are pre-bought)."""
    recs = spread(DEV_RECS, 60) + spread(TUNE_RECS, 40)
    rng = np.random.default_rng(3)
    checked_steps = 0
    for i, rec in enumerate(recs):
        L = layout_of(rec)
        base = BreachEnv(layouts=[L], prebought="random" if i % 2 else None)
        for k in range(8):
            tr = BreachEnv(layouts=[Y.transform(L, k)], prebought="random" if i % 2 else None)
            ob, _ = base.reset(seed=i)
            ot, _ = tr.reset(seed=i)
            perm = d4_action_perm(k)
            done = False
            while True:
                np.testing.assert_array_equal(ot["grid"], d4_transform_grid(ob["grid"], k))
                np.testing.assert_array_equal(ot["state"], ob["state"])
                np.testing.assert_array_equal(tr.action_masks(), d4_transform_mask(base.action_masks(), k))
                checked_steps += 1
                if done:
                    break
                a = int(rng.choice(np.flatnonzero(base.action_masks())))
                ob, rb, done, _, ib = base.step(a)
                ot, rt, done_t, _, it = tr.step(int(perm[a]))
                assert rb == rt and done == done_t
                if done:
                    assert ib["score"] == it["score"] and ib["d_num"] == it["d_num"] and ib["core_destroyed"] == it["core_destroyed"]
    assert checked_steps > 8 * 100


# ------------------------------------------------------------------ observation channels (hand-built layout)
def hand_layout():
    return {"buildings": [{"cell": (6, 6), "value": 3}, {"cell": (4, 9), "value": 1}],
            "core": {"cell": (8, 8)},
            "walls": [(0, 5), (13, 0), (7, 7)],                                   # two border walls, one interior wall
            "defenders": [{"cell": (2, 2), "type": "gun"}, {"cell": (10, 3), "type": "lancer"}, {"cell": (10, 10), "type": "flak"}]}


def test_observation_channels_on_a_hand_built_layout():
    L = hand_layout()
    env = BreachEnv(layouts=[L])
    o, _ = env.reset(seed=0)
    g, s = o["grid"], o["state"]
    assert g[0].sum() == 0 and g[7].sum() == 0 and g[8].sum() == 0                  # BUY phase: no units, no deploy tiles, no repair reach
    assert g[1, 6, 6] == 1.0 and g[1, 4, 9] == 1.0 and g[1, 8, 8] == 1.0 and (g[1] > 0).sum() == 3     # ch1 buildings + core at full hp
    assert (g[2] > 0).sum() == 3 and g[2, 2, 2] == 1.0 and (g[3] > 0).sum() == 3 and g[3, 7, 7] == 1.0
    assert g[4, 6, 6] == 1.0 and g[4, 4, 9] == pytest.approx(1 / 3) and g[4, 8, 8] == 1.0       # value / 3 (3 -> 1.0), core = 1
    assert g[5].sum() == 1 and g[5, 8, 8] == 1.0
    S = CFG.S
    gun, lancer, flak = CFG.defenders[CFG.defender_index["gun"]], CFG.defenders[CFG.defender_index["lancer"]], CFG.defenders[CFG.defender_index["flak"]]
    thr = lambda d: d.damage / S / d.period / 20.0
    assert g[6, 2, 2] == pytest.approx(thr(gun)) and g[6, 2, 6] == pytest.approx(thr(gun)) and g[6, 2, 7] == 0.0         # gun: range 4 incl. centre
    assert g[6, 10, 3] == 0.0 and g[6, 10, 4] == 0.0 and g[6, 10, 5] > 0 and g[6, 10, 9] > 0 and g[6, 10, 10] == 0.0   # lancer: hollow r < 2, outer 6
    assert g[6, 10, 10 + 2] >= thr(flak) and g[6, 10, 10] == 0.0 and g[6, 10, 11] == 0.0                              # flak: hollow centre r < 2
    assert g[6].max() <= 1.0
    assert s[0] == 1.0 and s[1:6].sum() == 0 and list(s[6:9]) == [1, 0, 0] and s[9] == 0 and s[10] == 0
    s = env.reset(seed=0, options={"prebought": [1, 0, 0, 0, 2]})[0]["state"]
    spent = CFG.units[0].cost + 2 * CFG.units[4].cost
    assert s[0] == pytest.approx((240 - spent) / 240) and s[1] == pytest.approx(1 / 20) and s[5] == pytest.approx(2 / 20)
    assert list(s[6:9]) == [0, 1, 0] and s[9] == pytest.approx(3 / 20)
    g = env._obs()["grid"]
    free = {BORDER_CELL[i] for i, rc in enumerate(BORDER) if rc not in {(0, 5), (13, 0)}}
    assert {c for c in range(196) if g[7].reshape(-1)[c] == 1.0} == free and len(free) == 50       # ch7: border minus the 2 walled tiles


def test_repair_reach_channel_counts_deployed_menders_in_range_two():
    def expected(cells):
        rr, cc = np.mgrid[0:14, 0:14]
        cnt = sum(((rr - r) ** 2 + (cc - c) ** 2 <= 4).astype(int) for r, c in cells)
        return np.minimum(cnt / 3.0, 1.0)
    env = BreachEnv(layouts=[hand_layout()], prebought=[0, 0, 0, 0, 4])
    o, _ = env.reset(seed=0)
    assert o["grid"][8].sum() == 0                                                              # nothing deployed yet
    placed = []
    for rc in ((0, 8), (0, 9)):
        o, *_ = env.step(DEPLOY_BASE + 4 * N_BORDER + BORDER.index(rc))
        placed.append(rc)
    g = o["grid"]
    assert g[0, 0, 8] == 0.25 and g[0, 0, 9] == 0.25 and g[0].sum() == 0.5
    np.testing.assert_allclose(g[8], expected(placed), atol=1e-7)
    assert g[8, 0, 8] == pytest.approx(2 / 3) and g[8, 0, 6] == pytest.approx(1 / 3) and g[8, 0, 5] == 0.0 and g[8, 2, 8] == pytest.approx(1 / 3)
    assert g[8, 2, 7] == 0.0                                                                    # (2, 7) is at distance sqrt(5) > 2 from (0, 8)
    for rc in ((0, 7), (0, 10)):                                                                # four Menders: reach is clipped at 1
        o, *_ = env.step(DEPLOY_BASE + 4 * N_BORDER + BORDER.index(rc))
        placed.append(rc)
    np.testing.assert_allclose(o["grid"][8], expected(placed), atol=1e-7)
    assert o["grid"][8, 0, 8] == 1.0 and o["grid"][8].max() == 1.0
    env.reset(seed=0)
    assert env._obs()["grid"][8].sum() == 0


def test_terminal_observation_shows_the_final_board_and_phase():
    L = layout_of(DEV_RECS[0])
    env = BreachEnv(layouts=[L], prebought="random")
    env.reset(seed=4)
    o0 = env._obs()
    done = False
    rng = np.random.default_rng(0)
    while not done:
        o, r, done, _, info = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
    assert list(o["state"][6:9]) == [0, 0, 1] and o["state"][10] == pytest.approx(info["ticks"] / 240) and o["state"][1:6].sum() == 0
    assert (o["grid"][1] <= o0["grid"][1]).all() and (o["grid"][3] <= o0["grid"][3]).all()          # hp only goes down
    assert o["grid"][7].sum() == 0
    core = L["core"]["cell"]
    assert (o["grid"][5, core[0], core[1]] == 0.0) == bool(info["core_destroyed"])
    assert env.observation_space.contains(o)


# ------------------------------------------------------------------ shaping flag (default off)
def test_shaping_off_is_terminal_only_and_on_sums_to_the_true_score():
    recs = spread(DEV_RECS, 6)
    saw_intermediate = False
    for pre in (None, "random"):
        for i in range(len(recs)):
            off = BreachEnv(records=recs, role="train", prebought=pre)
            on = BreachEnv(records=recs, role="train", prebought=pre, shaping=True)
            assert off.shaping is False and BreachEnv(records=recs, role="train").shaping is False      # default OFF
            acts, r_off, info_off = play(off, np.random.default_rng(i), options={"layout_index": i}, seed=i)
            on.reset(seed=i, options={"layout_index": i})
            outs = [on.step(a) for a in acts]
            r_on = [o[1] for o in outs]
            assert outs[-1][4]["score"] == info_off["score"]                                           # info["score"] is the true score either way
            assert all(x == 0.0 for x in r_off[:-1]) and r_off[-1] == info_off["score"]                # off: terminal only
            assert sum(r_on) == pytest.approx(info_off["score"], abs=1e-12)                            # on: return sums to the true score
            saw_intermediate |= any(x != 0.0 for x in r_on[:-1])
    assert saw_intermediate


def test_shaping_potential_is_zero_in_buy_and_before_first_deployment():
    env = BreachEnv(records=spread(DEV_RECS, 3), role="train", shaping=True)
    env.reset(seed=0, options={"layout_index": 1})
    for a in (0, 1, 2):
        assert env.step(a)[1] == 0.0
    assert env.step(END_BUY)[1] == 0.0
    acts = np.flatnonzero(env.action_masks()[:END_BUY])
    rewards = [env.step(int(acts[0]))[1]]
    assert env._phi == pytest.approx(rewards[0]) and rewards[0] >= 0.0


# ------------------------------------------------------------------ bounded caches
def test_lru_is_bounded_evicts_least_recently_used_and_zero_disables():
    c = LRU(3)
    for k in range(5):
        c[k] = k * 10
    assert len(c) == 3 and c.get(0) is None and c.get(4) == 40 and c.evictions == 2
    c.get(2)
    c[9] = 90
    assert c.get(3) is None and c.get(2) == 20                                    # 3 was least recently used
    z = LRU(0)
    z[1] = 1
    assert len(z) == 0 and z.get(1) is None
    u = LRU(None)
    for k in range(100):
        u[k] = k
    assert len(u) == 100


def test_movement_cache_bounded_and_results_identical_with_cache_on_off_and_tiny():
    from breachlens.armies import random_army
    rng = np.random.default_rng(5)
    for rec in spread(DEV_RECS, 8) + spread(TUNE_RECS, 4):
        L = layout_of(rec)
        armies = [random_army(rng, L, n_types=5) for _ in range(4)]            # includes Menders
        base = Prepared(L)
        variants = {"off": Prepared(L, field_cache=0, small_cache=0), "tiny": Prepared(L, field_cache=2, small_cache=3),
                    "unbounded": Prepared(L, field_cache=None, small_cache=None)}
        for army in armies:
            ref = simulate(base, army)
            for name, p in variants.items():
                r = simulate(p, army)
                assert (r.score, r.ticks, r.reason, r.destroyed, r.struct_hp, r.unit_hp, r.d_num) == \
                       (ref.score, ref.ticks, ref.reason, ref.destroyed, ref.struct_hp, ref.unit_hp, ref.d_num), name
        assert len(variants["tiny"]._fields) <= 2 and len(variants["tiny"]._region) <= 3 and len(variants["off"]._fields) == 0
        assert len(base._fields) <= 2048 and base._fields.hits > 0
    assert variants["tiny"]._fields.evictions > 0 and variants["tiny"]._fields.maxsize == 2


def test_env_layout_cache_is_bounded_and_does_not_change_results():
    recs = spread(DEV_RECS, 12)
    small = BreachEnv(records=recs, role="train", layout_cache=3, field_cache=2, small_cache=3)
    big = BreachEnv(records=recs, role="train")
    off = BreachEnv(records=recs, role="train", layout_cache=0, field_cache=0, small_cache=0)
    for i in range(12):
        outs = []
        for e in (small, big, off):
            rng = np.random.default_rng(i)
            acts, rews, info = play(e, rng, options={"layout_index": i}, seed=i)
            outs.append((acts, rews, info))
            assert len(e._static) <= max(e._layout_cache_size, 0)
        assert outs[0] == outs[1] == outs[2]
    assert len(small._static) == 3 and len(off._static) == 0


# ------------------------------------------------------------------ layout access guards (no FINAL content is ever loaded here)
FINAL_REC = next(r for r in P.json.loads(P.POOL_PATH.read_text())["records"] if r["split"] == "FINAL-familiar")
TRANSFER_REC = next(r for r in P.json.loads(P.POOL_PATH.read_text())["records"] if r["split"] == "FINAL-transfer")


@pytest.mark.parametrize("role", ["train", "validation", "tuning"])
def test_env_raises_on_final_and_transfer_ids_for_dev_side_roles(role):
    ledger = P.LEDGER_PATH.read_bytes()
    for rec in (FINAL_REC, TRANSFER_REC):
        with pytest.raises(PermissionError):
            BreachEnv(records=[rec], role=role)
        with pytest.raises(PermissionError):
            BreachEnv(records=[DEV_RECS[0], rec], role=role)
    for src in ("FINAL-familiar", "FINAL-transfer"):
        with pytest.raises(PermissionError):
            BreachEnv(source=src, role=role)
    assert P.LEDGER_PATH.read_bytes() == ledger


def test_env_role_split_matrix_and_eval_needs_unlock():
    ledger = P.LEDGER_PATH.read_bytes()
    assert BreachEnv(source="DEV-train", role="train").n_layouts == 150
    assert BreachEnv(source="DEV-validation", role="validation").n_layouts == 30
    assert BreachEnv(source="TUNE-train", role="train").n_layouts == 180 and BreachEnv(source="TUNE-val", role="validation").n_layouts == 60
    for role in ("train", "validation", "tuning"):                                       # LAB-DESIGN is design-role only (B7.1)
        with pytest.raises(PermissionError):
            BreachEnv(source="LAB-DESIGN", role=role)
    with pytest.raises(PermissionError):
        BreachEnv(source="DEV-validation", role="train")
    with pytest.raises(PermissionError):
        BreachEnv(source="TUNE-val", role="train")
    with pytest.raises(PermissionError):
        BreachEnv(source="DEV-train", role="eval")
    with pytest.raises(PermissionError):
        BreachEnv(source="FINAL-familiar", role="eval")                                  # no unlock: refused before anything is read or logged
    with pytest.raises(PermissionError):
        BreachEnv(source="FINAL-transfer", role="eval", unlock=False)
    env = BreachEnv(records=[FINAL_REC], role="eval")                                    # ids only: nothing loaded yet
    with pytest.raises(PermissionError):
        env.reset(seed=0)                                                                # content load refused without unlock
    with pytest.raises(ValueError):
        BreachEnv(source="DEV-train", role="build")
    with pytest.raises(ValueError):
        BreachEnv(source="DEV-train", layouts=[hand_layout()])
    assert P.LEDGER_PATH.read_bytes() == ledger and sum(e["purpose"] == "eval" for e in P.read_ledger()) == 238


def test_env_module_never_generates_layouts_itself():
    import ast
    tree = ast.parse((ROOT / "breachlens" / "env.py").read_text())
    names = {c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", "") for c in ast.walk(tree) if isinstance(c, ast.Call)}
    assert not names & {"generate", "candidate"}
