"""D4 augmentation: transforms, action-permutation tables, wrapper, and simulator equivariance."""
from __future__ import annotations

import numpy as np
import pytest

from rl.augment import (
    ACTION_PERM,
    ACTION_PERM_INV,
    D4Wrapper,
    N_SYMMETRIES,
    TILE_PERM,
    inverse_symmetry,
    transform_action,
    transform_army,
    transform_cell,
    transform_grid,
    transform_layout,
    transform_mask,
    transform_obs,
    untransform_action,
)
from siege.bases import generate_base_layout
from siege.combat import simulate_battle
from siege.env import (
    BORDER,
    DEPLOY_BASE,
    END_BUY,
    GRID,
    N_ACTIONS,
    N_BORDER,
    N_TROOPS,
    WAIT,
    SiegeEnv,
    decode_action,
)
from siege.layouts import load_layout_pool
from siege.troops import TROOP_COST, TROOP_TYPES

ALL = range(N_SYMMETRIES)


# -- group structure / transforms -----------------------------------------------------------------
def test_eight_distinct_symmetries_of_the_square():
    pts = [(0, 1), (2, 5)]  # a generic pair of cells distinguishes all 8 elements
    images = {tuple(transform_cell(s, *p) for p in pts) for s in ALL}
    assert len(images) == 8


@pytest.mark.parametrize("s", ALL)
def test_cell_transform_then_inverse_is_identity(s):
    inv = inverse_symmetry(s)
    for r in range(GRID):
        for c in range(GRID):
            assert transform_cell(inv, *transform_cell(s, r, c)) == (r, c)


@pytest.mark.parametrize("s", ALL)
def test_grid_transform_then_inverse_is_identity(s):
    grid = np.random.default_rng(s).random((8, GRID, GRID)).astype(np.float32)
    assert np.array_equal(transform_grid(transform_grid(grid, s), inverse_symmetry(s)), grid)
    # and the other order
    assert np.array_equal(transform_grid(transform_grid(grid, inverse_symmetry(s)), s), grid)


@pytest.mark.parametrize("s", ALL)
def test_grid_transform_moves_each_cell_exactly_as_transform_cell(s):
    grid = np.arange(GRID * GRID, dtype=np.float32).reshape(1, GRID, GRID)
    out = transform_grid(grid, s)
    for r in range(GRID):
        for c in range(GRID):
            nr, nc = transform_cell(s, r, c)
            assert out[0, nr, nc] == grid[0, r, c]


def test_identity_and_rotations_match_numpy():
    grid = np.random.default_rng(0).random((3, GRID, GRID))
    assert np.array_equal(transform_grid(grid, 0), grid)
    # one clockwise quarter turn == np.rot90 with k=-1 on the spatial axes
    assert np.array_equal(transform_grid(grid, 1), np.rot90(grid, k=-1, axes=(-2, -1)))
    # flip-then-no-rotation mirrors columns
    assert np.array_equal(transform_grid(grid, 4), grid[..., ::-1])


def test_state_vector_is_unchanged_and_obs_roundtrips():
    env = SiegeEnv()
    obs, _ = env.reset(seed=0)
    for s in ALL:
        t = transform_obs(obs, s)
        assert np.array_equal(t["state"], obs["state"])
        back = transform_obs(t, inverse_symmetry(s))
        assert np.array_equal(back["grid"], obs["grid"]) and np.array_equal(back["state"], obs["state"])


@pytest.mark.parametrize("s", ALL)
def test_border_mask_channel_is_invariant(s):
    obs, _ = SiegeEnv().reset(seed=1)
    assert np.array_equal(transform_grid(obs["grid"], s)[6], obs["grid"][6])


# -- action permutation table ---------------------------------------------------------------------
@pytest.mark.parametrize("s", ALL)
def test_action_permutation_is_a_bijection_with_correct_inverse(s):
    perm = ACTION_PERM[s]
    assert perm.shape == (N_ACTIONS,)
    assert sorted(perm.tolist()) == list(range(N_ACTIONS))
    assert np.array_equal(ACTION_PERM_INV[s][perm], np.arange(N_ACTIONS))
    assert np.array_equal(perm[ACTION_PERM_INV[s]], np.arange(N_ACTIONS))
    for a in range(N_ACTIONS):
        assert untransform_action(transform_action(a, s), s) == a


@pytest.mark.parametrize("s", ALL)
def test_buy_end_buy_wait_are_fixed_and_deploy_keeps_troop_type(s):
    perm = ACTION_PERM[s]
    for a in list(range(N_TROOPS)) + [END_BUY, WAIT]:
        assert perm[a] == a
    for a in range(DEPLOY_BASE, END_BUY):
        kind, k, t = decode_action(a)
        kind2, k2, t2 = decode_action(int(perm[a]))
        assert (kind, kind2) == ("deploy", "deploy") and k == k2
        # the tile moves exactly as its board cell does
        assert BORDER[t2] == transform_cell(s, *BORDER[t])


@pytest.mark.parametrize("s", ALL)
def test_tile_perm_is_a_bijection_on_the_60_border_tiles(s):
    assert sorted(TILE_PERM[s].tolist()) == list(range(N_BORDER))


def test_identity_symmetry_is_the_identity_permutation():
    assert np.array_equal(ACTION_PERM[0], np.arange(N_ACTIONS))


def test_permutation_tables_form_the_group_d4():
    # the 8 tables are distinct, and closed under composition
    tables = {tuple(ACTION_PERM[s]) for s in ALL}
    assert len(tables) == 8
    for a in ALL:
        for b in ALL:
            composed = tuple(ACTION_PERM[b][ACTION_PERM[a]])
            assert composed in tables


# -- mask legality --------------------------------------------------------------------------------
def _random_legal_actions(env, rng):
    """Yield legal random actions until the episode ends."""
    while True:
        yield int(rng.choice(np.flatnonzero(env.action_masks())))


@pytest.mark.parametrize("s", ALL)
def test_mask_legality_preserved_through_the_whole_episode(s):
    rng = np.random.default_rng(100 + s)
    for ep in range(10):
        w = D4Wrapper(SiegeEnv(), symmetry=s)
        w.reset(seed=ep)
        done, steps = False, 0
        while not done and steps < 300:
            steps += 1
            raw = w.env.unwrapped.action_masks()  # original coordinates
            view = w.action_masks()  # what the agent sees
            assert raw.sum() == view.sum()
            assert np.array_equal(view, transform_mask(raw, s))
            for a in np.flatnonzero(raw):  # legal original -> legal in view
                assert view[transform_action(a, s)]
            for a in np.flatnonzero(view):  # legal in view -> legal original
                assert raw[untransform_action(a, s)]
            _, _, done, _, _ = w.step(int(rng.choice(np.flatnonzero(view))))
        assert done


# -- end-to-end equivariance of the env and the simulator ----------------------------------------
def _hundred_layout_records():
    records = load_layout_pool()
    assert len(records) >= 100
    step = len(records) / 100
    picked = [records[int(i * step)] for i in range(100)]
    assert {r["family"] for r in picked} == {"ring_defence", "clustered_core", "scattered_outposts"}
    return picked


def _random_army_actions(env, rng, mode):
    """A full buy -> deploy action script for `env`'s current layout, in original coordinates."""
    actions = []
    gold = 200
    while True:
        affordable = [k for k in range(N_TROOPS) if TROOP_COST[k] <= gold]
        if not affordable or rng.random() < 0.04:
            break
        k = int(rng.choice(affordable))
        gold -= int(TROOP_COST[k])
        actions.append(k)
    n = len(actions)
    actions.append(END_BUY)
    if mode == "scatter":
        tiles = rng.choice(N_BORDER, size=n, replace=False)
    else:  # concentrated push on consecutive border tiles
        start = int(rng.integers(N_BORDER))
        tiles = [(start + i) % N_BORDER for i in range(n)]
    deploys = [DEPLOY_BASE + k * N_BORDER + int(t) for k, t in zip(actions[:n], tiles)]
    return actions + deploys


def test_simulator_and_env_are_d4_equivariant_on_100_layouts():
    """Same army, transformed layout + mapped actions => identical outcome, for all 8 symmetries.

    Compares (a) the bare simulator on transformed layout/army, and (b) a full SiegeEnv episode on a
    transformed layout stepped with permuted actions (obs must equal the transformed original obs at
    every step), and (c) the D4Wrapper presenting the view over the *untransformed* env.
    """
    rng = np.random.default_rng(7)
    mismatches = []
    for rec in _hundred_layout_records():
        plain = SiegeEnv(layout_pool=[rec])
        obs_plain, _ = plain.reset(seed=0)
        script = _random_army_actions(plain, rng, "scatter" if rec["seed"] % 2 else "push")
        base_layout = plain.layout

        # reference trajectory in original coordinates
        ref_obs, ref_rewards = [obs_plain], []
        for a in script:
            o, r, term, _, _ = plain.step(a)
            ref_obs.append(o)
            ref_rewards.append(r)
        if plain.battle_result is None:  # script ended before deploying everything: force the battle
            o, r, term, _, _ = plain.step(WAIT)
            ref_obs.append(o)
            ref_rewards.append(r)
        ref_dest = plain.battle_result.destruction
        ref_value = plain.battle_result.destroyed_value
        ref_army = list(plain.deployed_troops)

        for s in ALL:
            # (a) bare simulator
            res = simulate_battle(
                transform_army(ref_army, s),
                transform_layout(base_layout, s)["buildings"],
                transform_layout(base_layout, s)["walls"],
                transform_layout(base_layout, s)["turrets"],
                max_ticks=50,
            )
            if (res.destroyed_value, res.destruction) != (ref_value, ref_dest):
                mismatches.append((rec["id"], s, "simulator", res.destruction, ref_dest))

            # (b) env on a transformed layout, driven by permuted actions
            env_t = SiegeEnv(layout_pool=[rec])
            o_t, _ = env_t.reset(seed=0)
            env_t.layout = transform_layout(base_layout, s)
            o_t = env_t._obs()
            assert np.array_equal(o_t["grid"], transform_obs(ref_obs[0], s)["grid"]), (rec["id"], s, "reset obs")
            steps = list(script) + ([WAIT] if len(ref_obs) == len(script) + 2 else [])
            for i, a in enumerate(steps):
                o_t, r_t, _, _, _ = env_t.step(transform_action(a, s))
                exp = transform_obs(ref_obs[i + 1], s)
                if not (np.array_equal(o_t["grid"], exp["grid"]) and np.array_equal(o_t["state"], exp["state"])):
                    mismatches.append((rec["id"], s, f"obs@{i}"))
                    break
                if r_t != ref_rewards[i]:
                    mismatches.append((rec["id"], s, f"reward@{i}", r_t, ref_rewards[i]))
                    break

            # (c) wrapper over the untransformed env: agent acts in the view
            w = D4Wrapper(SiegeEnv(layout_pool=[rec]), symmetry=s)
            o_w, _ = w.reset(seed=0)
            assert np.array_equal(o_w["grid"], transform_obs(ref_obs[0], s)["grid"])
            for i, a in enumerate(steps):
                assert w.action_masks()[transform_action(a, s)], (rec["id"], s, i)
                o_w, r_w, _, _, _ = w.step(transform_action(a, s))
                exp = transform_obs(ref_obs[i + 1], s)
                assert np.array_equal(o_w["grid"], exp["grid"]) and np.array_equal(o_w["state"], exp["state"])
                assert r_w == ref_rewards[i]
            assert w.env.unwrapped.battle_result.destruction == ref_dest
    assert not mismatches, f"{len(mismatches)} equivariance failures, first few: {mismatches[:5]}"


def test_simulator_equivariant_on_fresh_random_layouts_all_families():
    """Beyond the frozen pool: 300 fresh layouts x 8 symmetries x varied armies, exact agreement."""
    from siege.armies import place_on_border
    from siege.bases import FAMILIES

    rng = np.random.default_rng(11)
    bad = 0
    for i in range(300):
        layout = generate_base_layout(np.random.default_rng(5000 + i), family=FAMILIES[i % 3])
        names, gold = [], 200
        while True:
            ok = [k for k in range(N_TROOPS) if TROOP_COST[k] <= gold]
            if not ok or rng.random() < 0.03:
                break
            k = int(rng.choice(ok))
            gold -= int(TROOP_COST[k])
            names.append(TROOP_TYPES[k].name)
        army = place_on_border(names, int(rng.integers(N_BORDER)))
        base = simulate_battle(army, layout["buildings"], layout["walls"], layout["turrets"], max_ticks=50)
        for s in ALL:
            tl = transform_layout(layout, s)
            r = simulate_battle(transform_army(army, s), tl["buildings"], tl["walls"], tl["turrets"], max_ticks=50)
            bad += (r.destroyed_value, r.destruction) != (base.destroyed_value, base.destruction)
    assert bad == 0


# -- wrapper behaviour ----------------------------------------------------------------------------
def test_wrapper_samples_all_eight_symmetries_and_is_seed_reproducible():
    w = D4Wrapper(SiegeEnv(), seed=0)
    seen = {w.reset()[1]["symmetry"] for _ in range(200)}
    assert seen == set(ALL)
    a = D4Wrapper(SiegeEnv(), seed=3)
    b = D4Wrapper(SiegeEnv(), seed=3)
    assert [a.reset()[1]["symmetry"] for _ in range(10)] == [b.reset()[1]["symmetry"] for _ in range(10)]
    assert D4Wrapper(SiegeEnv()).reset(seed=5)[1]["symmetry"] == D4Wrapper(SiegeEnv()).reset(seed=5)[1]["symmetry"]


def test_wrapper_works_with_maskable_ppo_helpers():
    from sb3_contrib.common.maskable.utils import get_action_masks

    w = D4Wrapper(SiegeEnv(), symmetry=5)
    w.reset(seed=0)
    assert np.array_equal(get_action_masks(w), w.action_masks())
    assert w.action_space.n == N_ACTIONS


def test_simulator_equivariance_holds_even_when_object_ids_are_shuffled():
    """Stronger than label-preserving equivariance: relabel buildings/turrets/walls arbitrarily.

    Since STEP 5c (simultaneous turret fire + invariant tie-breaks) this is exact; before it, 4/7000
    comparisons differed (wall-id ties) and turret order mattered when two turrets chose one troop.
    """
    from siege.armies import place_on_border
    from siege.bases import FAMILIES

    rng = np.random.default_rng(21)
    bad = 0
    for i in range(150):
        layout = generate_base_layout(np.random.default_rng(8000 + i), family=FAMILIES[i % 3])
        names, gold = [], 200
        while True:
            ok = [k for k in range(N_TROOPS) if TROOP_COST[k] <= gold]
            if not ok or rng.random() < 0.03:
                break
            k = int(rng.choice(ok))
            gold -= int(TROOP_COST[k])
            names.append(TROOP_TYPES[k].name)
        army = place_on_border(names, int(rng.integers(N_BORDER)))
        base = simulate_battle(army, layout["buildings"], layout["walls"], layout["turrets"], max_ticks=50)
        for s in ALL:
            tl = transform_layout(layout, s)
            b = [tl["buildings"][j] for j in rng.permutation(len(tl["buildings"]))]
            t = [tl["turrets"][j] for j in rng.permutation(len(tl["turrets"]))]
            items = list(tl["walls"].items())
            w = dict(items[j] for j in rng.permutation(len(items)))
            r = simulate_battle(transform_army(army, s), b, w, t, max_ticks=50)
            bad += (r.destroyed_value, r.destruction) != (base.destroyed_value, base.destruction)
    assert bad == 0
