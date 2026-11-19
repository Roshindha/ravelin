"""B9: D4 augmentation (rl/augment.py), D4Wrapper and test-time ensembling (rl/tta.py) for BreachEnv. DEV / TUNE layouts only."""
from __future__ import annotations

import numpy as np
import pytest

from breachlens import layouts as Y
from breachlens import pool as P
from breachlens._d4check import tr as ind_tr
from breachlens.actions import BORDER
from breachlens.constants import DEPLOY_BASE, END_BUY, GRID, N_ACTIONS, N_BORDER, N_GRID_CHANNELS, WAIT
from breachlens.env import BreachEnv, d4_action_perm, d4_cell_perm, d4_transform_grid
from breachlens.sim import Prepared, simulate
from rl.augment import (ACTION_PERM, ACTION_PERM_INV, CELL_DEST, D4Wrapper, N_SYMMETRIES, TILE_PERM, compose, inverse_symmetry,
                        transform_action, transform_army, transform_cell, transform_grid, transform_mask, transform_obs,
                        untransform_action, untransform_probs)
from rl.tta import tta_action, tta_scores

ALL = range(N_SYMMETRIES)
DEV_RECS = P.dev_records("train")
TUNE_RECS = P.design_records("TUNE-train", "train")
_L: dict = {}


def layout_of(rec):
    if rec["id"] not in _L:
        _L[rec["id"]] = P.load_layout(rec, "train")
    return _L[rec["id"]]


def spread(recs, n):
    return [recs[int(i)] for i in np.linspace(0, len(recs) - 1, n)]


# ------------------------------------------------------------------ group structure
def test_tables_match_the_independent_helpers_and_the_env_tables():
    for s in ALL:
        for r in range(GRID):
            for c in range(GRID):
                assert transform_cell(s, r, c) == ind_tr(s, (r, c))
        assert (CELL_DEST[s] == d4_cell_perm(s)).all()
        assert (ACTION_PERM[s] == d4_action_perm(s)).all()
    assert (ACTION_PERM[0] == np.arange(N_ACTIONS)).all()


def test_inverse_identity():
    for s in ALL:
        i = inverse_symmetry(s)
        for r, c in [(0, 0), (3, 11), (13, 2), (6, 6)]:
            assert transform_cell(i, *transform_cell(s, r, c)) == (r, c)
            assert transform_cell(s, *transform_cell(i, r, c)) == (r, c)
        assert compose(s, i) == 0 == compose(i, s)
        assert (ACTION_PERM_INV[s] == ACTION_PERM[i]).all()
        for a in (0, 4, 5, 100, 264, END_BUY, WAIT):
            assert untransform_action(transform_action(a, s), s) == a


def test_group_closure_identity_and_associativity():
    assert len({compose(a, b) for a in ALL for b in ALL}) == 8
    for a in ALL:
        assert compose(0, a) == a == compose(a, 0)
        for b in ALL:
            c = compose(a, b)
            assert 0 <= c < 8
            assert (ACTION_PERM[c] == ACTION_PERM[a][ACTION_PERM[b]]).all()      # tables compose like the group
            assert (CELL_DEST[c] == CELL_DEST[a][CELL_DEST[b]]).all()
            for d in ALL:
                assert compose(compose(a, b), d) == compose(a, compose(b, d))
    assert sorted(compose(1, 1) for _ in [0]) == [2] and compose(3, 1) == 0 and compose(4, 4) == 0   # r1.r1 = r2, r3.r1 = e, flip^2 = e


# ------------------------------------------------------------------ action bijection and masks
def test_action_permutation_is_a_bijection_per_symmetry_and_preserves_structure():
    for s in ALL:
        p = ACTION_PERM[s]
        assert sorted(p) == list(range(N_ACTIONS))
        assert (p[:DEPLOY_BASE] == np.arange(DEPLOY_BASE)).all() and p[END_BUY] == END_BUY and p[WAIT] == WAIT
        assert sorted(TILE_PERM[s]) == list(range(N_BORDER))
        for t in range(5):                                                    # a deploy keeps its unit type
            block = p[DEPLOY_BASE + t * N_BORDER: DEPLOY_BASE + (t + 1) * N_BORDER]
            assert sorted(block) == list(range(DEPLOY_BASE + t * N_BORDER, DEPLOY_BASE + (t + 1) * N_BORDER))
        for i, (r, c) in enumerate(BORDER):                                    # a tile moves to the image cell
            assert BORDER[TILE_PERM[s, i]] == transform_cell(s, r, c)


def test_grid_transform_is_exact_per_channel_incl_repair_and_state_is_invariant():
    rng = np.random.default_rng(0)
    g = rng.random((N_GRID_CHANNELS, GRID, GRID), dtype=np.float64).astype(np.float32)
    for s in ALL:
        t = transform_grid(g, s)
        assert (t == d4_transform_grid(g, s)).all()
        for ch in range(N_GRID_CHANNELS):                                      # incl. channel 8 (repair reach)
            for r, c in [(0, 0), (2, 9), (13, 5), (7, 7)]:
                assert t[ch][transform_cell(s, r, c)] == g[ch, r, c]
        assert (transform_grid(t, inverse_symmetry(s)) == g).all()
    st = rng.random(11).astype(np.float32)
    assert (transform_obs({"grid": g, "state": st}, 5)["state"] == st).all()


def test_mask_legality_under_transform_on_real_episodes():
    """The transformed mask equals the mask of the env run on the transformed layout, at every step, and a legal original action maps to a legal one."""
    rng = np.random.default_rng(1)
    for i, rec in enumerate(spread(DEV_RECS, 6) + spread(TUNE_RECS, 6)):
        L = layout_of(rec)
        base = BreachEnv(layouts=[L], prebought="random" if i % 2 else None)
        for s in (1, 4, 6):
            tr = BreachEnv(layouts=[Y.transform(L, s)], prebought="random" if i % 2 else None)
            base.reset(seed=i), tr.reset(seed=i)
            while not base.done:
                m = base.action_masks()
                mt = transform_mask(m, s)
                assert (mt == tr.action_masks()).all() and mt.sum() == m.sum()
                a = int(rng.choice(np.flatnonzero(m)))
                assert tr.action_masks()[transform_action(a, s)]
                base.step(a), tr.step(transform_action(a, s))


# ------------------------------------------------------------------ simulator + env equivariance, incl. Menders
def _random_army_actions(env, rng, with_mender: bool):
    """A full legal episode script (original coordinates) from a freshly reset env; forces >= 1 Mender when requested."""
    acts, probe = [], env.__class__(layouts=[env._synthetic[0]])
    probe.reset(seed=0)
    first = True
    while not probe.done:
        m = probe.action_masks()
        if with_mender and first and m[4]:
            a = 4
        else:
            a = int(rng.choice(np.flatnonzero(m)))
        first = False
        probe.step(a)
        acts.append(a)
    return acts


def test_env_and_simulator_are_d4_equivariant_on_100_layouts_incl_menders_through_the_wrapper():
    """100 layouts x 8 symmetries. D4Wrapper(symmetry=s) over the UNtransformed env must show exactly what the env on the transformed
    layout shows (obs, state, mask), and the same action sequence must give the same score / D / breach. >= 50 % of armies hold a Mender."""
    recs = spread(DEV_RECS, 50) + spread(TUNE_RECS, 50)
    rng = np.random.default_rng(7)
    n_mender, n_steps = 0, 0
    for i, rec in enumerate(recs):
        L = layout_of(rec)
        with_mender = i % 2 == 0
        script = _random_army_actions(BreachEnv(layouts=[L]), rng, with_mender)
        n_mender += any(a == 4 for a in script)
        for s in ALL:
            wrapped = D4Wrapper(BreachEnv(layouts=[L]), symmetry=s)
            direct = BreachEnv(layouts=[Y.transform(L, s)])
            ow, _ = wrapped.reset(seed=i)
            od, _ = direct.reset(seed=i)
            for a in script:
                a_view = transform_action(a, s)                               # the action the agent would choose in the transformed view
                assert (ow["grid"] == od["grid"]).all() and (ow["state"] == od["state"]).all()
                assert (wrapped.action_masks() == direct.action_masks()).all()
                n_steps += 1
                ow, rw, dw, _, iw = wrapped.step(a_view)
                od, rd, dd, _, idd = direct.step(a_view)
                assert rw == rd and dw == dd
            assert dw and iw["score"] == idd["score"] and iw["d_num"] == idd["d_num"] and iw["core_destroyed"] == idd["core_destroyed"]
            assert (ow["grid"] == od["grid"]).all()
    assert n_mender >= 50 and n_steps > 8 * 100


def test_bare_simulator_equivariance_with_transformed_army_helper():
    rng = np.random.default_rng(3)
    for rec in spread(TUNE_RECS, 12):
        L = layout_of(rec)
        army = [(int(rng.integers(5)), int(rng.integers(N_BORDER))) for _ in range(8)] + [(4, 3), (4, 30)]
        free = [t for t in range(N_BORDER) if BORDER[t] not in {tuple(w) for w in L["walls"]}]
        army = [(t, free[tile % len(free)]) for t, tile in army]
        army = list({tile: (t, tile) for t, tile in army}.values())
        ref = simulate(Prepared(L), army)
        for s in ALL:
            r2 = simulate(Prepared(Y.transform(L, s)), transform_army(army, s))
            assert (r2.d_num, r2.d_den, r2.core_destroyed, r2.ticks) == (ref.d_num, ref.d_den, ref.core_destroyed, ref.ticks)


# ------------------------------------------------------------------ wrapper behaviour
def test_wrapper_identity_and_random_symmetry_stream_is_seeded():
    L = layout_of(TUNE_RECS[0])
    plain = BreachEnv(layouts=[L])
    o0, _ = plain.reset(seed=3)
    w0 = D4Wrapper(BreachEnv(layouts=[L]), symmetry=0)
    ow, info = w0.reset(seed=3)
    assert (ow["grid"] == o0["grid"]).all() and info["symmetry"] == 0 and (w0.action_masks() == plain.action_masks()).all()
    a, b = D4Wrapper(BreachEnv(layouts=[L]), seed=0), D4Wrapper(BreachEnv(layouts=[L]), seed=0)
    sa = [a.reset(seed=k)[1]["symmetry"] for k in range(30)]
    sb = [b.reset(seed=k)[1]["symmetry"] for k in range(30)]
    assert sa == sb and len(set(sa)) >= 6
    with pytest.raises(ValueError):
        D4Wrapper(BreachEnv(layouts=[L]), symmetry=8)


# ------------------------------------------------------------------ TTA
def _obs_mask(i=0, steps=3):
    env = BreachEnv(layouts=[layout_of(TUNE_RECS[i])])
    env.reset(seed=0)
    rng = np.random.default_rng(i)
    for _ in range(steps):
        env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
    return env._obs(), env.action_masks()


def _masked_softmax(logits, mask):
    z = np.where(mask, logits, -np.inf)
    e = np.exp(z - z.max())
    return e / e.sum()


def _random_policy(seed=0):
    """A fixed, NON-equivariant policy: probabilities from a random linear map of (grid, state), masked."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(N_GRID_CHANNELS * GRID * GRID + 11, N_ACTIONS)) * 0.3
    return lambda obs, mask: _masked_softmax(np.concatenate([obs["grid"].ravel(), obs["state"]]) @ W, mask)


def test_tta_with_identity_only_equals_the_plain_policy():
    pol = _random_policy()
    for i in range(4):
        obs, mask = _obs_mask(i)
        assert (tta_scores(pol, obs, mask, syms=[0], aggregate="mean") == pol(obs, mask)).all()
        assert tta_action(pol, obs, mask, syms=[0], aggregate="mean") == int(np.argmax(pol(obs, mask)))


def test_tta_of_an_equivariant_policy_equals_the_policy_and_tta_is_itself_equivariant():
    base = _random_policy(1)

    def equivariant(obs, mask):                      # symmetrise: average of the 8 mapped-back views (an exactly D4-equivariant policy)
        return tta_scores(base, obs, mask, aggregate="mean")

    for i in range(3):
        obs, mask = _obs_mask(i, steps=4)
        p = tta_scores(base, obs, mask, aggregate="mean")
        assert abs(p.sum() - 1.0) < 1e-9 and (p[~mask] == 0).all()
        for s in (1, 4, 7):
            ps = tta_scores(base, transform_obs(obs, s), transform_mask(mask, s), aggregate="mean")
            np.testing.assert_allclose(ps, _apply(p, s), atol=1e-12)   # TTA(view_s) = view_s of TTA
            # a TTA-symmetrised policy is a fixed point of TTA
            np.testing.assert_allclose(tta_scores(equivariant, obs, mask, aggregate="mean"), p, atol=1e-12)
        assert int(np.argmax(np.where(mask, p, -1))) == tta_action(base, obs, mask, aggregate="mean")


def _apply(p, s):
    """p over original actions -> p over the transformed view's actions: out[ACTION_PERM[s][a]] = p[a]."""
    out = np.empty_like(p)
    out[ACTION_PERM[s]] = p
    return out


def test_tta_maps_q_values_back_through_the_inverse_permutation():
    obs, mask = _obs_mask(2, steps=5)
    q = np.where(mask, np.random.default_rng(5).normal(size=N_ACTIONS), 0.0)
    for s in ALL:
        assert (untransform_probs(_apply(q, s), s) == q).all()

    def true_equivariant_q(o, m):                    # knows the true Q in original coordinates; answers in the view it is shown
        return _apply(q, _sym_of(o, obs))

    np.testing.assert_allclose(tta_scores(true_equivariant_q, obs, mask, aggregate="mean"), q, atol=1e-12)
    assert tta_action(true_equivariant_q, obs, mask, aggregate="mean") == int(np.argmax(np.where(mask, q, -np.inf)))
    with pytest.raises(ValueError):
        tta_scores(true_equivariant_q, obs, mask, syms=[], aggregate="mean")


def _sym_of(o, obs):
    for s in ALL:
        if (transform_grid(obs["grid"], s) == o["grid"]).all():
            return s
    raise AssertionError


# ---- TTA aggregation pinned (Gate changelog (n)): mean log-probability over legal actions across the 8 views, then masked argmax
def test_tta_default_is_mean_logprob_over_legal_actions_then_masked_argmax():
    pol = _random_policy(3)
    for i in range(4):
        obs, mask = _obs_mask(i, steps=4)
        manual = 0.0
        for s in ALL:                                             # independent restatement: views -> inverse permutation -> log -> mean
            p = pol(transform_obs(obs, s), transform_mask(mask, s))
            manual = manual + np.log(np.where(mask, p[ACTION_PERM[s]], 1.0))
        manual = np.where(mask, manual / 8, -np.inf)
        sc = tta_scores(pol, obs, mask)
        np.testing.assert_allclose(sc[mask], manual[mask], atol=1e-12)
        assert np.isneginf(sc[~mask]).all()
        assert tta_action(pol, obs, mask) == int(np.argmax(manual))
        assert mask[tta_action(pol, obs, mask)]
        # identity-only logprob TTA == the plain policy's masked argmax
        assert tta_action(pol, obs, mask, syms=[0]) == int(np.argmax(np.where(mask, pol(obs, mask), -np.inf)))


def test_tta_logprob_differs_from_probability_mean_on_a_constructed_case_and_zero_prob_legal_is_minus_inf():
    obs, mask = _obs_mask(0, steps=2)
    a, b = np.flatnonzero(mask)[:2]
    # view-dependent outputs in ORIGINAL coordinates: action a gets 0.9 in 4 views and 0.001 in the other 4; action b a steady 0.3
    def probs_in_view(o, m):
        s = _sym_of(o, obs)
        out = np.zeros(N_ACTIONS)
        pa = 0.9 if s < 4 else 0.001
        out[ACTION_PERM[s][a]], out[ACTION_PERM[s][b]] = pa, 0.3
        return out
    assert tta_action(probs_in_view, obs, mask, aggregate="mean") == a          # arithmetic mean: 0.4505 > 0.3
    assert tta_action(probs_in_view, obs, mask) == b                            # geometric: sqrt(0.9*0.001)=0.03 < 0.3
    sc = tta_scores(probs_in_view, obs, mask)
    others = [x for x in np.flatnonzero(mask) if x not in (a, b)]
    if others:
        assert np.isneginf(sc[others]).all()                                     # legal but p = 0 in every view -> -inf
