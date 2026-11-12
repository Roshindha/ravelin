import numpy as np
from gymnasium.utils.env_checker import check_env

from siege.env import (
    BUY_BASE,
    DEPLOY_BASE,
    END_BUY,
    N_ACTIONS,
    N_BORDER,
    SiegeEnv,
    decode_action,
    encode_deploy,
)


def test_action_encode_decode_roundtrip():
    for troop_idx in range(4):
        for tile_idx in range(N_BORDER):
            action = encode_deploy(troop_idx, tile_idx)
            decoded = decode_action(action)
            assert decoded == ("deploy", troop_idx, tile_idx)
    assert decode_action(0) == ("buy", 0, None)
    assert decode_action(END_BUY) == ("end_buy", None, None)
    assert decode_action(END_BUY + 1) == ("wait", None, None)


def test_random_legal_rollouts_5000_no_crash():
    env = SiegeEnv()
    rng = np.random.default_rng(0)
    for _ in range(5000):
        obs, _ = env.reset(seed=int(rng.integers(0, 10_000)))
        while True:
            legal = np.flatnonzero(env.action_masks())
            assert legal.size > 0
            action = int(rng.choice(legal))
            obs, reward, terminated, truncated, _ = env.step(action)
            assert obs["grid"].shape == (8, 16, 16)
            assert obs["state"].shape == (9,)
            assert 0.0 <= reward <= 1.0 + 1e-6
            assert env.gold >= 0 and env.gold <= 200
            if terminated or truncated:
                break


def test_mask_invariants_and_budget():
    env = SiegeEnv()
    env.reset(seed=7)
    m = env.action_masks()
    assert m.shape == (N_ACTIONS,)
    assert m.any()
    assert (m[:BUY_BASE + 4] == (np.array([True, True, True, True]))).all()
    assert m[END_BUY]
    assert not m[END_BUY + 1] or env.phase in (0, 1)
    env.step(0)
    env.step(END_BUY)
    assert env.phase == 1
    mask = env.action_masks()
    assert np.any(mask[DEPLOY_BASE:END_BUY])
    assert not mask[:BUY_BASE + 4].any()
    assert env.gold >= 0


def test_check_env_smoke():
    env = SiegeEnv()
    check_env(env, skip_render_check=True)


# -- STEP 5c: Greedy must be a sensible reference ------------------------------------------------
def test_greedy_army_is_lanchester_value_ranked_not_five_splash():
    from siege.baselines import _greedy_army, lanchester_value
    from siege.troops import TROOP_TYPES

    army = _greedy_army({})
    assert sum(next(s.cost for s in TROOP_TYPES if s.name == n) for n in army) <= 200
    assert army.count("splash") < 5  # the v1 bug: damage/cost bought nothing but splash
    best = max(TROOP_TYPES, key=lanchester_value)
    assert army[0] == best.name
