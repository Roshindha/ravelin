import numpy as np
from siege import env as E
from siege.env import SiegeEnv


def test_constants():
    assert E.N_ACTIONS == 246
    assert len(E.BORDER) == 60 and len(set(E.BORDER)) == 60
    assert all(r in (0, 15) or c in (0, 15) for r, c in E.BORDER)


def test_reset_obs_shapes():
    env = SiegeEnv()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs["grid"].shape == (8, 16, 16) and obs["state"].shape == (9,)


def test_random_masked_rollout_100_steps():
    env = SiegeEnv()
    obs, _ = env.reset(seed=123)
    rng = np.random.default_rng(123)
    for _ in range(100):
        mask = env.action_masks()
        assert mask.shape == (246,) and mask.any()
        obs, r, term, trunc, _ = env.step(int(rng.choice(np.flatnonzero(mask))))
        assert env.observation_space.contains(obs)
        if term or trunc:
            obs, _ = env.reset()


def test_mask_phases():
    env = SiegeEnv()
    env.reset(seed=0)
    m = env.action_masks()
    assert m[:4].all() and m[E.END_BUY] and not m[E.WAIT]
    assert not m[E.DEPLOY_BASE:E.END_BUY].any()
    env.step(0)  # buy one type-0
    env.step(E.END_BUY)
    m = env.action_masks()
    assert not m[:4].any() and not m[E.END_BUY]
    assert m[E.DEPLOY_BASE:E.DEPLOY_BASE + 60].all()  # type 0 everywhere
    assert not m[E.DEPLOY_BASE + 60:E.END_BUY].any()  # none owned of others


def test_unaffordable_buy_masked():
    env = SiegeEnv()
    env.reset(seed=0)
    for _ in range(5):
        env.step(3)  # 5 * 40 = 200
    assert not env.action_masks()[:4].any()


def test_illegal_action_is_noop():
    env = SiegeEnv()
    env.reset(seed=0)
    obs, reward, term, trunc, info = env.step(E.WAIT)
    assert env.observation_space.contains(obs)
    assert reward == 0.0
    assert not term and not trunc
    assert info == {}


def test_determinism_same_seed():
    def run(seed):
        env = SiegeEnv()
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(100):
            a = int(rng.choice(np.flatnonzero(env.action_masks())))
            o, r, t, _, _ = env.step(a)
            out.append((a, r, t, o["state"].tobytes(), o["grid"].tobytes()))
            if t:
                env.reset()
        return out
    assert run(7) == run(7)


def test_gym_reset_contract():
    # Gymnasium's check_env samples unmasked random actions, which this env
    # rejects by design, so check only the reset/seed contract.
    from gymnasium.utils.env_checker import check_reset_seed_determinism
    check_reset_seed_determinism(SiegeEnv())


def test_reset_generates_base_layout():
    env = SiegeEnv()
    env.reset(seed=0)
    assert len(env.layout["buildings"]) >= 3
    assert len(env.layout["walls"]) > 0


def test_full_episode_runs_real_battle_and_damages_base():
    # Buy 5 tanks and dump them all on adjacent border tiles near one corner;
    # with a real deterministic simulator this should produce nonzero
    # destruction (the previous stub always returned 0).
    env = SiegeEnv()
    env.reset(seed=1)
    for _ in range(5):
        env.step(2)  # buy tank (index 2)
    env.step(E.END_BUY)
    mask = env.action_masks()
    lo = E.DEPLOY_BASE + 2 * E.N_BORDER
    hi = lo + E.N_BORDER
    tile_actions = np.flatnonzero(mask[lo:hi]) + lo
    reward = 0.0
    for a in tile_actions[:5]:
        _, r, term, _, _ = env.step(int(a))
        reward += r
    assert term
    assert env.battle_result is not None
    assert 0.0 <= env.destruction <= 1.0
    assert reward == env.battle_result.reward
