from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from siege.env import SiegeEnv


def test_maskable_ppo_trains_and_predicts_on_siege_env():
    # Not a convergence test -- just guards the MaskablePPO <-> SiegeEnv wiring
    # (Dict obs space, action_masks() plumbing) against silent breakage.
    env = SiegeEnv()
    model = MaskablePPO("MultiInputPolicy", env, verbose=0, n_steps=64, seed=0)
    model.learn(total_timesteps=64)

    obs, _ = env.reset(seed=0)
    masks = get_action_masks(env)
    action, _ = model.predict(obs, action_masks=masks, deterministic=True)
    assert masks[int(action)]  # predicted action must be legal under the mask
