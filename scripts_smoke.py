"""Smoke run: 100 random masked steps on SiegeEnv. Usage: python scripts_smoke.py"""
import numpy as np

from siege.env import SiegeEnv

env = SiegeEnv()
obs, _ = env.reset(seed=0)
rng = np.random.default_rng(0)
episodes = 0
for t in range(100):
    a = int(rng.choice(np.flatnonzero(env.action_masks())))
    obs, r, term, trunc, _ = env.step(a)
    if term or trunc:
        episodes += 1
        obs, _ = env.reset()
print(f"OK: 100 steps, {episodes} episodes finished, last state={obs['state'].round(2)}")
