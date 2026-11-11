"""Evaluate a trained MaskablePPO model on SiegeEnv over N held-out base layouts.

Usage:
    .venv/bin/python -m rl.eval --model models/maskable_ppo_siege --episodes 50
"""
from __future__ import annotations

import argparse

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from siege.env import SiegeEnv


def run_episode(model: MaskablePPO, env: SiegeEnv, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action_masks = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    return env.destruction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000)  # disjoint from training seed range
    args = parser.parse_args()

    env = SiegeEnv()
    model = MaskablePPO.load(args.model)

    destructions = [run_episode(model, env, seed=args.seed + i) for i in range(args.episodes)]
    print(
        f"episodes={args.episodes} "
        f"mean_destruction={np.mean(destructions):.3f} "
        f"std={np.std(destructions):.3f} "
        f"min={np.min(destructions):.3f} "
        f"max={np.max(destructions):.3f}"
    )


if __name__ == "__main__":
    main()
