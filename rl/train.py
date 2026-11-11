"""Train MaskablePPO on SiegeEnv.

Usage:
    .venv/bin/python -m rl.train --timesteps 200000 --out maskable_ppo_siege
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO

from siege.env import SiegeEnv

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="maskable_ppo_siege")
    args = parser.parse_args()

    env = SiegeEnv()
    model = MaskablePPO("MultiInputPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / args.out
    model.save(out_path)
    print(f"saved model to {out_path}.zip")


if __name__ == "__main__":
    main()
