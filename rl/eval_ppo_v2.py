"""B10: evaluation of a trained MaskablePPO on DEV-side sets (v2). Deterministic masked argmax on the canonical (identity) view.

* `ModelPolicy` is a `breachlens.baselines.Policy`, so `breachlens.evaluator.evaluate` scores it exactly like every baseline (breach rate headline,
  mean score, mean D, bootstrap CIs over layouts, required metadata).
* Roles: only train / validation sets (TUNE-train, TUNE-val, DEV-train, DEV-validation). The held-out sets are refused here; the held-out
  evaluation is B15 and goes through the write-once protocol.
* `masked_probs` is the model adapter for `rl.tta` (probabilities for ONE observation).

    .venv/bin/python -m rl.eval_ppo_v2 --run-dir models_v2/<run> --set TUNE-val
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO

from breachlens import evaluator as EV
from breachlens.baselines import Policy, _ep
from breachlens.constants import N_UNIT_TYPES
from breachlens.env import BreachEnv

SET_ROLE = {"TUNE-train": "train", "DEV-train": "train", "TUNE-val": "validation", "DEV-validation": "validation"}


class ModelPolicy(Policy):
    """Plays one deterministic episode per layout with `model.predict(..., deterministic=True)` (masked argmax)."""

    def __init__(self, model: MaskablePPO, name: str = "PPO") -> None:
        self.model, self.name = model, name

    def episodes(self, rec: dict, layout: dict, P) -> list[dict]:
        env = BreachEnv(layouts=[layout], role="train")        # the evaluator already loaded the layout through the pool loaders
        obs, _ = env.reset(seed=zlib.crc32(rec["id"].encode()), options={"layout_index": 0})
        done = False
        while not done:
            action, _ = self.model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, _, done, _, _ = env.step(int(action))
        army = (np.bincount([t for t, _ in env.army], minlength=N_UNIT_TYPES) + np.asarray(env.owned)).tolist()   # leftover owned units count as bought
        return [_ep(env.result, env.bought, len(env.army), army=army)]


def masked_probs(model: MaskablePPO):
    """probs_fn(obs, mask) -> (N_ACTIONS,) masked action probabilities of ONE observation (adapter for rl.tta)."""
    def fn(obs: dict, mask: np.ndarray) -> np.ndarray:
        obs_t, _ = model.policy.obs_to_tensor(obs)
        with th.no_grad():
            dist = model.policy.get_distribution(obs_t, action_masks=np.asarray(mask, dtype=bool).reshape(1, -1))
        return dist.distribution.probs[0].cpu().numpy().astype(np.float64)
    return fn


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--set", required=True, choices=sorted(SET_ROLE))
    p.add_argument("--model", default="final.zip")
    args = p.parse_args(argv)
    run_dir = Path(args.run_dir)
    model = MaskablePPO.load(run_dir / args.model, device="cpu")
    res = EV.evaluate(ModelPolicy(model, name=f"PPO:{run_dir.name}"), args.set, SET_ROLE[args.set])
    EV.write_csv(res["rows"], run_dir / f"eval_{args.set}_layouts.csv")
    EV.write_csv(res["summary"], run_dir / f"eval_{args.set}_summary.csv")
    for r in res["summary"]:
        print(f"{r['group']:>12}  n={r['n_layouts']:3d}  breach={r['breach_mean']:.3f}  score={r['score_mean']:.4f}  D={r['d_mean']:.3f}  bought={r['mean_n_bought']:.1f}")
    return res


if __name__ == "__main__":
    main()
