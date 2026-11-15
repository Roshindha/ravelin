"""D4 test-time ensembling (spec 7.1): combine a policy's masked action distribution (or Q-values) over the 8 symmetric views of a state.

For each symmetry s: transform obs and mask, query the policy, map the result back through the inverse action permutation, aggregate over
the views, then take the masked argmax. ``probs_fn(obs, mask) -> (N_ACTIONS,)`` is any callable returning probabilities / Q-values for ONE
observation (the model adapters live in rl/policy_io.py, B10+). No retraining, no env access.

Aggregation (pinned 2026-11-15, Gate changelog (n), before any trained model exists):
  * ``"logprob"`` (default; PPO plain / PPO+D4): average the log-probabilities of the LEGAL actions across the views (after the inverse action
    permutation), then masked argmax. Illegal actions score -inf. A legal action with probability 0 in any view scores -inf.
  * ``"mean"`` (DDQN Q-values, B11; also plain probability averaging, kept for tests): arithmetic mean of the mapped-back outputs.
"""
from __future__ import annotations

import numpy as np

from .augment import N_SYMMETRIES, transform_mask, transform_obs, untransform_probs

AGGREGATIONS = ("logprob", "mean")


def tta_scores(probs_fn, obs: dict, mask: np.ndarray, syms=range(N_SYMMETRIES), aggregate: str = "logprob") -> np.ndarray:
    """Aggregate over `syms` of the policy output of each transformed view, mapped back to original action indices.

    ``logprob``: mean log-probability on legal actions, -inf on illegal ones. ``mean``: mean of the raw outputs.
    """
    syms = list(syms)
    if not syms:
        raise ValueError("syms is empty")
    if aggregate not in AGGREGATIONS:
        raise ValueError(f"aggregate must be one of {AGGREGATIONS}")
    mask = np.asarray(mask, dtype=bool)
    acc = 0.0
    for s in syms:
        p = np.asarray(probs_fn(transform_obs(obs, s), transform_mask(mask, s)), dtype=np.float64)
        back = untransform_probs(p, s)
        if aggregate == "logprob":
            with np.errstate(divide="ignore"):
                back = np.where(mask, np.log(np.where(mask, back, 1.0)), -np.inf)
        acc = acc + back
    return acc / len(syms)


def tta_action(probs_fn, obs: dict, mask: np.ndarray, syms=range(N_SYMMETRIES), aggregate: str = "logprob") -> int:
    """Masked argmax of the TTA-aggregated scores (lowest index wins an exact tie, as np.argmax)."""
    sc = tta_scores(probs_fn, obs, mask, syms, aggregate)
    return int(np.argmax(np.where(mask, sc, -np.inf)))
