"""D4 test-time ensembling (spec 7.1): average a policy's masked action distribution (or Q-values) over the 8 symmetric views of a state.

For each symmetry s: transform obs and mask, query the policy, map the result back through the action permutation, average, then take the
masked argmax. ``probs_fn(obs, mask) -> (N_ACTIONS,)`` is any callable returning probabilities / Q-values for ONE observation (the model
adapters live in rl/policy_io.py, B10+). No retraining, no env access.
"""
from __future__ import annotations

import numpy as np

from .augment import N_SYMMETRIES, transform_mask, transform_obs, untransform_probs


def tta_scores(probs_fn, obs: dict, mask: np.ndarray, syms=range(N_SYMMETRIES)) -> np.ndarray:
    """Mean over `syms` of the policy output of each transformed view, mapped back to original action indices."""
    syms = list(syms)
    if not syms:
        raise ValueError("syms is empty")
    acc = 0.0
    for s in syms:
        p = np.asarray(probs_fn(transform_obs(obs, s), transform_mask(mask, s)), dtype=np.float64)
        acc = acc + untransform_probs(p, s)
    return acc / len(syms)


def tta_action(probs_fn, obs: dict, mask: np.ndarray, syms=range(N_SYMMETRIES)) -> int:
    """Masked argmax of the TTA-averaged scores (lowest index wins an exact tie, as np.argmax)."""
    sc = tta_scores(probs_fn, obs, mask, syms)
    return int(np.argmax(np.where(mask, sc, -np.inf)))
