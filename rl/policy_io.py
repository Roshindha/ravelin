"""Thin, model-agnostic access to what a MaskablePPO policy "thinks" at a given observation.

This is the hook for the later "Why this move?" explainability feature: it returns the full masked
action distribution, the value estimate and the top-k actions, without sampling or stepping the env.
"""
from __future__ import annotations

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO

from siege.env import N_ACTIONS, decode_action


def policy_outputs(model: MaskablePPO, obs: dict, action_masks: np.ndarray, top_k: int = 5) -> dict:
    """Return {'probs': (246,), 'logits': (246,) masked, 'value': float, 'top': [(action, prob, decoded)]}.

    Illegal actions get probability exactly 0 (same masking MaskablePPO applies when acting).
    """
    policy = model.policy
    obs_t, _ = policy.obs_to_tensor(obs)
    mask_t = th.as_tensor(np.asarray(action_masks, dtype=bool).reshape(1, N_ACTIONS), device=policy.device)
    with th.no_grad():
        dist = policy.get_distribution(obs_t, action_masks=mask_t)
        probs = dist.distribution.probs[0].cpu().numpy()
        logits = dist.distribution.logits[0].cpu().numpy()
        value = float(policy.predict_values(obs_t)[0, 0].cpu())
    order = np.argsort(-probs)[:top_k]
    top = [(int(a), float(probs[a]), decode_action(int(a))) for a in order]
    return {"probs": probs, "logits": logits, "value": value, "top": top}
