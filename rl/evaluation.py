"""Run a MaskablePPO policy over a fixed list of layout records (deterministic, identity symmetry)."""
from __future__ import annotations

import numpy as np
from sb3_contrib import MaskablePPO

from siege.env import SiegeEnv
from siege.troops import TROOP_TYPES

TROOP_NAMES = [spec.name for spec in TROOP_TYPES]


def evaluate_policy(model: MaskablePPO, records: list[dict], prebought: list[int] | None = None, deterministic: bool = True) -> list[dict]:
    """One episode per record. ``prebought=None`` -> the full task (agent shops); a count list -> deploy-only.

    Returns per-layout dicts: id, family, destruction, army (bought counts per troop type), n_steps.
    """
    env = SiegeEnv()
    rows = []
    for rec in records:
        obs, _ = env.reset(seed=int(rec["id"]), options={"layout_record": rec, "prebought": prebought if prebought else False})
        done, n = False, 0
        while not done:
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=deterministic)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done, n = terminated or truncated, n + 1
        bought = (env.owned + env.deployed).tolist()  # leftover owned troops count as bought-but-unused
        rows.append({"id": rec["id"], "family": rec["family"], "destruction": float(env.destruction),
                     "army": bought, "deployed": env.deployed.tolist(), "n_steps": n})
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {"overall": float(np.mean([r["destruction"] for r in rows]))}
    for fam in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] == fam]
        out[fam] = float(np.mean([r["destruction"] for r in sub]))
        out[f"{fam}_clear"] = float(np.mean([r["destruction"] >= 0.999 for r in sub]))
    out["mean_army"] = np.mean([r["army"] for r in rows], axis=0).tolist()
    return out
