"""STEP 5c, iteration 2 design aid: sweep splash_falloff on the TUNING seeds (fixed armies only).

This is part of tuning iteration 2 (one parameter, four values, same disjoint TUNING seeds; no FINAL data).
Usage: .venv/bin/python scripts_sweep_splash.py
"""
from __future__ import annotations

import dataclasses

import numpy as np

import siege.troops as troops
from scripts_tuning import tuning_seeds
from siege.bases import FAMILIES
from siege.layouts import ARMY_NAMES, generate_and_score


def set_falloff(value: float) -> None:
    new = tuple(dataclasses.replace(s, splash_falloff=value) if s.name == "splash" else s for s in troops.TROOP_TYPES)
    troops.TROOP_TYPES = new
    troops.TROOP_TYPE_BY_NAME.update({s.key: s for s in new})


def main() -> None:
    seeds = tuning_seeds(150)
    print(f"{'falloff':>8} {'family':<20}" + "".join(f"{a:>16}" for a in ("swarm", "ranger", "tank", "splash", "mixed_balanced", "mixed_breach")) + "   splash-army-strictly-best")
    for fo in (0.2, 0.3, 0.4, 0.5):
        set_falloff(fo)
        for family in FAMILIES:
            rows = [generate_and_score(sd, family)[1] for sd in seeds[family]]
            means = {a: np.mean([r[a] for r in rows]) for a in ARMY_NAMES}
            strict = np.mean([max(r["splash"], r["mixed_breach"]) > max(r[a] for a in ("swarm", "ranger", "tank", "mixed_balanced")) + 1e-9 for r in rows])
            print(f"{fo:>8} {family:<20}" + "".join(f"{means[a]:>16.3f}" for a in ("swarm", "ranger", "tank", "splash", "mixed_balanced", "mixed_breach")) + f"   {strict:.2f}")


if __name__ == "__main__":
    main()
