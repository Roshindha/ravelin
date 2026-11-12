"""Balance check: run fixed single-type and mixed armies against generated layouts from each
family using the real simulator, and report mean destruction by army type and family.

Usage: .venv/bin/python scripts_balance.py [--n 200]
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from siege.armies import MIXED_ARMIES, best_affordable_army, mixed_army, place_on_border
from siege.bases import FAMILIES, generate_base_layout
from siege.combat import simulate_battle
from siege.env import MAX_BATTLE_TICKS
from siege.troops import TROOP_TYPES

SINGLE_TYPES = [spec.name for spec in TROOP_TYPES]
ARMY_NAMES = SINGLE_TYPES + list(MIXED_ARMIES)


def army_for(name: str) -> list[str]:
    if name in MIXED_ARMIES:
        return mixed_army(MIXED_ARMIES[name])
    return best_affordable_army(name)


def score(army_names: list[str], layout: dict, start_idx: int) -> float:
    army = place_on_border(army_names, start_idx=start_idx)
    result = simulate_battle(army, layout["buildings"], layout["walls"], layout["turrets"], max_ticks=MAX_BATTLE_TICKS)
    return result.destruction


def run(n_per_family: int, seed_base: int = 0) -> dict:
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for family_idx, family in enumerate(FAMILIES):
        for i in range(n_per_family):
            rng = np.random.default_rng(seed_base + family_idx * 1_000_000 + i)
            layout = generate_base_layout(rng, family=family)
            start_idx = int(rng.integers(0, 60))  # same deploy point for every army on this layout
            for army_name in ARMY_NAMES:
                d = score(army_for(army_name), layout, start_idx)
                scores[(family, army_name)].append(d)
    return scores


def run_pool(pools: list[tuple[str | None, str | None]]) -> dict:
    """Score the armies on the frozen pool's actual layouts (same push point rule as acceptance).

    `pools` is a list of (pool, tag) filters passed to load_layout_pool. Default use: DEV-familiar
    plus the transfer family (which is entirely FINAL; only aggregate balance is reported).
    """
    from siege.layouts import generate_and_score, load_layout_pool

    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for pool, tag in pools:
        for rec in load_layout_pool(pool=pool, tag=tag):
            _, s = generate_and_score(rec["seed"], rec["family"])
            for name in ARMY_NAMES:
                scores[(rec["family"], name)].append(s[name])
    return scores


def print_table(scores: dict[tuple[str, str], list[float]]) -> None:
    header = f"{'family':<20}" + "".join(f"{name:>16}" for name in ARMY_NAMES)
    print(header)
    print("-" * len(header))
    for family in FAMILIES:
        row = f"{family:<20}"
        for name in ARMY_NAMES:
            vals = scores[(family, name)]
            row += f"{np.mean(vals):>16.3f}"
        print(row)
    print()
    print(f"{'best single-type':<20}" + "".join(f"{'':>16}" for _ in ARMY_NAMES))
    for family in FAMILIES:
        best_single = max(SINGLE_TYPES, key=lambda n: np.mean(scores[(family, n)]))
        best_single_mean = np.mean(scores[(family, best_single)])
        best_mixed = max(MIXED_ARMIES, key=lambda n: np.mean(scores[(family, n)]))
        best_mixed_mean = np.mean(scores[(family, best_mixed)])
        print(
            f"{family:<20} best single = {best_single} ({best_single_mean:.3f}),"
            f" best mixed = {best_mixed} ({best_mixed_mean:.3f}),"
            f" mixed_beats_single = {best_mixed_mean > best_single_mean}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--pool", action="store_true", help="score the frozen pool (DEV-familiar + transfer) instead of fresh seeds")
    args = parser.parse_args()

    if args.pool:
        print_table(run_pool([("DEV", "familiar"), (None, "transfer")]))
        return
    scores = run(args.n, seed_base=args.seed_base)
    print_table(scores)


if __name__ == "__main__":
    main()
