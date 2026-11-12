"""Layout pool generation: acceptance rules, DEV/FINAL split, familiar/transfer tagging.

A layout record is small and fully reproducible: {id, family, seed, pool, tag}. The actual
buildings/turrets/walls are never stored -- `generate_base_layout(np.random.default_rng(seed),
family=family)` deterministically reproduces the exact same layout, so the json file only needs
enough to regenerate it plus the bookkeeping (which pool/tag it belongs to).
"""
from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path

import numpy as np

from siege.armies import MIXED_ARMIES, best_affordable_army, mixed_army, place_on_border
from siege.bases import FAMILIES, generate_base_layout
from siege.combat import simulate_battle
from siege.troops import TROOP_TYPES

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LAYOUTS_JSON = RESULTS_DIR / "layouts.json"
LAYOUTS_SHA256 = RESULTS_DIR / "layouts.sha256"

MAX_BATTLE_TICKS = 50

SINGLE_TYPES = [spec.name for spec in TROOP_TYPES]
ARMY_NAMES = SINGLE_TYPES + list(MIXED_ARMIES)

# One family is held out entirely for transfer evaluation -- its layouts are never trained or
# tuned on, only used to check whether a policy generalizes to a base style it has never seen.
# ring_defence (one consolidated wall+core) is the simplest, most "classic" style, which makes it
# a meaningful generalization probe for a policy trained on the two multi-cluster/multi-outpost
# families.
TRANSFER_FAMILY = "ring_defence"
FAMILIAR_FAMILIES = tuple(f for f in FAMILIES if f != TRANSFER_FAMILY)

# 41 (all transfer) + 100 + 100 (80/20 DEV/FINAL split each) = 241.
TARGET_COUNTS = {"ring_defence": 41, "clustered_core": 100, "scattered_outposts": 100}
DEV_FRACTION = 0.8

MIN_FEASIBLE_DESTRUCTION = 0.10
RANGER_FULL_CLEAR = 0.99


def stable_family_hash(family: str) -> int:
    """Process-independent replacement for hash(family) (str hashes are salted per run)."""
    return zlib.crc32(family.encode("utf-8")) % 10_000


def _army_for(name: str) -> list[str]:
    if name in MIXED_ARMIES:
        return mixed_army(MIXED_ARMIES[name])
    return best_affordable_army(name)


def _score(army_names: list[str], layout: dict, start_idx: int) -> float:
    army = place_on_border(army_names, start_idx=start_idx)
    result = simulate_battle(army, layout["buildings"], layout["walls"], layout["turrets"], max_ticks=MAX_BATTLE_TICKS)
    return result.destruction


def is_reachable(layout: dict) -> bool:
    """Sanity check, not a walkable-path BFS: walls are destructible (melee breaches them) and
    ranged troops ignore walls entirely when targeting, so almost any building is "reachable" in
    the gameplay sense even behind a fully sealed wall ring. This only catches generator bugs: a
    building/turret placed off the board, or stacked directly on top of a wall tile.
    """
    from siege.env import GRID

    wall_positions = set(layout["walls"])
    for b in layout["buildings"]:
        x, y = b["x"], b["y"]
        if not (0 <= x < GRID and 0 <= y < GRID) or (x, y) in wall_positions:
            return False
    for t in layout["turrets"]:
        x, y = t["x"], t["y"]
        if not (0 <= x < GRID and 0 <= y < GRID) or (x, y) in wall_positions:
            return False
    return True


def generate_and_score(seed: int, family: str) -> tuple[dict, dict[str, float]]:
    """Generate a layout and score every fixed test army against it (clustered deploy, same
    shared push point for every army so the comparison is apples to apples)."""
    rng = np.random.default_rng(seed)
    layout = generate_base_layout(rng, family=family)
    start_idx = int(rng.integers(0, 60))
    scores = {name: _score(_army_for(name), layout, start_idx) for name in ARMY_NAMES}
    return layout, scores


def acceptance_reason(layout: dict, scores: dict[str, float]) -> str | None:
    """None means accept; otherwise a short rejection reason string."""
    total_value = sum(b["value"] for b in layout["buildings"])
    if not layout["buildings"] or total_value <= 0:
        return "empty"
    if not is_reachable(layout):
        return "unreachable"
    if max(scores.values()) < MIN_FEASIBLE_DESTRUCTION:
        return "too_hard"
    if scores["ranger"] >= RANGER_FULL_CLEAR:
        return "ranger_trivial"
    return None


def build_layout_pool(
    target_counts: dict[str, int] = TARGET_COUNTS,
    seed_base: int = 0,
    dev_fraction: float = DEV_FRACTION,
    split_seed: int = 1,
    rejection_log: dict[str, dict[str, int]] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Generate accepted layouts per family up to `target_counts`, then assign pool/tag.

    Returns (records, rejection_counts). If `rejection_log` is given it is filled with
    {family: {reason: count}}. Acceptance is decided per layout alone (never from pool membership).
    """
    rejection_counts: dict[str, int] = {}
    accepted_by_family: dict[str, list[int]] = {family: [] for family in target_counts}

    for family_idx, family in enumerate(target_counts):
        target = target_counts[family]
        seed = seed_base + family_idx * 10_000_000
        while len(accepted_by_family[family]) < target:
            layout, scores = generate_and_score(seed, family)
            reason = acceptance_reason(layout, scores)
            if reason is None:
                accepted_by_family[family].append(seed)
            else:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                if rejection_log is not None:
                    fam_log = rejection_log.setdefault(family, {})
                    fam_log[reason] = fam_log.get(reason, 0) + 1
            seed += 1

    records: list[dict] = []
    next_id = 0
    for family, seeds in accepted_by_family.items():
        if family == TRANSFER_FAMILY:
            for s in seeds:
                records.append({"id": next_id, "family": family, "seed": s, "pool": "FINAL", "tag": "transfer"})
                next_id += 1
            continue
        split_rng = np.random.default_rng(split_seed + stable_family_hash(family))
        order = split_rng.permutation(len(seeds))
        n_dev = int(round(len(seeds) * dev_fraction))
        for rank, idx in enumerate(order):
            s = seeds[int(idx)]
            pool = "DEV" if rank < n_dev else "FINAL"
            records.append({"id": next_id, "family": family, "seed": int(s), "pool": pool, "tag": "familiar"})
            next_id += 1

    return records, rejection_counts


def save_layout_pool(records: list[dict], path: Path = LAYOUTS_JSON, sha_path: Path = LAYOUTS_SHA256) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(records, indent=2, sort_keys=True)
    path.write_text(payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    sha_path.write_text(digest + "\n")
    return digest


def load_layout_pool(pool: str | None = None, tag: str | None = None, path: Path = LAYOUTS_JSON) -> list[dict]:
    records = json.loads(Path(path).read_text())
    if pool is not None:
        records = [r for r in records if r["pool"] == pool]
    if tag is not None:
        records = [r for r in records if r["tag"] == tag]
    return records
