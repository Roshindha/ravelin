"""STEP 5b/5c: rebuild + re-freeze the layout pool (v3 after the final balance pass).

Same generators, same acceptance rules, same split logic as v1/v2; only the simulator/stats (and so which
seeds pass acceptance) and the seed stream differ. Acceptance never looks at pool membership.
The seed base and per-family/per-rule rejection counts are written to results/layouts_build.json.

Usage: .venv/bin/python scripts_rebuild_pool.py
"""
from __future__ import annotations

import json
from collections import Counter

from scripts_tuning import FAMILY_STRIDE, TUNING_SEED_BASE
from siege.bases import FAMILIES
from siege.layouts import DEV_FRACTION, RESULTS_DIR, TARGET_COUNTS, build_layout_pool, save_layout_pool

SEED_BASE = 3_000_000  # fresh stream: v1 used 0, v2 5_000_000, TUNING 9_000_000 (per-family offsets of 10_000_000 apply to all)
VERSION = "v3"
SPLIT_SEED = 1


def main() -> None:
    log: dict[str, dict[str, int]] = {}
    records, rejections = build_layout_pool(seed_base=SEED_BASE, split_seed=SPLIT_SEED, rejection_log=log)
    digest = save_layout_pool(records)
    counts = Counter((r["family"], r["pool"], r["tag"]) for r in records)
    build = {
        "version": VERSION,
        "seed_base": SEED_BASE,
        "tuning_seed_base": TUNING_SEED_BASE,
        "tuning_family_stride": FAMILY_STRIDE,
        "tuning_layouts_per_family": 300,
        "tuning_seed_ranges": {f: [TUNING_SEED_BASE + i * FAMILY_STRIDE, TUNING_SEED_BASE + i * FAMILY_STRIDE + 299] for i, f in enumerate(FAMILIES)},
        "tuning_iterations_used": 2,
        "split_seed": SPLIT_SEED,
        "dev_fraction": DEV_FRACTION,
        "target_counts": TARGET_COUNTS,
        "rejections_total": rejections,
        "rejections_by_family": log,
        "counts": {f"{f}|{p}|{t}": n for (f, p, t), n in sorted(counts.items())},
        "sha256": digest,
    }
    (RESULTS_DIR / "layouts_build.json").write_text(json.dumps(build, indent=2, sort_keys=True))
    print(json.dumps(build, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
