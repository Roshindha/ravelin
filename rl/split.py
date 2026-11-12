"""Fixed train / validation split of the DEV-familiar layouts.

Validation = 20 DEV layouts (10 per familiar family), chosen once with a fixed seed and recorded in
results/val_split.json. FINAL and transfer layouts are never loaded by this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from siege.layouts import FAMILIAR_FAMILIES, LAYOUTS_JSON, RESULTS_DIR, load_layout_pool, stable_family_hash

SPLIT_PATH = RESULTS_DIR / "val_split.json"
N_VAL_PER_FAMILY = 10
SPLIT_SEED = 20260


def layouts_sha() -> str:
    return hashlib.sha256(Path(LAYOUTS_JSON).read_bytes()).hexdigest()


def make_split() -> dict:
    dev = load_layout_pool(pool="DEV", tag="familiar")
    val_ids: list[int] = []
    for fam in FAMILIAR_FAMILIES:
        ids = sorted(r["id"] for r in dev if r["family"] == fam)
        rng = np.random.default_rng(SPLIT_SEED + stable_family_hash(fam))
        val_ids += sorted(int(i) for i in rng.choice(ids, size=N_VAL_PER_FAMILY, replace=False))
    train_ids = sorted(r["id"] for r in dev if r["id"] not in set(val_ids))
    return {"layouts_sha256": layouts_sha(), "split_seed": SPLIT_SEED, "val_ids": val_ids, "train_ids": train_ids}


def write_split(path: Path = SPLIT_PATH) -> dict:
    split = make_split()
    path.write_text(json.dumps(split, indent=2))
    return split


def load_split(path: Path = SPLIT_PATH) -> tuple[list[dict], list[dict]]:
    """Return (train_records, val_records), creating the split file on first use. Verifies the pool sha."""
    if not Path(path).exists():
        write_split(path)
    split = json.loads(Path(path).read_text())
    if split["layouts_sha256"] != layouts_sha():
        raise RuntimeError("val_split.json was made for a different layouts.json; refusing to use it")
    dev = {r["id"]: r for r in load_layout_pool(pool="DEV", tag="familiar")}
    return [dev[i] for i in split["train_ids"]], [dev[i] for i in split["val_ids"]]
