"""B21: the precomputed "Watch the AI" bundle (docs/SPEC_V2.md, Gate changelog row (qq)). Pure standard library: no torch, no stable-baselines, no streamlit.

``data/app_bundle/`` holds, per DEV-validation layout, the ``episode()`` data of the app for the three seed-0 arms x {plain, TTA} (the output of
``scripts/make_replays.run_episode``: per decision step the top 5, chosen action, heatmap, value and labels, plus the battle frames), written by
``scripts/build_app_bundle.py`` from a clean tree together with its provenance in ``manifest.json``. The app reads it when torch is not importable or
``RAVELIN_DEPLOY=1``; otherwise it plays the live models.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "data" / "app_bundle"
BUNDLE_SPLIT = "DEV-validation"
BUNDLE_VERSION = 1


def torch_available() -> bool:
    """True when ``import torch`` could succeed (a blocked import, ``sys.modules['torch'] = None``, counts as unavailable)."""
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


def deploy_mode() -> bool:
    """Use the bundle (and write nothing for visitors): ``RAVELIN_DEPLOY=1``, or torch is not installed."""
    return os.environ.get("RAVELIN_DEPLOY") == "1" or not torch_available()


def episode_key(arm: str, tta: bool) -> str:
    return f"{arm}|{'tta' if tta else 'plain'}"


def layout_file(layout_id: str, bundle_dir: Path = BUNDLE_DIR) -> Path:
    return Path(bundle_dir) / f"{layout_id}.json.gz"


@lru_cache(maxsize=4)
def load_manifest(bundle_dir: Path = BUNDLE_DIR) -> dict:
    m = json.loads((Path(bundle_dir) / "manifest.json").read_text())
    if m.get("version") != BUNDLE_VERSION or m.get("split") != BUNDLE_SPLIT:
        raise ValueError(f"{bundle_dir}: not an app bundle of version {BUNDLE_VERSION} for {BUNDLE_SPLIT}")
    return m


def layout_ids(bundle_dir: Path = BUNDLE_DIR) -> list:
    return list(load_manifest(bundle_dir)["layouts"])


@lru_cache(maxsize=8)
def _load_layout(layout_id: str, bundle_dir: Path) -> dict:
    if layout_id not in load_manifest(bundle_dir)["layouts"]:
        raise PermissionError(f"layout {layout_id!r} is not in the app bundle (DEV-validation only)")
    with gzip.open(layout_file(layout_id, bundle_dir), "rt", encoding="utf-8") as f:
        return json.load(f)


def load_episode(layout_id: str, arm: str, tta: bool, bundle_dir: Path = BUNDLE_DIR) -> dict:
    """The ``episode()`` dict of one layout / arm / view, exactly as the live path would return it (a fresh copy)."""
    eps = _load_layout(layout_id, Path(bundle_dir))["episodes"]
    key = episode_key(arm, tta)
    if key not in eps:
        raise KeyError(f"no bundled episode for {key!r}")
    return json.loads(json.dumps(eps[key]))
