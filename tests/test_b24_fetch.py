"""scripts/fetch_checkpoints.py: verification and placement, offline (``--src-dir``); the real download is never exercised by the suite."""
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from rl import explain as X

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fetch_checkpoints", ROOT / "scripts" / "fetch_checkpoints.py")
F = importlib.util.module_from_spec(_spec)
sys.modules["fetch_checkpoints"] = F
_spec.loader.exec_module(F)


def _assets(d: Path, corrupt: str | None = None):
    sums = []
    for name in F.ASSETS:
        (d / name).write_bytes(b"payload-" + name.encode())
        sums.append(f"{hashlib.sha256((d / name).read_bytes()).hexdigest()}  {name}")
    (d / F.SUMS).write_text("\n".join(sums) + "\n")
    if corrupt:
        (d / corrupt).write_bytes(b"tampered")


def test_assets_land_exactly_where_the_code_loads_them():
    expect = {X.MODELS / run / ("final.zip" if algo == "ppo" else "final.pt") for run, algo in X.DISPLAY_MODELS.values()}
    assert {F.ROOT / "models_v2" / rel for rel in F.ASSETS.values()} == expect


def test_fetch_verifies_then_places(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "models"
    src.mkdir()
    _assets(src)
    placed = F.fetch(dest, src)
    assert sorted(p.relative_to(dest).as_posix() for p in placed) == sorted(F.ASSETS.values()) and all(p.read_bytes().startswith(b"payload-") for p in placed)


def test_a_hash_mismatch_places_nothing(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "models"
    src.mkdir()
    _assets(src, corrupt="grid_ddqn_s0.pt")
    with pytest.raises(SystemExit, match="SHA256 mismatch"):
        F.fetch(dest, src)
    assert not dest.exists()


def test_sums_parser_and_missing_entry(tmp_path):
    assert F.parse_sums("ab  x.zip\n\ncd *y.pt\n") == {"x.zip": "ab", "y.pt": "cd"}
    src = tmp_path / "s"
    src.mkdir()
    _assets(src)
    (src / F.SUMS).write_text("")
    with pytest.raises(SystemExit, match="no entry"):
        F.fetch(tmp_path / "m", src)
