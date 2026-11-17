"""Shared fixtures. B15d: the real FINAL ledger now holds one finished eval session, so every launcher test of a "would start" path runs against a
synthetic build-only ledger (5 build entries, 0 eval). The real ledger is never modified by any test."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REAL_LEDGER = Path(__file__).resolve().parents[1] / "data" / "v2" / "final_access.log"


@pytest.fixture
def build_only_ledger(tmp_path):
    """Path of a temp ledger holding only the 5 build entries of the real one (byte-identical lines)."""
    lines = [ln for ln in REAL_LEDGER.read_text().splitlines() if ln.strip() and json.loads(ln)["purpose"] == "build"]
    assert len(lines) == 5
    p = tmp_path / "build_only_final_access.log"
    p.write_text("\n".join(lines) + "\n")
    return p
