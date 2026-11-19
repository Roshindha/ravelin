"""B23: history rewritten to remove commit-message trailers only; docs/commit_map.tsv maps every old commit id to the new one."""
from __future__ import annotations

import subprocess
from pathlib import Path

from scripts import b14_grid as G

ROOT = Path(__file__).resolve().parent.parent
MAP = ROOT / "docs" / "commit_map.tsv"


def _rows():
    lines = MAP.read_text().splitlines()
    assert lines[0].startswith("# History rewritten 2026-11-19") and lines[1].startswith("# Frozen files reference the old ids")
    return [l.split("\t") for l in lines[2:]]


def _git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)


def test_map_is_well_formed_one_to_one():
    rows = _rows()
    assert len(rows) == 142
    assert all(len(a) == 40 and len(b) == 40 for a, b in rows)
    assert len({a for a, _ in rows}) == len(rows) and len({b for _, b in rows}) == len(rows)


def test_new_ids_exist_and_old_trailers_are_gone():
    if _git("rev-parse", "--is-shallow-repository").stdout.strip() == "true":
        return
    for _, new in _rows():
        assert _git("cat-file", "-e", new + "^{commit}").returncode == 0
    msgs = _git("log", "--all", "--format=%B").stdout.lower()
    assert "co-authored-by" not in msgs and "claude-session" not in msgs and "generated with [claude code]" not in msgs


def test_resolve_commit_translates_old_ids_by_full_id_and_prefix():
    old, new = _rows()[10]
    assert G.resolve_commit(old) == new and G.resolve_commit(old[:7]) == new
    assert G.resolve_commit("0" * 40) == "0" * 40


def test_launcher_state_with_an_old_commit_id_still_diffs_against_the_mapped_commit():
    old, new = _rows()[-1]
    assert G.code_changed_since(old) == G.code_changed_since(new)
