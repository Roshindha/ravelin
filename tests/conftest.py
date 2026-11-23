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


@pytest.fixture
def arena(monkeypatch, tmp_path):
    """B24: stand-in for the v2 component (the real JS is exercised in headless Chrome by tests/test_b24_arena_js.py and in the screenshots). It records what the app mounts
    and, when ``st.session_state['_sim_fight']`` / ``['_sim_illegal']`` hold a payload, delivers it exactly like the browser does: the component value lands in
    ``st.session_state[key]`` and the app's ``on_<trigger>_change`` callback runs. Everything from there on is the app's real code. Human runs go to a temp file."""
    import types

    import streamlit as st
    monkeypatch.setenv("BREACHLENS_DUEL_RUNS", str(tmp_path / "runs.jsonl"))
    mounted = {}

    def register(name, **kw):
        def mount(*, key, data=None, on_tile_change=None, on_fight_change=None, on_illegal_change=None, **_):
            mounted[key] = {"data": data, "name": name, "callbacks": {"fight": on_fight_change is not None, "illegal": on_illegal_change is not None}, **kw}
            for trig, cb in (("fight", on_fight_change), ("illegal", on_illegal_change)):
                sim = st.session_state.pop(f"_sim_{trig}", None)
                if sim is not None and cb is not None:
                    st.session_state[key] = {trig: sim}
                    cb()
            return types.SimpleNamespace()
        return mount
    monkeypatch.setattr(st.components.v2, "component", register)
    st.cache_resource.clear()
    yield mounted
    st.cache_resource.clear()
