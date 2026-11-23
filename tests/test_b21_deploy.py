"""B21: the deployable lightweight app (docs/SPEC_V2.md, Gate changelog row (qq)). No training, no new results.

* the app bundle (data/app_bundle) has provenance and equals the live explain / episode output for every DEV-validation layout x seed-0 arm x {plain, TTA}
  (actions and every integer exactly; floats within 1e-9);
* the app runs with the torch import blocked (reads the bundle) and with RAVELIN_DEPLOY=1 (never loads a model);
* on deploy the duel uses the simulator + duel cache only and visitors' plays are not logged (no file is opened for writing);
* requirements.txt is only what the deployed app imports (pinned), requirements-dev.txt adds the research stack.
"""
import builtins
import gzip
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import bundle as BU
from breachlens import evaluator as EV
from breachlens.config import sim_version
from breachlens.score import SCORE_VERSION
from rl import explain as X
from tests.test_b18eb_app import APP, LAYOUT, TIMEOUT, _tracked_state, app_script, click, game, pos_key

ROOT = Path(__file__).resolve().parents[1]
HAVE_MODELS = all((X.MODELS / run / ("final.zip" if algo == "ppo" else "final.pt")).exists() for run, algo in X.DISPLAY_MODELS.values())
FLOAT_TOL = 1e-9


@pytest.fixture(autouse=True)
def _fresh_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def assert_same(a, b, path="ep"):
    """Strings, bools, ints and structure exact; floats within 1e-9."""
    assert type(a) is type(b) or (isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool)), (path, a, b)
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            assert_same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert abs(a - b) <= FLOAT_TOL, (path, a, b)
    else:
        assert a == b, (path, a, b)


# ============================================================================================================================ the bundle
def test_bundle_manifest_provenance_and_file_hashes():
    m = BU.load_manifest()
    prov = m["provenance"]
    assert len(prov["code_commit"]) == 40 and "dirty" not in prov["code_commit"]
    assert prov["sim_version"] == sim_version() and prov["score_version"] == SCORE_VERSION
    assert set(m["models"]) == set(X.DISPLAY_MODELS) == set(m["arms"])
    for arm, (run, algo) in X.DISPLAY_MODELS.items():
        assert m["models"][arm]["run"] == run and m["models"][arm]["algo"] == algo and len(m["models"][arm]["sha256"]) == 64
    ids = EV.dev_layout_ids(BU.BUNDLE_SPLIT, entry_point="tests.test_b21_deploy")
    assert len(ids) == 30 and sorted(m["layouts"]) == sorted(ids)
    total = 0
    for lid, row in m["layouts"].items():
        f = BU.layout_file(lid)
        assert hashlib.sha256(f.read_bytes()).hexdigest() == row["sha256"] and f.stat().st_size == row["bytes"]
        total += row["bytes"]
        with gzip.open(f, "rt") as fh:
            eps = json.load(fh)["episodes"]
        assert sorted(eps) == sorted(BU.episode_key(a, t) for a in X.DISPLAY_MODELS for t in (False, True))
    assert total == m["total_bytes"] and total < 5 * 1024 * 1024


def test_bundle_refuses_other_splits_and_unknown_layouts():
    with pytest.raises(PermissionError):
        BU.load_episode("not-a-layout", "ppo_plain", False)
    train_id = EV.dev_layout_ids("DEV-train", entry_point="tests.test_b21_deploy")[0]
    with pytest.raises(PermissionError):
        BU.load_episode(train_id, "ppo_plain", False)


def test_build_script_refuses_a_dirty_tree(monkeypatch, tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import build_app_bundle as BB
        monkeypatch.setattr(BB.M, "tree_dirty", lambda: True)
        with pytest.raises(SystemExit, match="dirty tree"):
            BB.build(tmp_path / "b")
        assert not (tmp_path / "b").exists()
    finally:
        sys.path.remove(str(ROOT / "scripts"))


@pytest.mark.skipif(not HAVE_MODELS, reason="seed-0 checkpoints not present locally")
def test_bundle_equals_live_for_every_layout_arm_view():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import build_app_bundle as BB
        loaded = {arm: X.load_display_model(arm) for arm in X.DISPLAY_MODELS}
        n = 0
        for lid in BU.layout_ids():
            for arm in X.DISPLAY_MODELS:
                for tta in (False, True):
                    live = BB.live_episode(loaded, arm, BU.BUNDLE_SPLIT, lid, tta)
                    got = BU.load_episode(lid, arm, tta)
                    assert [s["action"] for s in got["steps"]] == [s["action"] for s in live["steps"]], (lid, arm, tta)      # exact
                    assert_same(got, json.loads(json.dumps(live)), f"{lid}|{arm}|{tta}")
                    n += 1
        assert n == 30 * 3 * 2
    finally:
        sys.path.remove(str(ROOT / "scripts"))


# ============================================================================================================================ the app without torch
def test_import_closure_of_the_deployed_app_has_no_research_stack():
    code = textwrap.dedent("""
        import sys
        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in {"torch", "stable_baselines3", "sb3_contrib", "pandas", "matplotlib", "scipy"}:
                    raise ImportError("blocked: " + name)
        sys.meta_path.insert(0, Block())
        sys.path[:0] = [%r, %r]
        import app.bundle, app.render, app.theme, app.game_view, app.icons, app.board_component
        import breachlens.duel, breachlens.evaluator, rl.explain, make_replays
        bad = [m for m in sys.modules if m.split(".")[0] in {"torch", "stable_baselines3", "sb3_contrib", "pandas", "matplotlib"}]
        assert not bad, bad
        assert app.bundle.deploy_mode()          # torch is not importable here
        print("ok")
    """) % (str(ROOT), str(ROOT / "scripts"))
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]


def watch(at):
    at.radio(key="mode_sel").set_value("Watch the AI").run()
    assert not at.exception
    return at


def new_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def test_app_runs_with_torch_blocked_and_reads_the_bundle(monkeypatch):
    for mod in ("torch", "stable_baselines3", "sb3_contrib"):
        monkeypatch.setitem(sys.modules, mod, None)                    # `import torch` now raises ImportError
    monkeypatch.delenv("RAVELIN_DEPLOY", raising=False)
    assert BU.deploy_mode() and not BU.torch_available()
    calls = []
    real = BU.load_episode
    monkeypatch.setattr(BU, "load_episode", lambda *a, **k: calls.append(a) or real(*a, **k))
    monkeypatch.setattr(X, "load_display_model", lambda *a, **k: pytest.fail("a model was loaded in deploy mode"))
    at = watch(new_app())
    assert calls, "the bundle was not used"
    assert at.selectbox(key="w_split").options == ["DEV-validation"] and at.selectbox(key="w_layout_DEV-validation").value == LAYOUT
    lens = at.session_state["w_lens"]
    ep = real(LAYOUT, "ppo_plain", True)
    assert lens["kind"] == "decision" and lens["chosen_action"] == ep["steps"][0]["action"] and lens["top"] == [t["a"] for t in ep["steps"][0]["top"]]
    for arm in X.DISPLAY_MODELS:                                       # every arm, both views, the last stop (the battle)
        at.selectbox(key="w_arm").select(arm).run()
        for view in ("plain", "TTA"):
            at.radio(key="w_view").set_value(view).run()
            assert not at.exception
            n = len(real(LAYOUT, arm, view == "TTA")["steps"])
            at.slider(key=pos_key(arm=arm, tta=view == "TTA")).set_value(n).run()
            assert not at.exception and at.session_state["w_lens"]["kind"] == "battle"


def test_deploy_env_never_loads_a_model(monkeypatch):
    monkeypatch.setenv("RAVELIN_DEPLOY", "1")
    monkeypatch.setattr(X, "load_display_model", lambda *a, **k: pytest.fail("a model was loaded with RAVELIN_DEPLOY=1"))
    at = watch(new_app())
    assert at.selectbox(key="w_split").options == ["DEV-validation"]
    assert at.session_state["w_lens"]["kind"] == "decision"


# ============================================================================================================================ the duel on deploy
def test_deploy_duel_uses_cache_and_logs_nothing(arena, monkeypatch, tmp_path):
    runs = tmp_path / "runs.jsonl"
    monkeypatch.setenv("RAVELIN_DEPLOY", "1")
    monkeypatch.setenv("BREACHLENS_DUEL_RUNS", str(runs))              # would be honoured by a non-deploy run; deploy must ignore it
    repo_file = ROOT / "results_v2" / "duel_human_runs.jsonl"
    size = repo_file.stat().st_size if repo_file.exists() else None
    written, real_open = [], builtins.open

    def spy(f, mode="r", *a, **k):
        if any(c in str(mode) for c in "wax+"):
            written.append(str(f))
        return real_open(f, mode, *a, **k)
    monkeypatch.setattr(builtins, "open", spy)
    before = _tracked_state()
    at = new_app()
    assert game(at)._runs_path is None
    app_script(at)
    assert game(at).finished
    click(at, "g_report_btn")                                          # the cache is read here, from the file, with no torch and no model
    assert "g_rows" in at.session_state and len(at.session_state["g_rows"]) >= 5
    assert not runs.exists() and not any(w.endswith(".jsonl") for w in written), written
    assert (repo_file.stat().st_size if repo_file.exists() else None) == size
    assert _tracked_state() == before


# ============================================================================================================================ requirements and config
def _reqs(name):
    out = {}
    for ln in (ROOT / name).read_text().splitlines():
        ln = ln.split("#")[0].strip()
        if ln and not ln.startswith("-r"):
            pkg, _, ver = ln.partition("==")
            assert ver and "=" not in ver and "<" not in ver and ">" not in ver, f"{name}: {ln!r} is not pinned with =="
            out[pkg.lower().replace("_", "-")] = ver
    return out


def test_requirements_are_pinned_and_split():
    app, dev = _reqs("requirements.txt"), _reqs("requirements-dev.txt")
    assert set(app) == {"numpy", "gymnasium", "streamlit"}                      # everything the deployed app imports directly, nothing else
    assert not {"torch", "stable-baselines3", "sb3-contrib", "pandas", "matplotlib", "pytest"} & set(app)
    assert any(ln.strip() == "-r requirements.txt" for ln in (ROOT / "requirements-dev.txt").read_text().splitlines())
    assert {"torch", "stable-baselines3", "sb3-contrib", "pytest", "pandas", "matplotlib"} <= set(dev) and not set(app) & set(dev)
    import importlib.metadata as md
    for pkg, ver in {**app, **dev}.items():
        assert md.version(pkg) == ver, f"{pkg}: pinned {ver}, installed {md.version(pkg)}"


def test_cloud_settings():
    import tomllib
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert cfg["server"]["headless"] is True and cfg["server"]["enableStaticServing"] is True and cfg["server"]["fileWatcherType"] == "none"
    assert cfg["browser"]["gatherUsageStats"] is False
    assert (ROOT / ".python-version").read_text().strip() == "3.11"
