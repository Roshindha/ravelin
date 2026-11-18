"""B18a: Streamlit app shell + replay / decision lens (docs/SPEC_V2.md, Gate changelog row (ll)). Uses streamlit.testing.v1.AppTest; DEV layouts only."""
import hashlib
import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from breachlens import evaluator as EV
from breachlens import pool as P
from breachlens.env import BreachEnv
from rl import explain as X

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
LEDGER_SHA = "20a8b2877c91c9cf45c3692617ad3bd08d860ef0f7783ca3610a9a3fc44c9dbd"
DEFAULT_LAYOUT = "spine_pods-300000056"                  # row (ll) (c)
TIMEOUT = 300


def new_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def replay_data(arm: str, tta: bool) -> dict:
    f = ROOT / "docs" / "replays" / f"spine_pods__{arm}__{'tta' if tta else 'plain'}.html"
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', f.read_text(), re.S)
    return json.loads(m.group(1))


def pos_slider(at: AppTest):
    return at.slider[0]


def test_default_view_matches_row_ll():
    at = new_app()
    assert not at.exception
    assert at.selectbox(key="split").value == "DEV-validation"
    assert at.selectbox(key="layout_DEV-validation").value == DEFAULT_LAYOUT
    assert at.selectbox(key="arm").value == "ppo_plain" and X.DISPLAY_MODELS["ppo_plain"][0] == "grid_ppo_plain_s0"
    assert at.radio(key="view").value == "TTA"
    assert pos_slider(at).value == 0
    lens = at.session_state["_lens"]
    assert lens["kind"] == "decision" and lens["pos"] == 0
    assert not any("IN-SAMPLE" in m.value for m in at.markdown)
    assert any("DEMO" in m.value for m in at.markdown)


def test_dev_train_shows_in_sample_badge():
    at = new_app()
    at.selectbox(key="split").select("DEV-train").run()
    assert not at.exception
    assert any("IN-SAMPLE" in m.value for m in at.markdown)
    ids = at.selectbox(key="layout_DEV-train").options
    assert ids == EV.dev_layout_ids("DEV-train")
    assert at.session_state["_lens"]["pos"] == 0


def test_no_way_to_select_a_non_dev_split_or_non_seed0_model():
    at = new_app()
    assert list(at.selectbox(key="split").options) == ["DEV-validation", "DEV-train"]
    assert list(at.selectbox(key="arm").options) == ["grid_ppo_plain_s0", "grid_ppo_d4_s0", "grid_ddqn_s0"]       # displayed names: seed 0 only
    assert list(at.radio(key="view").options) == ["TTA", "plain"]
    for bad in ("FINAL-familiar", "FINAL-transfer", "LAB-DESIGN", "TUNE-val", "DEV-test"):          # an unlisted value is refused when the script runs
        with pytest.raises(ValueError):
            new_app().selectbox(key="split").select(bad).run()
    with pytest.raises(ValueError):
        new_app().selectbox(key="arm").select("grid_ppo_plain_s1").run()
    # layout options are exactly the DEV ids of the chosen split (30 validation ids; every one is a DEV id)
    assert at.selectbox(key="layout_DEV-validation").options == EV.dev_layout_ids("DEV-validation")
    assert len(at.selectbox(key="layout_DEV-validation").options) == 30
    with pytest.raises(ValueError):
        new_app().selectbox(key="layout_DEV-validation").select("labyrinth-999").run()
    # no model or checkpoint other than seed 0 can be named by the source
    src = Path(APP).read_text()
    assert not re.search(r"_s[1-9]\b", src) and "FINAL" not in src.replace("never FINAL", "") .replace("FINAL / transfer", "")
    assert "load_pool_records" not in src and "role=" not in src


def test_no_sidebar_widgets_and_theme():
    at = new_app()
    assert len(at.sidebar) == 0 and not at.sidebar.children
    src = Path(APP).read_text()
    assert "st.sidebar" not in src and 'initial_sidebar_state="collapsed"' in src
    cfg = (ROOT / ".streamlit" / "config.toml").read_text()
    assert 'base = "dark"' in cfg and 'backgroundColor = "#07142b"' in cfg and 'font = "monospace"' in cfg


@pytest.mark.parametrize("tta", [True, False])
def test_decision_lens_equals_b16b_replay_and_b15_eval_path(tta, monkeypatch):
    import scripts_b15_final as F
    at = new_app()
    at.radio(key="view").set_value("TTA" if tta else "plain").run()
    assert not at.exception
    rep = replay_data("ppo_plain", tta)
    n_dec = len(rep["steps"])
    # B15 eval path: scripts_b15_final._task with env.step recorded
    rec, layout = EV.load_dev_layout("DEV-validation", layout_id=DEFAULT_LAYOUT)
    taken, orig = [], BreachEnv.step

    def rec_step(self, action):
        taken.append(int(action))
        return orig(self, action)
    run, algo = X.DISPLAY_MODELS["ppo_plain"]
    with monkeypatch.context() as mp:
        mp.setattr(BreachEnv, "step", rec_step)
        row = F._task(({"kind": "learner", "run": run, "algo": algo, "tta": tta, "name": "x"}, rec, layout, {"models_dir": str(X.MODELS)}))[0]
    assert len(taken) == n_dec
    shown = []
    for p in range(n_dec):
        at.slider[0].set_value(p).run()
        assert not at.exception
        lens = at.session_state["_lens"]
        assert lens["kind"] == "decision" and lens["pos"] == p
        shown.append(lens["chosen_action"])
        assert lens["chosen_action"] == rep["steps"][p]["action"]           # = the committed B16b replay
        assert lens["value"] == rep["steps"][p]["value"]
    assert shown == taken                                                    # = the B15 evaluator path's actions
    final = at.session_state["_lens"]["final_outcome"]
    assert final["score"] == row["score"] == rep["outcome"]["score"]
    assert final["core"] == bool(row["breach"]) == rep["outcome"]["core"]
    # last battle tick: the lens reports the true final state
    at.slider[0].set_value(n_dec + len(rep["battle"]["frames"]) - 1).run()
    lens = at.session_state["_lens"]
    assert lens["kind"] == "battle" and lens["tick"] == final["ticks"] and lens["core_destroyed"] == final["core"]
    assert any("Final (true) outcome" in m.value for m in at.markdown)


def test_ddqn_keeps_not_a_policy_and_value_label():
    at = new_app()
    at.selectbox(key="arm").select("ddqn").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "NOT a policy" in text and "not a policy" in text
    assert "SHAPED return" in text
    assert at.session_state["_lens"]["chosen_action"] == replay_data("ddqn", True)["steps"][0]["action"]


def test_headline_is_verbatim_from_results_md_and_no_other_aggregates():
    at = new_app()
    txt = (ROOT / "docs" / "RESULTS.md").read_text()
    a = txt.index("## Plain-language summary")
    quoted = txt[a:txt.index("\n## ", a + 5)].strip()
    assert any(m.value == quoted for m in at.markdown)


def test_ledger_sha_unchanged():
    assert hashlib.sha256(P.LEDGER_PATH.read_bytes()).hexdigest() == LEDGER_SHA
    new_app()
    assert hashlib.sha256(P.LEDGER_PATH.read_bytes()).hexdigest() == LEDGER_SHA


def test_screenshot_rule_positions_and_files():
    """Row (ll) addendum: screenshot 1 = first DEPLOY step, screenshot 2 = battle tick 120 (or the last tick); files are named by position."""
    from app import render as R
    at = new_app()                                           # default layout / arm / view
    assert (at.selectbox(key="layout_DEV-validation").value, at.selectbox(key="arm").value, at.radio(key="view").value) == (DEFAULT_LAYOUT, "ppo_plain", "TTA")
    ep = replay_data("ppo_plain", True)
    pos = R.screenshot_positions(ep)
    n_dec = len(ep["steps"])
    assert ep["steps"][pos["pos_deploy"]]["phase"] == "DEPLOY" and all(s["phase"] != "DEPLOY" for s in ep["steps"][:pos["pos_deploy"]])
    assert pos["tick"] == min(120, ep["outcome"]["ticks"]) and pos["pos_battle"] == n_dec + pos["tick"]
    assert (pos["pos_deploy"], pos["pos_battle"], pos["tick"]) == (9, 137, 120)            # the documented values
    shots = sorted(p.name for p in (ROOT / "docs" / "screenshots").glob("*.png"))
    assert shots == [f"b18a_pos{pos['pos_battle']}_battle_tick_{pos['tick']}.png", f"b18a_pos{pos['pos_deploy']}_first_deploy_step.png"]
    # the app shows exactly those frames at those positions
    at.slider[0].set_value(pos["pos_deploy"]).run()
    assert at.session_state["_lens"]["kind"] == "decision" and ep["steps"][pos["pos_deploy"]]["phase"] == "DEPLOY"
    at.slider[0].set_value(pos["pos_battle"]).run()
    assert at.session_state["_lens"]["kind"] == "battle" and at.session_state["_lens"]["tick"] == 120
