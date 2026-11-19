"""B18e-b: the finished game app ("Ravelin", app/streamlit_app.py) and the guarantees ported from the deleted legacy app (docs/SPEC_V2.md, Gate changelog row (oo)).
AppTest; DEV-validation / DEV-train only; human runs go to a temp file; no training, no FINAL / transfer / LAB-DESIGN.

Mapping, legacy test -> test here (every guarantee of the deleted tests/test_b18a_app.py and tests/test_b18b_duel_app.py is asserted against the NEW app):

  b18a test_default_view_matches_row_ll                  -> test_watch_default_view_matches_row_ll
  b18a test_dev_train_shows_in_sample_badge              -> test_dev_train_shows_in_sample_badge
  b18a test_no_way_to_select_a_non_dev_split_or_non_seed0_model -> test_no_way_to_select_a_non_dev_split_or_non_seed0_model
  b18a test_no_sidebar_widgets_and_theme                 -> test_no_sidebar_and_theme
  b18a test_decision_lens_equals_b16b_replay_and_b15_eval_path -> test_decision_lens_equals_b16b_replay_and_b15_eval_path (tta, plain)
  b18a test_ddqn_keeps_not_a_policy_and_value_label      -> test_ddqn_keeps_not_a_policy_and_value_label
  b18a test_headline_is_verbatim_from_results_md_and_no_other_aggregates -> test_headline_is_verbatim_from_results_md
  b18a test_ledger_sha_unchanged                         -> test_ledger_sha_unchanged
  b18a test_screenshot_rule_positions_and_files          -> test_watch_screenshot_rule_first_deploy_step_with_heatmap_on and test_screenshots_are_the_ravelin_set
  b18b test_scripted_human_matches_engine_directly       -> test_scripted_human_matches_engine_directly (dropdown and clicks)
  b18b test_nothing_about_opponents_before_resolution_and_cache_unread -> test_nothing_about_opponents_before_resolution_and_cache_unread
  b18b test_table_equals_cache_with_oracle_label_and_best_single -> test_table_equals_cache_with_oracle_label_and_best_single
  b18b test_illegal_actions_show_message_and_change_nothing -> test_illegal_actions_show_message_and_change_nothing (+ test_illegal_click_*)
  b18b test_undo_in_buy_and_refused_after_end_buy        -> test_undo_in_buy_and_refused_after_end_buy
  b18b test_only_dev_validation_ids_selectable           -> test_only_dev_validation_ids_selectable
  b18b test_human_run_appended_to_ignored_file_and_nothing_tracked_changes -> test_human_run_appended_to_ignored_file_and_nothing_tracked_changes
  b18b test_new_duel_resets_cleanly                      -> test_new_game_resets_cleanly
  b18b test_two_sessions_do_not_share_state              -> test_two_sessions_do_not_share_state
  b18b test_disclaimer_visible_and_no_sidebar            -> test_disclaimer_visible_once_and_no_sidebar
  b18b test_ledger_sha_unchanged                         -> test_ledger_sha_unchanged
"""
import builtins
import hashlib
import json
import pathlib
import re
import subprocess
import types
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import board_component as BC
from app import game_view as GV
from app import icons as IC
from app import render as R
from breachlens import duel as D
from breachlens import evaluator as EV
from breachlens import pool as P
from breachlens.actions import BORDER, encode_deploy
from breachlens.env import BreachEnv
from rl import explain as X

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
LEDGER_SHA = "20a8b2877c91c9cf45c3692617ad3bd08d860ef0f7783ca3610a9a3fc44c9dbd"
LAYOUT = "spine_pods-300000056"
BUYS = ["sniper"] * 4 + ["bulwark"] * 2                      # the row (mm) (g) script
MM_ACTIONS = [1, 1, 1, 1, 2, 2, 265, 57, 58, 59, 60, 113, 114]
TIMEOUT = 300


# ----------------------------------------------------------------------------------------------------------------------------- fixtures / helpers
@pytest.fixture
def runs(tmp_path, monkeypatch):
    p = tmp_path / "runs.jsonl"
    monkeypatch.setenv("BREACHLENS_DUEL_RUNS", str(p))
    return p


@pytest.fixture
def clicks(runs, monkeypatch):
    """Replace the v2 component by a stand-in that records what the app mounts and, when ``st.session_state['_sim_click']`` holds a border index, delivers it exactly
    like the browser does: the component's value lands in ``st.session_state[key]`` and the app's ``on_tile_change`` callback runs. The routing from there (callback ->
    ``Duel.deploy`` -> action 5 + 52*u + b) is the app's real code; the real JS is exercised in real Chrome (docs/dev_log.md) and checked statically below."""
    mounted = {}

    def register(name, **kw):
        def mount(*, key, data=None, on_tile_change=None, **_):
            mounted[key] = {"data": data, "has_callback": on_tile_change is not None, "name": name, **kw}
            sim = st.session_state.pop("_sim_click", None)
            if sim is not None and on_tile_change is not None:
                st.session_state[key] = types.SimpleNamespace(tile=sim)
                on_tile_change()
            return types.SimpleNamespace(tile=None)
        return mount
    monkeypatch.setattr(st.components.v2, "component", register)
    st.cache_resource.clear()
    yield mounted
    st.cache_resource.clear()


def new_app(query: dict | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    for k, v in (query or {}).items():
        at.query_params[k] = v
    return at.run()


def watch_app() -> AppTest:
    at = new_app()
    at.radio(key="mode_sel").set_value("Watch the AI").run()
    assert not at.exception
    return at


def md(at) -> str:
    return " ".join(m.value for m in at.markdown)


def texts(at) -> str:
    return " ".join([m.value for m in at.markdown] + [e.value for e in at.error] + [str(x.value) for x in at.text])


def game(at):
    return at.session_state["g_duel"]


def click(at, key):
    at.button(key=key).click().run()
    assert not at.exception


def lowest_tile(duel, unit):
    m = duel.env.action_masks()
    u = D.parse_unit(unit)
    return next(b for b in range(52) if m[encode_deploy(u, b)])


def app_script(at, buys=BUYS):
    """The row (mm) script through the buttons and the tile dropdown."""
    for u in buys:
        click(at, f"g_buy_{u}")
    click(at, "g_end")
    for u in buys:                                           # deploy in the order bought, each on the lowest legal tile
        g = game(at)
        at.selectbox(key="g_unit").select(u)
        at.selectbox(key="g_tile").select(lowest_tile(g, u))
        click(at, "g_deploy")
        if game(at).finished:
            break
    if not game(at).finished:
        click(at, "g_wait")


def click_tile(at, b):
    """A click / tap on border tile b of the board component (two runs: the callback runs mid-run in the stand-in, the second run redraws everything)."""
    at.session_state["_sim_click"] = b
    at.run()
    assert not at.exception
    at.run()
    assert not at.exception


def click_script(at, buys=BUYS):
    """The row (mm) script with the board: buy by buttons, END BUY, then select each unit and click its lowest legal tile."""
    for u in buys:
        click(at, f"g_buy_{u}")
    click(at, "g_end")
    for u in buys:
        g = game(at)
        b = lowest_tile(g, u)
        at.selectbox(key="g_unit").select(u).run()
        click_tile(at, b)
        if game(at).finished:
            break
    if not game(at).finished:
        click(at, "g_wait")


def direct_script(buys=BUYS, path=None):
    d = D.Duel(LAYOUT, human_runs_path=path)
    for u in buys:
        d.buy(u)
    d.end_buy()
    for u in buys:
        d.deploy(u, lowest_tile(d, u))
        if d.finished:
            break
    if not d.finished:
        d.wait()
    return d


def replay_data(arm: str, tta: bool) -> dict:
    f = ROOT / "docs" / "replays" / f"spine_pods__{arm}__{'tta' if tta else 'plain'}.html"
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', f.read_text(), re.S)
    return json.loads(m.group(1))


def pos_key(split="DEV-validation", layout=LAYOUT, arm="ppo_plain", tta=True):
    return f"w_pos|{split}|{layout}|{arm}|{tta}"


def iframes(at) -> list:
    """The srcdoc of every st.iframe on the page (the battle player)."""
    return [p.proto.srcdoc for p in _walk(at.main) if getattr(getattr(p, "proto", None), "srcdoc", "")]


def _walk(node):
    for _, ch in node.children.items():
        yield ch
        if hasattr(ch, "children"):
            yield from _walk(ch)


# ============================================================================================================================ Watch the AI (ported from test_b18a_app)
def test_watch_default_view_matches_row_ll():
    at = watch_app()
    assert at.selectbox(key="w_split").value == "DEV-validation"
    assert at.selectbox(key="w_layout_DEV-validation").value == LAYOUT
    assert at.selectbox(key="w_arm").value == "ppo_plain" and X.DISPLAY_MODELS["ppo_plain"][0] == "grid_ppo_plain_s0"
    assert at.radio(key="w_view").value == "TTA"
    assert at.slider(key=pos_key()).value == 0
    lens = at.session_state["w_lens"]
    assert lens["kind"] == "decision" and lens["pos"] == 0
    assert not any("IN-SAMPLE" in m.value for m in at.markdown)
    assert any("Demo, not a result" in m.value for m in at.markdown)


def test_dev_train_shows_in_sample_badge():
    at = watch_app()
    at.selectbox(key="w_split").select("DEV-train").run()
    assert not at.exception
    assert any("IN-SAMPLE" in m.value for m in at.markdown)
    assert at.selectbox(key="w_layout_DEV-train").options == EV.dev_layout_ids("DEV-train")
    assert at.session_state["w_lens"]["pos"] == 0


def test_no_way_to_select_a_non_dev_split_or_non_seed0_model():
    at = watch_app()
    assert list(at.selectbox(key="w_split").options) == ["DEV-validation", "DEV-train"]
    assert list(at.selectbox(key="w_arm").options) == ["grid_ppo_plain_s0", "grid_ppo_d4_s0", "grid_ddqn_s0"]       # displayed names: seed 0 only
    assert list(at.radio(key="w_view").options) == ["TTA", "plain"]
    for bad in ("FINAL-familiar", "FINAL-transfer", "LAB-DESIGN", "TUNE-val", "DEV-test"):          # an unlisted value is refused when the script runs
        with pytest.raises(ValueError):
            watch_app().selectbox(key="w_split").select(bad).run()
    with pytest.raises(ValueError):
        watch_app().selectbox(key="w_arm").select("grid_ppo_plain_s1").run()
    assert at.selectbox(key="w_layout_DEV-validation").options == EV.dev_layout_ids("DEV-validation")
    assert len(at.selectbox(key="w_layout_DEV-validation").options) == 30
    with pytest.raises(ValueError):
        watch_app().selectbox(key="w_layout_DEV-validation").select("labyrinth-999").run()
    src = Path(APP).read_text()                                  # no model or checkpoint other than seed 0 can be named by the source
    assert not re.search(r"_s[1-9]\b", src) and "FINAL" not in src and "load_pool_records" not in src and not re.search(r"(?<!\[)\brole\s*=", src)      # (a CSS attribute selector [role=...] is not the pool API's role= argument)


def test_no_sidebar_and_theme():
    for at in (new_app(), watch_app()):
        assert len(at.sidebar) == 0 and not at.sidebar.children
    src = Path(APP).read_text()
    assert "st.sidebar" not in src and 'initial_sidebar_state="collapsed"' in src
    cfg = (ROOT / ".streamlit" / "config.toml").read_text()
    assert 'base = "light"' in cfg and 'backgroundColor = "#FEFDFE"' in cfg and 'font = "Atkinson Hyperlegible Next"' in cfg


@pytest.mark.parametrize("tta", [True, False])
def test_decision_lens_equals_b16b_replay_and_b15_eval_path(tta, monkeypatch):
    from scripts import b15_final as F
    at = watch_app()
    at.radio(key="w_view").set_value("TTA" if tta else "plain").run()
    assert not at.exception
    rep = replay_data("ppo_plain", tta)
    n_dec = len(rep["steps"])
    rec, layout = EV.load_dev_layout("DEV-validation", layout_id=LAYOUT)       # the B15 eval path with env.step recorded
    taken, orig = [], BreachEnv.step

    def rec_step(self, action):
        taken.append(int(action))
        return orig(self, action)
    run, algo = X.DISPLAY_MODELS["ppo_plain"]
    with monkeypatch.context() as mp:
        mp.setattr(BreachEnv, "step", rec_step)
        row = F._task(({"kind": "learner", "run": run, "algo": algo, "tta": tta, "name": "x"}, rec, layout, {"models_dir": str(X.MODELS)}))[0]
    assert len(taken) == n_dec
    key, shown = pos_key(tta=tta), []
    for p in range(n_dec):
        at.slider(key=key).set_value(p).run()
        assert not at.exception
        lens = at.session_state["w_lens"]
        assert lens["kind"] == "decision" and lens["pos"] == p
        shown.append(lens["chosen_action"])
        assert lens["chosen_action"] == rep["steps"][p]["action"]           # = the committed B16b replay
        assert lens["value"] == rep["steps"][p]["value"]
    assert shown == taken                                                    # = the B15 evaluator path's actions
    final = at.session_state["w_lens"]["final_outcome"]
    assert final["score"] == row["score"] == rep["outcome"]["score"]
    assert final["core"] == bool(row["breach"]) == rep["outcome"]["core"]
    at.slider(key=key).set_value(n_dec).run()                                # the last stop: the battle, with the true final outcome
    lens = at.session_state["w_lens"]
    assert lens["kind"] == "battle" and lens["ticks"] == final["ticks"] and lens["core_destroyed"] == final["core"]
    assert f'{final["score"]:.4f}' in md(at)
    page = iframes(at)[0]
    data = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S).group(1))
    assert data["frames"][-1]["t"] == final["ticks"] and data["outcome"]["breach"] == final["core"]


def test_ddqn_keeps_not_a_policy_and_value_label():
    at = watch_app()
    at.selectbox(key="w_arm").select_index(2).run()
    assert not at.exception
    text = md(at)
    assert "NOT a policy" in text and "not a policy" in text
    assert "SHAPED return" in text and "shaped-return estimate" in text
    assert at.session_state["w_lens"]["chosen_action"] == replay_data("ddqn", True)["steps"][0]["action"]


def test_headline_is_verbatim_from_results_md():
    txt = (ROOT / "docs" / "RESULTS.md").read_text()
    a = txt.index("## Plain-language summary")
    quoted = txt[a:txt.index("\n## ", a + 5)].strip()
    assert any(m.value == quoted for m in new_app().markdown) and any(m.value == quoted for m in watch_app().markdown)


def test_ledger_sha_unchanged(runs):
    assert hashlib.sha256(P.LEDGER_PATH.read_bytes()).hexdigest() == LEDGER_SHA
    app_script(new_app())
    watch_app()
    assert hashlib.sha256(P.LEDGER_PATH.read_bytes()).hexdigest() == LEDGER_SHA


def test_watch_screenshot_rule_first_deploy_step_with_heatmap_on():
    """Row (oo) (g): the Watch shot = the first DEPLOY decision step (position 9) with the heatmap on."""
    ep = replay_data("ppo_plain", True)
    pos = R.screenshot_positions(ep)
    assert ep["steps"][pos["pos_deploy"]]["phase"] == "DEPLOY" and all(s["phase"] != "DEPLOY" for s in ep["steps"][:pos["pos_deploy"]])
    assert pos["pos_deploy"] == 9
    at = new_app({"mode": "watch", "pos": "9"})
    assert at.slider(key=pos_key()).value == 9 and at.session_state["w_lens"]["kind"] == "decision"
    assert ep["steps"][9]["phase"] == "DEPLOY" and at.toggle(key="w_heat").value is True
    assert "Heatmap overlay" in " ".join(t.label for t in at.toggle)


def test_screenshots_are_the_ravelin_set():
    shots = sorted(p.name for p in (ROOT / "docs" / "screenshots").glob("*.png"))
    assert shots == ["ravelin_1a_title_top_two_units_bought.png", "ravelin_1b_shop_two_units_bought.png", "ravelin_2_deploy_hovering_a_free_tile.png",
                     "ravelin_3_battle_midpoint_tick_16.png", "ravelin_4_end_sticker_BREACH_row_mm_play.png", "ravelin_5_end_sticker_HELD_one_scout_at_0_3.png",
                     "ravelin_6_report_cards.png", "ravelin_7_watch_first_deploy_step_heatmap_on.png", "ravelin_8_mobile_390x844_deploy.png"]
    from PIL import Image
    for s in shots:
        size = Image.open(ROOT / "docs" / "screenshots" / s).size
        assert size == ((390, 844) if "mobile" in s else (1500, 1000)), (s, size)


# ============================================================================================================================ Play (ported from test_b18b_duel_app)
def test_scripted_human_matches_engine_directly(runs):
    at = new_app()
    app_script(at)
    d = direct_script(path=None)
    assert game(at).finished and d.finished
    assert game(at).human_row() == d.human_row() and game(at).actions == d.actions
    assert "g_rows" not in at.session_state                                             # rows exist only after "Show report"
    click(at, "g_report_btn")
    assert at.session_state["g_rows"] == d.reveal()


OPP_MARKERS = ('class="ccard', "<table")


def test_nothing_about_opponents_before_resolution_and_cache_unread(runs, monkeypatch):
    opened = []
    real_open, real_read = builtins.open, pathlib.Path.read_text

    def spy_open(f, *a, **k):
        opened.append(str(f))
        return real_open(f, *a, **k)

    def spy_read(self, *a, **k):
        opened.append(str(self))
        return real_read(self, *a, **k)
    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(pathlib.Path, "read_text", spy_read)

    def hidden(at):
        assert not any(mk in m.value for m in at.markdown for mk in OPP_MARKERS)         # no opponent card, no comparison table (the About text only names the agents)
        assert "Full comparison table" not in [e.label for e in at.expander] and "g_rows" not in at.session_state

    at = new_app()
    for u in BUYS:
        click(at, f"g_buy_{u}")
        hidden(at)
    click(at, "g_end")
    hidden(at)
    assert not game(at).finished
    assert not any(o.endswith("duel_cache.json") for o in opened), "cache was opened before the battle resolved"
    g = game(at)
    at.selectbox(key="g_unit").select("sniper")
    at.selectbox(key="g_tile").select(lowest_tile(g, "sniper"))
    click(at, "g_deploy")
    assert not any(o.endswith("duel_cache.json") for o in opened)
    click(at, "g_wait")
    assert game(at).finished
    hidden(at)                                                       # battle over, report not requested yet: still hidden, cache still unread
    assert not any(o.endswith("duel_cache.json") for o in opened)
    click(at, "g_report_btn")
    assert any(o.endswith("duel_cache.json") for o in opened)       # the spy does see the cache once the report is requested
    assert any('class="ccard' in m.value for m in at.markdown)


def test_table_equals_cache_with_oracle_label_and_best_single(runs):
    at = new_app()
    app_script(at)
    click(at, "g_report_btn")
    cache = json.loads((ROOT / "data" / "v2" / "duel_cache.json").read_text())["layouts"][LAYOUT]["contestants"]
    rows = at.session_state["g_rows"]
    assert [r["contestant"] for r in rows] == ["human", *D.CONTESTANTS]
    for r in rows[1:]:
        for k in ("score", "d", "breach", "n_bought", "n_deployed", "bought", "deploy"):
            assert r[k] == cache[r["contestant"]][k]
    html = texts(at)
    for r in rows:
        assert r["label"].replace("&", "&amp;") in html or r["label"] in html
        assert f'{r["score"]:.3f}' in html
    assert "Oracle (search baseline, not the optimum)" in html and D.LABELS["best_single"] in html
    assert "search baseline, not the optimum" in html and "leaderboard" not in html.lower()


def test_illegal_actions_show_message_and_change_nothing(runs):
    at = new_app()
    assert at.button(key="g_end").disabled and at.button(key="g_undo").disabled          # nothing bought: the UI offers neither
    with pytest.raises(D.DuelError, match="buy at least one unit"):
        game(at).end_buy()                                                              # and the engine refuses if asked anyway
    assert game(at).actions == [] and game(at).phase == "BUY"
    for _ in range(40):                                                                 # buy until the engine / UI stops us (gold or unit cap)
        if at.button(key="g_buy_bulwark").disabled:
            break
        click(at, "g_buy_bulwark")
    assert at.button(key="g_buy_bulwark").disabled
    snap = (list(game(at).actions), game(at).env.gold, game(at).phase)
    with pytest.raises(D.DuelError):
        game(at).buy("bulwark")
    assert (list(game(at).actions), game(at).env.gold, game(at).phase) == snap
    click(at, "g_end")
    assert not at.error
    t = lowest_tile(game(at), "bulwark")
    at.selectbox(key="g_unit").select("bulwark")
    at.selectbox(key="g_tile").select(t)
    click(at, "g_deploy")
    assert not at.error and not game(at).finished
    snap = (list(game(at).actions), list(game(at).env.owned))
    D_before = game(at).env.gold
    with pytest.raises(D.DuelError, match="no mender left"):
        game(at).deploy("mender", lowest_tile(game(at), "bulwark"))                     # a unit that is not owned
    assert (list(game(at).actions), list(game(at).env.owned)) == snap and game(at).env.gold == D_before


def test_illegal_click_shows_the_engine_message_and_changes_nothing(clicks):
    at = new_app()
    for u in BUYS:
        click(at, f"g_buy_{u}")
    click(at, "g_end")
    click_tile(at, 0)
    assert not at.error and game(at).actions[-1] == 5 + 52 * D.parse_unit("sniper") + 0
    snap = (list(game(at).actions), list(game(at).env.owned), game(at).phase)
    click_tile(at, 0)                                                                   # occupied
    assert at.error and "occupied" in at.error[0].value
    assert (list(game(at).actions), list(game(at).env.owned), game(at).phase) == snap
    click_tile(at, 99)                                                                  # not a border tile at all (a tampered component value)
    assert at.error and "out of range" in at.error[0].value
    assert (list(game(at).actions), list(game(at).env.owned), game(at).phase) == snap
    click_tile(at, 1)                                                                   # and a legal tile still works afterwards
    assert not at.error and len(game(at).actions) == len(snap[0]) + 1


def test_undo_in_buy_and_refused_after_end_buy(runs):
    at = new_app()
    click(at, "g_buy_sniper")
    click(at, "g_buy_bulwark")
    gold2, acts2 = game(at).env.gold, list(game(at).actions)
    click(at, "g_undo")
    assert game(at).actions == acts2[:-1] and game(at).env.gold > gold2 and not at.error
    click(at, "g_undo")
    assert game(at).actions == [] and game(at).env.gold == 240
    assert at.button(key="g_undo").disabled                                              # nothing to undo
    with pytest.raises(D.DuelError, match="nothing to undo"):
        game(at).undo()
    click(at, "g_buy_sniper")
    click(at, "g_end")
    assert "g_undo" not in [b.key for b in at.button]                                    # the button is gone in DEPLOY
    g = game(at)
    snap = (list(g.actions), g.env.gold)
    with pytest.raises(D.DuelError):
        g.undo()                                                                         # and the engine refuses
    assert (list(g.actions), g.env.gold) == snap and g.phase == "DEPLOY"


def test_only_dev_validation_ids_selectable(runs):
    at = new_app()
    opts = list(at.selectbox(key="g_layout").options)
    assert opts == EV.dev_layout_ids("DEV-validation") and len(opts) == 30
    assert not set(opts) & set(EV.dev_layout_ids("DEV-train"))
    assert at.selectbox(key="g_layout").value == LAYOUT
    train_id = EV.dev_layout_ids("DEV-train")[0]
    for bad in (train_id, "labyrinth-999"):
        with pytest.raises(ValueError):
            new_app().selectbox(key="g_layout").select(bad).run()
    other = next(i for i in opts if i != LAYOUT)
    at.selectbox(key="g_layout").select(other).run()
    assert game(at).layout_id == other and game(at).phase == "BUY"


def _tracked_state():
    return subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout


def test_human_run_appended_to_ignored_file_and_nothing_tracked_changes(runs):
    repo_file = ROOT / "results_v2" / "duel_human_runs.jsonl"
    size = repo_file.stat().st_size if repo_file.exists() else None
    assert subprocess.run(["git", "check-ignore", "-q", "results_v2/duel_human_runs.jsonl"], cwd=ROOT).returncode == 0
    before = _tracked_state()
    at = new_app()
    app_script(at)
    lines = runs.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["layout_id"] == LAYOUT and rec["split"] == "DEV-validation" and rec["actions"] == game(at).actions and rec["row"] == game(at).human_row()
    assert _tracked_state() == before
    assert (repo_file.stat().st_size if repo_file.exists() else None) == size           # the real local file was not touched by the test
    app_script(new_app())                                                               # a second run appends, never rewrites
    assert len(runs.read_text().splitlines()) == 2


def test_new_game_resets_cleanly(runs):
    at = new_app()
    app_script(at)
    click(at, "g_report_btn")
    assert "g_rows" in at.session_state
    click(at, "g_new")
    g = game(at)
    assert g.phase == "BUY" and g.actions == [] and g.env.gold == 240 and not g.finished
    assert not any(k in at.session_state for k in ("g_rows", "g_battle", "g_structs", "g_report"))
    assert not any(mk in m.value for m in at.markdown for mk in OPP_MARKERS) and not at.error
    app_script(at)                                                                      # and it can be played again
    assert game(at).finished


def test_two_sessions_do_not_share_state(runs):
    a, b = new_app(), new_app()
    click(a, "g_buy_sniper")
    assert len(game(a).actions) == 1 and game(b).actions == [] and game(a) is not game(b)
    app_script(b)
    assert game(b).finished and not game(a).finished and "g_rows" not in a.session_state
    click(a, "g_buy_scout")
    assert len(game(a).actions) == 2


def test_disclaimer_visible_once_and_no_sidebar(runs):
    at = new_app()
    assert md(at).count(R.DISCLAIMER) == 1 and len(at.sidebar) == 0
    app_script(at)
    assert md(at).count(R.DISCLAIMER) == 1
    click(at, "g_report_btn")
    assert md(at).count(R.DISCLAIMER) == 1
    assert md(watch_app()).count(R.DISCLAIMER) == 1


# ============================================================================================================================ click-to-deploy (new)
def test_a_click_gives_the_same_action_and_outcome_as_the_dropdown(clicks):
    at = new_app()
    for u in ("sniper", "bulwark"):
        click(at, f"g_buy_{u}")
    click(at, "g_end")
    b = 7
    click_tile(at, b)                                                                    # the unit selector starts on the first owned unit (sniper)
    u = D.parse_unit("sniper")
    assert game(at).actions[-1] == 5 + 52 * u + b
    st2 = new_app()
    for u_ in ("sniper", "bulwark"):
        click(st2, f"g_buy_{u_}")
    click(st2, "g_end")
    st2.selectbox(key="g_unit").select("sniper")
    st2.selectbox(key="g_tile").select(b)
    click(st2, "g_deploy")
    assert game(st2).actions == game(at).actions and list(game(st2).env.owned) == list(game(at).env.owned)
    # to the end: the same last deployment through both paths gives the same finished duel
    for a_, via in ((at, "click"), (st2, "dropdown")):
        a_.selectbox(key="g_unit").select("bulwark").run()
        if via == "click":
            click_tile(a_, 9)
        else:
            a_.selectbox(key="g_tile").select(9)
            click(a_, "g_deploy")
    assert game(at).finished and game(st2).finished
    assert game(at).actions == game(st2).actions and game(at).human_row() == game(st2).human_row() and game(at).env.result.ticks == game(st2).env.result.ticks


def test_row_mm_script_entered_by_clicks(clicks, runs):
    at = new_app()
    click_script(at)
    g = game(at)
    assert g.actions == MM_ACTIONS
    row = g.human_row()
    assert (round(row["score"], 3), round(row["d"], 3), row["breach"], g.env.result.ticks) == (0.400, 0.0, True, 32)
    assert row == direct_script().human_row()
    # the board in the DEPLOY phase was the interactive one: every border tile listed, free ones ok, a ghost unit named
    at2 = new_app()
    for u in BUYS:
        click(at2, f"g_buy_{u}")
    click(at2, "g_end")
    m = clicks["g_board"]
    assert m["has_callback"] and len(m["data"]["tiles"]) == 52 and m["data"]["cell"] == GV.CELL
    assert [t["b"] for t in m["data"]["tiles"]] == list(range(52)) and all(t["ok"] for t in m["data"]["tiles"])
    assert (m["data"]["tiles"][3]["row"], m["data"]["tiles"][3]["col"]) == tuple(BORDER[3]) and m["data"]["unit"] == "sniper"
    assert not clicks["g_board_buy"]["has_callback"] and clicks["g_board_buy"]["data"]["tiles"] == []     # BUY draws a static board


def test_component_source_is_plain_html_js_with_no_network_and_reports_the_border_index():
    assert "bidi" not in BC.JS and "http" not in BC.HTML + BC.CSS + BC.JS
    assert "setTriggerValue('tile', t.b)" in BC.JS and "'data-b': t.b" in BC.JS and "addEventListener('click'" in BC.JS and "'keydown'" in BC.JS
    assert "pointerenter" in BC.JS and ":hover" in BC.CSS and "touch-action:manipulation" in BC.CSS and "width:100%" in BC.CSS and "preventDefault" in BC.JS
    reg = {}
    BC.make_component(lambda name, **kw: reg.update(name=name, **kw))
    assert reg["name"] == "ravelin_board" and set(reg) == {"name", "html", "css", "js"}
    src = Path(APP).read_text()
    assert "st.components.v2" in (ROOT / "app" / "board_component.py").read_text() and "components.v1" not in src and "components.html" not in src
    assert '"on_tile_change": on_click' in src and "act(\"deploy\"" in src


# ============================================================================================================================ st.iframe for the battle player
def test_battle_player_is_a_self_contained_st_iframe(runs):
    at = new_app({"demo": "battle"})
    pages = iframes(at)
    assert len(pages) == 1 and "components.v1" not in Path(APP).read_text() and "st.iframe(page" in Path(APP).read_text()
    page = pages[0]
    assert page.startswith("<!doctype html>") and 'id="play"' in page
    assert not re.search(r"""(https?:|src\s*=|href\s*=\s*['"]?[^#'"\s]|@import|<link|fetch\(|XMLHttpRequest|WebSocket|import\()""", page)
    assert "setInterval" in page and "Streamlit" not in page and "setTriggerValue" not in page      # ticks advance inside the page: no message to the app, no rerun per tick


# ============================================================================================================================ polish
def _event_replay(name="archipelago__ppo_d4__tta"):
    f = ROOT / "docs" / "replays" / f"{name}.html"
    return json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', f.read_text(), re.S).group(1))


@pytest.mark.parametrize("name", ["archipelago__ppo_d4__tta", "bunker_nest__ddqn__plain"])
def test_breach_banner_names_the_tick_the_core_fell_not_the_last_tick(name):
    d = _event_replay(name)
    structs, frames, o = d["structs"], d["battle"]["frames"], d["outcome"]
    ci = next(i for i, s in enumerate(structs) if s[0] == "c")
    first_dead = next(f["t"] for f in frames if f["s"][ci] <= 0)
    assert o["core"] and first_dead < o["ticks"] and GV.core_tick(structs, frames) == first_dead         # the core falls before the battle ends
    assert all(f["s"][ci] > 0 for f in frames if f["t"] < first_dead) and all(f["s"][ci] <= 0 for f in frames if f["t"] >= first_dead)
    page = GV.player_html(structs, frames, [t for t, _ in d["battle"]["units"]], {"breach": True, "ticks": o["ticks"], "reasonText": "x"})
    data = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S).group(1))
    assert data["outcome"]["coreTick"] == first_dead and data["outcome"]["ticks"] == o["ticks"] != first_dead
    assert "'core destroyed at tick ' + n(D.outcome.coreTick)" in page and "outcome.ticks" not in page.split("bs.innerHTML")[1].split(";")[0]


def test_breach_banner_in_the_row_mm_play_is_tick_24_not_32(runs):
    d = direct_script()
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("make_replays", ROOT / "scripts" / "make_replays.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("make_replays", mod)
    spec.loader.exec_module(mod)
    battle, structs = mod._battle(d.env), mod._struct_table(d.env._st.P)
    assert d.env.result.ticks == 32 and GV.core_tick(structs, battle["frames"]) == 24


def test_board_edge_numbers_are_ink_on_snow_inside_the_card():
    import xml.etree.ElementTree as ET
    root = ET.fromstring(GV.board_svg([["c", 2, 5, ""]], None, []))
    axis = next(g for g in root.iter("{http://www.w3.org/2000/svg}g") if g.get("class") == "axis")
    cols = [t for t in axis if float(t.get("y")) < 0]
    rows = [t for t in axis if float(t.get("x")) < 0]
    assert [t.text for t in cols] == [str(i) for i in range(14)] and [t.text for t in rows] == [str(i) for i in range(14)]
    assert all(t.get("fill") == GV.T.TEXT and GV.T.FONT_NUM in t.get("font-family") for t in axis)
    vb = [float(v) for v in root.get("viewBox").split()]
    assert vb[0] < 0 and vb[1] < 0 and vb[2] == GV.BOARD_PX + GV.GUT                     # the gutter is part of the same card image, outside the board rectangle
    assert GV.T.contrast(GV.T.TEXT, GV.T.BG) >= 4.5
    page = GV.player_html([["c", 3, 3, ""]], [{"t": 0, "u": [], "s": [1000]}], [], {"breach": False, "ticks": 0, "reasonText": "x"})
    assert page.count('<g class="axis"') == 1 and 'viewBox="%s"' % GV.VIEW in page


def test_buildings_are_small_houses_distinct_from_walls_and_the_keep():

    import numpy as np
    for n, pips in (("b1", 1), ("b2", 2), ("b3", 3)):
        assert IC.SYMBOLS[n].count("<circle") == pips                                    # 1-3 value pips
        assert "polygon" in IC.SYMBOLS[n] and "<rect" in IC.SYMBOLS[n]                    # pitched roof + body
    assert "wallstone" in IC.SYMBOLS["wall"] and "wallstone" not in IC.SYMBOLS["b1"] and IC.building_icon(3) == "b3"
    mask = {n: IC.silhouette_mask(n) for n in IC.SILHOUETTE}
    iou = lambda a, b: (mask[a] & mask[b]).sum() / (mask[a] | mask[b]).sum()
    for b in ("b1", "b2", "b3"):
        assert iou(b, "wall") < 0.8 and iou(b, "core") < 0.8
    assert np.array_equal(mask["b1"], mask["b3"])                                       # the three weights are one house; the pips carry the value


def test_no_two_pieces_share_a_silhouette_and_scout_is_a_winged_boot_not_a_spike():
    import itertools
    pieces = [n for n in IC.SILHOUETTE if n not in ("b2", "b3")]                          # b1 / b2 / b3 are one building, told apart by pips
    mask = {n: IC.silhouette_mask(n) for n in pieces}
    worst = []
    for a, b in itertools.combinations(pieces, 2):
        iou = (mask[a] & mask[b]).sum() / (mask[a] | mask[b]).sum()
        worst.append((iou, a, b))
    assert max(worst)[0] < 0.8, max(worst)
    sc, ln = mask["scout"], mask["lancer"]
    assert (sc & ln).sum() / (sc | ln).sum() < 0.5
    assert "<path" in IC.SYMBOLS["scout"] and "<polygon" not in IC.SYMBOLS["scout"].split("<path")[0] and T_ARROW not in IC.SYMBOLS["scout"]
    assert "<polygon" in IC.SYMBOLS["lancer"] and IC.TURRETS == ("gun", "lancer", "flak")
    assert len(set(IC.SYMBOLS.values())) == len(IC.ALL)


T_ARROW = "17.0,3.0 28.0,23.0 17.0,18.0 6.0,23.0"                                       # the old arrow polygon


def test_mono_only_for_numbers_prose_is_atkinson_next():
    css = re.search(r'PROTO_CSS = """(.*?)"""', Path(APP).read_text(), re.S).group(1) + GV.PLAYER_CSS + BC.CSS
    allowed = ("stat b", "price", "tok", "cnt", "nums b", "army span", ".n", ".num", "axis", "tip", "table.top td.num", "tick")
    for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if "var(--f-num)" in body:
            assert any(a in sel for a in allowed), sel.strip()
    from app import theme as TH
    mono_rules = [r for r in TH.app_css().split("}") if "var(--f-num)" in r and "{" in r]
    assert all("stSelectbox" not in r and "stMarkdown" not in r for r in mono_rules)       # dropdown text and prose are Atkinson Next
    page = GV.player_html([["c", 3, 3, ""]], [{"t": 0, "u": [], "s": [1000]}], [], {"breach": False, "ticks": 0, "reasonText": "x"})
    assert ".status{" in page and "font-family" not in re.search(r"\.status\{([^}]*)\}", page).group(1)           # status line = prose
    assert "class=\"n\"" in page.replace("\\\"", '"') or "class=\\\"n\\\"" in page or '<span class="n">' in page


def test_title_is_ravelin_in_the_page_and_the_tab():
    src = Path(APP).read_text()
    assert 'page_title="Ravelin"' in src and "<h1>Ravelin</h1>" in src and "Breach Lens" not in src
    assert any("<h1>Ravelin</h1>" in m.value for m in new_app().markdown)


def test_mobile_layout_rules_exist_and_buttons_are_44px():
    src = Path(APP).read_text()
    css = re.search(r'PROTO_CSS = """(.*?)"""', src, re.S).group(1)
    assert "@media (max-width:640px)" in css and "calc(50% - 8px)" in css and "min-height:44px" in css
    assert ".stButton button,[data-testid='stPopover'] button{min-height:44px}" in css and "[data-testid='stSelectbox'] button" in css
    assert "@media (max-width:520px){button{min-height:44px}" in GV.PLAYER_CSS and "#board{width:100%;height:auto" in GV.PLAYER_CSS
    assert "width:100%" in BC.CSS and "height:auto" in BC.CSS                                # the interactive board scales to its container
    assert 'content="width=device-width,initial-scale=1"' in GV.player_html([["c", 3, 3, ""]], [{"t": 0, "u": [], "s": [1000]}], [], {"breach": False, "ticks": 0, "reasonText": "x"})
    assert 'PLAYER_H = "content"' in src
