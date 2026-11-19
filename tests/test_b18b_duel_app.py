"""B18b: duel mode in the Streamlit app (docs/SPEC_V2.md, Gate changelog row (mm)). AppTest; DEV-validation only; human runs go to a temp file."""
import builtins
import hashlib
import json
import pathlib
import subprocess
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from breachlens import duel as D
from breachlens import evaluator as EV
from breachlens import pool as P
from breachlens.actions import encode_deploy

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "legacy_app.py")
LEDGER_SHA = "20a8b2877c91c9cf45c3692617ad3bd08d860ef0f7783ca3610a9a3fc44c9dbd"
LAYOUT = "spine_pods-300000056"
BUYS = ["sniper"] * 4 + ["bulwark"] * 2                      # the row (mm) (g) script
TIMEOUT = 300


@pytest.fixture
def runs(tmp_path, monkeypatch):
    p = tmp_path / "runs.jsonl"
    monkeypatch.setenv("BREACHLENS_DUEL_RUNS", str(p))
    return p


def duel_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.radio(key="mode").set_value("Duel").run()
    assert not at.exception
    return at


def game(at):
    return at.session_state["duel_game"]


def click(at, key):
    at.button(key=key).click().run()
    assert not at.exception


def lowest_tile(duel, unit):
    m = duel.env.action_masks()
    u = D.parse_unit(unit)
    return next(b for b in range(52) if m[encode_deploy(u, b)])


def app_script(at, buys=BUYS):
    for u in buys:
        click(at, f"duel_buy_{u}")
    click(at, "duel_end")
    for u in buys:                                           # deploy in the order bought, each on the lowest legal tile
        g = game(at)
        at.selectbox(key="duel_unit").select(u)
        at.number_input(key="duel_tile").set_value(lowest_tile(g, u))
        click(at, "duel_deploy")
        if game(at).finished:
            break
    if not game(at).finished:
        click(at, "duel_wait")


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


def texts(at):
    return " ".join([m.value for m in at.markdown] + [e.value for e in at.error] + [str(x.value) for x in at.text])


OPP_WORDS = ("Oracle", "BestSingle", "PPO", "DDQN", "search baseline")


def test_scripted_human_matches_engine_directly(runs, tmp_path):
    at = duel_app()
    app_script(at)
    d = direct_script(path=None)
    assert game(at).finished and d.finished
    assert game(at).human_row() == d.human_row()
    assert game(at).actions == d.actions
    assert at.session_state["duel_rows"] == d.reveal()


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
    at = duel_app()
    for u in BUYS:
        click(at, f"duel_buy_{u}")
        assert not any(w in texts(at) for w in OPP_WORDS)
    click(at, "duel_end")
    assert not any(w in texts(at) for w in OPP_WORDS)
    assert "duel_rows" not in at.session_state and not game(at).finished
    assert not any(o.endswith("duel_cache.json") for o in opened), "cache was opened before the battle resolved"
    g = game(at)
    at.selectbox(key="duel_unit").select("sniper")
    at.number_input(key="duel_tile").set_value(lowest_tile(g, "sniper"))
    click(at, "duel_deploy")
    assert not any(o.endswith("duel_cache.json") for o in opened) and "duel_rows" not in at.session_state
    click(at, "duel_wait")
    assert game(at).finished
    assert any(o.endswith("duel_cache.json") for o in opened)       # the spy does see the cache once the battle has resolved


def test_table_equals_cache_with_oracle_label_and_best_single(runs):
    at = duel_app()
    app_script(at)
    cache = json.loads((ROOT / "data" / "v2" / "duel_cache.json").read_text())["layouts"][LAYOUT]["contestants"]
    rows = at.session_state["duel_rows"]
    assert [r["contestant"] for r in rows] == ["human", *D.CONTESTANTS]
    for r in rows[1:]:
        for k in ("score", "d", "breach", "n_bought", "n_deployed", "bought", "deploy"):
            assert r[k] == cache[r["contestant"]][k]
    html = texts(at)
    for r in rows:
        assert r["label"].replace("&", "&amp;") in html or r["label"] in html
        assert f'{r["score"]:.3f}' in html
    assert "Oracle (search baseline, not the optimum)" in html and D.LABELS["best_single"] in html
    assert "leaderboard" not in html.lower()


def test_illegal_actions_show_message_and_change_nothing(runs):
    at = duel_app()
    click(at, "duel_end")                                          # nothing bought yet
    assert at.error and "buy at least one unit" in at.error[0].value
    assert game(at).actions == [] and game(at).phase == "BUY"
    for _ in range(40):                                            # buy until the engine refuses (gold or unit cap)
        click(at, "duel_buy_bulwark")
        if at.error:
            break
    assert at.error, "the engine never refused a buy"
    snap = (list(game(at).actions), game(at).env.gold, game(at).phase)
    click(at, "duel_buy_bulwark")
    assert at.error and (list(game(at).actions), game(at).env.gold, game(at).phase) == snap
    click(at, "duel_end")
    assert not at.error
    # deploy: unit not owned, then an occupied tile
    at.selectbox(key="duel_unit").select("mender")
    at.number_input(key="duel_tile").set_value(lowest_tile(game(at), "bulwark"))
    snap = (list(game(at).actions), list(game(at).env.owned))
    click(at, "duel_deploy")
    assert at.error and "no mender left" in at.error[0].value
    assert (list(game(at).actions), list(game(at).env.owned)) == snap
    t = lowest_tile(game(at), "bulwark")
    at.selectbox(key="duel_unit").select("bulwark")
    at.number_input(key="duel_tile").set_value(t)
    click(at, "duel_deploy")
    assert not at.error and not game(at).finished
    snap = (list(game(at).actions), list(game(at).env.owned))
    click(at, "duel_deploy")                                       # same tile again: occupied
    assert at.error and "occupied" in at.error[0].value
    assert (list(game(at).actions), list(game(at).env.owned)) == snap


def test_undo_in_buy_and_refused_after_end_buy(runs):
    at = duel_app()
    click(at, "duel_buy_sniper")
    click(at, "duel_buy_bulwark")
    gold2, acts2 = game(at).env.gold, list(game(at).actions)
    click(at, "duel_undo")
    assert game(at).actions == acts2[:-1] and game(at).env.gold > gold2 and not at.error
    click(at, "duel_undo")
    assert game(at).actions == [] and game(at).env.gold == 240
    click(at, "duel_undo")                                         # nothing to undo
    assert at.error and game(at).actions == []
    click(at, "duel_buy_sniper")
    click(at, "duel_end")
    assert "duel_undo" not in [b.key for b in at.button]           # the button is gone in DEPLOY
    g = game(at)
    snap = (list(g.actions), g.env.gold)
    with pytest.raises(D.DuelError):
        g.undo()                                                   # and the engine refuses
    assert (list(g.actions), g.env.gold) == snap and g.phase == "DEPLOY"


def test_only_dev_validation_ids_selectable(runs):
    at = duel_app()
    opts = list(at.selectbox(key="duel_layout").options)
    assert opts == EV.dev_layout_ids("DEV-validation") and len(opts) == 30
    assert not set(opts) & set(EV.dev_layout_ids("DEV-train"))
    assert at.selectbox(key="duel_layout").value == LAYOUT
    train_id = EV.dev_layout_ids("DEV-train")[0]
    for bad in (train_id, "labyrinth-999"):
        with pytest.raises(ValueError):
            duel_app().selectbox(key="duel_layout").select(bad).run()
    other = next(i for i in opts if i != LAYOUT)
    at.selectbox(key="duel_layout").select(other).run()
    assert game(at).layout_id == other and game(at).phase == "BUY"


def _tracked_state():
    return subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout


def test_human_run_appended_to_ignored_file_and_nothing_tracked_changes(runs):
    repo_file = ROOT / "results_v2" / "duel_human_runs.jsonl"
    before, size = _tracked_state(), (repo_file.stat().st_size if repo_file.exists() else None)
    assert subprocess.run(["git", "check-ignore", "-q", "results_v2/duel_human_runs.jsonl"], cwd=ROOT).returncode == 0
    at = duel_app()
    app_script(at)
    lines = runs.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["layout_id"] == LAYOUT and rec["split"] == "DEV-validation" and rec["actions"] == game(at).actions and rec["row"] == game(at).human_row()
    assert _tracked_state() == before
    assert (repo_file.stat().st_size if repo_file.exists() else None) == size      # the real local file was not touched by the test
    app_script(duel_app())                                       # a second run appends, never rewrites
    assert len(runs.read_text().splitlines()) == 2


def test_new_duel_resets_cleanly(runs):
    at = duel_app()
    app_script(at)
    assert "duel_rows" in at.session_state
    click(at, "duel_new")
    g = game(at)
    assert g.phase == "BUY" and g.actions == [] and g.env.gold == 240 and not g.finished
    assert not any(k in at.session_state for k in ("duel_rows", "duel_battle", "duel_structs", "duel_pos"))
    assert not any(w in texts(at) for w in OPP_WORDS) and not at.error
    app_script(at)                                               # and it can be played again
    assert game(at).finished


def test_two_sessions_do_not_share_state(runs):
    a, b = duel_app(), duel_app()
    click(a, "duel_buy_sniper")
    assert len(game(a).actions) == 1 and game(b).actions == [] and game(a) is not game(b)
    app_script(b)
    assert game(b).finished and not game(a).finished and "duel_rows" not in a.session_state
    click(a, "duel_buy_scout")
    assert len(game(a).actions) == 2


def test_disclaimer_visible_and_no_sidebar(runs):
    at = duel_app()
    assert any("Demo, not a result" in m.value for m in at.markdown)
    assert len(at.sidebar) == 0
    app_script(at)
    assert any("Demo, not a result" in m.value for m in at.markdown)


def test_ledger_sha_unchanged(runs):
    assert hashlib.sha256(P.LEDGER_PATH.read_bytes()).hexdigest() == LEDGER_SHA
    app_script(duel_app())
    assert hashlib.sha256(P.LEDGER_PATH.read_bytes()).hexdigest() == LEDGER_SHA
