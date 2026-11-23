"""B24: Play mode as a real-time-strategy board (app/arena.py, app/arena_data.py, app/board_component.py plan mode). Science frozen: nothing here changes the env, the simulator,
scoring, masks, unit stats, the duel cache or any protected file; a human run stays a legal action sequence of the same env.

Python side (this file): placement draft -> the env's action sequence, outcome identical to ``breachlens.duel``, raw fixed-point frames equal the simulator, damage events,
HUD totals, key moments, Fight timing, Watch-on-the-same-base, frozen files. The browser side (real Chrome, the DOM the player draws) is tests/test_b24_arena_js.py."""
import hashlib
import html as _html
import importlib.util
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import arena as AR
from app import arena_data as AD
from app import bundle as BU
from breachlens import duel as D
from breachlens import evaluator as EV
from breachlens.sim import Prepared, simulate
from tests.test_b18eb_app import BUYS, LAYOUT, MM_ACTIONS, app_script, fight, game, mm_draft, new_app  # noqa: F401  (new_app / game / fight drive the app with the component stand-in)

ROOT = Path(__file__).resolve().parents[1]
LEDGER_SHA = "20a8b2877c91c9cf45c3692617ad3bd08d860ef0f7783ca3610a9a3fc44c9dbd"
_spec = importlib.util.spec_from_file_location("make_replays", ROOT / "scripts" / "make_replays.py")
M = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("make_replays", M)
_spec.loader.exec_module(M)


def mm_duel():
    d = D.Duel(LAYOUT, human_runs_path=None)
    AD.play_draft(d, mm_draft())
    return d


# ----------------------------------------------------------------------------------------------------------------------------- the draft is a legal env sequence
def test_canonical_order_is_buys_end_buy_deploys_in_placement_order():
    draft = mm_draft()
    assert AD.canonical_actions(draft) == MM_ACTIONS == [1, 1, 1, 1, 2, 2, 265, 57, 58, 59, 60, 113, 114]
    mixed = [(2, 5), (1, 0), (0, 9), (1, 1)]                                              # buys sorted by type, deploys in the order placed
    assert AD.canonical_actions(mixed) == [0, 1, 1, 2, 265, 5 + 104 + 5, 5 + 52 + 0, 5 + 9, 5 + 52 + 1]
    assert AD.canonical_actions([]) == []


def test_the_row_mm_script_through_the_new_flow_is_row_mm():
    d = mm_duel()
    assert d.actions == MM_ACTIONS and d.finished
    row = d.human_row()
    assert (round(row["score"], 3), round(row["d"], 3), row["breach"], d.env.result.ticks) == (0.400, 0.0, True, 32)


def _random_draft(rng, layout_id):
    d = D.Duel(layout_id, human_runs_path=None)
    free = [b for b in range(52) if d.env._st.border_free[b]]
    gold, draft, taken = 240, [], set()
    for _ in range(rng.randint(1, 12)):
        u = rng.randrange(5)
        cost = (8, 28, 30, 36, 28)[u]
        tiles = [b for b in free if b not in taken]
        if cost > gold or not tiles or len(draft) >= 20:
            break
        b = rng.choice(tiles)
        draft.append((u, b)); taken.add(b); gold -= cost
    return draft


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_any_ui_draft_replays_through_breachlens_duel_with_an_identical_outcome(arena, seed):
    rng = random.Random(seed)
    at = new_app()
    draft = _random_draft(rng, LAYOUT)
    fight(at, draft)
    g = game(at)
    assert g.finished and g.actions == AD.canonical_actions(draft)
    twin = D.Duel(LAYOUT, human_runs_path=None)                                           # the same actions through the engine API, one by one
    for a in g.actions:
        twin.act(a)
    assert twin.human_row() == g.human_row() and twin.env.result.ticks == g.env.result.ticks and twin.env.result.struct_hp == g.env.result.struct_hp
    assert twin.env.result.unit_hp == g.env.result.unit_hp
    # and the simulator on the army as deployed, outside the env
    res = simulate(g.env._st.P, [(u, b) for u, b in draft])
    assert (res.d_num, res.d_den, res.core_destroyed, res.ticks) == (g.env.result.d_num, g.env.result.d_den, g.env.result.core_destroyed, g.env.result.ticks)


def test_two_click_orders_with_the_same_ordered_deploys_give_the_same_outcome():
    """Drafts reached by different click histories (extra placements removed again) are the same ordered list, hence the same actions and outcome; a different buy order of the
    same army still gives the same buys (sorted), so only the deploy order can matter."""
    a = mm_draft()
    b = [x for x in mm_draft() + [(0, 20), (3, 21)] if x not in ((0, 20), (3, 21))]       # placed two extra units and tapped them off again
    assert a == b and AD.canonical_actions(a) == AD.canonical_actions(b)
    sorted_buys = lambda dr: [x for x in AD.canonical_actions(dr) if x < 5]
    assert sorted_buys([(2, 4), (1, 0)]) == sorted_buys([(1, 0), (2, 4)])                 # same army, other order: same buys
    r1, r2 = (mm_duel().human_row(), mm_duel().human_row())
    assert r1 == r2


def test_engine_messages_for_a_tap_the_draft_cannot_take():
    assert "occupied" in AD.probe_message(LAYOUT, [(1, 0)], 1, 0)
    assert "cannot afford" in AD.probe_message(LAYOUT, [(2, i) for i in range(8)], 0, 8)
    assert "unit cap" in AD.probe_message(LAYOUT, [(0, i) for i in range(20)], 0, 30) or "cannot afford" in AD.probe_message(LAYOUT, [(0, i) for i in range(20)], 0, 30)
    assert AD.probe_message(LAYOUT, [(1, 0)], 1, 1) == ""                                 # a legal tap has no message
    st_ = D.Duel(LAYOUT, human_runs_path=None).env._st
    wall = next(b for b in range(52) if not st_.border_free[b]) if not all(st_.border_free) else None
    if wall is not None:
        assert "wall tile" in AD.probe_message(LAYOUT, [], 1, wall)


# ----------------------------------------------------------------------------------------------------------------------------- raw frames equal the simulator
def test_raw_frames_equal_the_simulator_at_every_tick():
    d = mm_duel()
    P, army = d.env._st.P, d.env.army
    b = AD.raw_battle(P, army)
    assert b["ticks"] == 32 and len(b["frames"]) == 33 and b["S"] == P.cfg.S == 840
    full = simulate(P, army, record=True)
    for t in range(1, 33):
        pre = simulate(P, army, max_ticks=t)
        f = b["frames"][t]
        assert f["t"] == t and f["s"] == list(pre.struct_hp)                                 # raw fixed-point, per structure
        assert [u[1] for u in f["u"]] == [h if st >= 0 else 0 for _, h, st in full.trace[t - 1]]
        assert [u[0] for u in f["u"]] == [c for c, _, _ in full.trace[t - 1]]
    assert b["frames"][0]["s"] == list(P.s_hp0) and [u[1] for u in b["frames"][0]["u"]] == b["umax"]
    assert b["frames"][-1]["s"] == list(d.env.result.struct_hp) and [u[1] for u in b["frames"][-1]["u"]] == list(d.env.result.unit_hp)


def test_raw_frames_agree_with_the_permille_replay_frames_of_the_same_battle():
    d = mm_duel()
    P = d.env._st.P
    b, old = AD.raw_battle(P, d.env.army), M._battle(d.env)
    assert [[u[0] for u in f["u"]] for f in b["frames"]] == [[u[0] for u in f["u"]] for f in old["frames"]]
    for fr, fo in zip(b["frames"], old["frames"]):
        assert [round(1000 * h / m) for h, m in zip(fr["s"], b["smax"])] == fo["s"]
        assert [round(1000 * u[1] / m) if u[2] >= 0 else 0 for u, m in zip(fr["u"], b["umax"])] == [u[1] for u in fo["u"]]


@pytest.mark.parametrize("arm", ["ppo_plain", "ppo_d4", "ddqn"])
def test_watch_the_ai_raw_frames_equal_the_bundled_permille_frames(arm):
    ids = BU.layout_ids()
    for lid in (ids[0], ids[12]):
        ep = BU.load_episode(lid, arm, True)
        P = Prepared(EV.load_dev_layout("DEV-validation", layout_id=lid, entry_point="tests")[1])
        b = AD.raw_battle(P, ep["battle"]["units"])
        assert len(b["frames"]) == len(ep["battle"]["frames"])
        for fr, fo in zip(b["frames"], ep["battle"]["frames"]):
            assert [round(1000 * h / m) for h, m in zip(fr["s"], b["smax"])] == fo["s"]
            assert [round(1000 * u[1] / m) if u[2] >= 0 else 0 for u, m in zip(fr["u"], b["umax"])] == [u[1] for u in fo["u"]] and [u[0] for u in fr["u"]] == [u[0] for u in fo["u"]]
        res = b["result"]
        assert (res.core_destroyed, res.ticks) == (bool(ep["outcome"]["core"]), int(ep["outcome"]["ticks"]))


# ----------------------------------------------------------------------------------------------------------------------------- damage events, HUD totals, key moments
def test_damage_events_equal_hp_deltas_between_frames():
    d = mm_duel()
    b = AD.raw_battle(d.env._st.P, d.env.army)
    ev = AD.damage_events(b["frames"])
    assert ev[0] == {"u": [], "s": []} and len(ev) == len(b["frames"])
    n_s = n_u = 0
    for t in range(1, len(b["frames"])):
        a, c = b["frames"][t - 1], b["frames"][t]
        assert dict(ev[t]["s"]) == {j: x - y for j, (x, y) in enumerate(zip(a["s"], c["s"])) if y < x}
        assert dict(ev[t]["u"]) == {i: x[1] - y[1] for i, (x, y) in enumerate(zip(a["u"], c["u"])) if y[1] < x[1]}
        n_s += len(ev[t]["s"]); n_u += len(ev[t]["u"])
    assert n_s > 0 and n_u > 0                                                              # both sides are hit in this battle
    total = sum(x for e in ev for _, x in e["s"])
    assert total == sum(b["frames"][0]["s"]) - sum(b["frames"][-1]["s"])                    # damage sums to the structure hp lost (nothing heals structures)


def test_hud_totals_are_raw_sums_and_key_moments_follow_the_rules():
    d = mm_duel()
    P = d.env._st.P
    b = AD.raw_battle(P, d.env.army)
    kinds = AD.struct_kinds(P)
    tot = AD.totals(b["frames"], kinds, b["umax"], b["smax"])
    fort = [i for i, k in enumerate(kinds) if k in ("b", "d")]
    for t in (0, 3, 10, 24, 32):
        f = b["frames"][t]
        assert tot[t]["army"] == sum(u[1] for u in f["u"]) and tot[t]["army_max"] == sum(b["umax"]) == sum(P.cfg.units[u].hp for u, _ in d.env.army)
        assert tot[t]["fort"] == sum(f["s"][i] for i in fort) and tot[t]["fort_max"] == sum(P.s_hp0[i] for i in fort)
        assert tot[t]["core"] == f["s"][P.core_sid] and tot[t]["core_max"] == P.s_hp0[P.core_sid]
    assert tot[0]["army"] == tot[0]["army_max"] and tot[0]["fort"] == tot[0]["fort_max"] and tot[-1]["core"] == 0
    mom = AD.moments(b["frames"], kinds)
    assert mom[0] == [] and any("destroyed" in m for m in mom) and any("core" in m for m in mom)
    for t in range(1, len(mom)):
        a, c = b["frames"][t - 1], b["frames"][t]
        assert ("core" in mom[t]) == (c["s"][P.core_sid] < a["s"][P.core_sid])
        assert ("lost" in mom[t]) == any(x[1] > 0 >= y[1] for x, y in zip(a["u"], c["u"]))
        assert ("destroyed" in mom[t]) == any(kinds[j] != "w" and x > 0 >= y for j, (x, y) in enumerate(zip(a["s"], c["s"])))


def test_the_core_falls_at_tick_24_in_the_row_mm_battle_and_the_page_says_so():
    d = mm_duel()
    P = d.env._st.P
    page = AR.battle_page(P, d.env.army, M._struct_table(P), d.env.result)
    data = json.loads(page.split('id="data">')[1].split("</script>")[0])
    assert data["outcome"]["coreTick"] == 24 and data["outcome"]["ticks"] == 32 and data["outcome"]["breach"] is True
    assert data["S"] == 840 and len(data["frames"]) == 33 and data["who"] == "your"
    assert 'id="m-mine"' in page and "Your army" in page and "Fortress (defenders + buildings)" in page and "Core" in page
    agent = AR.battle_page(P, d.env.army, M._struct_table(P), d.env.result, mode="agent")
    assert _html.escape("Agent's army") in agent and "Your army" not in agent.split('<div id="app">')[1].split('<div class="bar">')[0]


# ----------------------------------------------------------------------------------------------------------------------------- Fight timing (decision 3)
LONG = ("archipelago-301000216", [(2, 7), (1, 8), (1, 32)])                              # a full 240-tick battle (horizon)


def test_fight_time_for_a_full_240_tick_battle_is_under_a_second_and_a_half(monkeypatch):
    monkeypatch.setenv("RAVELIN_DEPLOY", "1")                                              # the deployed mode: simulator + bundle only
    d = D.Duel(LONG[0], human_runs_path=None)
    P = d.env._st.P
    AD._CACHE.clear()
    t0 = time.perf_counter()
    b = AD.raw_battle(P, LONG[1], key=LONG[0])
    cold = time.perf_counter() - t0
    assert b["ticks"] == 240 and b["result"].reason == "horizon"
    assert cold < 1.5, cold
    t0 = time.perf_counter()
    assert AD.raw_battle(P, LONG[1], key=LONG[0]) is b                                      # cached by (layout, ordered army)
    assert time.perf_counter() - t0 < 0.01
    assert AD.raw_battle(P, LONG[1][::-1], key=LONG[0]) is not b                            # another deploy order is another battle


# ----------------------------------------------------------------------------------------------------------------------------- Watch on the same base
def test_watch_buttons_after_the_comparison_cards_open_watch_the_ai_on_the_same_layout(arena, monkeypatch):
    monkeypatch.setenv("RAVELIN_DEPLOY", "1")                                              # the bundle serves the episodes; no model is loaded
    at = new_app()
    fight(at, mm_draft())
    assert not [b for b in at.button if b.key and b.key.startswith("g_watch_")]            # nothing about agents before the report
    at.button(key="g_report_btn").click().run()
    keys = [b.key for b in at.button if b.key and b.key.startswith("g_watch_")]
    assert keys == ["g_watch_ppo_plain", "g_watch_ppo_d4", "g_watch_ddqn"]
    labels = [b.label for b in at.button if b.key in keys]
    assert labels == ["Watch PPO plain + TTA on this base", "Watch PPO+D4 + TTA on this base", "Watch DDQN + TTA on this base"]
    at.button(key="g_watch_ppo_d4").click().run()
    assert not at.exception, at.exception
    assert at.radio(key="mode_sel").value == "Watch the AI"
    assert at.selectbox(key="w_layout_DEV-validation").value == LAYOUT and at.selectbox(key="w_arm").value == "ppo_d4" and at.radio(key="w_view").value == "TTA"
    assert at.selectbox(key="w_split").value == "DEV-validation"


def test_watch_keeps_pause_at_every_decision_and_gets_the_same_in_board_player(arena, monkeypatch):
    monkeypatch.setenv("RAVELIN_DEPLOY", "1")
    at = new_app({"mode": "watch"})
    assert not at.exception, at.exception
    assert at.toggle(key="w_pause_dec").value is True and "w_skip" not in [b.key for b in at.button]      # default: step through every buy and deploy
    at.toggle(key="w_pause_dec").set_value(False).run()
    assert "w_skip" in [b.key for b in at.button]
    at.button(key="w_skip").click().run()
    assert not at.exception
    page = next(p.proto.srcdoc for p in _iframes(at))
    assert _html.escape("Agent's army") in page and 'id="step"' in page and 'id="keys" checked' in page and "Pause on key moments" in page
    data = json.loads(page.split('id="data">')[1].split("</script>")[0])
    assert data["S"] == 840 and data["who"] == "the agent's"


def _iframes(at):
    def walk(node):
        for _, ch in node.children.items():
            yield ch
            if hasattr(ch, "children"):
                yield from walk(ch)
    return [p for p in walk(at.main) if getattr(getattr(p, "proto", None), "srcdoc", "")]


# ----------------------------------------------------------------------------------------------------------------------------- science is frozen
PROTECTED = ["breachlens/sim.py", "breachlens/env.py", "breachlens/score.py", "breachlens/actions.py", "breachlens/layouts.py", "breachlens/constants.py", "breachlens/config.py",
             "breachlens/duel.py", "data/v2/unit_stats.json", "data/v2/pool_v1.json", "data/v2/pool_v1.sha256", "data/v2/design_sets.json", "data/v2/generator_hash.json",
             "data/v2/final_access.log", "data/v2/duel_cache.json", "scripts/make_replays.py", "scripts/build_duel_cache.py"]


def _git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)


def test_ledger_duel_cache_and_protected_files_are_byte_identical_to_main():
    assert hashlib.sha256((ROOT / "data" / "v2" / "final_access.log").read_bytes()).hexdigest() == LEDGER_SHA
    if _git("rev-parse", "--verify", "-q", "main").returncode != 0:
        pytest.skip("no main branch to compare with")
    for rel in PROTECTED:
        ref = subprocess.run(["git", "show", f"main:{rel}"], cwd=ROOT, capture_output=True)
        assert ref.returncode == 0, rel
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == hashlib.sha256(ref.stdout).hexdigest(), rel


def test_opponents_stay_hidden_in_the_arena_data_until_the_battle_ends(arena):
    at = new_app()
    blob = json.dumps(arena["g_arena"]["data"])
    assert "oracle" not in blob.lower() and "best_single" not in blob and "ppo_" not in blob and "ddqn" not in blob.lower()
    assert "g_rows" not in at.session_state
    fight(at, mm_draft())
    assert "g_rows" not in at.session_state and "g_watch_ppo_plain" not in [b.key for b in at.button]


def test_placement_data_has_green_tiles_and_costs_from_the_engine(arena):
    from app import theme as TH
    new_app()
    data = arena["g_arena"]["data"]
    assert data["okColor"] == TH.TILE_OK == TH.UNIT_FILL["bulwark"]
    assert [c["cost"] for c in data["cards"]] == [8, 28, 30, 36, 28] and data["gold"] == 240 and data["cap"] == 20
    assert {"mode", "gid", "svg", "cell", "cards", "gold", "cap", "tiles", "okColor", "draft", "msg"} == set(data)


def test_the_app_still_runs_with_the_torch_import_blocked_in_play_mode(arena, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=300).run()
    assert not at.exception
    fight(at, mm_draft())
    assert game(at).actions == MM_ACTIONS
