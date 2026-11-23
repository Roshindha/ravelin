"""B18e-a: the short-battle check (the engine is right, the battle really lasts 1 tick) and the game-feel prototype (app/streamlit_app.py, app/icons.py,
app/game_view.py). DEV-validation only; no training, no FINAL / transfer / LAB-DESIGN."""
import base64
import importlib.util
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import arena as AR
from app import arena_data as AD
from app import game_view as GV
from app import icons as IC
from app import render as R
from app import theme as T
from breachlens import duel as D
from breachlens.actions import encode_buy, encode_deploy
from breachlens.constants import END_BUY
from breachlens.env import BreachEnv
from breachlens.sim import simulate
from tests._arena_util import mini_page

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
LAYOUT = "spine_pods-300000056"
_spec = importlib.util.spec_from_file_location("make_replays", ROOT / "scripts" / "make_replays.py")
M = importlib.util.module_from_spec(_spec)
sys.modules["make_replays"] = M
_spec.loader.exec_module(M)


# ------------------------------------------------------------------------------------------------ item 1: 1 Scout at (0,3) is a real 1-tick battle
def test_one_scout_at_0_3_ends_in_one_tick_in_engine_env_ui_frames_and_sim():
    tile = next(i for i, b in enumerate(D.BORDER) if tuple(b) == (0, 3))
    duel = D.Duel(LAYOUT, human_runs_path=None)
    duel.buy("scout"); duel.end_buy(); duel.deploy("scout", tile)
    assert duel.finished and duel.env.result.ticks == 1 and duel.env.result.reason == "all_dead" and duel.env.result.score == 0.0
    frames = M._battle(duel.env)["frames"]                                  # what the UI slider / player show: ticks 0 and 1, nothing is dropped
    assert [f["t"] for f in frames] == [0, 1] and frames[0]["u"][0][1] == 1000 and frames[1]["u"][0][1] == 0
    raw = BreachEnv(layouts=[duel.layout], role="train")                   # the same actions through the bare env
    raw.reset(seed=D.reset_seed(LAYOUT), options={"layout_index": 0})
    for a in (encode_buy(0), END_BUY, encode_deploy(0, tile)):
        raw.step(a)
    assert (raw.result.ticks, raw.result.score) == (1, 0.0)
    assert simulate(duel.env._st.P, duel.env.army).ticks == 1               # and the simulator called directly
    # cause: a Lancer (5.4 tiles, 15120 dmg) and a Gun (3.6 tiles, 10080 dmg) cover (0,3); Scout hp is 13440
    P = duel.env._st.P
    assert P.cfg.units[0].hp == 13440 and sorted(d.damage for d in P.def_stats)[-1] == 15120


# ------------------------------------------------------------------------------------------------ icons
def test_icons_are_valid_distinct_svg_and_theme_coloured():
    assert set(IC.SYMBOLS) == set(IC.ALL) and len(set(IC.SYMBOLS.values())) == len(IC.ALL)          # every silhouette is different
    for n in IC.ALL:
        root = ET.fromstring(IC.svg_icon(n))
        assert root.tag.endswith("svg") and root.attrib["viewBox"] == "0 0 32 32"
    assert len({IC.SYMBOLS[t] for t in IC.TURRETS}) == 3 and all(T.DANGER in IC.SYMBOLS[t] for t in IC.TURRETS)
    for u in IC.UNITS:
        assert T.UNIT_FILL[u] in IC.SYMBOLS[u]
    assert "url(#wallstone)" in IC.SYMBOLS["wall"] and "wallstone" in IC.pattern_defs()
    assert all(f'id="ic-{n}"' in IC.symbol_defs() for n in IC.ALL)
    assert IC.building_icon(0) == "b1" and IC.building_icon(2) == "b2" and IC.building_icon(9) == "b3"
    assert not re.findall(r"https?://(?!www\.w3\.org/2000/svg)", Path(IC.__file__).read_text())


def test_board_svg_is_valid_and_hero_sized():
    structs = [["c", 2, 5, ""], ["d", 3, 5, "lancer"], ["w", 4, 5, ""], ["b", 5, 8, 3]]
    svg = GV.board_svg(structs, None, [(1, 0, 3, 1000)], overlay={(0, 1): "x"}, zone=[(0, 2)], selected=(0, 2), preview=(0, 0, 2), chosen=(0, 3))
    ET.fromstring(svg)
    assert GV.BOARD_PX >= 640 and f'width="{GV.FULL_PX}"' in svg and "#ic-core" in svg and "#ic-b3" in svg and "#ic-lancer" in svg and "#ic-sniper" in svg


# ------------------------------------------------------------------------------------------------ player
@pytest.fixture(scope="module")
def played():
    duel = D.Duel(LAYOUT, human_runs_path=None)
    for u in ("scout", "scout", "sniper", "bulwark"):
        duel.buy(u)
    duel.end_buy()
    for u, b in (("scout", 18), ("scout", 20), ("sniper", 22), ("bulwark", 24)):
        duel.deploy(u, b)
    return duel, M._battle(duel.env), M._struct_table(duel.env._st.P)


def test_player_is_self_contained_and_built_from_the_raw_frames(played):
    duel, battle, structs = played
    raw = AD.raw_battle(duel.env._st.P, duel.env.army)
    page = AR.player_html(structs, raw, {"breach": False, "ticks": duel.env.result.ticks, "reasonText": "x"}, AD.ranges(duel.env._st.P))
    assert not re.search(r"""(https?:|src\s*=|href\s*=\s*['"]?[^#'"\s]|@import|<link|fetch\(|XMLHttpRequest|WebSocket|import\()""", page)
    assert page.count("@font-face") == len(T.FONTS) and "data:font/woff2;base64," in page
    data = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S).group(1))
    assert data["frames"] == raw["frames"] and data["structs"] == structs and data["units"] == [t for t, _ in battle["units"]]
    for needle in ("id=\"play\"", 'data-s="1"', 'data-s="2"', 'data-s="4"', "Tick ", "BREACH", "HELD", "prefers-reduced-motion", "aria-live", 'id="step"', 'id="cont"', 'id="keys" checked'):
        assert needle in page, needle
    assert not re.search(r"glow|box-shadow|text-shadow|drop-shadow|filter:|gradient", page, re.I)
    durations = [float(x) for x in re.findall(r"(?:animation|transition)[^;{}]*?\s(\.\d+|0?\.\d+)s", page)] + [0.2]
    assert max(durations) <= 0.3 and min(d for d in durations if d > 0) >= 0.15


def test_player_has_one_slot_per_frame_tick_and_unit(played):
    duel, battle, _ = played
    frames = battle["frames"]
    assert frames[0]["t"] == 0 and frames[-1]["t"] == duel.env.result.ticks
    assert all(len(f["u"]) == len(battle["units"]) for f in frames)


# ------------------------------------------------------------------------------------------------ the app
@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setenv("BREACHLENS_DUEL_RUNS", str(tmp_path / "runs.jsonl"))
    return tmp_path / "runs.jsonl"


def md(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_prototype_play_flow_and_opponents_stay_hidden_until_report(arena, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the duel cache was opened before the report was requested")
    monkeypatch.setattr(D, "load_cache", boom)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    text = md(at)
    assert "Ravelin" in text and "Breach Lens" not in text and R.DISCLAIMER in text and text.count(R.DISCLAIMER) == 1
    assert [r.label for r in at.radio][0] == "Mode" and list(at.radio[0].options) == ["Play", "Watch the AI"]
    assert arena["g_arena"]["data"]["mode"] == "plan" and "g_buy_scout" not in [b.key for b in at.button]      # no quantity / "start deploying" step any more
    at.session_state["_sim_fight"] = {"draft": [[0, 3], [0, 4], [1, 5]], "n": 1}                                   # place two Scouts and a Sniper, press Fight
    at.run()
    at.run()
    assert not at.exception and at.session_state["g_duel"].finished
    assert [b.key for b in at.button if b.key == "g_report_btn"] == ["g_report_btn"]
    assert "g_rows" not in at.session_state and "Full comparison table" not in [e.label for e in at.expander]          # no opponent row, no table (the About text only names them)
    monkeypatch.undo()
    at.button(key="g_report_btn").click().run()
    assert not at.exception
    text = md(at)
    for name in ("You", "PPO plain + TTA", "PPO+D4 + TTA", "DDQN + TTA", "BestSingle", "Oracle", "search baseline, not the optimum"):
        assert name in text, name


def test_prototype_honesty_items_and_about_panel(arena):
    at = AppTest.from_file(APP, default_timeout=120).run()
    text = md(at)
    assert "DEV-validation" in text
    assert any("About this demo" == e.label for e in at.expander)
    src = (ROOT / "app" / "streamlit_app.py").read_text()
    for needle in ("What is TTA", "Who are the agents", "results_headline()", "IN-SAMPLE", "not a policy"):
        assert needle in src, needle


def test_legacy_app_is_removed_and_no_dependency_was_added():
    for gone in ("legacy_app.py", "duel_ui.py", "game_prototype.py"):
        assert not (ROOT / "app" / gone).exists(), gone
    assert not (ROOT / "tests" / "test_b18a_app.py").exists() and not (ROOT / "tests" / "test_b18b_duel_app.py").exists()
    reqs = (ROOT / "requirements.txt").read_text().lower()
    assert "image-coordinates" not in reqs and "components" not in reqs and "streamlit==1.67.0" in reqs



# ------------------------------------------------------------------------------------------------ B18e-a follow-up: "printed board-game box" art direction
def proto_css() -> str:
    return re.search(r'PROTO_CSS = """(.*?)"""', (ROOT / "app" / "streamlit_app.py").read_text(), re.S).group(1)


def test_contrast_pairs_for_the_field_cards_and_stickers_are_registered():
    pairs = {n: (fg, bg, k) for n, fg, bg, k in T.TEXT_PAIRS}
    assert pairs["display text (>= 28px) snow on azure field"] == (T.BG, T.PRIMARY, "large")                  # snow on azure: large-only (4.04:1)
    assert pairs["ink on snow card"][2] == "normal" and pairs["ink on coral sticker (price tag, breach, active phase, BREACH banner)"][:2] == (T.TEXT, T.DANGER)
    assert pairs["snow halftone dots on azure field (decorative)"][2] == "decorative" and pairs["azure tape stripe on snow deploy tile (decorative)"][2] == "decorative"
    assert T.contrast(T.BG, T.PRIMARY) < 4.5 and T.contrast(T.BG, T.PRIMARY) >= 3.0
    assert T.contrast(T.FIELD_DOT_BLEND, T.FIELD) >= T.THRESHOLD["decorative"]
    assert (T.FIELD, T.FIELD_INK, T.CARD_BG, T.CARD_BORDER) == (T.PRIMARY, T.BG, T.BG, T.TEXT)
    assert (T.CARD_BORDER_W, T.CARD_RADIUS, T.FRAME_W, T.STICKER_BORDER_W) == (2.5, 10, 3, 2) and T.BANNER_ROT == -4 and all(abs(r) <= 2 for r in T.STICKER_ROT)


def test_no_normal_size_text_sits_directly_on_the_azure_field():
    assert T.SIZE_TITLE >= T.SIZE_FIELD_TEXT_MIN and T.SIZE_TAGLINE >= T.SIZE_FIELD_TEXT_MIN == 28 and T.FIELD_INK == T.BG
    css = proto_css()
    for sel in (".titleblock h1", ".titleblock .tagline"):                      # the only rules that put snow text on the field: both display-sized
        rule = re.search(re.escape(sel) + r"\{([^}]*)\}", css).group(1)
        assert "color:var(--field-ink)" in rule and int(re.search(r"font-size:(\d+|__[TG]__)px", rule).group(1).replace("__T__", str(T.SIZE_TITLE)).replace("__G__", str(T.SIZE_TAGLINE))) >= 28
    assert css.count("color:var(--field-ink)") == 2                              # nothing else uses the field ink
    at = AppTest.from_file(APP, default_timeout=120).run()
    for demo in (None, "deploy"):
        if demo:
            at.query_params["demo"] = demo
            at = AppTest.from_file(APP, default_timeout=120)
            at.query_params["demo"] = demo
            at.run()
        for _, el in at.main.children.items():                                   # every top-level element is the title block, a card, a card grid or the About / table expander
            kind = type(el).__name__
            if kind == "Markdown":
                assert any(k in el.value for k in ('class="titleblock"', 'class="card', "<style>")), el.value[:80]
            elif kind == "Block":
                assert el.key and el.key.startswith(("card_", "grid_")), el.key
            else:
                assert kind == "Expander", kind


def test_stickers_cards_and_pattern_follow_the_spec():
    css = proto_css()
    assert "background-color:var(--field)" in css and "background-image:var(--dots)" in css
    assert not re.search(r"shadow|glow|blur|gradient|backdrop|filter", css, re.I)                       # no shadow / glow / gradient / glass in the prototype CSS
    caps = [m.start() for m in re.finditer(r"text-transform:uppercase", css)]
    assert len(caps) == 1 and ".sticker{" in css[css.rfind("\n", 0, caps[0]) - 200: caps[0]]            # uppercase only on stickers
    assert "font-weight:600" in re.search(r"\.sticker\{([^}]*)\}", css).group(1) and "var(--f-body)" in re.search(r"\.sticker\{([^}]*)\}", css).group(1)
    svg = base64.b64decode(T.field_pattern_uri().split(",", 1)[1]).decode()
    assert f'width="{T.FIELD_DOT_GRID}"' in svg and f'fill-opacity="{T.FIELD_DOT_OPACITY}"' in svg and T.FIELD_DOT in svg and "Gradient" not in svg and 0.10 <= T.FIELD_DOT_OPACITY <= 0.12
    assert T.FIELD_DOT_GRID == 6
    assert "id=\"tape\"" in IC.pattern_defs() and T.TAPE in IC.pattern_defs()
    assert "prefers-reduced-motion" in css and "rotate(var(--rot" in css
    d = IC.disc_svg("sniper")
    ET.fromstring(d)
    assert T.UNIT_FILL["sniper"] in d


def test_player_banner_is_a_rotated_sticker():
    page = mini_page({"breach": True, "ticks": 0, "reasonText": "x"})
    assert f"rotate({T.BANNER_ROT}deg)" in page and "#banner.breach{background:var(--danger)" in page and "#banner.held{background:var(--bg)" in page
    assert page.count("text-transform:uppercase") == 1 and 'class="zone"' not in page and "url(#tape)" in page


# ------------------------------------------------------------------------------------------------ B18e-a follow-up: the row (mm) script through the default app
MM_BUYS = ["sniper"] * 4 + ["bulwark"] * 2                                 # row (mm) (g): buy sniper x4, bulwark x2, END BUY, deploy in order on the lowest legal tiles (0-5)


def _lowest(duel, unit):
    from breachlens.actions import encode_deploy
    m, u = duel.env.action_masks(), D.parse_unit(unit)
    return next(b for b in range(52) if m[encode_deploy(u, b)])


def test_row_mm_script_is_identical_through_breachlens_duel_and_the_default_app(arena):
    d = D.Duel(LAYOUT, human_runs_path=None)
    for u in MM_BUYS:
        d.buy(u)
    d.end_buy()
    for u in MM_BUYS:
        d.deploy(u, _lowest(d, u))
        if d.finished:
            break
    row = d.human_row()
    assert (round(row["score"], 3), round(row["d"], 3), row["breach"], d.env.result.ticks) == (0.400, 0.0, True, 32)          # the B18b numbers
    assert d.actions == [1, 1, 1, 1, 2, 2, 265, 57, 58, 59, 60, 113, 114]                                                        # tiles 0..5 = actions 5 + 52u + b
    # the same script through the app is tests/test_b18eb_app.py::test_row_mm_script_through_the_arena_is_the_row_mm_outcome (placement draft -> Fight)
    assert AD.canonical_actions([(D.parse_unit(u), i) for i, u in enumerate(MM_BUYS)]) == d.actions


def test_tagline_uses_the_real_fraunces_italic():
    css = proto_css()
    assert re.search(r"\.tagline\{[^}]*font-style:italic", css) and "font-family:var(--f-head)" in re.search(r"\.tagline\{([^}]*)\}", css).group(1)
    assert (T.FONT_DIR / T.FONT_ITALIC[1]).read_bytes()[:4] == b"wOF2" and T.FONT_ITALIC[1] in T.italic_face_css()
    assert "italic_face_css()" in (ROOT / "app" / "streamlit_app.py").read_text()
