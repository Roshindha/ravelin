"""B24: the browser side of the arena, in real (headless) Chrome through Playwright: the DOM the battle player draws and the placement board's events.

Skipped when Playwright or a Chrome / Chromium build is not installed (it is not in requirements-dev.txt; ``pip install playwright`` is enough with a system Chrome).
The assertions compare what the page SHOWS (HUD bars, per-piece bars and rings, floating numbers, banner) with the simulator's recorded raw fixed-point hp from
``app.arena_data`` (which tests/test_b24_arena.py ties to ``simulate`` tick by tick)."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from app import arena as AR
from app import arena_data as AD
from app import board_component as BC
from app import theme as T
from breachlens import duel as D
from tests.test_b18eb_app import LAYOUT, MM_ACTIONS, mm_draft, new_app  # noqa: F401

pw = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_replays", ROOT / "scripts" / "make_replays.py")
M = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("make_replays", M)
_spec.loader.exec_module(M)


@pytest.fixture(scope="module")
def browser():
    with pw.sync_playwright() as p:
        b = None
        for kw in ({"channel": "chrome"}, {}):
            try:
                b = p.chromium.launch(**kw)
                break
            except Exception:                                                      # noqa: BLE001 - no browser of this kind installed
                continue
        if b is None:
            pytest.skip("no Chrome / Chromium available to Playwright")
        yield b
        b.close()


def _rgb(hex_color: str) -> str:
    r, g, b = T.rgb(hex_color)
    return f"rgb({r}, {g}, {b})"


@pytest.fixture(scope="module")
def mm():
    d = D.Duel(LAYOUT, human_runs_path=None)
    AD.play_draft(d, mm_draft())
    P = d.env._st.P
    b = AD.raw_battle(P, d.env.army)
    kinds = AD.struct_kinds(P)
    page = AR.battle_page(P, d.env.army, M._struct_table(P), d.env.result, autoplay=False)
    return {"duel": d, "P": P, "b": b, "kinds": kinds, "page": page, "tot": AD.totals(b["frames"], kinds, b["umax"], b["smax"]), "ev": AD.damage_events(b["frames"]),
            "mom": AD.moments(b["frames"], kinds)}


def open_player(browser, html, **ctx):
    c = browser.new_context(**ctx)
    pg = c.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.set_content(html)
    pg.wait_for_function("window.rv !== undefined")
    return pg, errs


def fmt(raw: int, S: int = 840) -> str:
    return f"{round(raw / S):,}"


# ----------------------------------------------------------------------------------------------------------------------------- HUD bars
def test_the_three_hud_bars_show_the_simulators_raw_sums_at_several_ticks(browser, mm):
    pg, errs = open_player(browser, mm["page"])
    for t in (0, 1, 3, 10, 18, 24, 32):
        pg.evaluate(f"rv.go({t})")
        tot = mm["tot"][t]
        for key, (cur, mx) in {"mine": (tot["army"], tot["army_max"]), "fort": (tot["fort"], tot["fort_max"]), "core": (tot["core"], tot["core_max"])}.items():
            tr = pg.locator(f"#t-{key}")
            assert tr.get_attribute("role") == "progressbar"
            assert (int(tr.get_attribute("aria-valuenow")), int(tr.get_attribute("aria-valuemax")), tr.get_attribute("aria-valuemin")) == (cur, mx, "0")
            assert pg.locator(f"#v-{key}").inner_text() == f"{fmt(cur)} / {fmt(mx)}"                 # the number label, e.g. "1,240 / 1,800"
            assert tr.get_attribute("aria-valuetext") == f"{fmt(cur)} of {fmt(mx)}" and tr.get_attribute("aria-label")
            width = float(pg.evaluate(f"document.getElementById('f-{key}').style.width.replace('%','')"))
            assert abs(width - 100 * cur / mx) < 1e-3
    assert mm["tot"][0]["army"] == mm["tot"][0]["army_max"] and mm["tot"][32]["core"] == 0
    assert errs == []


def test_hud_labels_icons_and_colours_per_side(browser, mm):
    pg, _ = open_player(browser, mm["page"])
    assert [pg.locator(f"#m-{k} .lab").inner_text() for k in ("mine", "fort", "core")] == ["Your army", "Fortress", "Core"]
    fill = lambda k: pg.evaluate(f"getComputedStyle(document.getElementById('f-{k}')).backgroundColor")
    assert fill("mine") == _rgb(T.PRIMARY) == "rgb(50, 131, 190)" and fill("fort") == fill("core") == _rgb(T.DANGER) == "rgb(253, 97, 75)"
    assert pg.locator("#m-mine svg").count() == 1 and pg.locator("#m-fort svg").count() == 1                   # shield and tower icons
    assert "Agent&#x27;s army" in AR.battle_page(mm["P"], mm["duel"].env.army, M._struct_table(mm["P"]), mm["duel"].env.result, mode="agent")
    mono = pg.evaluate("getComputedStyle(document.getElementById('v-mine')).fontFamily")
    assert "Atkinson Hyperlegible Mono" in mono
    core_h = pg.evaluate("document.getElementById('t-core').getBoundingClientRect().height")
    mine_h = pg.evaluate("document.getElementById('t-mine').getBoundingClientRect().height")
    assert core_h < mine_h                                                                                      # the core bar is the smaller one


def test_a_bar_flashes_when_it_drops_and_animates_smoothly(browser, mm):
    pg, _ = open_player(browser, mm["page"])
    t = next(i for i in range(1, 33) if mm["tot"][i]["fort"] < mm["tot"][i - 1]["fort"])
    pg.evaluate(f"rv.go({t - 1})")
    pg.evaluate("rv.step(1)")
    assert pg.evaluate("document.getElementById('m-fort').classList.contains('drop')")
    assert "width" in pg.evaluate("getComputedStyle(document.getElementById('f-fort')).transitionProperty")
    pg2, _ = open_player(browser, mm["page"], reduced_motion="reduce")
    assert pg2.evaluate("getComputedStyle(document.getElementById('f-fort')).transitionDuration") in ("0s", "0.000001s")        # reduced motion: no animation


# ----------------------------------------------------------------------------------------------------------------------------- per-piece bars and rings
def test_troop_bars_and_fortress_rings_equal_the_per_object_hp(browser, mm):
    b, kinds = mm["b"], mm["kinds"]
    pg, _ = open_player(browser, mm["page"])
    cell_w = 46 - 12
    for t in (0, 2, 5, 9, 14, 24, 32):
        pg.evaluate(f"rv.go({t})")
        f = b["frames"][t]
        got_u = pg.evaluate("[...document.querySelectorAll('.u')].map(g => [+g.dataset.i, +g.dataset.hp, +g.querySelector('.ubar').getAttribute('width'), g.classList.contains('dead')])")
        for i, hp, w, dead in got_u:
            assert hp == f["u"][i][1] and abs(w - cell_w * hp / b["umax"][i]) < 1e-6 and dead == (hp <= 0)
        got_s = pg.evaluate("[...document.querySelectorAll('.s')].map(g => [+g.dataset.i, +g.dataset.hp, g.classList.contains('dead'), g.querySelector('.ring') ? g.querySelector('.ring').getAttribute('stroke-dasharray') : null, "
                            "g.querySelector('.ring') ? g.querySelector('.ring').getAttribute('stroke-dashoffset') : null])")
        for j, hp, dead, dash, off in got_s:
            assert hp == f["s"][j] and dead == (hp <= 0)
            if kinds[j] == "w":
                assert dash is None                                                                          # walls have no ring
            else:
                frac = hp / b["smax"][j]
                assert abs(float(dash.split()[0]) - frac) < 1e-9 and abs(float(off) + (1 - frac)) < 1e-9     # the arc left is the share of hp; the lost part starts at 12 o'clock
    # shapes differ by side: a bar above each troop, a ring around each defender / building
    assert pg.locator(".u .ubar").count() == len(b["units"]) and pg.locator(".s .ring").count() == sum(1 for k in kinds if k != "w")
    assert pg.locator(".s .ubar").count() == 0 and pg.locator(".u .ring").count() == 0
    stroke = pg.evaluate("getComputedStyle(document.querySelector('.ring')).stroke")
    assert stroke == _rgb(T.DANGER) and pg.evaluate("document.querySelector('.ubar').getAttribute('fill')") == T.PRIMARY


def test_a_dead_piece_shows_the_destroyed_mark_and_an_empty_bar_or_ring(browser, mm):
    b, kinds = mm["b"], mm["kinds"]
    pg, _ = open_player(browser, mm["page"])
    pg.evaluate("rv.go(32)")
    dead_s = [j for j, k in enumerate(kinds) if k != "w" and b["frames"][32]["s"][j] <= 0]
    assert dead_s
    j = dead_s[0]
    pg.wait_for_timeout(350)                                                                                     # the mark fades in (.2 s)
    assert pg.evaluate(f"document.querySelector('.s[data-i=\"{j}\"] .x') && getComputedStyle(document.querySelector('.s[data-i=\"{j}\"] .x')).opacity") == "1"
    assert pg.evaluate(f"document.querySelector('.s[data-i=\"{j}\"] .ring').getAttribute('stroke-dasharray')").startswith("0 ")
    alive_s = next(j for j, k in enumerate(kinds) if k != "w" and b["frames"][32]["s"][j] > 0)
    assert pg.evaluate(f"getComputedStyle(document.querySelector('.s[data-i=\"{alive_s}\"] .x')).opacity") == "0"


# ----------------------------------------------------------------------------------------------------------------------------- floating damage numbers
def test_every_floating_number_equals_the_raw_hp_delta_between_frames(browser, mm):
    kinds, ev = mm["kinds"], mm["ev"]
    pg, errs = open_player(browser, mm["page"])
    seen_fort = seen_mine = 0
    for t in range(1, 33):
        pg.evaluate(f"rv.go({t - 1})")
        pg.evaluate("document.querySelectorAll('#board text.dn').forEach(e => e.parentNode.parentNode.remove())")
        pg.evaluate("rv.step(1)")
        shown = pg.evaluate("[...document.querySelectorAll('#board text.dn:not(.left)')].map(e => [e.classList.contains('fort') ? 'fort' : 'mine', +e.dataset.raw, e.textContent])")
        want = [("fort", x, j) for j, x in ev[t]["s"] if kinds[j] != "w"] + [("mine", x, i) for i, x in ev[t]["u"]]
        assert sorted((k, r) for k, r, _ in shown) == sorted((k, r) for k, r, _ in want), t                  # one number per piece per tick, no more, no less
        for k, r, txt in shown:
            assert txt == "-" + str(float(f"{r / 840:.2f}")).rstrip("0").rstrip(".") if not float(f"{r / 840:.2f}").is_integer() else txt == f"-{int(r / 840)}"
        seen_fort += sum(1 for s in shown if s[0] == "fort"); seen_mine += sum(1 for s in shown if s[0] == "mine")
    assert seen_fort > 0 and seen_mine > 0 and errs == []


def test_numbers_use_the_text_safe_shades_and_a_left_over_label_that_goes_away(browser, mm):
    ev = mm["ev"]
    t = next(i for i in range(1, 33) if ev[i]["s"] and ev[i]["u"])
    pg, _ = open_player(browser, mm["page"])
    pg.evaluate(f"rv.go({t - 1})")
    pg.evaluate("rv.step(1)")
    col = lambda cls: pg.evaluate(f"getComputedStyle(document.querySelector('#board text.dn.{cls}')).fill")
    assert col("fort") == _rgb(T.PRIMARY_TEXT) == "rgb(36, 95, 140)" and col("mine") == _rgb(T.DANGER_TEXT) == "rgb(196, 64, 44)"
    assert _rgb(T.PRIMARY) != col("fort") and _rgb(T.DANGER) != col("mine")                                    # never the light azure / coral as small text
    left = pg.evaluate("[...document.querySelectorAll('#board text.dn.left')].map(e => [e.textContent, +e.dataset.left])")
    assert left and all(re.fullmatch(r"\d+/\d+", txt) for txt, _ in left)
    frame = mm["b"]["frames"][t]
    for txt, raw in left:
        assert raw in frame["s"] or raw in [u[1] for u in frame["u"]]                                           # hp left is the piece's raw hp after the hit
    pg.wait_for_timeout(750)
    assert pg.evaluate("document.querySelectorAll('#board text.dn.left').length") == 0                          # the label lasts ~600 ms
    pg.wait_for_timeout(500)
    assert pg.evaluate("document.querySelectorAll('#board text.dn').length") == 0


def test_numbers_are_not_drawn_when_scrubbing_or_stepping_back(browser, mm):
    pg, _ = open_player(browser, mm["page"])
    pg.evaluate("rv.go(10)")
    assert pg.evaluate("document.querySelectorAll('#board text.dn').length") == 0
    pg.evaluate("rv.step(-1)")
    assert pg.evaluate("document.querySelectorAll('#board text.dn').length") == 0


# ----------------------------------------------------------------------------------------------------------------------------- playback controls
def test_pause_on_key_moments_continue_space_and_arrow_keys(browser, mm):
    mom = mm["mom"]
    first = next(t for t, m in enumerate(mom) if m)
    pg, _ = open_player(browser, mm["page"].replace('"tickMs":480', '"tickMs":20'))
    assert pg.evaluate("document.getElementById('keys').checked")                                              # default ON
    pg.evaluate("rv.play()")
    pg.wait_for_function("rv.state().held !== null", timeout=5000)
    st = pg.evaluate("rv.state()")
    assert st["pos"] == first and not st["playing"] and pg.locator("#hold.on").count() == 1
    assert pg.locator("#why").inner_text().startswith("Paused: ")
    pg.keyboard.press("Space")                                                                                  # Space = Continue
    assert pg.evaluate("rv.state().held") is None or pg.evaluate("rv.state().pos") >= first
    pg.wait_for_timeout(200)
    nxt = [t for t, m in enumerate(mom) if m and t > first]
    if nxt:
        pg.wait_for_function("rv.state().held !== null || rv.state().pos >= 32", timeout=5000)
        assert pg.evaluate("rv.state().pos") in nxt + [32]
    pg.evaluate("rv.go(5)")
    pg.keyboard.press("ArrowRight")
    assert pg.evaluate("rv.state().pos") == 6
    pg.keyboard.press("ArrowLeft")
    assert pg.evaluate("rv.state().pos") == 5
    pg.click("#step")
    assert pg.evaluate("rv.state().pos") == 6
    pg.click("#back")
    assert pg.evaluate("rv.state().pos") == 5


def test_the_key_moment_toggle_off_plays_through_and_every_speed_button_exists(browser, mm):
    pg, _ = open_player(browser, mm["page"].replace('"tickMs":480', '"tickMs":10'))
    pg.uncheck("#keys")
    pg.click('button[data-s="4"]')
    assert pg.evaluate("rv.state().speed") == 4 and [pg.locator(f'button[data-s="{s}"]').count() for s in (1, 2, 4)] == [1, 1, 1]
    pg.evaluate("rv.play()")
    pg.wait_for_function("rv.state().pos >= 32", timeout=5000)
    assert not pg.evaluate("rv.state().playing") and pg.locator("#play").inner_text() == "Replay"
    pg.keyboard.press("Space")                                                                                  # Space on the finished battle restarts it
    assert pg.evaluate("rv.state().pos") <= 2


def test_moments_in_the_page_are_the_same_as_arena_data_moments(browser, mm):
    mom = mm["mom"]
    pg, _ = open_player(browser, mm["page"].replace('"tickMs":480', '"tickMs":10'))
    pg.evaluate("rv.play()")
    stops = []
    for _ in range(40):
        pg.wait_for_function("rv.state().held !== null || rv.state().pos >= 32", timeout=5000)
        s = pg.evaluate("rv.state()")
        if s["held"] is None:
            break
        stops.append(s["pos"])
        pg.keyboard.press("Space")
    assert stops == [t for t, m in enumerate(mom) if m and t < 32]


def test_status_strip_and_banner_at_the_real_core_death_tick(browser, mm):
    pg, _ = open_player(browser, mm["page"])
    pg.evaluate("rv.go(23)")
    assert pg.evaluate("document.getElementById('banner').classList.contains('show')") is False
    pg.evaluate("rv.go(24)")
    assert pg.evaluate("document.getElementById('banner').classList.contains('show')") is True
    assert pg.locator("#bt").inner_text() == "BREACH" and "tick 24" in pg.locator("#bs").inner_text().lower()
    pg.evaluate("rv.go(32)")
    assert pg.evaluate("document.getElementById('banner').classList.contains('show')") is True                 # and it stays
    s = pg.locator("#status").inner_text()
    assert s.startswith("Tick 32") and "troops alive" in s and "buildings destroyed" in s and "building value destroyed" in s and "core hp 0%" in s
    alive = sum(1 for u in mm["b"]["frames"][32]["u"] if u[1] > 0)
    assert f"{alive} of 6 troops alive" in s


def test_a_held_battle_gets_the_held_sticker_on_the_last_tick(browser):
    d = D.Duel(LAYOUT, human_runs_path=None)
    AD.play_draft(d, [(0, 3)])
    P = d.env._st.P
    page = AR.battle_page(P, d.env.army, M._struct_table(P), d.env.result, autoplay=False)
    pg, _ = open_player(browser, page)
    last = d.env.result.ticks
    pg.evaluate(f"rv.go({last - 1})") if last > 1 else None
    pg.evaluate(f"rv.go({last})")
    assert pg.locator("#bt").inner_text() == "HELD" and pg.evaluate("document.getElementById('banner').classList.contains('show')")


def test_ranges_and_paths_overlays_toggle(browser, mm):
    pg, _ = open_player(browser, mm["page"])
    pg.evaluate("rv.go(6)")
    assert pg.locator("circle.rng").count() == 0 and pg.locator("polyline.path").count() == 0
    pg.check("#rng")
    assert pg.locator("circle.rng.d").count() >= 1 and pg.locator("circle.rng.u").count() >= 1
    pg.check("#pth")
    assert pg.locator("polyline.path").count() >= 1
    pg.evaluate("rv.go(0)")
    assert pg.locator("polyline.path").count() == 0                                                              # nothing walked yet
    pg.uncheck("#rng"); pg.uncheck("#pth")
    assert pg.locator("circle.rng").count() == 0


# ----------------------------------------------------------------------------------------------------------------------------- mobile (390 px)
def test_battle_player_at_390px_fits_and_buttons_are_44px(browser, mm):
    pg, _ = open_player(browser, mm["page"], viewport={"width": 390, "height": 844}, device_scale_factor=2, has_touch=True, is_mobile=True)
    assert pg.evaluate("document.documentElement.scrollWidth") <= 390
    for sel in ("#play", "#step", "#back", '.speed button[data-s="1"]', '.speed button[data-s="2"]', '.speed button[data-s="4"]', "#scrub"):
        assert pg.evaluate(f"document.querySelector('{sel}').getBoundingClientRect().height") >= 44, sel
    assert pg.evaluate("[...document.querySelectorAll('.opts label')].every(l => l.getBoundingClientRect().height >= 44)")
    pg.evaluate("rv.go(5)")
    pg.evaluate("document.getElementById('hold').classList.add('on')")
    assert pg.evaluate("document.querySelector('#cont').getBoundingClientRect().height") >= 44
    assert pg.evaluate("document.getElementById('board').getBoundingClientRect().width") <= 390
    pg.tap("#step")
    assert pg.evaluate("rv.state().pos") == 6


# ----------------------------------------------------------------------------------------------------------------------------- placement board
@pytest.fixture
def plan_data(arena):
    new_app()
    return json.loads(json.dumps(arena["g_arena"]["data"]))


def open_plan(browser, data, **ctx):
    c = browser.new_context(**ctx)
    pg = c.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    html = ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><style>" + T.root_vars() + BC.CSS + "</style>"
            "<div id=root><div class='bl-board'></div></div><script type=module>const src = " + json.dumps(BC.JS).replace("</", "<\\/") +
            "; const mod = await import(URL.createObjectURL(new Blob([src], {type: 'text/javascript'}))); window.__t = [];"
            "window.__cleanup = mod.default({data: " + json.dumps(data).replace("</", "<\\/") + ", setTriggerValue: (k, v) => window.__t.push([k, JSON.parse(JSON.stringify(v))]),"
            " parentElement: document.getElementById('root')}); window.__ready = true;</script>")
    pg.set_content(html)
    pg.wait_for_function("window.__ready === true")
    return pg, errs


def tile(pg, b, **kw):
    pg.click(f'rect.hit[data-b="{b}"]', **kw)


def triggers(pg):
    return pg.evaluate("window.__t")


def draft_of(pg):
    return pg.evaluate("[...document.querySelectorAll('.ghostp')].map(g => [+g.dataset.u, +g.dataset.b])")


def test_select_a_card_with_keys_and_tap_a_green_tile_to_place_it(browser, plan_data):
    pg, errs = open_plan(browser, plan_data)
    assert pg.evaluate("document.querySelectorAll('.card[aria-pressed=true]').length") == 1 and "240 gold left" in pg.locator("#pl-gold").inner_text()
    assert pg.locator("rect.hit.legal").count() == 52                                                          # every free border tile is highlighted
    assert pg.evaluate("getComputedStyle(document.querySelector('rect.hit.legal')).fill") == _rgb(T.TILE_OK)
    pg.keyboard.press("2")
    assert pg.locator('.card[data-u="1"]').get_attribute("aria-pressed") == "true"
    tile(pg, 0)
    assert draft_of(pg) == [[1, 0]] and "212 gold left" in pg.locator("#pl-gold").inner_text() and "1 / 20 units" in pg.locator("#pl-army").inner_text()
    pg.keyboard.press("3")
    tile(pg, 4)
    assert draft_of(pg) == [[1, 0], [2, 4]] and "182 gold left" in pg.locator("#pl-gold").inner_text()
    assert pg.locator("#pl-fight").inner_text() == "Fight! 2 units, 182 gold left"
    assert pg.locator("rect.hit.legal").count() == 50 and errs == [] and triggers(pg) == []                    # placing sends nothing to the engine
    assert pg.locator(".ghostp use").count() == 2 and pg.locator(".ghostp text").first.text_content() == "1"


def test_shift_tap_places_up_to_five_of_the_unit_on_adjacent_free_tiles(browser, plan_data):
    pg, _ = open_plan(browser, plan_data)
    pg.keyboard.press("1")
    tile(pg, 10, modifiers=["Shift"])
    assert draft_of(pg) == [[0, b] for b in (10, 11, 12, 13, 14)]
    pg.click("#pl-clear")
    pg.keyboard.press("2")                                                                                       # 28 gold each: 240 pays for 8, Shift places 5
    tile(pg, 30, modifiers=["Shift"])
    assert len(draft_of(pg)) == 5 and [b for _, b in draft_of(pg)] == [30, 31, 32, 33, 34] and "100 gold left" in pg.locator("#pl-gold").inner_text()
    tile(pg, 40, modifiers=["Shift"])                                                                            # only 3 more fit in the gold
    assert len(draft_of(pg)) == 8 and "16 gold left" in pg.locator("#pl-gold").inner_text()


def test_line_of_five_toggle_is_the_touch_version_of_shift_tap(browser, plan_data):
    pg, _ = open_plan(browser, plan_data, viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True)
    pg.keyboard.press("1")
    pg.click("#pl-line")
    assert pg.locator("#pl-line").get_attribute("aria-pressed") == "true"
    pg.tap('rect.hit[data-b="0"]')
    assert [b for _, b in draft_of(pg)] == [0, 1, 2, 3, 4]
    assert pg.evaluate("document.documentElement.scrollWidth") <= 390
    for sel in (".card", "#pl-line", "#pl-undo", "#pl-clear", "#pl-fight"):
        assert pg.evaluate(f"[...document.querySelectorAll('{sel}')].every(e => e.getBoundingClientRect().height >= 44)"), sel


def test_tap_a_ghost_to_remove_it_with_refund_undo_and_clear(browser, plan_data):
    pg, _ = open_plan(browser, plan_data)
    pg.keyboard.press("2")
    for b in (0, 1, 2):
        tile(pg, b)
    assert "156 gold left" in pg.locator("#pl-gold").inner_text()
    tile(pg, 1)                                                                                                 # tap a placed ghost: removed, refunded
    assert draft_of(pg) == [[1, 0], [1, 2]] and "184 gold left" in pg.locator("#pl-gold").inner_text()
    pg.click("#pl-undo")
    assert draft_of(pg) == [[1, 0]] and "212 gold left" in pg.locator("#pl-gold").inner_text()
    pg.keyboard.press("z")
    assert draft_of(pg) == [] and pg.locator("#pl-fight").is_disabled() and pg.locator("#pl-undo").is_disabled()
    tile(pg, 3)
    pg.click("#pl-clear")
    assert draft_of(pg) == [] and "240 gold left" in pg.locator("#pl-gold").inner_text() and pg.locator("#pl-fight").inner_text().startswith("Fight (place")


def test_an_unaffordable_tap_asks_the_app_for_the_engines_message_and_changes_nothing(browser, plan_data):
    pg, _ = open_plan(browser, plan_data)
    pg.keyboard.press("3")
    for b in range(8):
        tile(pg, b)                                                                                             # 8 Bulwarks = 240 gold
    assert pg.locator("rect.hit.legal").count() == 0 and "0 gold left" in pg.locator("#pl-gold").inner_text()
    tile(pg, 20)
    t = triggers(pg)
    assert [k for k, _ in t] == ["illegal"] and t[0][1]["unit"] == 2 and t[0][1]["tile"] == 20 and t[0][1]["draft"] == [[2, b] for b in range(8)]
    assert len(draft_of(pg)) == 8                                                                                # nothing placed
    # the app answers with the engine's message
    assert "cannot afford" in AD.probe_message(LAYOUT, t[0][1]["draft"], t[0][1]["unit"], t[0][1]["tile"])


def test_unit_cap_and_a_tap_on_a_wall_tile_are_refused_client_side_too(browser, plan_data):
    data = json.loads(json.dumps(plan_data))
    blocked = [t["b"] for t in data["tiles"] if not t["ok"]]
    pg, _ = open_plan(browser, data)
    pg.keyboard.press("1")
    tile(pg, 0, modifiers=["Shift"]); tile(pg, 5, modifiers=["Shift"]); tile(pg, 10, modifiers=["Shift"]); tile(pg, 15, modifiers=["Shift"])
    assert len(draft_of(pg)) == 20                                                                               # the cap
    tile(pg, 40)
    assert triggers(pg)[-1][0] == "illegal" and "unit cap" in AD.probe_message(LAYOUT, draft_of(pg), 0, 40)
    if blocked:
        pg2, _ = open_plan(browser, data)
        tile(pg2, blocked[0])
        assert triggers(pg2)[-1][0] == "illegal"


def test_fight_sends_the_draft_and_the_engine_plays_the_row_mm_script(browser, plan_data):
    pg, errs = open_plan(browser, plan_data)
    pg.keyboard.press("2")
    for b in range(4):
        tile(pg, b)
    pg.keyboard.press("3")
    for b in (4, 5):
        tile(pg, b)
    assert pg.locator("#pl-fight").inner_text() == "Fight! 6 units, 68 gold left"
    pg.click("#pl-fight")
    kind, payload = triggers(pg)[-1]
    assert kind == "fight" and payload["draft"] == [[1, 0], [1, 1], [1, 2], [1, 3], [2, 4], [2, 5]] == [list(x) for x in mm_draft()]
    d = D.Duel(LAYOUT, human_runs_path=None)
    AD.play_draft(d, payload["draft"])
    assert d.actions == MM_ACTIONS and (round(d.human_row()["score"], 3), d.human_row()["breach"], d.env.result.ticks) == (0.4, True, 32)
    assert errs == []


def test_two_click_orders_with_the_same_ordered_deploys_send_the_same_draft_and_give_the_same_outcome(browser, plan_data):
    """Order A: the straight script. Order B: extra units placed, a card changed, ghosts tapped off, Undo and Clear used on the way: the final ordered deploys are the same."""
    def finish(pg):
        pg.click("#pl-fight")
        return triggers(pg)[-1][1]["draft"]
    a, _ = open_plan(browser, plan_data)
    a.keyboard.press("2")
    for b in range(4):
        tile(a, b)
    a.keyboard.press("3")
    for b in (4, 5):
        tile(a, b)
    b_, _ = open_plan(browser, plan_data)
    b_.keyboard.press("1"); tile(b_, 20); tile(b_, 21); b_.keyboard.press("4"); tile(b_, 22)
    b_.click("#pl-clear")
    b_.keyboard.press("2"); tile(b_, 0); tile(b_, 1); tile(b_, 9); tile(b_, 9)                                   # placed then tapped off again
    b_.keyboard.press("3"); tile(b_, 13); b_.click("#pl-undo")
    b_.keyboard.press("2"); tile(b_, 2); tile(b_, 3)
    b_.keyboard.press("3"); tile(b_, 4); tile(b_, 5)
    da, db = finish(a), finish(b_)
    assert da == db
    r = []
    for dr in (da, db):
        d = D.Duel(LAYOUT, human_runs_path=None)
        AD.play_draft(d, dr)
        r.append((d.actions, d.human_row(), d.env.result.ticks))
    assert r[0] == r[1]


def test_the_plan_board_source_stays_free_of_network_and_hex_colours():
    assert "http" not in BC.JS and "http" not in BC.CSS and not re.findall(r"#[0-9A-Fa-f]{6}\b", BC.JS + BC.CSS)
