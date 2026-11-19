"""B18c / B18d: visual theme and content fixes (docs/SPEC_V2.md, Gate changelog row (nn)). Pure file / module checks; no training, no evaluation."""
import re
import tomllib
from pathlib import Path

import pytest

from app import render as R
from app import theme as T

ROOT = Path(__file__).resolve().parents[1]
APP_FILES = sorted((ROOT / "app").glob("*.py"))
HEX = re.compile(r"(?<![&\w])#[0-9a-fA-F]{3,8}\b")


@pytest.mark.parametrize("name,fg,bg,kind", T.TEXT_PAIRS, ids=[p[0] for p in T.TEXT_PAIRS])
def test_contrast_meets_wcag_aa(name, fg, bg, kind):
    assert T.contrast(fg, bg) >= T.THRESHOLD[kind], (name, fg, bg, round(T.contrast(fg, bg), 2))


def test_contrast_function_known_values():
    assert T.contrast("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert T.contrast("#3283BE", "#FEFDFE") == pytest.approx(4.04, abs=0.01)       # azure on the page: below 4.5, so large / bold / graphic only
    assert T.contrast("#FD614B", "#FEFDFE") == pytest.approx(2.96, abs=0.01)       # coral on the page: below 3, so shape only, never text


def test_azure_is_large_only_and_coral_is_shape_only():
    for name, fg, bg, kind in T.TEXT_PAIRS:
        if kind == "normal":
            assert T.PRIMARY not in (fg, bg) and fg != T.DANGER, name                # normal-size text: never azure, never coral text (coral may be the pill background)
    assert any(fg == T.ON_PRIMARY and bg == T.PRIMARY and kind == "large" for _, fg, bg, kind in T.TEXT_PAIRS)       # white on azure: declared bold-only
    assert not any(fg == T.DANGER and kind not in ("shape", "large") for _, fg, _, kind in T.TEXT_PAIRS)
    # source scan: azure / coral are never a text colour (CSS `color:` or SVG / canvas text fill) in app code, theme CSS or the replay script
    srcs = [p for p in APP_FILES] + [ROOT / "scripts" / "make_replays.py"]
    bad = []
    for p in srcs:
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r"(?<![-\w])color:\s*var\(--(primary|danger)\)", line):          # CSS text colour (background / border / accent-color are fine)
                bad.append(f"{p.name}:{i}")
            if ("<text" in line or "fillText" in line) and re.search(r"fill=\"\{T\.(PRIMARY|DANGER)\}\"", line):
                bad.append(f"{p.name}:{i}")
    assert not bad, bad
    assert T.TEXT_PAIRS and all(T.contrast(fg, bg) >= T.THRESHOLD[k] for _, fg, bg, k in T.TEXT_PAIRS)


def test_palette_is_the_spec():
    assert (T.BG, T.PANEL, T.PANEL_BORDER, T.TEXT, T.MUTED) == ("#FEFDFE", "#F3F7FB", "#D5E2EE", "#132A3E", "#5B6B7A")
    assert (T.PRIMARY, T.ON_PRIMARY, T.PRIMARY_TEXT) == ("#3283BE", "#FEFDFE", "#245F8C")
    assert (T.DANGER, T.ON_DANGER, T.DANGER_TEXT) == ("#FD614B", "#132A3E", "#C4402C")
    assert (T.CORE_FILL, T.CORE_BORDER, T.OUTLINE, T.CHOSEN_FRAME) == ("#132A3E", "#FD614B", "#132A3E", "#132A3E")
    assert T.UNIT_FILL == {"scout": "#F2C14E", "sniper": "#3283BE", "bulwark": "#6BAF92", "blaster": "#A788DD", "mender": "#F4A6C0"}
    assert T.UNIT_INK == {"scout": "#132A3E", "sniper": "#FEFDFE", "bulwark": "#132A3E", "blaster": "#132A3E", "mender": "#132A3E"}
    assert (T.BOARD_BG, T.GRID_LINE, T.WALL_FILL, T.WALL_HATCH, T.ZEBRA, T.HUMAN_ROW) == ("#EEF4FA", "#D5E2EE", "#C9D6E3", "#A9BCCF", "#F3F7FB", "#3283BE")
    assert T.HEAT_STOPS == ("#FEFDFE", "#3283BE", "#132A3E")
    assert (T.SIZE_BASE, T.SIZE_H1, T.SIZE_H2, T.SIZE_TABLE, T.RADIUS) == (16, 32, 22, 15, 6)
    assert (T.FONT_HEADING, T.FONT_BODY, T.FONT_NUM) == ("Fraunces", "Atkinson Hyperlegible Next", "Atkinson Hyperlegible Mono")


def test_old_palette_and_fonts_are_gone():
    old = ["#250104", "#3b0a0f", "#5a1a20", "#f4ebdd", "#c9afa8", "#d71a06", "#cdf73d", "#f6e7c8", "#9fb8ad", "#ff8a3d", "#e9a6c0", "#2e0306", "#4a1218"]
    files = [*APP_FILES, ROOT / "scripts" / "make_replays.py", ROOT / ".streamlit" / "config.toml", *(ROOT / "docs" / "replays").glob("*.html"), *(ROOT / "app" / "static" / "fonts").glob("*.txt")]
    for p in files:
        t = p.read_text().lower()
        assert not [c for c in old if c.lower() in t], p.name
        assert not re.search(r"maroon|tomato|neon|cream|bricolage|instrument sans|plex", t), p.name
    assert sorted(f.name for f in (ROOT / "app" / "static" / "fonts").glob("*.woff2")) == sorted([fn for _, fn, _, _ in T.FONTS] + [T.FONT_ITALIC[1]])


def test_hex_colours_only_in_theme():
    offenders = []
    for p in [*APP_FILES, ROOT / "scripts" / "make_replays.py"]:
        if p.name == "theme.py":
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if HEX.search(line) or re.search(r"rgba?\(\s*\d", line):
                offenders.append(f"{p.relative_to(ROOT)}:{i}")
    assert not offenders, offenders


def test_config_toml_matches_theme_and_enables_static_serving():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    th = cfg["theme"]
    for k, v in T.config_toml_theme().items():
        assert th[k].upper() == v.upper(), k
    assert th["base"] == "light" and th["borderColor"].upper() == T.PANEL_BORDER and th["font"] == T.FONT_BODY and th["headingFont"] == T.FONT_HEADING and th["codeFont"] == T.FONT_NUM
    assert cfg["server"]["enableStaticServing"] is True
    for ff in th["fontFaces"]:                                   # relative, local URLs only
        assert ff["url"].startswith(T.STATIC_URL_PREFIX) and (ROOT / "app" / ff["url"].removeprefix("app/")).exists()
    assert {f["family"] for f in th["fontFaces"]} == {T.FONT_HEADING, T.FONT_BODY, T.FONT_NUM}


def test_fonts_and_license_files_present():
    for fam, fn, weight, lic in (*T.FONTS, T.FONT_ITALIC):
        f, l = T.FONT_DIR / fn, T.FONT_DIR / lic
        assert f.exists() and f.stat().st_size > 5_000 and f.read_bytes()[:4] == b"wOF2", fn
        assert l.exists() and "SIL OPEN FONT LICENSE" in l.read_text().upper(), lic
        assert not re.search(r'with Reserved Font Name', l.read_text(), re.I), lic           # no RFN declared: the subsets may keep the family names
    assert (T.FONT_DIR / "SOURCES.txt").exists()
    src = (T.FONT_DIR / "SOURCES.txt").read_text()
    assert "Fraunces-Italic" in src and "b24448c43702fac4" in src and "Fraunces-Italic-opsz-wght600-700.woff2" in src
    assert "font-style:italic" in T.italic_face_css() and T.FONT_ITALIC[1] in T.italic_face_css() and T.FONT_ITALIC[1] not in T.font_face_css(True)


def test_no_network_urls_in_app_or_replays():
    bad = re.compile(r"""(url\(\s*['"]?https?:|src\s*=\s*['"]?(https?:)?//|href\s*=\s*['"]?https?:|@import|<link\b|fonts\.googleapis|fonts\.gstatic|cdn\.)""", re.I)
    files = [*APP_FILES, ROOT / ".streamlit" / "config.toml", ROOT / "scripts" / "make_replays.py", *(ROOT / "docs" / "replays").glob("*.html")]
    for p in files:
        assert not bad.search(p.read_text()), p.name
    for p in (ROOT / "docs" / "replays").glob("*.html"):
        assert "http" not in p.read_text().lower(), p.name          # replays: no URL of any kind (not even an svg namespace)
    for p in APP_FILES:                                          # the only http literal allowed in app code is the svg namespace
        assert not re.findall(r"https?://(?!www\.w3\.org/2000/svg)", p.read_text()), p.name


def test_replays_embed_fonts_and_are_themed():
    t = (ROOT / "docs" / "replays" / "spine_pods__ppo_plain__tta.html").read_text()
    assert t.count("@font-face") == len(T.FONTS) and "data:font/woff2;base64," in t
    for c in (T.BG, T.PANEL, T.DANGER, T.PRIMARY, T.TEXT):
        assert c in t
    assert R.DISCLAIMER in t.replace("&#x27;", "'")


def test_disclaimer_text_a1_and_one_per_screen():
    assert R.DISCLAIMER == ("Demo, not a result. DEV-validation layouts were held out from training and only monitored during it; "
                            "they were never used to select models.")
    assert "model selection" not in R.DISCLAIMER and "used for model selection" not in (ROOT / "breachlens" / "duel.py").read_text()
    src = "".join(p.read_text() for p in APP_FILES if p.name != "theme.py")
    assert src.count("DISCLAIMER}") == 1                          # rendered in exactly one place (the line under the top bar)


def test_legend_covers_units_defender_shapes_core_walls_buildings():
    kinds = dict(R.legend_items())
    for u in T.UNIT_ORDER:
        assert "unit:" + u in kinds
    assert {"gun", "lancer", "flak", "core", "wall", "building"} <= set(kinds)
    assert "square" in kinds["gun"] and "triangle" in kinds["lancer"] and "hexagon" in kinds["flak"]
    assert "number" in R.LEGEND_BUILDING and "value" in R.LEGEND_BUILDING
    assert "no separate defender-type channel" in R.AGENTS_SEE and "footprint radius" in R.AGENTS_SEE
    svg = R.legend_svg()
    assert svg.count("<polygon") >= 3 and "<rect" in svg


def test_defender_shapes_are_type_specific_in_both_modes():
    assert "<rect" in R.defender_shape("gun", 10, 10) and R.defender_shape("lancer", 10, 10).count(",") == 3
    assert len(R.defender_shape("flak", 10, 10).split("points=")[1].split('"')[1].split()) == 6
    import app.duel_ui as _  # noqa: F401  (import only: the duel board no longer blanks the defender type)
    assert 'if k == "d" else ex' not in (ROOT / "app" / "duel_ui.py").read_text()


def test_deploy_tiles_are_row_col_not_tile_at():
    assert R.deploy_text("sniper", 0, 3) == "Sniper (0,3)"
    assert R.pretty_label("Deploy sniper at border tile 3 (row 0, col 3)") == ("Deploy Sniper (0,3)", "border tile 3")
    assert R.pretty_label("Buy bulwark") == ("Buy bulwark", "")
    assert "{u}@{b}" not in (ROOT / "app" / "duel_ui.py").read_text()


def test_board_svg_uses_theme_and_has_no_blueprint_texture():
    ep = {"steps": [], "structs": [["c", 2, 5, ""], ["d", 3, 5, "lancer"], ["w", 4, 5, ""], ["b", 5, 8, 3]],
          "battle": {"frames": [{"t": 0, "u": [], "s": [1000] * 4}], "units": []}}
    svg, _ = R.grid_svg(ep, [[0, 0]], 14, 0, "total")
    assert T.BOARD_BG in svg and T.GRID_LINE in svg and "url(#hatch)" in svg and "<polygon" in svg
    assert "07142b" not in svg.lower() and "3f7fb8" not in svg.lower()
