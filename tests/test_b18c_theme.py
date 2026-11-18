"""B18c: visual theme and content fixes (docs/SPEC_V2.md, Gate changelog row (nn)). Pure file / module checks; no training, no evaluation."""
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
    assert T.contrast("#D71A06", "#250104") == pytest.approx(3.72, abs=0.01)       # tomato on maroon: large / shapes only (never small text)
    assert ("tomato large text (>= 18px bold) on page", T.TOMATO, T.BG, "large") in T.TEXT_PAIRS
    assert not any(fg == T.TOMATO and kind == "normal" for _, fg, _, kind in T.TEXT_PAIRS)


def test_palette_is_the_spec():
    assert (T.BG, T.PANEL, T.PANEL_BORDER, T.TEXT, T.MUTED) == ("#250104", "#3B0A0F", "#5A1A20", "#F4EBDD", "#C9AFA8")
    assert (T.TOMATO, T.NEON, T.CORE_FILL, T.CORE_BORDER) == ("#D71A06", "#CDF73D", "#FEFDFE", "#D71A06")
    assert T.UNIT_FILL == {"scout": "#F6E7C8", "sniper": "#CDF73D", "bulwark": "#9FB8AD", "blaster": "#FF8A3D", "mender": "#E9A6C0"}
    assert (T.BOARD_BG, T.GRID_LINE, T.WALL_FILL, T.WALL_HATCH) == ("#2E0306", "#4A1218", "#3B0A0F", "#5A1A20")
    assert T.HEAT_STOPS == ("#250104", "#D71A06", "#CDF73D")
    assert (T.SIZE_BASE, T.SIZE_H1, T.SIZE_H2, T.SIZE_TABLE, T.RADIUS) == (16, 32, 22, 15, 6)


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
    assert th["borderColor"].upper() == T.PANEL_BORDER and th["font"] == T.FONT_BODY and th["headingFont"] == T.FONT_HEADING and th["codeFont"] == T.FONT_NUM
    assert cfg["server"]["enableStaticServing"] is True
    for ff in th["fontFaces"]:                                   # relative, local URLs only
        assert ff["url"].startswith(T.STATIC_URL_PREFIX) and (ROOT / "app" / ff["url"].removeprefix("app/")).exists()
    assert {f["family"] for f in th["fontFaces"]} == {T.FONT_HEADING, T.FONT_BODY, T.FONT_NUM}


def test_fonts_and_license_files_present():
    for fam, fn, weight, lic in T.FONTS:
        f, l = T.FONT_DIR / fn, T.FONT_DIR / lic
        assert f.exists() and f.stat().st_size > 5_000 and f.read_bytes()[:4] == b"wOF2", fn
        assert l.exists() and "SIL OPEN FONT LICENSE" in l.read_text().upper(), lic


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
    for c in (T.BG, T.PANEL, T.TOMATO, T.NEON):
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
