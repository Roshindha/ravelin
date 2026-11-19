"""B18c / B18d: the single source of truth for the visual theme: palette v2 + fonts v2 (docs/SPEC_V2.md, Gate changelog row (nn) and its B18d addendum).

Every hex colour of ``app/`` and ``scripts/make_replays.py`` is defined HERE and nowhere else (test-enforced). Pure module: no streamlit import, so the
replay generator and the renderer can use it. Fonts are OFL, bundled under ``app/static/fonts/`` (no CDN, no network request).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

# ----------------------------------------------------------------------------------------------------------------------------- palette (v2, light)
BG = "#FEFDFE"               # page background (snowy white, flat)
PANEL = "#F3F7FB"
PANEL_BORDER = "#D5E2EE"
TEXT = "#132A3E"             # ink
MUTED = "#5B6B7A"
PRIMARY = "#3283BE"          # azure navy: selected controls, primary buttons, scrubber, heading rule, unit Sniper. LARGE / BOLD / GRAPHIC ONLY (4.04:1 on the page).
ON_PRIMARY = BG              # text on a primary button: bold, >= 16px
PRIMARY_TEXT = "#245F8C"     # azure text smaller than 18px
DANGER = "#FD614B"           # coral red: defenders, breach pill background, core ring. SHAPE ONLY (2.96:1 on the page): never a text colour.
ON_DANGER = TEXT             # text on the breach pill
DANGER_TEXT = "#C4402C"      # coral-coloured text
OUTLINE = TEXT               # 1.5px outline of units and defenders; 3px frame of the chosen action
CORE_FILL = TEXT             # ink diamond ...
CORE_BORDER = DANGER         # ... with a 2px coral ring
CORE_INK = BG                # the 'C' on the ink diamond
UNIT_FILL = {"scout": "#F2C14E", "sniper": PRIMARY, "bulwark": "#6BAF92", "blaster": "#A788DD", "mender": "#F4A6C0"}
UNIT_INK = {"scout": TEXT, "sniper": BG, "bulwark": TEXT, "blaster": TEXT, "mender": TEXT}
UNIT_ORDER = ("scout", "sniper", "bulwark", "blaster", "mender")          # the engine's UNIT_TYPES order (index t)
UNIT_ABBR = ("Sc", "Sn", "Bu", "Bl", "Me")
WALL_FILL = "#C9D6E3"
WALL_HATCH = "#A9BCCF"
BOARD_BG = "#EEF4FA"
GRID_LINE = PANEL_BORDER
ZEBRA = PANEL
HEAT_STOPS = (BG, PRIMARY, TEXT)                                        # sequential snow -> azure -> ink
# derived roles (aliases only, no new colours)
BUILDING_STROKE = TEXT
BUILDING_INK = TEXT
HP_BACK = PANEL_BORDER
HP_FILL = TEXT
ATTACK_RING = TEXT           # ring (offset from the unit outline) on a unit that attacked this tick
CHOSEN_FRAME = TEXT          # 3px outline of the chosen deploy tile and of the chosen top-5 row
CHOSEN_KEYLINE = BG          # 1px light fringe on both sides of the chosen tile frame, so it stays visible on dark heatmap cells
AXIS_INK = MUTED
HUMAN_ROW = PRIMARY          # 3px left border of the human row in the comparison table

# ----------------------------------------------------------------------------------------------------------------------------- typography
FONT_HEADING = "Fraunces"
FONT_BODY = "Atkinson Hyperlegible Next"
FONT_NUM = "Atkinson Hyperlegible Mono"
SIZE_BASE, SIZE_H1, SIZE_H2, SIZE_TABLE = 16, 32, 22, 15
RADIUS = 6
FONT_DIR = Path(__file__).resolve().parent / "static" / "fonts"
# (family, file, CSS weight, license file). Variable woff2 subsets (Latin): Fraunces keeps its opsz axis; none of the three OFL files has a Reserved Font Name.
FONTS = (
    (FONT_HEADING, "Fraunces-opsz-wght600-700.woff2", "600 700", "OFL-Fraunces.txt"),
    (FONT_BODY, "AtkinsonHyperlegibleNext-wght400-600.woff2", "400 600", "OFL-AtkinsonHyperlegibleNext.txt"),
    (FONT_NUM, "AtkinsonHyperlegibleMono-wght400-600.woff2", "400 600", "OFL-AtkinsonHyperlegibleMono.txt"),
)
FONT_ITALIC = (FONT_HEADING, "Fraunces-Italic-opsz-wght600-700.woff2", "600 700", "OFL-Fraunces.txt")      # real italic (tagline of the game app); not embedded in replays
STATIC_URL_PREFIX = "app/static/fonts/"                                 # Streamlit static serving (relative URL, same origin)

# ----------------------------------------------------------------------------------------------------------------------------- contrast
# (name, foreground, background, kind). kind: "normal" text 4.5:1; "large" = bold text >= 16px / large text / a graphic or UI shape, 3:1;
# "shape" = coral fills (identity carried by the ink outline pair, ratio reported only);
# "decorative" = hairlines and tints that carry no information alone (>= 1.1:1, ratio reported).
TEXT_PAIRS = (
    ("body text on page", TEXT, BG, "normal"),
    ("body text on panel / zebra row", TEXT, PANEL, "normal"),
    ("body text on board", TEXT, BOARD_BG, "normal"),
    ("muted text on page", MUTED, BG, "normal"),
    ("muted text on panel / zebra row", MUTED, PANEL, "normal"),
    ("muted text on board", MUTED, BOARD_BG, "normal"),
    ("azure text (< 18px) on page", PRIMARY_TEXT, BG, "normal"),
    ("azure text (< 18px) on panel", PRIMARY_TEXT, PANEL, "normal"),
    ("coral text on page", DANGER_TEXT, BG, "normal"),
    ("coral text on panel", DANGER_TEXT, PANEL, "normal"),
    ("breach pill: ink text on coral", ON_DANGER, DANGER, "normal"),
    ("building outline and number on board", BUILDING_INK, BOARD_BG, "normal"),
    ("core letter on ink diamond", CORE_INK, CORE_FILL, "normal"),
    ("primary button: snow text (bold, >= 16px) on azure", ON_PRIMARY, PRIMARY, "large"),
    ("azure on page (selected control, scrubber, rule)", PRIMARY, BG, "large"),
    ("azure on panel (human row border, selected control)", PRIMARY, PANEL, "large"),
    ("azure on board (heatmap mid stop, Sniper)", PRIMARY, BOARD_BG, "large"),
    ("chosen-action frame (ink) on board", OUTLINE, BOARD_BG, "large"),
    ("unit / defender outline (ink) on board", OUTLINE, BOARD_BG, "large"),
    ("defender outline (ink) on coral fill", OUTLINE, DANGER, "large"),
    ("core ring: coral on ink diamond", CORE_BORDER, CORE_FILL, "large"),
    ("coral defender fill on board (shape only; the ink outline pair above carries it)", DANGER, BOARD_BG, "shape"),
    ("coral on page (shape only, never text)", DANGER, BG, "shape"),
    ("wall on board (decorative; the hatch and legend identify it)", WALL_FILL, BOARD_BG, "decorative"),
    ("wall hatch on wall", WALL_HATCH, WALL_FILL, "decorative"),
    ("grid line on board", GRID_LINE, BOARD_BG, "decorative"),
    ("panel border on page", PANEL_BORDER, BG, "decorative"),
) + tuple((f"unit letter on {u}", UNIT_INK[u], UNIT_FILL[u], "large" if u == "sniper" else "normal") for u in UNIT_ORDER)
THRESHOLD = {"normal": 4.5, "large": 3.0, "shape": 0.0, "decorative": 1.1}
LARGE_ONLY = (PRIMARY,)                                                 # never the foreground of normal-size text (checked by a test)
SHAPE_ONLY = (DANGER,)                                                  # never a text colour (checked by a test)


def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast(fg: str, bg: str) -> float:
    """WCAG 2.x contrast ratio."""
    a, b = sorted((luminance(fg), luminance(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def heat_color(f: float) -> str:
    """0..1 -> snow .. azure .. ink, as an rgb() string."""
    s = [rgb(c) for c in HEAT_STOPS]
    p = min(max(f, 0.0), 1.0) * (len(s) - 1)
    i = min(int(p), len(s) - 2)
    t = p - i
    return "rgb(%d,%d,%d)" % tuple(round(s[i][k] + (s[i + 1][k] - s[i][k]) * t) for k in range(3))


def heat_scale_css() -> str:
    """The one gradient allowed by the spec: the heatmap scale."""
    return f"linear-gradient(90deg,{HEAT_STOPS[0]},{HEAT_STOPS[1]},{HEAT_STOPS[2]})"


# ----------------------------------------------------------------------------------------------------------------------------- B18e-a: "printed board-game box" (prototype only)
# Palette v2 only: the page is a flat azure FIELD with snow halftone dots; the only text that may sit directly on the field is large display text (snow,
# Fraunces >= 28 px: 4.04:1, the large-text rule). Every other piece of text sits inside a snow CARD (ink border) or a flat STICKER.
FIELD = PRIMARY
FIELD_INK = BG               # display text on the field: Fraunces >= SIZE_TITLE_MIN px only
FIELD_DOT = BG               # halftone dots
FIELD_DOT_OPACITY = 0.11
FIELD_DOT_GRID = 6           # px between dots
SIZE_TITLE, SIZE_TAGLINE, SIZE_FIELD_TEXT_MIN = 64, 28, 28
CARD_BG, CARD_BORDER = BG, TEXT
CARD_BORDER_W, CARD_RADIUS, FRAME_W = 2.5, 10, 3        # cards 2.5px ink, board frame 3px ink
STICKER_FILL, STICKER_HOT, STICKER_INK, STICKER_BORDER_W = BG, DANGER, TEXT, 2
TAPE = PRIMARY               # diagonal tape stripes on the border deploy tiles
STICKER_ROT = (-1.5, 1.0, -0.6, 1.8)                    # degrees, a few stickers "placed by hand" (end banner: -4)
BANNER_ROT = -4


def mix(fg: str, bg: str, t: float) -> str:
    """fg over bg at opacity t, as #rrggbb (computed, e.g. the halftone dot as seen on the field)."""
    a, b = rgb(fg), rgb(bg)
    return "#" + "".join(f"{round(a[i] * t + b[i] * (1 - t)):02X}" for i in range(3))


FIELD_DOT_BLEND = mix(FIELD_DOT, FIELD, FIELD_DOT_OPACITY)


def field_pattern_uri() -> str:
    """The halftone as a data: URI (an SVG pattern tile, no CSS gradient, no network)."""
    g = FIELD_DOT_GRID
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{g}" height="{g}"><circle cx="{g / 2}" cy="{g / 2}" r="1.1" fill="{FIELD_DOT}" '
           f'fill-opacity="{FIELD_DOT_OPACITY}"/></svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


TEXT_PAIRS = TEXT_PAIRS + (
    ("display text (>= 28px) snow on azure field", FIELD_INK, FIELD, "large"),
    ("ink on snow card", TEXT, CARD_BG, "normal"),
    ("ink on snow sticker", STICKER_INK, STICKER_FILL, "normal"),
    ("ink on coral sticker (price tag, breach, active phase, BREACH banner)", STICKER_INK, STICKER_HOT, "normal"),
    ("snow on ink sticker (HELD banner)", BG, TEXT, "normal"),
    ("ink card border on azure field (graphic)", CARD_BORDER, FIELD, "large"),
    ("snow halftone dots on azure field (decorative)", FIELD_DOT_BLEND, FIELD, "decorative"),
    ("azure tape stripe on snow deploy tile (decorative)", TAPE, BG, "decorative"),
)


def game_root_vars() -> str:
    return (f":root{{--field:{FIELD};--field-ink:{FIELD_INK};--card-bw:{CARD_BORDER_W}px;--card-r:{CARD_RADIUS}px;--frame-w:{FRAME_W}px;--sticker-bw:{STICKER_BORDER_W}px;"
            f"--dots:url({field_pattern_uri()});--dot-grid:{FIELD_DOT_GRID}px}}")


def js_theme() -> str:
    """The palette as a JSON object for the replay pages' script (so the script holds no colour literal)."""
    return json.dumps({
        "bg": BG, "panel": PANEL, "line": PANEL_BORDER, "text": TEXT, "muted": MUTED, "danger": DANGER, "primary": PRIMARY, "onPrimary": ON_PRIMARY,
        "outline": OUTLINE, "coreFill": CORE_FILL, "coreBorder": CORE_BORDER, "coreInk": CORE_INK, "unitFill": [UNIT_FILL[u] for u in UNIT_ORDER],
        "unitInk": [UNIT_INK[u] for u in UNIT_ORDER], "unitAbbr": list(UNIT_ABBR),
        "wallFill": WALL_FILL, "wallHatch": WALL_HATCH, "boardBg": BOARD_BG, "gridLine": GRID_LINE, "buildingStroke": BUILDING_STROKE,
        "buildingInk": BUILDING_INK, "hpBack": HP_BACK, "hpFill": HP_FILL, "attackRing": ATTACK_RING, "chosenFrame": CHOSEN_FRAME, "chosenKeyline": CHOSEN_KEYLINE,
        "heat": [list(rgb(c)) for c in HEAT_STOPS], "fontHeading": FONT_HEADING, "fontBody": FONT_BODY, "fontNum": FONT_NUM,
    }, separators=(",", ":"), sort_keys=True)


# ----------------------------------------------------------------------------------------------------------------------------- css
def font_face_css(embed: bool) -> str:
    """@font-face rules. embed=False: relative static-serving URLs (the app); embed=True: base64 data URIs (self-contained replays)."""
    out = []
    for fam, fn, weight, _ in FONTS:
        if embed:
            src = "url(data:font/woff2;base64," + base64.b64encode((FONT_DIR / fn).read_bytes()).decode() + ") format('woff2')"
        else:
            src = f"url({STATIC_URL_PREFIX}{fn}) format('woff2')"
        out.append(f"@font-face{{font-family:'{fam}';font-style:normal;font-weight:{weight};font-display:swap;src:{src}}}")
    return "".join(out)


def italic_face_css() -> str:
    """@font-face for the real Fraunces italic (relative static-serving URL, same origin; the replays do not use it)."""
    fam, fn, weight, _ = FONT_ITALIC
    return f"@font-face{{font-family:'{fam}';font-style:italic;font-weight:{weight};font-display:swap;src:url({STATIC_URL_PREFIX}{fn}) format('woff2')}}"


def root_vars() -> str:
    return (f":root{{--bg:{BG};--panel:{PANEL};--line:{PANEL_BORDER};--fg:{TEXT};--mut:{MUTED};--danger:{DANGER};--danger-text:{DANGER_TEXT};--on-danger:{ON_DANGER};"
            f"--primary:{PRIMARY};--primary-text:{PRIMARY_TEXT};--on-primary:{ON_PRIMARY};--board:{BOARD_BG};--grid:{GRID_LINE};--zebra:{ZEBRA};--outline:{OUTLINE};"
            f"--r:{RADIUS}px;--f-head:'{FONT_HEADING}',Georgia,serif;--f-body:'{FONT_BODY}',system-ui,sans-serif;--f-num:'{FONT_NUM}',ui-monospace,monospace}}")


def common_css() -> str:
    """Rules shared by the app and the replay pages."""
    return (
        f"body{{background:var(--bg);color:var(--fg);font-family:var(--f-body);font-size:{SIZE_BASE}px;line-height:1.5}}"
        f"h1,h2,h3{{font-family:var(--f-head);font-weight:700;font-optical-sizing:auto;letter-spacing:0;text-transform:none}}"
        f"h1{{font-size:{SIZE_H1}px}}h2{{font-size:{SIZE_H2}px}}"
        "h1,h2{border-bottom:2px solid var(--primary);padding-bottom:.15em}"
        ".num,code,td.num,.mono{font-family:var(--f-num);font-variant-numeric:tabular-nums;font-feature-settings:'tnum'}"
        ".mut{color:var(--mut)}"
        f".disclaimer{{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);color:var(--mut);padding:3px 12px;font-size:14px;margin:0 0 10px 0}}"
        f".pill{{display:inline-block;border-radius:999px;padding:1px 10px;font-size:14px;font-weight:600;line-height:1.5}}"
        f".pill.breach,.yes.pill{{background:var(--danger);color:var(--on-danger)}}"
        f".pill.held,.no.pill{{border:1px solid var(--fg);color:var(--fg)}}"
    )


def app_css() -> str:
    """The Streamlit page: flat light background, Atkinson Hyperlegible Next body, Fraunces headings, Atkinson Hyperlegible Mono numbers, 6px radius, no shadows."""
    return (
        "<style>" + font_face_css(False) + root_vars() + common_css() +
        f".stApp{{background:var(--bg);color:var(--fg);font-family:var(--f-body);font-size:{SIZE_BASE}px}}"
        "[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl']{display:none}"
        ".block-container{padding-top:1rem;max-width:1400px}"
        "[data-testid='stHeader']{background:transparent}"
        "*{box-shadow:none!important;text-shadow:none!important}"
        "[data-testid='stMarkdownContainer'],[data-testid='stWidgetLabel'] p,.stButton button p,label,input,select{font-family:var(--f-body)}"
        "[data-testid='stNumberInput'] input,[data-testid='stSliderThumbValue'],"
        "[data-testid='stSliderTickBarMin'],[data-testid='stSliderTickBarMax']{font-family:var(--f-num);font-variant-numeric:tabular-nums}"
        ".bar{display:flex;flex-wrap:wrap;gap:8px;margin:2px 0 10px 0}"
        f".badge{{border:1px solid var(--line);background:var(--panel);color:var(--fg);padding:2px 10px;font-size:14px;border-radius:var(--r)}}"
        f".badge.warn{{border-color:var(--danger);color:var(--fg);font-weight:600}}"
        f".panel{{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:10px 14px;margin-bottom:10px}}"
        f".outcome{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--primary);border-radius:var(--r);padding:6px 12px;margin:6px 0}}"
        ".outcome .k{color:var(--primary-text);font-weight:600}"
        f".legend{{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:8px 12px;font-size:14px}}"
        ".legend .row{display:flex;align-items:center;gap:8px;margin:3px 0}.legend svg{flex:none}"
        f".stButton button{{border-radius:var(--r);border:1px solid var(--line);background:var(--panel);color:var(--fg);padding:.25rem .4rem;font-family:var(--f-body)}}"
        ".stButton button p{font-size:15px;white-space:normal;overflow:visible;text-overflow:clip;font-weight:600;line-height:1.25}"
        ".stButton button [data-testid='stMarkdownContainer']{overflow:visible;text-overflow:clip}"        # Atkinson is wider than the old body font: no ellipsis on button labels
        f".stButton button[kind='primary']{{background:var(--primary);border-color:var(--primary);color:var(--on-primary)}}"
        ".stButton button[kind='primary'] p{color:var(--on-primary);font-size:16px;font-weight:600}"
        f"table.top{{width:100%;border-collapse:collapse;font-size:{SIZE_TABLE}px}}"
        f"table.top td,table.top th{{padding:5px 8px;text-align:left;border:0;border-left:3px solid transparent}}"
        "table.top th{color:var(--mut);font-weight:600;border-bottom:1px solid var(--line)}"
        "table.top tr:nth-child(even) td{background:var(--zebra)}"
        "table.top tr.chosen td{border-top:3px solid var(--outline);border-bottom:3px solid var(--outline);border-left:0}"
        "table.top tr.chosen td:first-child{border-left:3px solid var(--outline)}table.top tr.chosen td:last-child{border-right:3px solid var(--outline)}"
        "table.top tr.human td:first-child{border-left-color:var(--primary)}"
        "table.top td.num,table.top th.num{text-align:right}"
        "</style>"
    )


def config_toml_theme() -> dict:
    """The colours that .streamlit/config.toml must carry (the config file cannot import this module; a test compares them)."""
    return {"primaryColor": PRIMARY, "backgroundColor": BG, "secondaryBackgroundColor": PANEL, "textColor": TEXT, "borderColor": PANEL_BORDER}
