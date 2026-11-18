"""B18c: the single source of truth for the visual theme (docs/SPEC_V2.md, Gate changelog row (nn)).

Every hex colour of ``app/`` and ``scripts/make_replays.py`` is defined HERE and nowhere else (test-enforced). Pure module: no streamlit import, so the
replay generator and the renderer can use it. Fonts are OFL, bundled under ``app/static/fonts/`` (no CDN, no network request).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

# ----------------------------------------------------------------------------------------------------------------------------- palette
BG = "#250104"               # page background (flat)
PANEL = "#3B0A0F"
PANEL_BORDER = "#5A1A20"
TEXT = "#F4EBDD"             # cream
MUTED = "#C9AFA8"
TOMATO = "#D71A06"           # accent 1: defenders, breach, danger (shapes, text >= 18px bold, or button background with ON_TOMATO text)
ON_TOMATO = "#FEFDFE"
NEON = "#CDF73D"             # accent 2: chosen action, primary button, current scrubber position, key numbers
ON_NEON = BG
CORE_FILL = "#FEFDFE"
CORE_BORDER = TOMATO
UNIT_FILL = {"scout": "#F6E7C8", "sniper": NEON, "bulwark": "#9FB8AD", "blaster": "#FF8A3D", "mender": "#E9A6C0"}
UNIT_INK = BG
UNIT_ORDER = ("scout", "sniper", "bulwark", "blaster", "mender")          # the engine's UNIT_TYPES order (index t)
UNIT_ABBR = ("Sc", "Sn", "Bu", "Bl", "Me")
WALL_FILL = PANEL
WALL_HATCH = PANEL_BORDER
BOARD_BG = "#2E0306"
GRID_LINE = "#4A1218"
ZEBRA = BOARD_BG
HEAT_STOPS = (BG, TOMATO, NEON)                                         # sequential maroon -> tomato -> neon yellow
# derived roles (aliases only, no new colours)
BUILDING_STROKE = TEXT
BUILDING_INK = TEXT
HP_BACK = PANEL_BORDER
HP_FILL = TEXT
ATTACK_RING = CORE_FILL
CHOSEN_FRAME = NEON
AXIS_INK = MUTED

# ----------------------------------------------------------------------------------------------------------------------------- typography
FONT_HEADING = "Bricolage Grotesque"
FONT_BODY = "Instrument Sans"
FONT_NUM = "IBM Plex Mono"
SIZE_BASE, SIZE_H1, SIZE_H2, SIZE_TABLE = 16, 32, 22, 15
RADIUS = 6
FONT_DIR = Path(__file__).resolve().parent / "static" / "fonts"
# (family, file, CSS weight, license file). Variable fonts are pinned to wdth 100 and the weight range used; IBM Plex Mono is static.
FONTS = (
    (FONT_HEADING, "BricolageGrotesque-wght600-800.woff2", "600 800", "OFL-BricolageGrotesque.txt"),
    (FONT_BODY, "InstrumentSans-wght400-600.woff2", "400 600", "OFL-InstrumentSans.txt"),
    (FONT_NUM, "IBMPlexMono-Regular.woff2", "400", "OFL-IBMPlexMono.txt"),
    (FONT_NUM, "IBMPlexMono-SemiBold.woff2", "600", "OFL-IBMPlexMono.txt"),
)
STATIC_URL_PREFIX = "app/static/fonts/"                                 # Streamlit static serving (relative URL, same origin)

# ----------------------------------------------------------------------------------------------------------------------------- contrast
# (name, foreground, background, kind). kind: "normal" text needs 4.5:1; "large" (>= 18px bold, or a graphic / UI shape) needs 3:1.
TEXT_PAIRS = (
    ("body text on page", TEXT, BG, "normal"),
    ("body text on panel", TEXT, PANEL, "normal"),
    ("body text on board / zebra row", TEXT, BOARD_BG, "normal"),
    ("muted text on page", MUTED, BG, "normal"),
    ("muted text on panel", MUTED, PANEL, "normal"),
    ("muted text on board / zebra row", MUTED, BOARD_BG, "normal"),
    ("primary button: dark text on neon", ON_NEON, NEON, "normal"),
    ("key numbers: neon on page", NEON, BG, "normal"),
    ("key numbers: neon on panel", NEON, PANEL, "normal"),
    ("breach pill / tomato button: text on tomato", ON_TOMATO, TOMATO, "normal"),
    ("tomato large text (>= 18px bold) on page", TOMATO, BG, "large"),
    ("tomato shapes (defenders, core border) on board", TOMATO, BOARD_BG, "large"),
    ("core diamond on board", CORE_FILL, BOARD_BG, "large"),
    ("chosen-tile frame on board", NEON, BOARD_BG, "large"),
    ("building outline and number on board", BUILDING_INK, BOARD_BG, "normal"),
    ("held pill: cream outline text on page", TEXT, BG, "normal"),
) + tuple((f"unit letter on {u}", UNIT_INK, UNIT_FILL[u], "normal") for u in UNIT_ORDER)
THRESHOLD = {"normal": 4.5, "large": 3.0}


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
    """0..1 -> maroon .. tomato .. neon yellow, as an rgb() string."""
    s = [rgb(c) for c in HEAT_STOPS]
    p = min(max(f, 0.0), 1.0) * (len(s) - 1)
    i = min(int(p), len(s) - 2)
    t = p - i
    return "rgb(%d,%d,%d)" % tuple(round(s[i][k] + (s[i + 1][k] - s[i][k]) * t) for k in range(3))


def heat_scale_css() -> str:
    """The one gradient allowed by the spec: the heatmap scale."""
    return f"linear-gradient(90deg,{HEAT_STOPS[0]},{HEAT_STOPS[1]},{HEAT_STOPS[2]})"


def js_theme() -> str:
    """The palette as a JSON object for the replay pages' script (so the script holds no colour literal)."""
    return json.dumps({
        "bg": BG, "panel": PANEL, "line": PANEL_BORDER, "text": TEXT, "muted": MUTED, "tomato": TOMATO, "neon": NEON, "onTomato": ON_TOMATO,
        "coreFill": CORE_FILL, "coreBorder": CORE_BORDER, "unitFill": [UNIT_FILL[u] for u in UNIT_ORDER], "unitInk": UNIT_INK, "unitAbbr": list(UNIT_ABBR),
        "wallFill": WALL_FILL, "wallHatch": WALL_HATCH, "boardBg": BOARD_BG, "gridLine": GRID_LINE, "buildingStroke": BUILDING_STROKE,
        "buildingInk": BUILDING_INK, "hpBack": HP_BACK, "hpFill": HP_FILL, "attackRing": ATTACK_RING, "chosenFrame": CHOSEN_FRAME,
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


def root_vars() -> str:
    return (f":root{{--bg:{BG};--panel:{PANEL};--line:{PANEL_BORDER};--fg:{TEXT};--mut:{MUTED};--tomato:{TOMATO};--on-tomato:{ON_TOMATO};--neon:{NEON};"
            f"--on-neon:{ON_NEON};--board:{BOARD_BG};--grid:{GRID_LINE};--r:{RADIUS}px;--f-head:'{FONT_HEADING}',system-ui,sans-serif;"
            f"--f-body:'{FONT_BODY}',system-ui,sans-serif;--f-num:'{FONT_NUM}',ui-monospace,monospace}}")


def common_css() -> str:
    """Rules shared by the app and the replay pages."""
    return (
        f"body{{background:var(--bg);color:var(--fg);font-family:var(--f-body);font-size:{SIZE_BASE}px;line-height:1.5}}"
        f"h1,h2,h3{{font-family:var(--f-head);font-weight:700;letter-spacing:0;text-transform:none}}"
        f"h1{{font-size:{SIZE_H1}px}}h2{{font-size:{SIZE_H2}px}}"
        ".num,code,td.num,.mono{font-family:var(--f-num);font-variant-numeric:tabular-nums;font-feature-settings:'tnum'}"
        ".mut{color:var(--mut)}"
        f".disclaimer{{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);color:var(--mut);padding:3px 12px;font-size:14px;margin:0 0 10px 0}}"
        f".pill{{display:inline-block;border-radius:999px;padding:1px 10px;font-size:14px;font-weight:600;line-height:1.5}}"
        f".pill.breach,.yes.pill{{background:var(--tomato);color:var(--on-tomato)}}"
        f".pill.held,.no.pill{{border:1px solid var(--fg);color:var(--fg)}}"
    )


def app_css() -> str:
    """The Streamlit page: flat background, Instrument Sans body, Bricolage headings, Plex Mono numbers, 6px radius, no shadows."""
    return (
        "<style>" + font_face_css(False) + root_vars() + common_css() +
        f".stApp{{background:var(--bg);color:var(--fg);font-family:var(--f-body);font-size:{SIZE_BASE}px}}"
        "[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl']{display:none}"
        ".block-container{padding-top:1rem;max-width:1400px}"
        "[data-testid='stHeader']{background:transparent}"
        "*{box-shadow:none!important;text-shadow:none!important}"
        "[data-testid='stMarkdownContainer'],[data-testid='stWidgetLabel'] p,.stButton button p,label,input,select{font-family:var(--f-body)}"
        "[data-testid='stNumberInput'] input,[data-baseweb='select'] [data-testid='stSelectboxVirtualDropdown'],[data-testid='stSliderThumbValue'],"
        "[data-testid='stSliderTickBarMin'],[data-testid='stSliderTickBarMax']{font-family:var(--f-num);font-variant-numeric:tabular-nums}"
        ".bar{display:flex;flex-wrap:wrap;gap:8px;margin:2px 0 10px 0}"
        f".badge{{border:1px solid var(--line);background:var(--panel);color:var(--fg);padding:2px 10px;font-size:14px;border-radius:var(--r)}}"
        f".badge.warn{{border-color:var(--tomato);color:var(--fg);font-weight:600}}"
        f".panel{{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:10px 14px;margin-bottom:10px}}"
        f".outcome{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--neon);border-radius:var(--r);padding:6px 12px;margin:6px 0}}"
        ".outcome .k{color:var(--neon);font-weight:600}"
        f".legend{{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:8px 12px;font-size:14px}}"
        ".legend .row{display:flex;align-items:center;gap:8px;margin:3px 0}.legend svg{flex:none}"
        f".stButton button{{border-radius:var(--r);border:1px solid var(--line);background:var(--panel);color:var(--fg);padding:.25rem .6rem;font-family:var(--f-body)}}"
        ".stButton button p{font-size:15px;white-space:nowrap;overflow:visible;text-overflow:clip;font-weight:600}"
        f".stButton button[kind='primary']{{background:var(--neon);border-color:var(--neon);color:var(--on-neon)}}"
        ".stButton button[kind='primary'] p{color:var(--on-neon)}"
        f"table.top{{width:100%;border-collapse:collapse;font-size:{SIZE_TABLE}px}}"
        f"table.top td,table.top th{{padding:5px 8px;text-align:left;border:0;border-left:3px solid transparent}}"
        "table.top th{color:var(--mut);font-weight:600;border-bottom:1px solid var(--line)}"
        "table.top tr:nth-child(even) td{background:var(--board)}"
        "table.top tr.chosen td:first-child{border-left-color:var(--neon)}"
        "table.top td.num,table.top th.num{text-align:right}"
        "</style>"
    )


def config_toml_theme() -> dict:
    """The colours that .streamlit/config.toml must carry (the config file cannot import this module; a test compares them)."""
    return {"primaryColor": NEON, "backgroundColor": BG, "secondaryBackgroundColor": PANEL, "textColor": TEXT}
