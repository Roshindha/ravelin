"""B18a / B18c: pure rendering helpers for the Streamlit app (no streamlit import): the board SVG, the legend, label helpers. Colours and fonts come from app/theme.py."""
from __future__ import annotations

import base64
import functools
import html
import math
import re

from app import theme as T

CELL = 38
UCOL = [T.UNIT_FILL[u] for u in T.UNIT_ORDER]                  # scout, sniper, bulwark, blaster, mender
UAB = list(T.UNIT_ABBR)
UNIT_NAMES = list(T.UNIT_ORDER)
BG = T.BOARD_BG
LINE = T.GRID_LINE
FOOT = 26                                                      # height of the heat-scale strip under the board

# One disclaimer per screen (row (nn) A1 / A2): a single slim line under the top bar.
DISCLAIMER = ("Demo, not a result. DEV-validation layouts were held out from training and only monitored during it; "
              "they were never used to select models.")
AGENTS_SEE = ("Agents see defender positions; they get no separate defender-type channel, but the type is recoverable from the footprint radius in the "
              "threat channel (observation spec).")
LEGEND_BUILDING = "Building (number = value weight)"

heat_color = T.heat_color


# ----------------------------------------------------------------------------------------------------------------------------- labels
_DEPLOY = re.compile(r"^Deploy (\w+) at border tile (\d+) \(row (\d+), col (\d+)\)$")


def fmt_tile(row: int, col: int) -> str:
    return f"({row},{col})"


def pretty_label(label: str) -> tuple[str, str]:
    """(display text, tooltip). 'Deploy sniper at border tile 3 (row 0, col 3)' -> ('Deploy Sniper (0,3)', 'border tile 3'); other labels pass through."""
    m = _DEPLOY.match(label)
    if not m:
        return label, ""
    u, b, r, c = m.groups()
    return f"Deploy {u.capitalize()} {fmt_tile(int(r), int(c))}", f"border tile {b}"


def deploy_text(unit: str, row: int, col: int) -> str:
    return f"{unit.capitalize()} {fmt_tile(row, col)}"


# ----------------------------------------------------------------------------------------------------------------------------- svg pieces
@functools.lru_cache(maxsize=1)
def font_style() -> str:
    """@font-face with base64 data URIs INSIDE the svg: an svg shown as an <img> cannot use the page's fonts, and no network request may be made."""
    parts = []
    for fam, fn, weight, _ in T.FONTS:
        if fam == T.FONT_HEADING:                                   # the board SVG has no heading text
            continue
        b64 = base64.b64encode((T.FONT_DIR / fn).read_bytes()).decode()
        parts.append(f"@font-face{{font-family:'{fam}';font-weight:{weight};src:url(data:font/woff2;base64,{b64}) format('woff2')}}")
    return "<style>" + "".join(parts) + "</style>"


def _hp_bar(x, y, hp):
    w = CELL - 10
    return (f'<rect x="{x + 5}" y="{y + CELL - 7}" width="{w}" height="3" fill="{T.HP_BACK}"/>'
            f'<rect x="{x + 5}" y="{y + CELL - 7}" width="{w * hp / 1000:.1f}" height="3" fill="{T.HP_FILL}"/>')


def defender_shape(kind: str, cx: float, cy: float, scale: float = 1.0) -> str:
    """Gun = square, Lancer = triangle, Flak = hexagon (coral fill, 1.5px ink outline). Unknown type: circle."""
    k = (kind or "").lower()
    r = 13 * scale
    if k == "gun":
        return f'<rect x="{cx - r + 1:.1f}" y="{cy - r + 1:.1f}" width="{2 * r - 2:.1f}" height="{2 * r - 2:.1f}" fill="{T.DANGER}" stroke="{T.OUTLINE}" stroke-width="1.5"/>'
    if k == "lancer":
        pts = f"{cx:.1f},{cy - r:.1f} {cx + r:.1f},{cy + r - 2:.1f} {cx - r:.1f},{cy + r - 2:.1f}"
        return f'<polygon points="{pts}" fill="{T.DANGER}" stroke="{T.OUTLINE}" stroke-width="1.5"/>'
    if k == "flak":
        pts = " ".join(f"{cx + r * math.cos(math.radians(60 * i)):.1f},{cy + r * math.sin(math.radians(60 * i)):.1f}" for i in range(6))
        return f'<polygon points="{pts}" fill="{T.DANGER}" stroke="{T.OUTLINE}" stroke-width="1.5"/>'
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r - 1:.1f}" fill="{T.DANGER}" stroke="{T.OUTLINE}" stroke-width="1.5"/>'


def _struct(s, hp):
    k, r, c, ex = s
    x, y = c * CELL, r * CELL
    dead = hp <= 0
    g = f'<g opacity="{0.25 if dead else 1}">'
    m = CELL / 2
    if k == "w":
        g += f'<rect x="{x + 2}" y="{y + 2}" width="{CELL - 4}" height="{CELL - 4}" fill="url(#hatch)" stroke="{T.WALL_HATCH}" stroke-width="1"/>'
    elif k == "b":
        g += (f'<rect x="{x + 5}" y="{y + 5}" width="{CELL - 10}" height="{CELL - 10}" fill="none" stroke="{T.BUILDING_STROKE}" stroke-width="2"/>'
              f'<text x="{x + m}" y="{y + m + 5}" text-anchor="middle" fill="{T.BUILDING_INK}" font-size="14" font-family="\'{T.FONT_NUM}\'">{html.escape(str(ex))}</text>')
    elif k == "c":
        g += (f'<polygon points="{x + m},{y + 3} {x + CELL - 3},{y + m} {x + m},{y + CELL - 3} {x + 3},{y + m}" fill="{T.CORE_FILL}" stroke="{T.CORE_BORDER}" stroke-width="2"/>'
              f'<text x="{x + m}" y="{y + m + 4}" text-anchor="middle" fill="{T.CORE_INK}" font-size="12" font-weight="600" font-family="\'{T.FONT_BODY}\'">C</text>')
    else:
        g += defender_shape(str(ex), x + m, y + m)
    g += "</g>"
    if k not in ("w",) and 0 < hp < 1000:
        g += _hp_bar(x, y, hp)
    if dead and k != "w":
        g += f'<path d="M{x + 7} {y + 7}L{x + CELL - 7} {y + CELL - 7}M{x + CELL - 7} {y + 7}L{x + 7} {y + CELL - 7}" stroke="{T.OUTLINE}" stroke-width="2"/>'
    return g


def _unit(t, col, row, hp, st):
    if hp <= 0:
        return ""
    x, y = col * CELL, row * CELL
    ring = f'<circle cx="{x + CELL / 2}" cy="{y + CELL / 2}" r="16" fill="none" stroke="{T.ATTACK_RING}" stroke-width="2"/>' if st == 2 else ""
    return (f'<circle cx="{x + CELL / 2}" cy="{y + CELL / 2}" r="12" fill="{UCOL[t]}" stroke="{T.OUTLINE}" stroke-width="1.5"/>{ring}'
            f'<text x="{x + CELL / 2}" y="{y + CELL / 2 + 4}" text-anchor="middle" fill="{T.UNIT_INK[T.UNIT_ORDER[t]]}" font-size="11" font-weight="600" font-family="\'{T.FONT_BODY}\'">{UAB[t]}</text>'
            + (_hp_bar(x, y, hp) if hp < 1000 else ""))


def _defs(fonts: bool = True) -> str:
    return ((font_style() if fonts else "") + "<defs>"
            f'<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="{T.WALL_FILL}"/>'
            f'<line x1="0" y1="0" x2="0" y2="6" stroke="{T.WALL_HATCH}" stroke-width="2"/></pattern>'
            f'<linearGradient id="heatscale" x1="0" x2="1" y1="0" y2="0">'
            + "".join(f'<stop offset="{i / (len(T.HEAT_STOPS) - 1):.2f}" stop-color="{c}"/>' for i, c in enumerate(T.HEAT_STOPS)) + "</linearGradient></defs>")


def _heat_scale(x, y, mx: float | None) -> str:
    if mx is None:
        return ""
    return (f'<rect x="{x}" y="{y}" width="120" height="8" fill="url(#heatscale)" stroke="{T.PANEL_BORDER}" stroke-width="1"/>'
            f'<text x="{x}" y="{y + 20}" fill="{T.MUTED}" font-size="11" font-family="\'{T.FONT_NUM}\'">0</text>'
            f'<text x="{x + 120}" y="{y + 20}" text-anchor="end" fill="{T.MUTED}" font-size="11" font-family="\'{T.FONT_NUM}\'">{mx:.4f}</text>'
            f'<text x="{x + 130}" y="{y + 8}" fill="{T.MUTED}" font-size="12" font-family="\'{T.FONT_BODY}\'">deploy probability on border tiles</text>')


def grid_svg(ep: dict, border: list, n: int, pos: int, heat_mode: str, tile_labels: list | None = None) -> tuple[str, str]:
    """(svg, heat caption) for scrubber position `pos`: decision steps first (0 .. n_dec-1), then battle frames."""
    n_dec = len(ep["steps"])
    W = n * CELL
    pad = 20
    out = [_defs(), f'<rect x="0" y="0" width="{W}" height="{W}" fill="{T.BOARD_BG}"/>']
    caption, scale_max = "", None
    if pos < n_dec:
        s = ep["steps"][pos]
        m = s["action"] - 5
        if heat_mode == "total":
            vals, caption = s["heat_tile"], "per-tile total over all unit types"
        else:
            u = m // len(border) if 0 <= m < 5 * len(border) else -1
            if u >= 0:
                caption = f"unit type of the chosen action: {UNIT_NAMES[u]}"
            else:
                u = max(range(5), key=lambda i: (s["heat_unit"][i], -i))
                caption = f"no deploy chosen at this step; showing the unit type with the most deploy probability ({UNIT_NAMES[u]})"
            vals = s["heat"][u]
        mx = max(vals + [1e-12])
        if s["heat_total"] > 0:
            scale_max = mx
            for b, v in enumerate(vals):
                r, c = border[b]
                out.append(f'<rect x="{c * CELL + 1}" y="{r * CELL + 1}" width="{CELL - 2}" height="{CELL - 2}" fill="{heat_color(v / mx)}" opacity="{0.9 if v > 0 else 0.35}"/>')
            caption += f"; colour scale 0 to {mx:.4f} (max over border tiles); total deploy probability {s['heat_total']:.4f}"
        else:
            caption = "no deploy action is legal at this step, so the heatmap is empty (total deploy probability 0)"
        structs = [(st, 1000) for st in ep["structs"]]
        units = [(t, border[b][1], border[b][0], 1000, 0) for t, b in s["army"]]
    else:
        f = ep["battle"]["frames"][pos - n_dec]
        structs = [(st, f["s"][i]) for i, st in enumerate(ep["structs"])]
        units = [(ep["battle"]["units"][i][0], u[0] % n, u[0] // n, u[1], u[2]) for i, u in enumerate(f["u"])]
    for i in range(n + 1):                                             # thin 1px grid lines
        out.append(f'<line x1="{i * CELL}" y1="0" x2="{i * CELL}" y2="{W}" stroke="{T.GRID_LINE}" stroke-width="1"/>'
                   f'<line x1="0" y1="{i * CELL}" x2="{W}" y2="{i * CELL}" stroke="{T.GRID_LINE}" stroke-width="1"/>')
    for i in range(n):
        out.append(f'<text x="{i * CELL + CELL / 2}" y="-6" text-anchor="middle" fill="{T.AXIS_INK}" font-size="11" font-family="\'{T.FONT_NUM}\'">{i}</text>'
                   f'<text x="-7" y="{i * CELL + CELL / 2 + 4}" text-anchor="end" fill="{T.AXIS_INK}" font-size="11" font-family="\'{T.FONT_NUM}\'">{i}</text>')
    out += [_struct(s, hp) for s, hp in structs]
    for b in tile_labels or []:                                        # duel: border tile numbers on the free tiles (the handle for the tile input)
        r, c = border[b]
        out.append(f'<text x="{c * CELL + CELL / 2}" y="{r * CELL + CELL / 2 + 4}" text-anchor="middle" fill="{T.MUTED}" font-size="11" '
                   f'font-family="\'{T.FONT_NUM}\'"><title>border tile {b}: row {r}, col {c}</title>{b}</text>')
    out += [_unit(*u) for u in units]
    if pos < n_dec:
        m = ep["steps"][pos]["action"] - 5
        if 0 <= m < 5 * len(border):
            r, c = border[m % len(border)]
            fr = f'x="{c * CELL + 2}" y="{r * CELL + 2}" width="{CELL - 4}" height="{CELL - 4}" fill="none"'
            out.append(f'<rect {fr} stroke="{T.CHOSEN_KEYLINE}" stroke-width="5"/><rect {fr} stroke="{T.CHOSEN_FRAME}" stroke-width="3"/>')
    out.append(f'<rect x="0" y="0" width="{W}" height="{W}" fill="none" stroke="{T.PANEL_BORDER}" stroke-width="1"/>')
    out.append(_heat_scale(0, W + 12, scale_max))
    H = W + FOOT + 14
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-pad} {-pad} {W + pad} {H + pad - 6}" width="100%" style="max-width:{W + pad}px" '
           f'role="img" aria-label="{n} by {n} board">' + "".join(out) + "</svg>")
    return svg, caption


# ----------------------------------------------------------------------------------------------------------------------------- legend
def legend_items() -> list[tuple[str, str]]:
    """(kind, text) in drawing order; the svg and the tests both use this list."""
    return ([("unit:" + u, f"{T.UNIT_ABBR[i]} {u.capitalize()}") for i, u in enumerate(T.UNIT_ORDER)]
            + [("gun", "Gun (square)"), ("lancer", "Lancer (triangle)"), ("flak", "Flak (hexagon)"),
               ("core", "Core"), ("wall", "Wall (hatched)"), ("building", LEGEND_BUILDING)])


def legend_svg(xmlns: bool = True, fonts: bool = True) -> str:
    """Compact legend: unit letters + colours, defender shapes, core, walls, buildings. `xmlns=False` for inline use in an HTML page."""
    items, rowh, w = legend_items(), 26, 270
    out = [font_style() if fonts else "", _defs(False)]
    for i, (kind, text) in enumerate(items):
        y, cx, cy = i * rowh + 4, 14, i * rowh + 14
        if kind.startswith("unit:"):
            t = T.UNIT_ORDER.index(kind[5:])
            out.append(f'<circle cx="{cx}" cy="{cy}" r="10" fill="{UCOL[t]}" stroke="{T.OUTLINE}" stroke-width="1.5"/><text x="{cx}" y="{cy + 4}" text-anchor="middle" fill="{T.UNIT_INK[kind[5:]]}" font-size="10" '
                       f'font-weight="600" font-family="\'{T.FONT_BODY}\'">{UAB[t]}</text>')
        elif kind in ("gun", "lancer", "flak"):
            out.append(defender_shape(kind, cx, cy, 0.8))
        elif kind == "core":
            out.append(f'<polygon points="{cx},{cy - 11} {cx + 11},{cy} {cx},{cy + 11} {cx - 11},{cy}" fill="{T.CORE_FILL}" stroke="{T.CORE_BORDER}" stroke-width="2"/>')
        elif kind == "wall":
            out.append(f'<rect x="{cx - 10}" y="{cy - 10}" width="20" height="20" fill="url(#hatch)" stroke="{T.WALL_HATCH}"/>')
        else:
            out.append(f'<rect x="{cx - 9}" y="{cy - 9}" width="18" height="18" fill="none" stroke="{T.BUILDING_STROKE}" stroke-width="2"/>'
                       f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" fill="{T.BUILDING_INK}" font-size="11" font-family="\'{T.FONT_NUM}\'">2</text>')
        out.append(f'<text x="{cx + 22}" y="{cy + 5}" fill="{T.TEXT}" font-size="14" font-family="\'{T.FONT_BODY}\'">{html.escape(text)}</text>')
    h = len(items) * rowh + 8
    ns = ' xmlns="http://www.w3.org/2000/svg"' if xmlns else ""
    return f'<svg{ns} viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="legend">' + "".join(out) + "</svg>"


# ----------------------------------------------------------------------------------------------------------------------------- screenshot rule
SHOT_TICK = 120


def screenshot_positions(ep: dict) -> dict:
    """Row (ll) addendum: screenshot 1 = the first DEPLOY decision step; screenshot 2 = battle tick 120 (or the last tick if the battle ends earlier)."""
    n_dec = len(ep["steps"])
    first_deploy = next(i for i, s in enumerate(ep["steps"]) if s["phase"] == "DEPLOY")
    last_tick = ep["battle"]["frames"][-1]["t"]
    tick = min(SHOT_TICK, last_tick)
    return {"first_deploy_step": first_deploy, "tick": tick, "pos_deploy": first_deploy, "pos_battle": n_dec + tick}
