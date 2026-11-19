"""B18e-a: original SVG icons for the game-feel prototype (docs/SPEC_V2.md, Gate changelog row (nn) palette v2). Pure module: no streamlit import.

Every icon is drawn here from plain shapes (viewBox 0 0 32 32) with colour tokens from ``app/theme.py``; nothing is copied from any game, icon set or
earlier project. Silhouettes are distinct on purpose (the shape carries the identity, colour only supports it):

units      Scout = winged boot | Sniper = crosshair | Bulwark = shield | Blaster = burst | Mender = plus
turrets    Gun = squat block with a short barrel | Lancer = tall spear on a plinth | Flak = hexagon with three fanned barrels (all coral)
structures Core = fortified keep with battlements and a gate | Building = small pitched-roof house with a chimney and 1-3 value pips | Wall = masonry (SVG pattern)
misc       coin, troops (pawn), contestant glyphs (you / learner / script / search)
"""
from __future__ import annotations

import base64
import math

from app import theme as T

INK, SNOW, CORAL = T.OUTLINE, T.BG, T.DANGER
SW = 1.6                                                                        # outline width in the 32-unit box
UNITS = tuple(T.UNIT_ORDER)                                                     # scout, sniper, bulwark, blaster, mender
TURRETS = ("gun", "lancer", "flak")
BUILDINGS = ("b1", "b2", "b3")                                                  # building with 1 / 2 / 3 value pips
MISC = ("core", "wall", "coin", "troops", "you", "learner", "script", "search")
ALL = UNITS + TURRETS + BUILDINGS + MISC


def _pts(points) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _burst(cx, cy, r_out, r_in, n) -> str:
    return _pts([(cx + (r_out if i % 2 == 0 else r_in) * math.cos(math.pi * i / n - math.pi / 2),
                  cy + (r_out if i % 2 == 0 else r_in) * math.sin(math.pi * i / n - math.pi / 2)) for i in range(2 * n)])


def _stroke(w: float = SW, color: str = INK) -> str:
    return f'stroke="{color}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"'


_F = T.UNIT_FILL
_I = T.UNIT_INK


def _body() -> dict:
    """name -> inner SVG markup (no <svg> wrapper)."""
    s = _stroke()
    d = {}
    # ---- units
    d["scout"] = (f'<path d="M12 4h8v13l6 3c2 1 3 2 3 4v3H11z" fill="{_F["scout"]}" {s}/>'
                  f'<path d="M11 24h18" fill="none" {_stroke(1.3, INK)}/>'
                  f'<path d="M12 8C8 4 3 5 2 9c3-1 5 0 7 2-3 0-5 2-5 5 3-2 5-2 8-1z" fill="{SNOW}" {s}/>')
    d["sniper"] = (f'<circle cx="16" cy="16" r="9" fill="{_F["sniper"]}" {s}/>'
                   f'<circle cx="16" cy="16" r="3.2" fill="none" {_stroke(1.6, _I["sniper"])}/>'
                   f'<path d="M16 4v8M16 20v8M4 16h8M20 16h8" fill="none" {_stroke(1.8, INK)}/>'
                   f'<path d="M16 9v4M16 19v4M9 16h4M19 16h4" fill="none" {_stroke(1.6, _I["sniper"])}/>')
    d["bulwark"] = (f'<path d="M16 3L27 7v9c0 7-5 11-11 13C10 27 5 23 5 16V7z" fill="{_F["bulwark"]}" {s}/>'
                    f'<path d="M16 3v26M5 14h22" fill="none" {_stroke(1.3, INK)}/>')
    d["blaster"] = (f'<polygon points="{_burst(16, 16, 13.5, 6.5, 8)}" fill="{_F["blaster"]}" {s}/>'
                    f'<circle cx="16" cy="16" r="3.2" fill="{INK}"/>')
    d["mender"] = (f'<path d="M12 4h8v8h8v8h-8v8h-8v-8H4v-8h8z" fill="{_F["mender"]}" {s}/>'
                   f'<circle cx="16" cy="16" r="2.2" fill="{INK}"/>')
    # ---- turrets (coral, ink outline)
    c = f'fill="{CORAL}" {s}'
    d["gun"] = (f'<rect x="4" y="18" width="24" height="9" rx="1.5" {c}/>'
                f'<path d="M9 18a7 7 0 0114 0z" {c}/><rect x="20" y="10" width="9" height="4" rx="1" {c}/>')
    d["lancer"] = (f'<polygon points="{_pts([(16, 2), (21, 22), (11, 22)])}" {c}/><rect x="7" y="22" width="18" height="6" rx="1.5" {c}/>'
                   f'<path d="M16 7v11" fill="none" {_stroke(1.3, INK)}/>')
    d["flak"] = (f'<polygon points="{_pts([(16 + 10 * math.cos(math.radians(60 * i + 30)), 21 + 8 * math.sin(math.radians(60 * i + 30))) for i in range(6)])}" {c}/>'
                 f'<path d="M16 14V4M12 15L6 6M20 15l6-9" fill="none" {_stroke(3.4, INK)}/><path d="M16 14V4M12 15L6 6M20 15l6-9" fill="none" {_stroke(1.4, CORAL)}/>')
    # ---- structures
    d["core"] = (f'<path d="M4 29V12h4V8h4v4h2V8h4v4h2V8h4v4h4v17z" fill="{T.CORE_FILL}" stroke="{T.CORE_BORDER}" stroke-width="2" stroke-linejoin="round"/>'
                 f'<path d="M12 29v-8a4 4 0 018 0v8z" fill="{T.CORE_INK}" {_stroke(1.2, INK)}/>'
                 f'<path d="M16 3v6M16 3l5 2-5 2" fill="{T.CORE_BORDER}" {_stroke(1.4, T.CORE_BORDER)}/>')
    for n in (1, 2, 3):
        pips = "".join(f'<circle cx="{16 + (i - (n - 1) / 2) * 4.6:.1f}" cy="22.5" r="1.7" fill="{INK}"/>' for i in range(n))
        d[f"b{n}"] = (f'<rect x="8" y="13" width="16" height="16" fill="{SNOW}" {_stroke(2)}/>'
                      f'<polygon points="{_pts([(5, 14), (16, 3), (27, 14)])}" fill="{INK}" {_stroke(1.6)}/>'
                      f'<rect x="21" y="5" width="3.4" height="6" fill="{INK}"/>{pips}')
    d["wall"] = f'<rect x="1.5" y="1.5" width="29" height="29" fill="url(#wallstone)" {_stroke(1.2, T.WALL_HATCH)}/>'
    # ---- misc
    d["coin"] = (f'<circle cx="16" cy="16" r="12" fill="{_F["scout"]}" {s}/><circle cx="16" cy="16" r="7.5" fill="none" {_stroke(1.3, INK)}/>'
                 f'<path d="M16 11v10M13.5 13.5h4a1.8 1.8 0 010 3.5h-3a1.8 1.8 0 000 3.5h4" fill="none" {_stroke(1.3, INK)}/>')
    d["troops"] = (f'<circle cx="16" cy="9" r="5" fill="{INK}"/><path d="M7 29c0-7 3-12 9-12s9 5 9 12z" fill="{INK}"/>')
    d["you"] = (f'<path d="M7 29V4" fill="none" {_stroke(2.4)}/><path d="M7 5h18l-5 6 5 6H7z" fill="{_F["scout"]}" {s}/>')
    d["learner"] = (f'<path d="M7 9l9 7-9 7M16 16h10" fill="none" {_stroke(1.6)}/>'
                    + "".join(f'<circle cx="{x}" cy="{y}" r="3.6" fill="{_F["sniper"]}" {s}/>' for x, y in ((7, 9), (7, 23), (26, 16))) + f'<circle cx="16" cy="16" r="3.6" fill="{SNOW}" {s}/>')
    d["script"] = (f'<rect x="6" y="3" width="20" height="26" rx="2" fill="{SNOW}" {s}/>'
                   f'<path d="M10 10h12M10 15h12M10 20h7" fill="none" {_stroke(1.8)}/>')
    d["search"] = (f'<circle cx="13" cy="13" r="8" fill="{SNOW}" {s}/><path d="M19 19l8 8" fill="none" {_stroke(3.4)}/>'
                   f'<path d="M9 13h8M13 9v8" fill="none" {_stroke(1.5)}/>')
    return d


SYMBOLS = _body()
assert set(SYMBOLS) == set(ALL)


def pattern_defs() -> str:
    """The masonry pattern the wall icon (and the board) fill with (two courses of offset stones, hairline mortar) and the azure TAPE stripes that mark
    the border start zone."""
    return (f'<pattern id="tape" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="9" height="9" fill="{T.BG}"/>'
            f'<rect width="4.5" height="9" fill="{T.TAPE}"/></pattern>'
            f'<pattern id="wallstone" width="16" height="16" patternUnits="userSpaceOnUse"><rect width="16" height="16" fill="{T.WALL_FILL}"/>'
            f'<path d="M0 8h16M8 0v8M0 8v8M16 8v8" fill="none" stroke="{T.WALL_HATCH}" stroke-width="1.4"/></pattern>')


def symbol_defs(names=ALL) -> str:
    """<defs> with one <symbol id="ic-NAME"> per icon (use with <use href="#ic-NAME" width= height=>) plus the wall pattern."""
    syms = "".join(f'<symbol id="ic-{n}" viewBox="0 0 32 32">{SYMBOLS[n]}</symbol>' for n in names)
    return f"<defs>{pattern_defs()}{syms}</defs>"


def svg_icon(name: str, size: int = 32) -> str:
    """A standalone <svg> for one icon."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="{size}" height="{size}" role="img" aria-label="{name}">'
            f'{pattern_defs() if name == "wall" else ""}{SYMBOLS[name]}</svg>')


def data_uri(name: str, size: int = 32) -> str:
    """The icon as a data: URI for an <img> inside markdown HTML (no network request)."""
    return "data:image/svg+xml;base64," + base64.b64encode(svg_icon(name, size).encode()).decode()


def img(name: str, size: int = 32, alt: str = "", cls: str = "") -> str:
    return f'<img class="ic {cls}" src="{data_uri(name, size)}" width="{size}" height="{size}" alt="{alt or name}">'


def disc_svg(name: str, size: int = 64) -> str:
    """A unit as a game piece: a disc in the unit colour (ink outline), a snow centre, the icon on it."""
    fill = T.UNIT_FILL.get(name, T.STICKER_FILL)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" width="{size}" height="{size}" role="img" aria-label="{name}">'
            f'<circle cx="20" cy="20" r="18.5" fill="{fill}" {_stroke(2)}/><circle cx="20" cy="20" r="13.5" fill="{T.BG}" {_stroke(1.2)}/>'
            f'<g transform="translate(8 8) scale(.75)">{SYMBOLS[name]}</g></svg>')


def disc_img(name: str, size: int = 64) -> str:
    uri = "data:image/svg+xml;base64," + base64.b64encode(disc_svg(name, size).encode()).decode()
    return f'<img class="ic disc" src="{uri}" width="{size}" height="{size}" alt="{name}">'


def building_icon(value: int) -> str:
    return f"b{min(max(int(value), 1), 3)}"


def _circle(cx, cy, r, n=48):
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)) for k in range(n)]


# The outline of each piece as filled polygons in the 32-box (curves sampled / simplified): used only by the distinctness test (no two pieces may share a
# silhouette). Each follows the primary shapes drawn in ``_body``.
SILHOUETTE = {
    "scout": [[(12, 4), (20, 4), (20, 17), (26, 20), (29, 24), (29, 27), (11, 27)], [(12, 8), (8, 5.5), (3, 6), (2, 9), (5, 9), (9, 11), (4, 16), (12, 15)]],
    "sniper": [_circle(16, 16, 9)],
    "bulwark": [[(16, 3), (27, 7), (27, 16), (22, 25), (16, 29), (10, 25), (5, 16), (5, 7)]],
    "blaster": [[(16 + (13.5 if k % 2 == 0 else 6.5) * math.cos(math.pi * k / 8 - math.pi / 2), 16 + (13.5 if k % 2 == 0 else 6.5) * math.sin(math.pi * k / 8 - math.pi / 2)) for k in range(16)]],
    "mender": [[(12, 4), (20, 4), (20, 12), (28, 12), (28, 20), (20, 20), (20, 28), (12, 28), (12, 20), (4, 20), (4, 12), (12, 12)]],
    "gun": [[(4, 18), (28, 18), (28, 27), (4, 27)], [(9, 18), (11, 13), (16, 11), (21, 13), (23, 18)], [(20, 10), (29, 10), (29, 14), (20, 14)]],
    "lancer": [[(16, 2), (21, 22), (11, 22)], [(7, 22), (25, 22), (25, 28), (7, 28)]],
    "flak": [[(16 + 10 * math.cos(math.radians(60 * k + 30)), 21 + 8 * math.sin(math.radians(60 * k + 30))) for k in range(6)], [(14.7, 14), (17.3, 14), (17.3, 4), (14.7, 4)],
             [(10.5, 15), (13.5, 16), (7.5, 6), (4.5, 7)], [(18.5, 16), (21.5, 15), (27.5, 7), (24.5, 6)]],
    "core": [[(4, 29), (4, 12), (8, 12), (8, 8), (12, 8), (12, 12), (14, 12), (14, 8), (18, 8), (18, 12), (20, 12), (20, 8), (24, 8), (24, 12), (28, 12), (28, 29)]],
    "wall": [[(1.5, 1.5), (30.5, 1.5), (30.5, 30.5), (1.5, 30.5)]],
}
for _n in ("b1", "b2", "b3"):
    SILHOUETTE[_n] = [[(8, 13), (24, 13), (24, 29), (8, 29)], [(5, 14), (16, 3), (27, 14)], [(21, 5), (24.4, 5), (24.4, 11), (21, 11)]]


def silhouette_mask(name: str):
    """The silhouette as a 64x64 boolean numpy mask (the 32-box sampled at half-unit centres)."""
    import numpy as np
    from matplotlib.path import Path
    g = (np.arange(64) + 0.5) / 2
    pts = np.array([(x, y) for y in g for x in g])
    m = np.zeros(len(pts), bool)
    for poly in SILHOUETTE[name]:
        m |= Path(poly).contains_points(pts)
    return m.reshape(64, 64)
