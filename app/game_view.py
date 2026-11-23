"""B18e-a: board SVG of the game-feel prototype (docs/SPEC_V2.md, Gate changelog row (nn)). Pure module: no streamlit import.

``board_svg`` draws the static board (placement, watch decision steps). The battle player moved to ``app/arena.py`` in B24. Game logic, scoring and outcomes are
untouched: this module only draws.
"""
from __future__ import annotations

import html
import json

from app import icons as IC
from app import theme as T

GRID = 14
CELL = 46                                                    # 14 * 46 = 644 px board
BOARD_PX = GRID * CELL
GUT = 20                                                     # gutter above / left of the board for the row and column numbers (ink on the snow card)
VIEW = f"{-GUT} {-GUT} {BOARD_PX + GUT} {BOARD_PX + GUT}"
FULL_PX = BOARD_PX + GUT


def axis_svg() -> str:
    """Small column numbers above and row numbers left of the board (ink; Atkinson Hyperlegible Mono, numbers only). Tile (row, col) = the engine's border table."""
    f = f'font-family="\'{T.FONT_NUM}\', monospace" font-size="12" font-weight="600" fill="{T.TEXT}"'
    cols = "".join(f'<text x="{i * CELL + CELL / 2}" y="-7" text-anchor="middle" {f}>{i}</text>' for i in range(GRID))
    rows = "".join(f'<text x="-7" y="{i * CELL + CELL / 2 + 4}" text-anchor="end" {f}>{i}</text>' for i in range(GRID))
    return f'<g class="axis" aria-hidden="true">{cols}{rows}</g>'


def core_tick(structs: list, frames: list):
    """The first battle tick at which the core's hp is 0 (read from the frames), or None if the core never falls."""
    ci = next(i for i, s in enumerate(structs) if s[0] == "c")
    return next((int(f["t"]) for f in frames if f["s"][ci] <= 0), None)


def _esc_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":")).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def _struct_icon(k: str, ex) -> str:
    if k == "c":
        return "core"
    if k == "w":
        return "wall"
    if k == "b":
        return IC.building_icon(int(ex))
    return str(ex).lower() if str(ex).lower() in IC.TURRETS else "gun"


def _use(name: str, x: float, y: float, size: float = CELL, opacity: float = 1.0, pad: float = 2) -> str:
    op = f' opacity="{opacity}"' if opacity != 1 else ""
    return f'<use href="#ic-{name}" x="{x + pad:.1f}" y="{y + pad:.1f}" width="{size - 2 * pad:.1f}" height="{size - 2 * pad:.1f}"{op}/>'


def _hp_bar(x, y, hp, w=CELL - 12) -> str:
    return (f'<rect x="{x + 6}" y="{y + CELL - 6}" width="{w}" height="4" rx="1" fill="{T.HP_BACK}"/>'
            f'<rect x="{x + 6}" y="{y + CELL - 6}" width="{w * hp / 1000:.1f}" height="4" rx="1" fill="{T.HP_FILL}"/>')


def board_svg(structs: list, s_hp: list | None = None, units: list | None = None, *, overlay: dict | None = None, zone: list | None = None,
              selected: tuple | None = None, preview: tuple | None = None, chosen: tuple | None = None) -> str:
    """The board as one SVG string (for st.image). structs = [[kind, row, col, extra]]; s_hp = permille per struct (None = all full);
    units = [(type index, row, col, hp permille)]; overlay = {(row, col): colour} heatmap on border tiles; zone = the border start zone (azure tape stripes); selected = tile the player is about to use (dashed frame); preview = (type index, row, col) translucent icon on that tile;
    chosen = (row, col) the AI's chosen tile (3 px ink frame with a light keyline, as in the replays)."""
    out = [IC.symbol_defs(), axis_svg(), f'<rect width="{BOARD_PX}" height="{BOARD_PX}" rx="10" fill="{T.BOARD_BG}"/>']
    for i in range(1, GRID):
        out.append(f'<path d="M{i * CELL} 0V{BOARD_PX}M0 {i * CELL}H{BOARD_PX}" stroke="{T.GRID_LINE}" stroke-width="1" fill="none"/>')
    for r, c in zone or []:
        out.append(f'<rect x="{c * CELL + 2}" y="{r * CELL + 2}" width="{CELL - 4}" height="{CELL - 4}" rx="4" fill="url(#tape)" stroke="{T.PRIMARY}" stroke-width="1.5"/>')
    for (r, c), col in (overlay or {}).items():
        out.append(f'<rect x="{c * CELL + 2}" y="{r * CELL + 2}" width="{CELL - 4}" height="{CELL - 4}" rx="4" fill="{col}" opacity="0.92"/>')
    hp = s_hp or [1000] * len(structs)
    for (k, r, c, ex), h in zip(structs, hp):
        x, y = c * CELL, r * CELL
        dead = h <= 0
        out.append(_use(_struct_icon(k, ex), x, y, opacity=0.25 if dead else 1, pad=1 if k == "w" else 3))
        if k != "w" and 0 < h < 1000:
            out.append(_hp_bar(x, y, h))
        if dead and k != "w":
            out.append(f'<path d="M{x + 10} {y + 10}L{x + CELL - 10} {y + CELL - 10}M{x + CELL - 10} {y + 10}L{x + 10} {y + CELL - 10}" stroke="{T.OUTLINE}" stroke-width="2.5" stroke-linecap="round"/>')
    for t, r, c, h in units or []:
        if h > 0:
            out.append(_use(IC.UNITS[t], c * CELL, r * CELL, pad=5))
            if h < 1000:
                out.append(_hp_bar(c * CELL, r * CELL, h))
    if preview:
        t, r, c = preview
        out.append(_use(IC.UNITS[t], c * CELL, r * CELL, opacity=0.55, pad=5))
    if selected:
        r, c = selected
        out.append(f'<rect x="{c * CELL + 2}" y="{r * CELL + 2}" width="{CELL - 4}" height="{CELL - 4}" rx="4" fill="none" stroke="{T.OUTLINE}" stroke-width="2.5" stroke-dasharray="5 3"/>')
    if chosen:
        r, c = chosen
        fr = f'x="{c * CELL + 2}" y="{r * CELL + 2}" width="{CELL - 4}" height="{CELL - 4}" rx="4" fill="none"'
        out.append(f'<rect {fr} stroke="{T.CHOSEN_KEYLINE}" stroke-width="6"/><rect {fr} stroke="{T.CHOSEN_FRAME}" stroke-width="3"/>')
    out.append(f'<rect x="{T.FRAME_W / 2}" y="{T.FRAME_W / 2}" width="{BOARD_PX - T.FRAME_W}" height="{BOARD_PX - T.FRAME_W}" rx="10" fill="none" stroke="{T.OUTLINE}" stroke-width="{T.FRAME_W}"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{VIEW}" width="{FULL_PX}" height="{FULL_PX}" role="img" '
            f'aria-label="{GRID} by {GRID} board">' + "".join(out) + "</svg>")


def outcome_reason_text(reason: str) -> str:
    return {"all_dead": "all units destroyed", "no_attackers": "no unit could attack", "stalemate": "stalemate"}.get(reason, html.escape(str(reason)).replace("_", " "))
