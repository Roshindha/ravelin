"""B18a: pure rendering helpers for the Streamlit app (no streamlit import): the blueprint SVG grid and the unit / colour tables."""
from __future__ import annotations

import html

CELL = 38
UCOL = ["#4fc3f7", "#c792ea", "#7ee787", "#ff9e64", "#ffe27a"]          # scout, sniper, bulwark, blaster, mender
UAB = ["Sc", "Sn", "Bu", "Bl", "Me"]
LINE = "#3f7fb8"
BG = "#07142b"


def heat_color(f: float) -> str:
    """0..1 -> dark navy .. amber .. pale yellow (sequential, readable on the blueprint background)."""
    stops = [(13, 34, 68), (140, 80, 10), (255, 160, 20), (255, 236, 150)]
    p = min(max(f, 0.0), 1.0) * 3
    i = min(int(p), 2)
    t = p - i
    return "rgb(%d,%d,%d)" % tuple(round(stops[i][k] + (stops[i + 1][k] - stops[i][k]) * t) for k in range(3))


def _hp_bar(x, y, hp):
    w = CELL - 10
    return (f'<rect x="{x + 5}" y="{y + CELL - 7}" width="{w}" height="3" fill="#000"/>'
            f'<rect x="{x + 5}" y="{y + CELL - 7}" width="{w * hp / 1000:.1f}" height="3" fill="#6fd08c"/>')


def _struct(s, hp):
    k, r, c, ex = s
    x, y = c * CELL, r * CELL
    dead = hp <= 0
    op = 0.25 if dead else 1
    g = f'<g opacity="{op}">'
    if k == "w":
        g += f'<rect x="{x + 3}" y="{y + 3}" width="{CELL - 6}" height="{CELL - 6}" fill="#56688a"/>'
    elif k == "b":
        g += (f'<rect x="{x + 5}" y="{y + 5}" width="{CELL - 10}" height="{CELL - 10}" fill="none" stroke="#7cc4ff" stroke-width="2"/>'
              f'<text x="{x + CELL / 2}" y="{y + CELL / 2 + 4}" text-anchor="middle" fill="#cfe6ff" font-size="12">{html.escape(str(ex))}</text>')
    elif k == "c":
        m = CELL / 2
        g += (f'<polygon points="{x + m},{y + 3} {x + CELL - 3},{y + m} {x + m},{y + CELL - 3} {x + 3},{y + m}" fill="#ffd166"/>'
              f'<text x="{x + m}" y="{y + m + 4}" text-anchor="middle" fill="#07142b" font-size="12" font-weight="bold">C</text>')
    else:
        g += (f'<circle cx="{x + CELL / 2}" cy="{y + CELL / 2}" r="{CELL / 2 - 6}" fill="#e5484d"/>'
              f'<text x="{x + CELL / 2}" y="{y + CELL / 2 + 4}" text-anchor="middle" fill="#fff" font-size="11" font-weight="bold">{html.escape(str(ex)[:1].upper())}</text>')
    g += "</g>"
    if k != "w" and 0 < hp < 1000:
        g += _hp_bar(x, y, hp)
    if dead and k != "w":
        g += (f'<path d="M{x + 7} {y + 7}L{x + CELL - 7} {y + CELL - 7}M{x + CELL - 7} {y + 7}L{x + 7} {y + CELL - 7}" stroke="#ff6b6b" stroke-width="2"/>')
    return g


def _unit(t, col, row, hp, st):
    if hp <= 0:
        return ""
    x, y = col * CELL, row * CELL
    ring = ' stroke="#fff" stroke-width="2"' if st == 2 else ""
    return (f'<circle cx="{x + CELL / 2}" cy="{y + CELL / 2}" r="12" fill="{UCOL[t]}"{ring}/>'
            f'<text x="{x + CELL / 2}" y="{y + CELL / 2 + 4}" text-anchor="middle" fill="#07142b" font-size="10" font-weight="bold">{UAB[t]}</text>'
            + (_hp_bar(x, y - 0, hp) if hp < 1000 else ""))


def grid_svg(ep: dict, border: list, n: int, pos: int, heat_mode: str) -> tuple[str, str]:
    """(svg, heat caption) for scrubber position `pos`: decision steps first (0 .. n_dec-1), then battle frames."""
    n_dec = len(ep["steps"])
    W = n * CELL
    pad = 18
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-pad} {-pad} {W + pad} {W + pad}" width="100%" style="max-width:{W + pad}px;'
           f'background:{BG};font-family:ui-monospace,Menlo,monospace" role="img" aria-label="{n} by {n} grid">']
    caption = ""
    if pos < n_dec:
        s = ep["steps"][pos]
        m = s["action"] - 5
        if heat_mode == "total":
            vals, caption = s["heat_tile"], "per-tile total over all unit types"
        else:
            u = m // len(border) if 0 <= m < 5 * len(border) else -1
            if u >= 0:
                caption = f"unit type of the chosen action: {['scout', 'sniper', 'bulwark', 'blaster', 'mender'][u]}"
            else:
                u = max(range(5), key=lambda i: (s["heat_unit"][i], -i))
                caption = f"no deploy chosen at this step; showing the unit type with the most deploy probability ({['scout', 'sniper', 'bulwark', 'blaster', 'mender'][u]})"
            vals = s["heat"][u]
        mx = max(vals + [1e-12])
        if s["heat_total"] > 0:
            for b, v in enumerate(vals):
                r, c = border[b]
                out.append(f'<rect x="{c * CELL + 1}" y="{r * CELL + 1}" width="{CELL - 2}" height="{CELL - 2}" fill="{heat_color(v / mx)}" opacity="{0.9 if v > 0 else 0.12}"/>')
            caption += f"; colour scale 0 to {mx:.4f} (max over border tiles); total deploy probability {s['heat_total']:.4f}"
        else:
            caption = "no deploy action is legal at this step, so the heatmap is empty (total deploy probability 0)"
        structs = [(st, 1000) for st in ep["structs"]]
        units = [(t, border[b][1], border[b][0], 1000, 0) for t, b in s["army"]]
    else:
        f = ep["battle"]["frames"][pos - n_dec]
        structs = [(st, f["s"][i]) for i, st in enumerate(ep["structs"])]
        units = [(ep["battle"]["units"][i][0], u[0] % n, u[0] // n, u[1], u[2]) for i, u in enumerate(f["u"])]
    for i in range(n + 1):                                             # thin light-blue grid lines
        out.append(f'<line x1="{i * CELL}" y1="0" x2="{i * CELL}" y2="{W}" stroke="{LINE}" stroke-width="0.6" opacity="0.7"/>'
                   f'<line x1="0" y1="{i * CELL}" x2="{W}" y2="{i * CELL}" stroke="{LINE}" stroke-width="0.6" opacity="0.7"/>')
    for i in range(n):
        out.append(f'<text x="{i * CELL + CELL / 2}" y="-5" text-anchor="middle" fill="#6fa8d8" font-size="9">{i}</text>'
                   f'<text x="-6" y="{i * CELL + CELL / 2 + 3}" text-anchor="end" fill="#6fa8d8" font-size="9">{i}</text>')
    out.append(f'<rect x="0" y="0" width="{W}" height="{W}" fill="none" stroke="#7cc4ff" stroke-width="1.5"/>')
    out += [_struct(s, hp) for s, hp in structs]
    out += [_unit(*u) for u in units]
    if pos < n_dec:
        m = ep["steps"][pos]["action"] - 5
        if 0 <= m < 5 * len(border):
            r, c = border[m % len(border)]
            out.append(f'<rect x="{c * CELL + 2}" y="{r * CELL + 2}" width="{CELL - 4}" height="{CELL - 4}" fill="none" stroke="#fff" stroke-width="2.5"/>')
    out.append("</svg>")
    return "".join(out), caption


# ----------------------------------------------------------------------------------------------------------------------------- screenshot rule
SHOT_TICK = 120


def screenshot_positions(ep: dict) -> dict:
    """Row (ll) addendum: screenshot 1 = the first DEPLOY decision step; screenshot 2 = battle tick 120 (or the last tick if the battle ends earlier)."""
    n_dec = len(ep["steps"])
    first_deploy = next(i for i, s in enumerate(ep["steps"]) if s["phase"] == "DEPLOY")
    last_tick = ep["battle"]["frames"][-1]["t"]
    tick = min(SHOT_TICK, last_tick)
    return {"first_deploy_step": first_deploy, "tick": tick, "pos_deploy": first_deploy, "pos_battle": n_dec + tick}
