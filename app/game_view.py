"""B18e-a: board SVG and battle player of the game-feel prototype (docs/SPEC_V2.md, Gate changelog row (nn)). Pure module: no streamlit import.

``board_svg`` draws the static board (buy / deploy / watch decision steps). ``player_html`` returns ONE self-contained HTML page (no network, no external file,
fonts embedded as data URIs) that animates a battle from the same per-tick frames the replays use (``scripts/make_replays._battle``): play / pause, 1x / 2x / 4x,
tick counter, hp bars, a short hit / destroyed flash and an end banner. All motion is 150-250 ms and switched off under prefers-reduced-motion. Game logic,
scoring and outcomes are untouched: this module only draws what the frames say.
"""
from __future__ import annotations

import html
import json

from app import icons as IC
from breachlens.actions import BORDER
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


# ----------------------------------------------------------------------------------------------------------------------------- player
PLAYER_CSS = T.font_face_css(True) + T.root_vars() + """
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--f-body);font-size:15px}
#app{max-width:__PX__px;margin:0 auto;padding:2px 0}
.bar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:0 0 8px}
.bar .grow{flex:1 1 auto}
button{font:inherit;font-weight:600;color:var(--fg);background:var(--bg);border:2px solid var(--fg);border-radius:var(--r);padding:4px 14px;cursor:pointer;
 transition:background-color .15s ease}
button:hover{background:var(--panel)}button:focus-visible,input:focus-visible{outline:3px solid var(--fg);outline-offset:2px}
button[aria-pressed=true]{background:var(--danger);color:var(--fg);border-color:var(--fg)}
button.go{background:var(--primary);color:var(--on-primary);border-color:var(--fg);font-size:17px;min-width:96px}
.tick{font-size:16px;min-width:116px;text-align:right}.n{font-family:var(--f-num);font-variant-numeric:tabular-nums}
.speed{display:inline-flex;gap:4px}
#wrap{position:relative;line-height:0}
#board{width:100%;height:auto;display:block}
.u{transition:transform var(--mv,.2s) ease,opacity .2s ease}.u.dead{opacity:0}
.s.dead>.ico{opacity:.25}.s .x{opacity:0;transition:opacity .2s ease}.s.dead .x{opacity:1}
.ring{fill:none;stroke:var(--fg);stroke-width:2.5;opacity:0;pointer-events:none}
.ring.hit{animation:flash .22s ease-out}.ring.boom{animation:boom .25s ease-out}
@keyframes flash{0%{opacity:1}100%{opacity:0}}
@keyframes boom{0%{opacity:1;transform:scale(.8)}100%{opacity:0;transform:scale(1.5)}}
.ring{transform-box:fill-box;transform-origin:center}
.atk{fill:none;stroke:var(--fg);stroke-width:2;opacity:0;transition:opacity .15s ease}.atk.on{opacity:1}
#banner{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%) rotate(-4deg);line-height:1.05;text-align:center;border-radius:8px;padding:12px 36px 14px;
 opacity:0;pointer-events:none;border:4px solid var(--fg);text-transform:uppercase}
#banner.show{opacity:1;animation:pop .22s ease-out}
@keyframes pop{0%{transform:translate(-50%,-50%) rotate(-4deg) scale(.9);opacity:0}100%{transform:translate(-50%,-50%) rotate(-4deg) scale(1);opacity:1}}
#banner b{display:block;font-family:var(--f-body);font-weight:600;font-size:58px;letter-spacing:.07em}
#banner #bs{display:block;font-size:15px;font-weight:600;letter-spacing:.04em;margin-top:6px}
#banner.breach{background:var(--danger);color:var(--fg)}
#banner.held{background:var(--bg);color:var(--fg)}
.status{font-size:14px;color:var(--mut);margin-top:8px;min-height:21px}
input[type=range]{flex:1 1 160px;accent-color:var(--primary);min-width:120px}
.zone{fill:url(#tape);stroke:var(--primary);stroke-width:1.5}
@media (max-width:520px){button{min-height:44px}button.go{min-width:84px}#banner b{font-size:36px}#banner{padding:8px 18px 10px}.tick{min-width:0;flex:1 1 100%;text-align:left}input[type=range]{min-height:44px}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
""".replace("__PX__", str(FULL_PX))

PLAYER_JS = r"""
const T = JSON.parse(document.getElementById('theme').textContent);
const D = JSON.parse(document.getElementById('data').textContent);
const N = D.grid, CELL = D.cell, F = D.frames, last = F.length - 1;
const $ = id => document.getElementById(id);
const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let pos = 0, timer = null, speed = 1;
function el(tag, attrs, parent) {
  const e = document.createElementNS(board.namespaceURI, tag);   // the inline <svg> already carries the SVG namespace
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function hpBar(g, w) {
  el('rect', {x: 6, y: CELL - 6, width: w, height: 4, rx: 1, fill: T.hpBack}, g);
  return el('rect', {x: 6, y: CELL - 6, width: w, height: 4, rx: 1, fill: T.hpFill}, g);
}
function ringOn(g, cls) {
  const r = g.querySelector('.ring');
  if (reduce) return;
  r.classList.remove('hit', 'boom'); void r.getBoundingClientRect(); r.classList.add(cls);
}
const board = $('board'), layer = el('g', {}, board);
const n = x => '<span class="n">' + x + '</span>';
for (let i = 1; i < N; i++) el('path', {d: 'M' + i * CELL + ' 0V' + N * CELL + 'M0 ' + i * CELL + 'H' + N * CELL, stroke: T.gridLine, 'stroke-width': 1, fill: 'none'}, layer);
D.zone.forEach(([r, c]) => el('rect', {class: 'zone', x: c * CELL + 2, y: r * CELL + 2, width: CELL - 4, height: CELL - 4, rx: 4}, layer));
const S = D.structs.map(([k, r, c, ex], i) => {
  const g = el('g', {class: 's', transform: 'translate(' + c * CELL + ',' + r * CELL + ')'}, layer);
  const ic = k === 'c' ? 'core' : k === 'w' ? 'wall' : k === 'b' ? 'b' + Math.min(Math.max(+ex, 1), 3) : (String(ex).toLowerCase());
  const pad = k === 'w' ? 1 : 3;
  const u = el('use', {class: 'ico', x: pad, y: pad, width: CELL - 2 * pad, height: CELL - 2 * pad}, g); u.setAttribute('href', '#ic-' + ic);
  const x = el('path', {class: 'x', d: 'M10 10L' + (CELL - 10) + ' ' + (CELL - 10) + 'M' + (CELL - 10) + ' 10L10 ' + (CELL - 10), stroke: T.outline, 'stroke-width': 2.5, 'stroke-linecap': 'round', fill: 'none'}, g);
  el('rect', {class: 'ring', x: 2, y: 2, width: CELL - 4, height: CELL - 4, rx: 4}, g);
  let bar = null;
  if (k !== 'w') { const b = el('g', {class: 'hp', style: 'display:none'}, g); bar = {g: b, fill: hpBar(b, CELL - 12)}; }
  return {g, bar, k, hp: 1000};
});
const U = D.units.map((t, i) => {
  const g = el('g', {class: 'u'}, layer);
  const a = el('circle', {class: 'atk', cx: CELL / 2, cy: CELL / 2, r: CELL / 2 - 2}, g);
  const u = el('use', {x: 5, y: 5, width: CELL - 10, height: CELL - 10}, g); u.setAttribute('href', '#ic-' + D.unitNames[t]);
  el('rect', {class: 'ring', x: 4, y: 4, width: CELL - 8, height: CELL - 8, rx: 8}, g);
  const b = el('g', {class: 'hp', style: 'display:none'}, g);
  return {g, atk: a, bar: {g: b, fill: hpBar(b, CELL - 12)}, hp: 1000, cell: -1};
});
function setBar(o, hp) {
  o.bar.g.style.display = hp < 1000 && hp > 0 ? '' : 'none';
  o.bar.fill.setAttribute('width', Math.max(0, (CELL - 12) * hp / 1000));
}
function show(i, animate) {
  const f = F[i];
  let alive = 0, bd = 0, core = false;
  f.s.forEach((hp, j) => {
    const o = S[j], dead = hp <= 0;
    if (animate && hp < o.hp) ringOn(o.g, dead ? 'boom' : 'hit');
    o.g.classList.toggle('dead', dead);
    if (o.bar) setBar(o, hp);
    o.hp = hp;
    if (D.structs[j][0] === 'b' && dead) bd++;
    if (D.structs[j][0] === 'c') core = dead;
  });
  f.u.forEach((u, j) => {
    const o = U[j], hp = u[1], cell = u[0];
    if (hp > 0) alive++;
    o.g.style.setProperty('--mv', reduce || !animate ? '0s' : Math.min(0.2, 0.8 * D.tickMs / speed / 1000) + 's');
    o.g.style.transform = 'translate(' + (cell % N) * CELL + 'px,' + Math.floor(cell / N) * CELL + 'px)';
    if (animate && hp < o.hp) ringOn(o.g, hp <= 0 ? 'boom' : 'hit');
    o.g.classList.toggle('dead', hp <= 0);
    o.atk.classList.toggle('on', u[2] === 2 && hp > 0);
    setBar(o, hp);
    o.hp = hp;
  });
  $('tick').innerHTML = 'Tick ' + n(f.t) + ' / ' + n(F[last].t);
  $('scrub').value = i;
  $('status').innerHTML = n(alive) + ' of ' + n(U.length) + ' units alive · ' + n(bd) + ' of ' + n(D.nb) + ' buildings destroyed · core ' + (core ? 'destroyed' : 'standing');
  const b = $('banner');
  b.classList.toggle('show', i === last);
}
function stop() { if (timer) { clearInterval(timer); timer = null; } $('play').textContent = (pos >= last ? 'Replay' : 'Play'); }
function go(i, animate) { pos = Math.max(0, Math.min(last, i)); show(pos, animate); if (pos >= last) stop(); }
function play() {
  if (timer) { stop(); return; }
  if (pos >= last) go(0, false);
  $('play').textContent = 'Pause';
  timer = setInterval(() => go(pos + 1, true), D.tickMs / speed);
}
function setSpeed(s) {
  speed = s;
  document.querySelectorAll('.speed button').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.s === s)));
  if (timer) { clearInterval(timer); timer = setInterval(() => go(pos + 1, true), D.tickMs / speed); }
}
$('play').onclick = play;
document.querySelectorAll('.speed button').forEach(b => b.onclick = () => setSpeed(+b.dataset.s));
$('scrub').max = last; $('scrub').oninput = e => { stop(); go(+e.target.value, false); };
const b = $('banner'), bt = $('bt'), bs = $('bs');
b.classList.add(D.outcome.breach ? 'breach' : 'held');
bt.textContent = D.outcome.breach ? 'BREACH' : 'HELD';
bs.innerHTML = D.outcome.breach ? 'core destroyed at tick ' + n(D.outcome.coreTick) : 'the core stood · ' + D.outcome.reasonText;
board.appendChild($('frame'));                                  // the frame is drawn last so that nothing covers it
go(Math.min(D.start, last), false);
if (D.autoplay && !reduce && D.start < last) play();
"""


def player_html(structs: list, frames: list, army_types: list, outcome: dict, *, tick_ms: int = 480, autoplay: bool = True, start: int = 0) -> str:
    """One self-contained page. structs = [[kind, row, col, extra]]; frames = ``_battle(env)["frames"]`` ({t, u: [[cell, hp, state]], s: [hp]});
    army_types = unit type index per deployed unit; outcome = {breach, ticks, reasonText}."""
    ct = core_tick(structs, frames)
    outcome = {**outcome, "coreTick": ct if ct is not None else int(outcome["ticks"])}        # the banner names the tick the core fell (the battle can go on after it)
    data = {"grid": GRID, "cell": CELL, "structs": structs, "frames": frames, "units": [int(t) for t in army_types], "unitNames": list(IC.UNITS),
            "nb": sum(1 for s in structs if s[0] == "b"), "outcome": outcome, "tickMs": tick_ms, "autoplay": bool(autoplay), "start": int(start), "zone": [list(b) for b in BORDER]}
    sym = f'<svg width="0" height="0" style="position:absolute" aria-hidden="true">{IC.symbol_defs()}</svg>'
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Battle</title><style>{PLAYER_CSS}</style></head><body>{sym}<div id="app">'
            f'<div class="bar"><button id="play" class="go" type="button">Play</button><span class="speed" role="group" aria-label="speed">'
            f'<button type="button" data-s="1" aria-pressed="true">1x</button><button type="button" data-s="2" aria-pressed="false">2x</button>'
            f'<button type="button" data-s="4" aria-pressed="false">4x</button></span>'
            f'<input type="range" id="scrub" min="0" max="0" value="0" aria-label="battle tick"><span class="tick" id="tick">Tick 0</span></div>'
            f'<div id="wrap"><svg id="board" viewBox="{VIEW}" role="img" aria-label="battle board">{axis_svg()}<rect width="{BOARD_PX}" height="{BOARD_PX}" rx="10" fill="{T.BOARD_BG}"/><rect id="frame" x="{T.FRAME_W / 2}" y="{T.FRAME_W / 2}" width="{BOARD_PX - T.FRAME_W}" height="{BOARD_PX - T.FRAME_W}" rx="10" fill="none" stroke="{T.OUTLINE}" stroke-width="{T.FRAME_W}"/></svg>'
            f'<div id="banner" role="status"><b id="bt"></b><span id="bs"></span></div></div>'
            f'<div class="status" id="status" aria-live="polite"></div></div>'
            f'<script type="application/json" id="theme">{T.js_theme()}</script><script type="application/json" id="data">{_esc_json(data)}</script>'
            f'<script>{PLAYER_JS}</script></body></html>')


def outcome_reason_text(reason: str) -> str:
    return {"all_dead": "all units destroyed", "no_attackers": "no unit could attack", "stalemate": "stalemate"}.get(reason, html.escape(str(reason)).replace("_", " "))
