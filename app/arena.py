"""B24: the in-board battle player of Ravelin (pure module, no streamlit import).

``player_html`` returns ONE self-contained HTML page (no network, fonts embedded) for ``st.iframe``. It animates a battle from RAW fixed-point frames
(``app.arena_data.raw_battle``; the simulator is not touched), entirely in the browser, so playback never reruns Streamlit:

* controls: Play / Pause (Space), Step back / Step 1 tick (arrow keys), speed 1x / 2x / 4x, scrubber, "Pause on key moments" (a defender or building is destroyed,
  one of your troops dies, the core takes damage) with a Continue button, "Ranges" and "Paths" overlays;
* health, two clearly different styles. HUD bars above the board: your army (azure, shield), the fortress (coral, tower) and the core (smaller coral bar), each with a
  text label, ``role=progressbar`` and ``aria-valuenow`` / ``aria-valuemax`` in raw fixed-point hp, animated between ticks and flashing when they drop.
  On the board: troops carry a thin azure bar ABOVE the piece; defenders and buildings carry a coral RING that drains clockwise (shape, not colour alone);
  a dead piece shows the destroyed mark;
* floating damage numbers, one per piece per tick, equal to the drop in raw hp between two frames (``data-raw``), plus an "hp left / max" label for ~600 ms:
  damage to the fortress in the azure-text shade, damage taken by troops in the coral-text shade (text-safe shades of ``app/theme.py``);
* a live status strip and the BREACH / HELD sticker at the tick the core really falls.
"""
from __future__ import annotations

import html

from app import arena_data as AD
from app import game_view as GV
from app import icons as IC
from app import theme as T
from breachlens.actions import BORDER

CSS = T.font_face_css(True) + T.root_vars() + """
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--f-body);font-size:15px}
#app{max-width:__PX__px;margin:0 auto;padding:2px 0}
.n{font-family:var(--f-num);font-variant-numeric:tabular-nums}
.hud{display:grid;grid-template-columns:1fr 1fr;gap:6px 14px;margin:0 0 8px}
.meter{display:flex;align-items:center;gap:8px;min-width:0}.meter svg{flex:none}
.meter .lab{font-weight:600;white-space:nowrap}
.meter .track{position:relative;flex:1 1 auto;height:14px;background:var(--line);border:2px solid var(--fg);border-radius:7px;overflow:hidden;min-width:60px}
.meter .fill{position:absolute;left:0;top:0;bottom:0;width:100%;transition:width .25s ease}
.meter.mine .fill{background:var(--primary)}.meter.fort .fill,.meter.core .fill{background:var(--danger)}
.meter .val{font-size:14px;white-space:nowrap}
.meter.drop .track{animation:drop .3s ease-out}
@keyframes drop{0%{border-color:var(--danger-text);border-width:4px}100%{border-color:var(--fg);border-width:2px}}
.meter.core{grid-column:1 / -1;max-width:60%}.meter.core .track{height:9px;border-width:1.5px}.meter.core .lab,.meter.core .val{font-size:13px}
.bar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:0 0 8px}
button{font:inherit;font-weight:600;color:var(--fg);background:var(--bg);border:2px solid var(--fg);border-radius:var(--r);padding:4px 14px;cursor:pointer;min-height:44px;
 transition:background-color .15s ease}
button:hover{background:var(--panel)}button:focus-visible,input:focus-visible{outline:3px solid var(--fg);outline-offset:2px}
button[aria-pressed=true]{background:var(--danger);color:var(--fg)}
button.go{background:var(--primary);color:var(--on-primary);font-size:17px;min-width:96px}
.speed{display:inline-flex;gap:4px}.tick{font-size:16px;min-width:116px;text-align:right}
input[type=range]{flex:1 1 160px;accent-color:var(--primary);min-width:120px;min-height:44px}
.opts{display:flex;gap:6px 16px;flex-wrap:wrap;margin:0 0 8px}.opts label{display:inline-flex;align-items:center;gap:6px;min-height:44px;cursor:pointer;font-weight:600}
.opts input{width:20px;height:20px;accent-color:var(--primary)}
#hold{display:none;align-items:center;gap:10px;background:var(--bg);border:2px solid var(--fg);border-radius:var(--r);padding:4px 10px;margin:0 0 8px;font-weight:600}
#hold.on{display:flex}#hold .why{flex:1 1 auto}
#wrap{position:relative;line-height:0}#board{width:100%;height:auto;display:block}
.u{transition:transform var(--mv,.2s) ease}.u .ico2{transition:opacity .2s ease}.u.dead .ico2{opacity:.25}
.s.dead>.ico{opacity:.25}.x{opacity:0;transition:opacity .2s ease;pointer-events:none}.dead>.x,.dead .x{opacity:1}
.flash{fill:none;stroke:var(--fg);stroke-width:2.5;opacity:0;pointer-events:none;transform-box:fill-box;transform-origin:center}
.flash.hit{animation:flash .22s ease-out}.flash.boom{animation:boom .25s ease-out}
@keyframes flash{0%{opacity:1}100%{opacity:0}}@keyframes boom{0%{opacity:1;transform:scale(.8)}100%{opacity:0;transform:scale(1.5)}}
.atk{fill:none;stroke:var(--fg);stroke-width:2;opacity:0;transition:opacity .15s ease}.atk.on{opacity:1}
.ringtrack{fill:none;stroke:var(--line);stroke-width:3.5}.ring{fill:none;stroke:var(--danger);stroke-width:3.5;stroke-linecap:butt;transition:stroke-dashoffset .25s ease,stroke-dasharray .25s ease}
.ubar{transition:width .25s ease}
.dn{font-family:var(--f-num);font-weight:600;font-size:15px;text-anchor:middle;paint-order:stroke;stroke:var(--bg);stroke-width:3.5px;stroke-linejoin:round;pointer-events:none}
.dn.fort{fill:var(--primary-text)}.dn.mine{fill:var(--danger-text)}.dn.left{font-size:11px;font-weight:400;fill:var(--fg);stroke-width:3px}
.fl{animation:rise var(--fl,.9s) ease-out forwards}
@keyframes rise{0%{opacity:1;transform:translateY(0)}100%{opacity:0;transform:translateY(-22px)}}
.rng{fill:none;stroke-width:1.5;stroke-dasharray:4 3;opacity:.8;pointer-events:none}.rng.d{stroke:var(--danger)}.rng.u{stroke:var(--primary)}
.path{fill:none;stroke:var(--primary);stroke-width:2;opacity:.55;stroke-linejoin:round;stroke-dasharray:2 3;pointer-events:none}
.zone{fill:url(#tape);stroke:var(--primary);stroke-width:1.5}
#banner{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%) rotate(-4deg);line-height:1.05;text-align:center;border-radius:8px;padding:12px 36px 14px;
 opacity:0;pointer-events:none;border:4px solid var(--fg);text-transform:uppercase}
#banner.show{opacity:1;animation:pop .22s ease-out}
@keyframes pop{0%{transform:translate(-50%,-50%) rotate(-4deg) scale(.9);opacity:0}100%{transform:translate(-50%,-50%) rotate(-4deg) scale(1);opacity:1}}
#banner b{display:block;font-family:var(--f-body);font-weight:600;font-size:58px;letter-spacing:.07em}
#banner #bs{display:block;font-size:15px;font-weight:600;letter-spacing:.04em;margin-top:6px}
#banner.breach{background:var(--danger);color:var(--fg)}#banner.held{background:var(--bg);color:var(--fg)}
.status{font-size:14px;color:var(--mut);margin-top:8px;min-height:21px}
@media (max-width:520px){#banner b{font-size:36px}#banner{padding:8px 18px 10px}.tick{min-width:0;flex:1 1 100%;text-align:left}.hud{grid-template-columns:1fr}.meter.core{max-width:100%}
 button.go{min-width:84px}.meter .val{font-size:13px}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
""".replace("__PX__", str(GV.FULL_PX))

JS = r"""
const T = JSON.parse(document.getElementById('theme').textContent);
const D = JSON.parse(document.getElementById('data').textContent);
const N = D.grid, CELL = D.cell, F = D.frames, last = F.length - 1, S = D.S, K = D.structs.map(s => s[0]);
const $ = id => document.getElementById(id);
const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const nf = new Intl.NumberFormat('en-US');
const hp = raw => nf.format(Math.round(raw / S));
const dmg = raw => String(parseFloat((raw / S).toFixed(2)));
let pos = 0, timer = null, speed = 1, held = null, shown = -1, playing = false;
const board = $('board');
function el(tag, attrs, parent) {
  const e = document.createElementNS(board.namespaceURI, tag);        // the inline <svg> already carries the SVG namespace
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
const n = x => '<span class="n">' + x + '</span>';
const layer = el('g', {}, board), rngL = el('g', {}, board), pathL = el('g', {}, board), unitL = el('g', {}, board), fx = el('g', {}, board);
for (let i = 1; i < N; i++) el('path', {d: 'M' + i * CELL + ' 0V' + N * CELL + 'M0 ' + i * CELL + 'H' + N * CELL, stroke: T.gridLine, 'stroke-width': 1, fill: 'none'}, layer);
D.zone.forEach(([r, c]) => el('rect', {class: 'zone', x: c * CELL + 2, y: r * CELL + 2, width: CELL - 4, height: CELL - 4, rx: 4}, layer));
// ring path: starts at 12 o'clock and runs clockwise; pathLength = 1, so the dash is the share of hp left and the lost part is the arc that follows the start
function ringPath(inset, r) {
  const a = inset, b = CELL - inset, m = CELL / 2;
  return 'M' + m + ' ' + a + 'H' + (b - r) + 'A' + r + ' ' + r + ' 0 0 1 ' + b + ' ' + (a + r) + 'V' + (b - r) + 'A' + r + ' ' + r + ' 0 0 1 ' + (b - r) + ' ' + b +
         'H' + (a + r) + 'A' + r + ' ' + r + ' 0 0 1 ' + a + ' ' + (b - r) + 'V' + (a + r) + 'A' + r + ' ' + r + ' 0 0 1 ' + (a + r) + ' ' + a + 'H' + m;
}
function setRing(o, frac) {
  const f = Math.max(0, Math.min(1, frac));
  o.ring.setAttribute('stroke-dasharray', f + ' 2');
  o.ring.setAttribute('stroke-dashoffset', String(-(1 - f)));
}
const cross = (g) => el('path', {class: 'x', d: 'M10 10L' + (CELL - 10) + ' ' + (CELL - 10) + 'M' + (CELL - 10) + ' 10L10 ' + (CELL - 10), stroke: T.outline, 'stroke-width': 2.5, 'stroke-linecap': 'round', fill: 'none'}, g);
const S_ = D.structs.map(([k, r, c, ex], i) => {
  const g = el('g', {class: 's', transform: 'translate(' + c * CELL + ',' + r * CELL + ')', 'data-kind': k, 'data-i': i}, layer);
  const ic = k === 'c' ? 'core' : k === 'w' ? 'wall' : k === 'b' ? 'b' + Math.min(Math.max(+ex, 1), 3) : (String(ex).toLowerCase());
  const pad = k === 'w' ? 1 : 6;
  const u = el('use', {class: 'ico', x: pad, y: pad, width: CELL - 2 * pad, height: CELL - 2 * pad}, g); u.setAttribute('href', '#ic-' + ic);
  cross(g);
  const o = {g, k, hp: D.smax[i], max: D.smax[i], i};
  if (k !== 'w') {
    el('path', {class: 'ringtrack', d: ringPath(3, 6)}, g);
    o.ring = el('path', {class: 'ring', d: ringPath(3, 6), pathLength: 1}, g);
    setRing(o, 1);
  }
  o.flash = el('rect', {class: 'flash', x: 2, y: 2, width: CELL - 4, height: CELL - 4, rx: 4}, g);
  return o;
});
const U_ = D.units.map((t, i) => {
  const g = el('g', {class: 'u', 'data-i': i}, unitL);
  el('rect', {x: 6, y: 1, width: CELL - 12, height: 5, rx: 2, fill: T.line, stroke: T.outline, 'stroke-width': 1}, g);
  const bar = el('rect', {class: 'ubar', x: 6, y: 1, width: CELL - 12, height: 5, rx: 2, fill: T.primary}, g);
  const a = el('circle', {class: 'atk', cx: CELL / 2, cy: CELL / 2 + 2, r: CELL / 2 - 4}, g);
  const u = el('use', {class: 'ico2', x: 6, y: 8, width: CELL - 12, height: CELL - 12}, g); u.setAttribute('href', '#ic-' + D.unitNames[t]);
  cross(g);
  const fl = el('rect', {class: 'flash', x: 4, y: 6, width: CELL - 8, height: CELL - 8, rx: 8}, g);
  return {g, bar, atk: a, flash: fl, hp: D.umax[i], max: D.umax[i], cell: -1, i};
});
function flash(o, cls) {
  if (reduce) return;
  o.flash.classList.remove('hit', 'boom'); void o.flash.getBoundingClientRect(); o.flash.classList.add(cls);
}
// ---- HUD
const meters = {
  mine: {root: $('m-mine'), track: $('t-mine'), fill: $('f-mine'), val: $('v-mine'), last: null},
  fort: {root: $('m-fort'), track: $('t-fort'), fill: $('f-fort'), val: $('v-fort'), last: null},
  core: {root: $('m-core'), track: $('t-core'), fill: $('f-core'), val: $('v-core'), last: null},
};
function setMeter(m, cur, max, animate) {
  m.track.setAttribute('aria-valuenow', cur); m.track.setAttribute('aria-valuemax', max);
  m.track.setAttribute('aria-valuetext', hp(cur) + ' of ' + hp(max));
  m.track.dataset.raw = cur; m.track.dataset.max = max;
  m.fill.style.width = (max ? 100 * cur / max : 0) + '%';
  m.val.textContent = hp(cur) + ' / ' + hp(max);
  if (animate && m.last !== null && cur < m.last && !reduce) { m.root.classList.remove('drop'); void m.root.offsetWidth; m.root.classList.add('drop'); }
  m.last = cur;
}
const fortIdx = K.map((k, i) => k === 'w' ? -1 : i).filter(i => i >= 0), coreI = K.indexOf('c');
const fortMax = fortIdx.reduce((a, i) => a + D.smax[i], 0), armyMax = D.umax.reduce((a, b) => a + b, 0);
// ---- floating numbers: one per piece per tick, the drop in raw hp between two frames
function floater(x, y, raw, left, max, cls, ofs) {
  const dur = reduce ? 0.9 : Math.max(0.5, Math.min(0.9, 1.6 * D.tickMs / speed / 1000));
  const g = el('g', {transform: 'translate(' + x + ',' + (y - ofs * 13) + ')'}, fx);
  const a = el('g', {class: 'fl', style: '--fl:' + dur + 's'}, g);
  const t1 = el('text', {class: 'dn ' + cls, x: 0, y: 0, 'data-raw': raw}, a); t1.textContent = '-' + dmg(raw);
  const t2 = el('text', {class: 'dn left', x: 0, y: 12, 'data-left': left}, a); t2.textContent = Math.round(left / S) + '/' + Math.round(max / S);
  setTimeout(() => t2.remove(), 600);
  setTimeout(() => g.remove(), dur * 1000 + 60);
}
// ---- key moments (same rule as app/arena_data.moments)
function moments(i) {
  if (i < 1) return [];
  const a = F[i - 1], b = F[i], m = [];
  if (b.s.some((y, j) => K[j] !== 'w' && a.s[j] > 0 && y <= 0)) m.push('a defender or building was destroyed');
  if (b.u.some((y, j) => a.u[j][1] > 0 && y[1] <= 0)) m.push('one of ' + D.who + ' troops died');
  if (b.s[coreI] < a.s[coreI]) m.push('the core took damage');
  return m;
}
let rangesOn = false, pathsOn = false;
const nb = D.structs.filter(s => s[0] === 'b').length, valTot = D.structs.reduce((a, s) => a + (s[0] === 'b' ? +s[3] : 0), 0);
function drawRanges(f) {
  rngL.textContent = '';
  if (!rangesOn) return;
  D.ranges.d.forEach(([r, c, mn, mx], j) => {
    const alive = f.s[D.defIdx[j]] > 0;
    if (!alive) return;
    el('circle', {class: 'rng d', cx: (c + .5) * CELL, cy: (r + .5) * CELL, r: mx * CELL}, rngL);
    if (mn > 0) el('circle', {class: 'rng d', cx: (c + .5) * CELL, cy: (r + .5) * CELL, r: mn * CELL}, rngL);
  });
  f.u.forEach((u, j) => { if (u[1] > 0) el('circle', {class: 'rng u', cx: (u[0] % N + .5) * CELL, cy: (Math.floor(u[0] / N) + .5) * CELL, r: Math.max(.2, D.ranges.u[D.units[j]]) * CELL}, rngL); });
}
function drawPaths(i) {
  pathL.textContent = '';
  if (!pathsOn) return;
  D.units.forEach((_, j) => {
    const pts = [];
    for (let t = 0; t <= i; t++) { const u = F[t].u[j]; if (t > 0 && u[1] <= 0 && F[t - 1].u[j][1] <= 0) break; pts.push(((u[0] % N + .5) * CELL).toFixed(1) + ',' + ((Math.floor(u[0] / N) + .5) * CELL).toFixed(1)); }
    if (pts.length > 1) el('polyline', {class: 'path', points: pts.join(' ')}, pathL);
  });
}
function show(i, animate) {
  const f = F[i], prev = F[i - 1];
  const step = animate && prev && i === shown + 1;
  const cellCount = {};
  f.s.forEach((h, j) => {
    const o = S_[j], dead = h <= 0;
    if (step && h < o.hp && K[j] !== 'w') {
      flash(o, dead ? 'boom' : 'hit');
      const c = D.structs[j]; const k = c[1] * N + c[2]; const ofs = cellCount[k] = (cellCount[k] || 0);
      cellCount[k]++;
      floater((c[2] + .5) * CELL, c[1] * CELL + 14, o.hp - h, h, o.max, 'fort', ofs);
    }
    o.g.classList.toggle('dead', dead);
    o.g.dataset.hp = h;
    if (o.ring) { setRing(o, h / o.max); o.ring.dataset.hp = h; }
    o.hp = h;
  });
  f.u.forEach((u, j) => {
    const o = U_[j], h = u[1], cell = u[0];
    o.g.style.setProperty('--mv', reduce || !animate ? '0s' : Math.min(0.2, 0.8 * D.tickMs / speed / 1000) + 's');
    o.g.style.transform = 'translate(' + (cell % N) * CELL + 'px,' + Math.floor(cell / N) * CELL + 'px)';
    if (step && h < o.hp) {
      flash(o, h <= 0 ? 'boom' : 'hit');
      const k = cell; const ofs = cellCount[k] = (cellCount[k] || 0); cellCount[k]++;
      floater((cell % N + .5) * CELL, Math.floor(cell / N) * CELL + 14, o.hp - h, h, o.max, 'mine', ofs);
    }
    o.g.classList.toggle('dead', h <= 0);
    o.atk.classList.toggle('on', u[2] === 2 && h > 0);
    o.bar.setAttribute('width', Math.max(0, (CELL - 12) * h / o.max)); o.bar.dataset.hp = h; o.g.dataset.hp = h;
    o.hp = h;
  });
  const army = f.u.reduce((a, u) => a + u[1], 0), fort = fortIdx.reduce((a, j) => a + f.s[j], 0), core = f.s[coreI];
  setMeter(meters.mine, army, armyMax, step); setMeter(meters.fort, fort, fortMax, step); setMeter(meters.core, core, D.smax[coreI], step);
  const alive = f.u.filter(u => u[1] > 0).length;
  let bd = 0, val = 0;
  f.s.forEach((h, j) => { if (K[j] === 'b' && h <= 0) { bd++; val += +D.structs[j][3]; } });
  $('tick').innerHTML = 'Tick ' + n(f.t) + ' / ' + n(F[last].t);
  $('scrub').value = i;
  $('status').innerHTML = 'Tick ' + n(f.t) + ' &middot; ' + n(alive) + ' of ' + n(F[0].u.length) + ' troops alive &middot; ' + n(bd) + ' of ' + n(nb) + ' buildings destroyed &middot; ' +
    n(valTot ? Math.round(100 * val / valTot) : 0) + '% of building value destroyed &middot; core hp ' + n(core > 0 ? Math.round(100 * core / D.smax[coreI]) : 0) + '%';
  const b = $('banner'), o = D.outcome;
  b.classList.toggle('show', o.breach ? i >= o.coreTick : i === last);
  drawRanges(f); drawPaths(i);
  shown = i;
}
function release() { held = null; $('hold').classList.remove('on'); }
function stop() { playing = false; if (timer) { clearTimeout(timer); timer = null; } $('play').textContent = pos >= last ? 'Replay' : 'Play'; $('play').dataset.state = 'paused'; }
function hold(why) { stop(); held = why; $('why').textContent = 'Paused: ' + why; $('hold').classList.add('on'); }
function go(i, animate) {
  pos = Math.max(0, Math.min(last, i)); show(pos, animate);
  if (pos >= last) { stop(); release(); return; }
  if (animate && playing && $('keys').checked) { const m = moments(pos); if (m.length) hold(m.join(' and ')); }
}
function loop() { if (!playing) return; timer = setTimeout(() => { go(pos + 1, true); loop(); }, D.tickMs / speed); }
function play() {
  if (held) release();
  else if (playing) { stop(); return; }
  if (pos >= last) go(0, false);
  playing = true; $('play').textContent = 'Pause'; $('play').dataset.state = 'playing';
  loop();
}
function stepBy(d) { stop(); release(); const i = Math.max(0, Math.min(last, pos + d)); pos = i; show(i, d === 1); }
function setSpeed(s) { speed = s; document.querySelectorAll('.speed button').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.s === s))); }
$('play').onclick = play; $('cont').onclick = play;
$('step').onclick = () => stepBy(1); $('back').onclick = () => stepBy(-1);
document.querySelectorAll('.speed button').forEach(b => b.onclick = () => setSpeed(+b.dataset.s));
$('scrub').max = last; $('scrub').oninput = e => { stop(); release(); pos = +e.target.value; show(pos, false); };
$('rng').onchange = e => { rangesOn = e.target.checked; drawRanges(F[pos]); };
$('pth').onchange = e => { pathsOn = e.target.checked; drawPaths(pos); };
document.addEventListener('keydown', e => {
  if (e.target && (e.target.tagName === 'INPUT' && e.target.type !== 'range')) return;
  if (e.key === ' ' || e.code === 'Space') { e.preventDefault(); play(); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); stepBy(1); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); stepBy(-1); }
});
const bn = $('banner'), bt = $('bt'), bs = $('bs');
bn.classList.add(D.outcome.breach ? 'breach' : 'held');
bt.textContent = D.outcome.breach ? 'BREACH' : 'HELD';
bs.innerHTML = D.outcome.breach ? 'core destroyed at tick ' + n(D.outcome.coreTick) : 'the core stood &middot; ' + D.outcome.reasonText;
board.appendChild($('frame'));
window.rv = {go: i => { stop(); release(); pos = i; show(i, false); }, step: stepBy, play, frames: F, state: () => ({pos, held, speed, playing})};
pos = Math.min(D.start, last); show(pos, false);
if (D.autoplay && !reduce && D.start < last) play();
"""

_SHIELD = ('<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path d="M12 2.5l8 3v6c0 5-3.4 8.3-8 10-4.6-1.7-8-5-8-10v-6z" fill="{f}" stroke="{i}" '
           'stroke-width="1.8" stroke-linejoin="round"/><path d="M8 11.5l3 3 5-6" fill="none" stroke="{i}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>')
_TOWER = ('<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path d="M6 21V8h2.6V5.5h2V8h2.8V5.5h2V8H18v13z" fill="{f}" stroke="{i}" stroke-width="1.8" '
          'stroke-linejoin="round"/><path d="M10.5 21v-4.5a1.5 1.5 0 013 0V21" fill="{b}" stroke="{i}" stroke-width="1.5"/></svg>')
_KEEP = ('<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M12 3l8 9-8 9-8-9z" fill="{f}" stroke="{i}" stroke-width="1.8" stroke-linejoin="round"/></svg>')


def hud_html(army_label: str) -> str:
    """The three HUD bars (empty; the script fills them). Labels are text; every track is a progressbar."""
    f, i, b = T.PRIMARY, T.OUTLINE, T.BG

    def meter(key: str, cls: str, label: str, icon: str) -> str:
        return (f'<div class="meter {cls}" id="m-{key}">{icon}<span class="lab">{html.escape(label)}</span>'
                f'<div class="track" id="t-{key}" role="progressbar" aria-label="{html.escape(label)} hp" aria-valuemin="0" aria-valuenow="0" aria-valuemax="0">'
                f'<div class="fill" id="f-{key}"></div></div><span class="val n" id="v-{key}"></span></div>')
    return ('<div class="hud" aria-label="health">' + meter("mine", "mine", army_label, _SHIELD.format(f=f, i=i)) +
            meter("fort", "fort", "Fortress", _TOWER.format(f=T.DANGER, i=i, b=b)) + meter("core", "core", "Core", _KEEP.format(f=T.DANGER, i=i)) + "</div>")


def player_html(structs: list, battle: dict, outcome: dict, ranges: dict, *, mode: str = "human", tick_ms: int = 480, autoplay: bool = True, start: int = 0,
                pause_keys: bool = True) -> str:
    """One self-contained page. structs = ``make_replays._struct_table``; battle = ``arena_data.raw_battle`` (RAW frames); outcome = {breach, ticks, reasonText};
    ranges = ``arena_data.ranges``; mode 'human' or 'agent' only changes the label ("Your army" / "Agent's army")."""
    frames = battle["frames"]
    ct = GV.core_tick(structs, frames)
    outcome = {**outcome, "coreTick": ct if ct is not None else int(outcome["ticks"])}
    army_types = [int(t) for t, _ in battle["units"]]
    data = {"grid": GV.GRID, "cell": GV.CELL, "S": int(battle["S"]), "structs": structs, "frames": frames, "units": army_types, "unitNames": list(IC.UNITS),
            "umax": [int(x) for x in battle["umax"]], "smax": [int(x) for x in battle["smax"]], "ranges": ranges,
            "defIdx": [i for i, s in enumerate(structs) if s[0] == "d"], "who": "your" if mode == "human" else "the agent's", "outcome": outcome, "tickMs": tick_ms,
            "autoplay": bool(autoplay), "start": int(start), "zone": [list(b) for b in BORDER]}
    sym = f'<svg width="0" height="0" style="position:absolute" aria-hidden="true">{IC.symbol_defs()}</svg>'
    label = "Your army" if mode == "human" else "Agent's army"
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Battle</title><style>{CSS}</style></head><body>{sym}<div id="app">{hud_html(label)}'
            f'<div class="bar"><button id="play" class="go" type="button" title="Space">Play</button>'
            f'<button id="back" type="button" title="Left arrow">&larr; Back 1 tick</button><button id="step" type="button" title="Right arrow">Step 1 tick &rarr;</button>'
            f'<span class="speed" role="group" aria-label="speed"><button type="button" data-s="1" aria-pressed="true">1x</button>'
            f'<button type="button" data-s="2" aria-pressed="false">2x</button><button type="button" data-s="4" aria-pressed="false">4x</button></span>'
            f'<input type="range" id="scrub" min="0" max="0" value="0" aria-label="battle tick"><span class="tick" id="tick">Tick 0</span></div>'
            f'<div class="opts"><label><input type="checkbox" id="keys"{" checked" if pause_keys else ""}>Pause on key moments</label>'
            f'<label><input type="checkbox" id="rng">Ranges</label><label><input type="checkbox" id="pth">Paths</label></div>'
            f'<div id="hold" role="status"><span class="why" id="why"></span><button id="cont" type="button" class="go">Continue</button></div>'
            f'<div id="wrap"><svg id="board" viewBox="{GV.VIEW}" role="img" aria-label="battle board">{GV.axis_svg()}<rect width="{GV.BOARD_PX}" height="{GV.BOARD_PX}" rx="10" fill="{T.BOARD_BG}"/>'
            f'<rect id="frame" x="{T.FRAME_W / 2}" y="{T.FRAME_W / 2}" width="{GV.BOARD_PX - T.FRAME_W}" height="{GV.BOARD_PX - T.FRAME_W}" rx="10" fill="none" stroke="{T.OUTLINE}" stroke-width="{T.FRAME_W}"/></svg>'
            f'<div id="banner" role="status"><b id="bt"></b><span id="bs"></span></div></div>'
            f'<div class="status" id="status" aria-live="polite"></div></div>'
            f'<script type="application/json" id="theme">{T.js_theme()}</script><script type="application/json" id="data">{GV._esc_json(data)}</script>'
            f'<script>{JS}</script></body></html>')


def battle_page(P, army, kinds_structs: list, res, *, mode="human", autoplay=True, start=0, key=None, pause_keys=True) -> str:
    """Convenience: raw frames + ranges + outcome -> the player page (``res`` = the env's BattleResult, whose outcome the page must show)."""
    b = AD.raw_battle(P, army, key=key)
    out = {"breach": bool(res.core_destroyed), "ticks": int(res.ticks), "reasonText": GV.outcome_reason_text(res.reason)}
    return player_html(kinds_structs, b, out, AD.ranges(P), mode=mode, autoplay=autoplay, start=start, pause_keys=pause_keys)
