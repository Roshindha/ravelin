"""B18e-b: the interactive board (docs/SPEC_V2.md, Gate changelog row (oo) (d)). A small custom component with NO build step: inline HTML / CSS / JS strings
for ``st.components.v2.component`` (Streamlit 1.67's supported bidirectional component API). No package, no network request, no new dependency.

Python -> component: ``data = {svg, tiles, unit}`` where ``svg`` is ``app.game_view.board_svg`` (trusted, built here), ``tiles`` lists every border tile as
``{b, row, col, ok}`` (``ok`` = free and unoccupied) and ``unit`` is the icon name of the unit that a click would deploy (for the hover preview).
Component -> Python: ``setTriggerValue('tile', b)`` with the border index ``b`` on a click, a tap (pointer events: touch works) or Enter / Space on a focused tile.
The app turns ``b`` into the action ``5 + 52*u + b`` and sends it through the duel engine, exactly like the Deploy button. Tiles that are not ``ok`` still report
the click, so the engine's own message is what the player reads (it never changes any state). The board is ``width:100%`` and scales on narrow screens.

BUY and Watch-the-AI use the same component with ``tiles = []`` (a static board that takes the page fonts).

B24 "plan" mode (``data.mode == "plan"``): the one-tap placement board of Play. The draft (``[[unit type, border tile], ...]`` in placement order) lives in the browser:
select a unit card (click, or keys 1-5), tap a green border tile to place it (cost counted against the gold, a ghost piece shown), Shift+tap (or the "Line of 5" toggle on
a phone) places up to 5 of the unit on adjacent free tiles, tap a ghost to remove it (refund), Undo / Clear, and "Fight". A tap the engine would refuse sends
``setTriggerValue('illegal', {unit, tile, draft, n})`` and the app answers with the engine's own message (``app.arena_data.probe_message``); "Fight" sends
``setTriggerValue('fight', {draft, n})``, which the app turns into the env's legal sequence (``app.arena_data.canonical_actions``: buys, END_BUY, deploys in placement order).
"""
from __future__ import annotations

HTML = '<div class="bl-board"></div>'

CSS = """
.bl-board{width:100%;max-width:664px;margin:0 auto;line-height:0;touch-action:manipulation}
.bl-board svg{width:100%;height:auto;display:block}
.bl-board .axis text{font-family:var(--f-num,'Atkinson Hyperlegible Mono',monospace);font-variant-numeric:tabular-nums}
.bl-board .hit{fill:transparent;stroke:transparent;stroke-width:3;rx:4;cursor:pointer;outline:none}
.bl-board .hit.ok:hover,.bl-board .hit.ok:focus-visible{fill:var(--primary);fill-opacity:.28;stroke:var(--fg)}
.bl-board .hit.no{cursor:not-allowed}
.bl-board .hit.no:hover,.bl-board .hit.no:focus-visible{fill:var(--danger);fill-opacity:.35;stroke:var(--fg)}
.bl-board .ghost{pointer-events:none;opacity:.6}
.bl-board .tip{pointer-events:none;font-family:var(--f-num,'Atkinson Hyperlegible Mono',monospace);font-size:12px;font-weight:600;fill:var(--fg)}
.pl{max-width:664px;margin:0 auto;padding-top:4px;font-family:var(--f-body,system-ui,sans-serif);color:var(--fg)}
.pl .row{display:flex;align-items:center;justify-content:center;gap:10px;flex-wrap:wrap;margin:0 0 8px}
.pl .cards{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:0 0 8px}
.pl .card{position:relative;background:var(--bg);color:var(--fg);border:2.5px solid var(--fg);border-radius:10px;padding:6px 4px 6px;text-align:center;cursor:pointer;font:inherit;min-height:44px;line-height:1.2}
.pl .card[aria-pressed=true]{background:var(--primary);color:var(--on-primary)}.pl .card:disabled{opacity:.45;cursor:not-allowed}
.pl .card:focus-visible,.pl button:focus-visible{outline:3px solid var(--fg);outline-offset:2px}
.pl .card img{display:block;margin:0 auto}.pl .card .nm{font-weight:700;font-size:15px}.pl .card .cost{font-family:var(--f-num,monospace);font-weight:600;font-size:14px}
.pl .card .key{position:absolute;top:3px;left:6px;font-family:var(--f-num,monospace);font-size:12px;font-weight:600}
.pl .card .cnt{position:absolute;top:3px;right:5px;background:var(--fg);color:var(--bg);border-radius:999px;min-width:22px;line-height:22px;font-family:var(--f-num,monospace);font-size:13px;font-weight:600}
.pl .stat{font-family:var(--f-num,monospace);font-weight:600;font-size:16px}
.pl button.b{font:inherit;font-weight:600;color:var(--fg);background:var(--bg);border:2px solid var(--fg);border-radius:6px;padding:4px 14px;min-height:44px;cursor:pointer}
.pl button.b[aria-pressed=true]{background:var(--danger)}.pl button.b:disabled{opacity:.45;cursor:not-allowed}
.pl button.fight{background:var(--primary);color:var(--on-primary);font-size:17px}.pl button.fight:disabled{background:var(--bg);color:var(--fg)}
.pl .msg{min-height:22px;text-align:center;font-weight:600;color:var(--fg)}
.pl .hit.ok.legal{fill:var(--ok);fill-opacity:.9;stroke:var(--fg);stroke-width:1.5}
.pl .ghostp{pointer-events:none}.pl .gnum{font-family:var(--f-num,monospace);font-size:11px;font-weight:700;fill:var(--fg);paint-order:stroke;stroke:var(--bg);stroke-width:3px;pointer-events:none}
@media (max-width:520px){.pl .cards{gap:4px}.pl .card .nm{font-size:13px}.pl .card img{width:44px;height:44px}}
@media (prefers-reduced-motion:reduce){.pl *{transition:none!important;animation:none!important}}
"""

JS = """
const plans = (window.__ravelinPlans = window.__ravelinPlans || {});
function staticBoard(component) {
  const { data, setTriggerValue, parentElement } = component;
  const host = parentElement.querySelector('.bl-board');
  host.innerHTML = data.svg;                                    // trusted: app.game_view.board_svg
  const svg = host.querySelector('svg');
  const NS = svg.namespaceURI, CELL = data.cell;
  const mk = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
  let ghost = null;
  if (data.unit && data.tiles.length) {
    ghost = mk('use', {class: 'ghost', width: CELL - 10, height: CELL - 10, visibility: 'hidden'});
    ghost.setAttribute('href', '#ic-' + data.unit);
    svg.appendChild(ghost);
  }
  const showGhost = t => { if (ghost && t.ok) { ghost.setAttribute('x', t.col * CELL + 5); ghost.setAttribute('y', t.row * CELL + 5); ghost.setAttribute('visibility', 'visible'); } };
  const hideGhost = () => { if (ghost) ghost.setAttribute('visibility', 'hidden'); };
  const send = t => setTriggerValue('tile', t.b);
  data.tiles.forEach(t => {
    const r = mk('rect', {class: 'hit ' + (t.ok ? 'ok' : 'no'), x: t.col * CELL + 2, y: t.row * CELL + 2, width: CELL - 4, height: CELL - 4, tabindex: 0, role: 'button',
                          'data-b': t.b, 'aria-label': (t.ok ? 'Deploy at (' : 'Tile (') + t.row + ',' + t.col + ')'});
    r.addEventListener('click', () => send(t));
    r.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); send(t); } });
    r.addEventListener('pointerenter', () => showGhost(t));
    r.addEventListener('focus', () => showGhost(t));
    r.addEventListener('pointerleave', hideGhost);
    r.addEventListener('blur', hideGhost);
    svg.appendChild(r);
  });
  return () => { host.innerHTML = ''; };
}
function planner(component) {
  const { data, setTriggerValue, parentElement } = component;
  const host = parentElement.querySelector('.bl-board');
  const st = plans[data.gid] || (plans[data.gid] = {sel: 0, draft: (data.draft || []).map(d => [d[0], d[1]]), line: false});
  const cost = data.cards.map(c => c.cost);
  const free = new Set(data.tiles.filter(t => t.ok).map(t => t.b));
  const tileOf = {}; data.tiles.forEach(t => { tileOf[t.b] = t; });
  const spent = () => st.draft.reduce((a, d) => a + cost[d[0]], 0);
  const left = () => data.gold - spent();
  const taken = () => new Set(st.draft.map(d => d[1]));
  const NB = data.tiles.length;
  host.style.setProperty('--ok', data.okColor);
  host.innerHTML = '<div class="pl"><div class="row"><span class="stat" id="pl-gold"></span><span class="stat" id="pl-army"></span></div><div class="cards" id="pl-cards"></div>' +
    '<div class="row"><button class="b" id="pl-line" type="button" aria-pressed="false" title="Shift+tap">Line of 5</button><button class="b" id="pl-undo" type="button" title="Z">Undo</button>' +
    '<button class="b" id="pl-clear" type="button">Clear</button><button class="b fight" id="pl-fight" type="button" title="Enter">Fight</button></div>' +
    '<div class="msg" id="pl-msg" role="status" aria-live="polite"></div><div id="pl-svg" style="line-height:0">' + data.svg + '</div></div>';
  const $ = id => host.querySelector('#' + id);
  const svg = host.querySelector('#pl-svg svg');
  const NS = svg.namespaceURI, CELL = data.cell;
  const mk = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const gl = mk('g', {}, svg);                                   // placed ghosts
  const hl = mk('g', {}, svg);                                   // hit rects
  const msg = t => { $('pl-msg').textContent = t || ''; };
  msg(data.msg);
  function canPlace(u, b) {
    if (st.draft.length >= data.cap) return false;
    if (cost[u] > left()) return false;
    return free.has(b) && !taken().has(b);
  }
  function render() {
    $('pl-gold').textContent = left() + ' gold left';
    $('pl-army').textContent = st.draft.length + ' / ' + data.cap + ' units';
    const f = $('pl-fight');
    f.textContent = st.draft.length ? 'Fight! ' + st.draft.length + ' unit' + (st.draft.length > 1 ? 's' : '') + ', ' + left() + ' gold left' : 'Fight (place a unit first)';
    f.disabled = !st.draft.length;
    $('pl-undo').disabled = !st.draft.length; $('pl-clear').disabled = !st.draft.length;
    $('pl-line').setAttribute('aria-pressed', String(st.line));
    const cards = $('pl-cards'); cards.innerHTML = '';
    data.cards.forEach((c, u) => {
      const n = st.draft.filter(d => d[0] === u).length;
      const b = document.createElement('button');
      b.className = 'card'; b.type = 'button'; b.setAttribute('aria-pressed', String(st.sel === u)); b.dataset.u = u;
      b.setAttribute('aria-label', c.name + ', ' + c.cost + ' gold, key ' + (u + 1));
      b.innerHTML = '<span class="key">' + (u + 1) + '</span>' + (n ? '<span class="cnt">&times;' + n + '</span>' : '') + c.img + '<div class="nm">' + c.name + '</div><div class="cost">' + c.cost + ' gold</div>';
      b.onclick = () => { st.sel = u; msg(''); render(); };
      cards.appendChild(b);
    });
    gl.textContent = '';
    st.draft.forEach((d, k) => {
      const t = tileOf[d[1]];
      const g = mk('g', {class: 'ghostp', 'data-b': d[1], 'data-u': d[0]}, gl);
      const u = mk('use', {x: t.col * CELL + 5, y: t.row * CELL + 5, width: CELL - 10, height: CELL - 10, opacity: 0.8}, g); u.setAttribute('href', '#ic-' + data.cards[d[0]].icon);
      const tx = mk('text', {class: 'gnum', x: t.col * CELL + CELL - 6, y: t.row * CELL + 12, 'text-anchor': 'end'}, g); tx.textContent = k + 1;
    });
    hl.textContent = '';
    const tk = taken();
    data.tiles.forEach(t => {
      const own = tk.has(t.b), ok = !own && canPlace(st.sel, t.b);
      const r = mk('rect', {class: 'hit ' + (t.ok ? 'ok' : 'no') + (ok ? ' legal' : ''), x: t.col * CELL + 2, y: t.row * CELL + 2, width: CELL - 4, height: CELL - 4, tabindex: 0, role: 'button',
                            'data-b': t.b, 'aria-label': (own ? 'Remove unit at (' : 'Place at (') + t.row + ',' + t.col + ')'}, hl);
      r.addEventListener('click', e => tap(t.b, e.shiftKey || st.line));
      r.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); tap(t.b, e.shiftKey || st.line); } });
    });
  }
  function illegal(u, b) {
    setTriggerValue('illegal', {unit: u, tile: b, draft: st.draft.map(d => [d[0], d[1]]), n: ++st.n});
    msg('Checking with the engine...');
  }
  function tap(b, line) {
    const i = st.draft.findIndex(d => d[1] === b);
    if (i >= 0) { st.draft.splice(i, 1); msg(''); render(); return; }          // tap a ghost: remove it, refund
    const u = st.sel;
    if (!canPlace(u, b)) { illegal(u, b); return; }
    const want = line ? 5 : 1;
    const tk = taken();
    // adjacent tiles only: walk up the border while tiles are free, then down
    const run = [b];
    for (let s = 1; run.length < want && s < NB; s++) { const x = (b + s) % NB; if (!free.has(x) || tk.has(x)) break; run.push(x); }
    for (let s = 1; run.length < want && s < NB; s++) { const x = ((b - s) % NB + NB) % NB; if (!free.has(x) || tk.has(x)) break; run.push(x); }
    for (const x of run) { if (!canPlace(u, x)) break; st.draft.push([u, x]); }
    msg(''); render();
  }
  st.n = st.n || 0;
  $('pl-line').onclick = () => { st.line = !st.line; render(); };
  $('pl-undo').onclick = () => { st.draft.pop(); msg(''); render(); };
  $('pl-clear').onclick = () => { st.draft = []; msg(''); render(); };
  $('pl-fight').onclick = () => { if (st.draft.length) setTriggerValue('fight', {draft: st.draft.map(d => [d[0], d[1]]), n: ++st.n}); };
  const key = e => {
    if (e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
    if (e.key >= '1' && e.key <= String(data.cards.length)) { st.sel = +e.key - 1; msg(''); render(); }
    else if (e.key === 'z' || e.key === 'Z') { st.draft.pop(); msg(''); render(); }
    else if (e.key === 'Enter' && st.draft.length && !(e.target && e.target.closest && e.target.closest('.hit'))) { $('pl-fight').click(); }
  };
  document.addEventListener('keydown', key);
  render();
  return () => { document.removeEventListener('keydown', key); host.innerHTML = ''; };
}
export default function (component) {
  return component.data.mode === 'plan' ? planner(component) : staticBoard(component);
}
"""


def make_component(register=None):
    """Register the component once and return its mounting command. ``register`` is ``st.components.v2.component`` (a parameter so tests can swap it)."""
    if register is None:
        import streamlit as st
        register = st.components.v2.component
    return register("ravelin_board", html=HTML, css=CSS, js=JS)
