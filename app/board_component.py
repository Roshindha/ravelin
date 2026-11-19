"""B18e-b: the interactive board (docs/SPEC_V2.md, Gate changelog row (oo) (d)). A small custom component with NO build step: inline HTML / CSS / JS strings
for ``st.components.v2.component`` (Streamlit 1.67's supported bidirectional component API). No package, no network request, no new dependency.

Python -> component: ``data = {svg, tiles, unit}`` where ``svg`` is ``app.game_view.board_svg`` (trusted, built here), ``tiles`` lists every border tile as
``{b, row, col, ok}`` (``ok`` = free and unoccupied) and ``unit`` is the icon name of the unit that a click would deploy (for the hover preview).
Component -> Python: ``setTriggerValue('tile', b)`` with the border index ``b`` on a click, a tap (pointer events: touch works) or Enter / Space on a focused tile.
The app turns ``b`` into the action ``5 + 52*u + b`` and sends it through the duel engine, exactly like the Deploy button. Tiles that are not ``ok`` still report
the click, so the engine's own message is what the player reads (it never changes any state). The board is ``width:100%`` and scales on narrow screens.

BUY and Watch-the-AI use the same component with ``tiles = []`` (a static board that takes the page fonts).
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
"""

JS = """
export default function (component) {
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
"""


def make_component(register=None):
    """Register the component once and return its mounting command. ``register`` is ``st.components.v2.component`` (a parameter so tests can swap it)."""
    if register is None:
        import streamlit as st
        register = st.components.v2.component
    return register("ravelin_board", html=HTML, css=CSS, js=JS)
