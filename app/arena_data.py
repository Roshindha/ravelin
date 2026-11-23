"""B24: the data side of the arena (pure module, no streamlit import; the simulator, env, scoring and caches are used exactly as they are).

* ``canonical_actions(draft)``: a placement draft ``[(unit type, border tile), ...]`` (in placement order) becomes the env's legal action sequence: one BUY per
  placed unit (sorted by type, so the same army always gives the same buys), END_BUY, then the deploys in placement order. The battle starts with the last deploy.
  Two click orders that produce the same ordered deploys therefore produce the same actions and the same outcome.
* ``play_draft(duel, draft)`` sends those actions through ``breachlens.duel.Duel`` (the same engine, masks and messages as the CLI); ``probe_message`` returns the
  engine's own message for a placement that the draft cannot take.
* ``raw_battle`` records one battle with RAW fixed-point hp per object per tick: unit hp from ``simulate(record=True)``, structure hp from prefix re-runs
  ``simulate(max_ticks=t)`` (the simulator is not changed; the prefix hp is checked against the recorded trace). Results are cached by (layout id, ordered army).
* ``damage_events`` / ``totals`` / ``moments``: what the player shows, derived from those frames only (damage = the drop in raw hp between two frames).
"""
from __future__ import annotations

from collections import OrderedDict

from breachlens.actions import BORDER, encode_buy, encode_deploy
from breachlens.constants import END_BUY
from breachlens.sim import cell_of, simulate

CACHE_SIZE = 16
_CACHE: OrderedDict = OrderedDict()


def canonical_actions(draft) -> list:
    """Buys (sorted by unit type), END_BUY, deploys in placement order. An empty draft gives no actions."""
    draft = [(int(u), int(b)) for u, b in draft]
    if not draft:
        return []
    return [encode_buy(u) for u in sorted(u for u, _ in draft)] + [END_BUY] + [encode_deploy(u, b) for u, b in draft]


def play_draft(duel, draft) -> list:
    """Apply the canonical action sequence of ``draft`` to a fresh ``Duel`` (its own ``buy`` / ``end_buy`` / ``deploy`` calls, so a tampered tile or unit gives the engine's
    message); a DuelError propagates. Returns the actions."""
    from breachlens.constants import UNIT_TYPES
    from breachlens.duel import DuelError
    draft = [(int(u), int(b)) for u, b in draft]
    if not draft:
        return []
    if any(not 0 <= u < len(UNIT_TYPES) for u, _ in draft):
        raise DuelError(f"unknown unit index in the draft; choose one of: {', '.join(UNIT_TYPES)}")
    for u in sorted(u for u, _ in draft):
        duel.buy(UNIT_TYPES[u])
    duel.end_buy()
    for u, b in draft:
        duel.deploy(UNIT_TYPES[u], b)
    return list(duel.actions)


def probe_message(layout_id: str, draft, unit: int, tile: int) -> str:
    """The engine's message if ``draft`` + one more (unit, tile) is not a legal placement, else an empty string. A scratch Duel is used (nothing is logged)."""
    from breachlens.duel import Duel, DuelError
    try:
        d = Duel(layout_id, human_runs_path=None)
        play_draft(d, [*map(tuple, draft), (int(unit), int(tile))])
    except DuelError as e:
        return str(e)
    return ""


def struct_kinds(P) -> list:
    return ["b"] * P.n_buildings + ["c"] + ["d"] * P.n_def + ["w"] * P.n_walls


def raw_battle(P, army, key=None) -> dict:
    """{frames, units, umax, smax}: frames[t] = {t, u: [[cell, raw hp, state]], s: [raw hp per structure]} for t = 0 .. ticks (frame 0 = the start)."""
    ck = (key, tuple((int(t), int(b)) for t, b in army)) if key is not None else None
    if ck is not None and ck in _CACHE:
        _CACHE.move_to_end(ck)
        return _CACHE[ck]
    army = [(int(t), int(b)) for t, b in army]
    full = simulate(P, army, record=True)
    umax = [P.cfg.units[t].hp for t, _ in army]
    smax = list(P.s_hp0)
    frames = [{"t": 0, "u": [[cell_of(BORDER[b]), umax[i], 0] for i, (_, b) in enumerate(army)], "s": list(smax)}]
    pre = None
    for t in range(1, full.ticks + 1):
        pre = simulate(P, army, max_ticks=t)
        row = full.trace[t - 1]
        assert [h for _, h, _ in row] == list(pre.unit_hp), "prefix run diverges from the recorded trace"
        frames.append({"t": t, "u": [[c, h if st >= 0 else 0, st] for c, h, st in row], "s": list(pre.struct_hp)})
    if pre is not None:
        assert list(pre.struct_hp) == list(full.struct_hp)
    out = {"frames": frames, "units": [[t, b] for t, b in army], "umax": umax, "smax": smax, "S": P.cfg.S, "ticks": full.ticks, "result": full}
    if ck is not None:
        _CACHE[ck] = out
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    return out


def damage_events(frames: list) -> list:
    """Per frame t >= 1: {"u": [(unit, raw loss)], "s": [(structure, raw loss)]}; a loss is hp(t-1) - hp(t) > 0 (heals are not damage)."""
    out = [{"u": [], "s": []}]
    for a, b in zip(frames, frames[1:]):
        out.append({"u": [(i, x[1] - y[1]) for i, (x, y) in enumerate(zip(a["u"], b["u"])) if y[1] < x[1]],
                    "s": [(j, x - y) for j, (x, y) in enumerate(zip(a["s"], b["s"])) if y < x]})
    return out


def totals(frames: list, kinds: list, umax: list, smax: list) -> list:
    """Per frame: raw sums shown by the three bars. army = all deployed troops; fort = every defender and building (walls and the core excluded: the core has its own bar); core = the core."""
    fort_idx = [i for i, k in enumerate(kinds) if k in ("b", "d")]
    ci = kinds.index("c")
    fmax = sum(smax[i] for i in fort_idx)
    return [{"army": sum(u[1] for u in f["u"]), "army_max": sum(umax), "fort": sum(f["s"][i] for i in fort_idx), "fort_max": fmax,
             "core": f["s"][ci], "core_max": smax[ci]} for f in frames]


def moments(frames: list, kinds: list) -> list:
    """Per frame: the key moments that happen on that tick: 'destroyed' (a defender or building falls), 'lost' (one of the player's troops dies), 'core' (the core takes damage)."""
    ci = kinds.index("c")
    out = [[]]
    for a, b in zip(frames, frames[1:]):
        m = []
        if any(kinds[j] in ("b", "c", "d") and x > 0 >= y for j, (x, y) in enumerate(zip(a["s"], b["s"]))):
            m.append("destroyed")
        if any(x[1] > 0 >= y[1] for x, y in zip(a["u"], b["u"])):
            m.append("lost")
        if b["s"][ci] < a["s"][ci]:
            m.append("core")
        out.append(m)
    return out


def ranges(P) -> dict:
    """Static range circles in tiles: defenders [row, col, min, max] and the unit types' attack range."""
    from breachlens.sim import COL, ROW
    return {"d": [[ROW[P.s_cell[s]], COL[P.s_cell[s]], ds.min_range_x10 / 10, ds.max_range_x10 / 10] for s, ds in zip(P.def_sids, P.def_stats)],
            "u": [u.range_x10 / 10 for u in P.cfg.units]}
