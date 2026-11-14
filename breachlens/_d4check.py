"""Independent D4 helpers for the B3 equivariance tests and the smoke script (not used by the simulator).

The 8 symmetries are written out here from scratch (they do NOT import `sim.D4_TABLES`), so a bug in the
simulator's tables cannot hide itself. Symmetry k < 4: rotate k times by 90 deg; k >= 4: flip columns, then rotate k-4 times.
"""
from __future__ import annotations

import copy

from .actions import BORDER
from .constants import GRID

N = GRID


def tr(k: int, rc):
    r, c = rc
    if k >= 4:
        c = N - 1 - c
    for _ in range(k % 4):
        r, c = c, N - 1 - r
    return (r, c)


def tr_inv(k: int, rc):
    for r in range(N):
        for c in range(N):
            if tr(k, (r, c)) == tuple(rc):
                return (r, c)
    raise ValueError(rc)


def transform_layout(L: dict, k: int) -> dict:
    """Same list orders (so structure ids are carried along), cells transformed."""
    out = copy.deepcopy(L)
    for b in out["buildings"]:
        b["cell"] = tr(k, b["cell"])
    out["core"]["cell"] = tr(k, out["core"]["cell"])
    out["walls"] = [tr(k, w) for w in out["walls"]]
    for d in out["defenders"]:
        d["cell"] = tr(k, d["cell"])
    return out


def transform_army(army, k: int):
    return [(t, tr(k, tile if isinstance(tile, tuple) else BORDER[tile])) for t, tile in army]


def structure_cells(L: dict) -> list:
    """Cell per structure id, in the order sim.Prepared assigns ids: buildings, core, defenders, walls."""
    return ([tuple(b["cell"]) for b in L["buildings"]] + [tuple(L["core"]["cell"])]
            + [tuple(d["cell"]) for d in L["defenders"]] + [tuple(w) for w in L["walls"]])


def summary(res, L: dict, k: int = 0) -> dict:
    """Integer outcome of a battle on layout L (already in frame k), mapped back to the original frame."""
    cells = structure_cells(L)
    back = {c: tr_inv(k, c) for c in set(cells)}
    return {
        "ticks": res.ticks, "reason": res.reason, "d_num": res.d_num, "core": res.core_destroyed,
        "struct_hp_total": sum(res.struct_hp), "unit_hp_total": sum(res.unit_hp),
        "units_alive": res.units_alive,
        "destroyed": frozenset(back[cells[s]] for s in res.destroyed),
        "struct_hp_by_cell": tuple(sorted((back[c], hp) for c, hp in zip(cells, res.struct_hp))),
        "unit_hp_multiset": tuple(sorted(res.unit_hp)),
    }


PRIMARY = ("ticks", "reason", "d_num", "core", "struct_hp_total", "unit_hp_total", "destroyed")


def diff(a: dict, b: dict, fields=None) -> list:
    return [f for f in (fields or a) if a[f] != b[f]]


def stabilizer(L: dict, army) -> list:
    """Symmetries k that map (layout, army) onto itself (as multisets)."""
    def key(Lx, ax):
        return (sorted((tuple(b["cell"]), b["value"]) for b in Lx["buildings"]), tuple(Lx["core"]["cell"]),
                sorted(map(tuple, Lx["walls"])), sorted((tuple(d["cell"]), d["type"]) for d in Lx["defenders"]),
                sorted((t, tuple(c)) for t, c in ax))
    base = key(L, transform_army(army, 0))
    return [k for k in range(8) if key(transform_layout(L, k), transform_army(army, k)) == base]


def equivariance_chunk(args):
    """Run n random battles x 8 symmetries. args = (seed, n, shuffle). Returns a dict of counts + failing cases.

    shuffle=True additionally permutes army order and the order of every layout list (adversarial id/order shuffle)
    independently in each frame, so structure and unit ids differ between frames.
    """
    import numpy as np

    from ._bench_layouts import FAMILIES, make_layout, random_army
    from .sim import Prepared, simulate

    seed, n, shuffle = args
    rng = np.random.default_rng(seed)
    bad, extra, menders, fb, eq = [], 0, 0, 0, 0
    for b in range(n):
        fam = FAMILIES[int(rng.integers(4))]
        ls = int(rng.integers(25))
        L = make_layout(fam, ls)
        army = random_army(rng, L, min_units=1, max_units=20, n_types=5)
        menders += any(t == 4 for t, _ in army)
        base = None
        for k in range(8):
            Lk = transform_layout(L, k)
            ak = transform_army(army, k)
            if shuffle:
                Lk["buildings"] = [Lk["buildings"][i] for i in rng.permutation(len(Lk["buildings"]))]
                Lk["walls"] = [Lk["walls"][i] for i in rng.permutation(len(Lk["walls"]))]
                Lk["defenders"] = [Lk["defenders"][i] for i in rng.permutation(len(Lk["defenders"]))]
                ak = [ak[i] for i in rng.permutation(len(ak))]
            r = simulate(Prepared(Lk), ak)
            fb += r.fallback_calls
            eq += r.canon_equal_picks
            s = summary(r, Lk, k)
            if base is None:
                base = s
                continue
            f = diff(base, s, PRIMARY)
            if f:
                bad.append({"battle": b, "family": fam, "layout_seed": ls, "sym": k, "fields": f,
                            "base": {x: base[x] for x in f}, "got": {x: s[x] for x in f},
                            "stabilizer": stabilizer(L, army), "army": army})
            elif diff(base, s, ("struct_hp_by_cell",)):
                extra += 1
    return {"n": n, "menders": menders, "fallback_calls": fb, "canon_equal_picks": eq, "bad": bad, "extra": extra}


def run_parallel(total: int, seed: int, shuffle: bool, procs: int = 6, chunk: int = 250):
    import multiprocessing as mp
    jobs = [(seed * 10_000 + j, min(chunk, total - j * chunk), shuffle) for j in range((total + chunk - 1) // chunk)]
    with mp.get_context("spawn").Pool(procs) as pool:
        parts = pool.map(equivariance_chunk, jobs)
    return {"n": sum(p["n"] for p in parts), "menders": sum(p["menders"] for p in parts),
            "fallback_calls": sum(p["fallback_calls"] for p in parts),
            "canon_equal_picks": sum(p["canon_equal_picks"] for p in parts),
            "bad": [c for p in parts for c in p["bad"]], "extra": sum(p["extra"] for p in parts)}
