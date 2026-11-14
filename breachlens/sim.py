"""BreachLens battle simulator, B3 scope (docs/SPEC_V2.md 3.1, 3.1b, 3.3, 3.4 rule 7 keys 1-4).

Scout / Sniper / Bulwark / Blaster / Mender, Gun / Lancer / Flak defenders, walls, buildings, core,
fixed-point hp (scale S), tick order, nearest-in-preference targeting, BFS movement with the full rule-7
key chain, early exit. All outcome-deciding arithmetic is integer.

Tie handling (D4 equivariance). Object ids and raw coordinates never decide an outcome:
  * attacks and heals that tie on D4-invariant keys SPLIT exactly (floor division of fixed-point hp);
  * decisions that cannot be split (movement step, movement target, splash centre, flak centre, wall
    to chew) go through `canon` (rule 7 key 4): each tied candidate gets the start-of-phase world
    serialized with the candidate decision marked; the lexicographically smallest serialization over the
    8 D4 transforms wins. `fallback_calls` counts these genuine canonical-form invocations.
  * If two candidates have EQUAL canonical strings their post-decision worlds are D4 images of one
    another (a symmetry of the world swaps them), so any pick gives D4-equivalent outcomes; the first
    in iteration order is taken and `canon_equal_picks` is counted. That pick is the only place where
    iteration order is consulted, and it is outcome-equivalent up to a symmetry of the world.
  * Every tied decision reads the start-of-movement-phase snapshot (plus the deciding unit's own steps
    this tick), never other units' decisions of the same tick, so processing order cannot matter.

Layout dict (cells are (row, col)):
  {"buildings": [{"cell": (r, c), "value": 1..3, "hp": human_hp (optional)}],
   "core": {"cell": (r, c), "hp": human_hp (optional)},
   "walls": [(r, c), ...],
   "defenders": [{"cell": (r, c), "type": "gun" | "lancer" | "flak"}]}
Army: list of (unit_type_idx, border_tile_idx). A (row, col) tuple in place of the tile index is a test hook
for interior start cells.
"""
from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import dataclass, field

from .actions import BORDER
from .config import CFG, CLASS_BUILDING, CLASS_TURRET, CLASS_WALL
from .constants import GRID, MAX_BATTLE_TICKS

N = GRID
NCELL = N * N
ROW = tuple(c // N for c in range(NCELL))
COL = tuple(c % N for c in range(NCELL))
D2 = tuple(tuple((ROW[a] - ROW[b]) ** 2 + (COL[a] - COL[b]) ** 2 for b in range(NCELL)) for a in range(NCELL))
MENDER = 4


def _build_d4():
    rot = [ (c % N) * N + (N - 1 - c // N) for c in range(NCELL)]   # (r, c) -> (c, N-1-r)
    flip = [ (c // N) * N + (N - 1 - c % N) for c in range(NCELL)]  # (r, c) -> (r, N-1-c)
    tabs = []
    for f in (False, True):
        cur = list(range(NCELL))
        for _ in range(4):
            cur = [rot[x] for x in cur]
            tabs.append(tuple(flip[x] for x in cur) if f else tuple(cur))
    return tuple(tabs)


D4_TABLES = _build_d4()   # D4_TABLES[k][cell] = image of cell under symmetry k (all 8 distinct)

# 8-connected moves: (neighbour, orthogonal cells that must be passable for a diagonal step)
MOVES = []
for _c in range(NCELL):
    _r, _q = ROW[_c], COL[_c]
    _l = []
    for _dr in (-1, 0, 1):
        for _dc in (-1, 0, 1):
            if _dr == 0 and _dc == 0:
                continue
            _nr, _nq = _r + _dr, _q + _dc
            if not (0 <= _nr < N and 0 <= _nq < N):
                continue
            _n = _nr * N + _nq
            _l.append((_n, (_r * N + _nq, _nr * N + _q) if (_dr and _dc) else None))
    MOVES.append(tuple(_l))
MOVES = tuple(MOVES)

_RANGE_CELLS: dict = {}
_DISC: dict = {}


def cells_in_range(center: int, r_x10: int) -> tuple:
    """Cells whose squared distance to `center` satisfies 100*d2 <= r_x10^2 (integer test)."""
    key = (center, r_x10)
    v = _RANGE_CELLS.get(key)
    if v is None:
        lim = r_x10 * r_x10
        v = tuple(c for c in range(NCELL) if 100 * D2[center][c] <= lim)
        _RANGE_CELLS[key] = v
    return v


def disc_others(center: int, r_x10: int) -> tuple:
    """Cells within r (excluding the centre)."""
    key = (center, r_x10)
    v = _DISC.get(key)
    if v is None:
        v = tuple(c for c in cells_in_range(center, r_x10) if c != center)
        _DISC[key] = v
    return v


def cell_of(rc) -> int:
    r, c = rc
    if not (0 <= r < N and 0 <= c < N):
        raise ValueError(f"cell {rc} outside the {N}x{N} grid")
    return r * N + c


@dataclass
class BattleResult:
    ticks: int
    reason: str                 # cleared | all_dead | no_attackers | stalemate | horizon
    core_destroyed: bool
    d_num: int                  # value-weighted destroyed (ints)
    d_den: int
    score: float                # reporting only; derived from the ints above
    destroyed: tuple            # destroyed building sids (value buildings and core)
    units_alive: int
    fallback_calls: int         # genuine canonical-form (rule 7 key 4) invocations
    walls_destroyed: int = 0
    defenders_destroyed: int = 0
    max_idle_run: int = 0       # longest run of ticks any unit made no progress (move or attack)
    struct_hp: tuple = ()       # final fixed-point hp per structure sid
    unit_hp: tuple = ()         # final fixed-point hp per unit (army order)
    trace: list = field(default_factory=list)  # record mode: per tick [(cell, hp, status)] per unit
    canon_equal_picks: int = 0  # canonical-form calls whose best candidates had EQUAL strings (symmetric pick)
    # B5 instrumentation (does not influence any decision): per unit type (5) counters
    attack_ticks: tuple = ()        # unit-ticks on which a unit of the type attacked (primary target chosen)
    wall_attack_ticks: tuple = ()   # ... of which the primary target was a wall
    alive_ticks: tuple = ()         # unit-ticks alive at the start of a tick
    walls_hit: tuple = ()           # distinct walls that were a primary target of a unit of the type
    wall_attack_any_ticks: int = 0  # ticks on which at least one unit attacked a wall

    @property
    def d(self) -> float:
        return self.d_num / self.d_den if self.d_den else 0.0


class Prepared:
    """Static, battle-independent view of a layout plus caches shared across battles on it."""

    def __init__(self, layout: dict, cfg=CFG):
        self.cfg = cfg
        cell, cls, hp, val, dtype = [], [], [], [], []
        self.n_buildings = len(layout["buildings"])
        for b in layout["buildings"]:
            cell.append(cell_of(b["cell"])); cls.append(CLASS_BUILDING)
            hp.append(int(b["hp"]) * cfg.S if b.get("hp") else cfg.building_hp_default)
            val.append(int(b["value"])); dtype.append(-1)
        core = layout["core"]
        self.core_sid = len(cell)
        cell.append(cell_of(core["cell"])); cls.append(CLASS_BUILDING)
        hp.append(int(core["hp"]) * cfg.S if core.get("hp") else cfg.core_hp)
        val.append(0); dtype.append(-1)
        self.def_sids = []
        for d in layout["defenders"]:
            di = cfg.defender_index[d["type"]]
            self.def_sids.append(len(cell))
            cell.append(cell_of(d["cell"])); cls.append(CLASS_TURRET)
            hp.append(cfg.defenders[di].hp); val.append(0); dtype.append(di)
        self.wall_base = len(cell)
        self.wall_bit_of_cell = [0] * NCELL
        for k, w in enumerate(layout["walls"]):
            c = cell_of(w)
            cell.append(c); cls.append(CLASS_WALL); hp.append(cfg.wall_hp); val.append(0); dtype.append(-1)
            self.wall_bit_of_cell[c] = 1 << k
        self.n_walls = len(layout["walls"])
        self.s_cell, self.s_class, self.s_hp0, self.s_value, self.s_dtype = cell, cls, hp, val, dtype
        self.n_s = len(cell)
        self.d_den = sum(val[: self.n_buildings])
        self.value_sids = tuple(range(self.n_buildings)) + (self.core_sid,)
        self.cls_members = [[s for s in range(self.n_s) if cls[s] == k] for k in range(3)]
        self.full_wallmask = (1 << self.n_walls) - 1
        # defender cover: cover[cell] = ((did, d2), ...) for defenders whose [min, max] range covers cell
        self.n_def = len(self.def_sids)
        self.def_stats = [cfg.defenders[dtype[s]] for s in self.def_sids]
        cover = [[] for _ in range(NCELL)]
        for did, s in enumerate(self.def_sids):
            ds, dc = self.def_stats[did], cell[s]
            for c in range(NCELL):
                d100 = 100 * D2[dc][c]
                if ds.min_range_x10 ** 2 <= d100 <= ds.max_range_x10 ** 2:
                    cover[c].append((did, D2[dc][c]))
        self.cover = tuple(tuple(x) for x in cover)
        self.cover_ids = tuple(tuple(d for d, _ in x) for x in cover)
        # caches
        self._cand: dict = {}
        self._fields: dict = {}
        self._region: dict = {}
        self._blocked: dict = {}
        self._splash_nb: dict = {}

    # candidate structures in attack range of a unit of type `ut` standing on `cell`, best-first
    def cand(self, ut: int, cell: int) -> tuple:
        key = ut * NCELL + cell
        v = self._cand.get(key)
        if v is None:
            u = self.cfg.units[ut]
            rank = u.class_rank
            lim = u.range_x10 * u.range_x10
            lst = []
            for s in range(self.n_s):
                d2 = D2[cell][self.s_cell[s]]
                if 100 * d2 <= lim and rank[self.s_class[s]] < len(u.pref):
                    lst.append((rank[self.s_class[s]], d2, s))
            lst.sort()
            v = tuple(lst)
            self._cand[key] = v
        return v

    def splash_nb(self, sid: int, r_x10: int) -> tuple:
        key = (sid, r_x10)
        v = self._splash_nb.get(key)
        if v is None:
            c0 = self.s_cell[sid]
            lim = r_x10 * r_x10
            v = tuple(s for s in range(self.n_s) if s != sid and 100 * D2[c0][self.s_cell[s]] <= lim)
            self._splash_nb[key] = v
        return v

    def field(self, sid: int, r_x10: int, wallmask: int) -> list:
        """BFS steps to the nearest passable cell from which `sid` is within r_x10 (255 = unreachable)."""
        key = (sid, r_x10, wallmask)
        f = self._fields.get(key)
        if f is not None:
            return f
        wb = self.wall_bit_of_cell
        f = [255] * NCELL
        frontier = []
        for c in cells_in_range(self.s_cell[sid], r_x10):
            if not (wallmask & wb[c]):
                f[c] = 0
                frontier.append(c)
        d = 0
        while frontier:
            d += 1
            nxt = []
            for c in frontier:
                for n, req in MOVES[c]:
                    if f[n] != 255 or (wallmask & wb[n]):
                        continue
                    if req is not None and ((wallmask & wb[req[0]]) or (wallmask & wb[req[1]])):
                        continue
                    f[n] = d
                    nxt.append(n)
            frontier = nxt
        self._fields[key] = f
        return f

    def region(self, p: int, wallmask: int) -> tuple:
        """(region_id, frozenset of cells) reachable from p over passable cells."""
        key = (p, wallmask)
        v = self._region.get(key)
        if v is None:
            wb = self.wall_bit_of_cell
            seen = {p}
            stack = [p]
            while stack:
                c = stack.pop()
                for n, req in MOVES[c]:
                    if n in seen or (wallmask & wb[n]):
                        continue
                    if req is not None and ((wallmask & wb[req[0]]) or (wallmask & wb[req[1]])):
                        continue
                    seen.add(n)
                    stack.append(n)
            v = (min(seen), frozenset(seen))
            self._region[key] = v
        return v


def prepare(layout: dict, cfg=CFG) -> Prepared:
    return Prepared(layout, cfg)


def _hpfrac(hp: int, mx: int) -> int:
    return hp * 1_000_000 // mx


def simulate(layout, army, record: bool = False, max_ticks: int = MAX_BATTLE_TICKS,
             _no_attackers_exit: bool = True) -> BattleResult:
    P = layout if isinstance(layout, Prepared) else Prepared(layout)
    cfg = P.cfg
    S = cfg.S
    units_cfg = cfg.units
    n_s = P.n_s
    s_cell, s_class, s_value, s_dtype = P.s_cell, P.s_class, P.s_value, P.s_dtype
    s_hp = list(P.s_hp0)
    s_max = P.s_hp0
    s_alive = [True] * n_s
    wallmask = P.full_wallmask
    wall_base = P.wall_base

    # --- units
    u_type, u_cell, u_hp, u_max, u_acc = [], [], [], [], []
    for ut, tile in army:
        rc = tile if isinstance(tile, tuple) else BORDER[tile]  # tuple = test hook: interior start cell
        c = cell_of(rc)
        if P.wall_bit_of_cell[c]:
            raise ValueError(f"unit deployed on a wall cell {rc}")
        u_type.append(ut); u_cell.append(c)
        u_hp.append(units_cfg[ut].hp); u_max.append(units_cfg[ut].hp); u_acc.append(0)
    nu = len(u_type)
    has_mender = MENDER in u_type
    u_alive = [True] * nu
    alive_units = list(range(nu))
    n_ut = len(units_cfg)
    c_att, c_wall, c_alive = [0] * n_ut, [0] * n_ut, [0] * n_ut
    c_walls_hit = [set() for _ in range(n_ut)]
    wall_any = 0
    attackers = [units_cfg[t].damage > 0 for t in u_type]
    n_attackers = sum(1 for i in alive_units if attackers[i])
    u_idle = [0] * nu
    max_idle = 0

    d_alive = [True] * P.n_def
    fallback_calls = 0
    equal_picks = 0
    version = 0          # bumped whenever any structure dies (invalidates movement-target cache)
    mt_cache: dict = {}
    cen = None           # centroid sums (m, sx, sy) of surviving value buildings + core
    live_value = set(P.value_sids)
    trace = []
    destroyed_order = []
    idle_ticks = 0
    reason = "horizon"
    tick = 0
    pend: dict = {}      # unit -> steps it will take this tick (set at the start of the movement phase)

    def centroid():
        m = sx = sy = 0
        for s in live_value:
            m += 1; sx += ROW[s_cell[s]]; sy += COL[s_cell[s]]
        return m, sx, sy

    cen = centroid()

    # ---------------- rule 7 key 4: canonical-form fallback
    def canon_pick(cands, marks_of):
        """Rule 7 key 4. cands: tied decisions; marks_of(c) -> (marks_s, marks_u, override) describing the world
        after decision c (which entities are tagged, which units sit on another cell). The candidate whose canonical
        serialization (lexicographic minimum over the 8 D4 transforms) is smallest wins; EQUAL strings mean the
        post-decision worlds are D4 images of each other, so the first of them is taken (counted in equal_picks)."""
        nonlocal fallback_calls, equal_picks
        fallback_calls += 1
        ents, idx_s, idx_u = [], {}, {}
        for sd in range(n_s):
            if s_alive[sd]:
                idx_s[sd] = len(ents)
                ents.append((s_cell[sd], 0, s_class[sd], s_dtype[sd], s_value[sd], s_hp[sd], s_max[sd], ()))
        for i in alive_units:
            idx_u[i] = len(ents)
            ents.append((u_cell[i], 1, u_type[i], u_hp[i], u_max[i], u_acc[i], pend.get(i, -1), ()))
        # the sorted, transformed base world is built once and shared by all candidates
        bases = [sorted((tab[e[0]],) + e[1:] for e in ents) for tab in D4_TABLES]
        best, chosen, n_best = None, None, 0
        for c in cands:
            ms, mu, ov = marks_of(c)
            edits = []   # (index in ents, new cell, new mark)
            for sd, mk in ms.items():
                edits.append((idx_s[sd], ents[idx_s[sd]][0], mk))
            for i in set(mu) | set(ov):
                k = idx_u[i]
                edits.append((k, ov.get(i, ents[k][0]), mu.get(i, ())))
            key = None
            for tab, base in zip(D4_TABLES, bases):
                lst = base[:]
                for k, _, _ in edits:
                    e = ents[k]
                    del lst[bisect_left(lst, (tab[e[0]],) + e[1:])]
                for k, cell, mk in edits:
                    e = ents[k]
                    insort(lst, (tab[cell],) + e[1:-1] + (mk,))
                t = tuple(lst)
                if key is None or t < key:
                    key = t
            if best is None or key < best:
                best, chosen, n_best = key, c, 1
            elif key == best:
                n_best += 1
        if n_best > 1:
            equal_picks += 1
        return chosen

    def move_target(i, ut, cell, rem):
        """Nearest alive structure of the most-preferred non-empty class (movement target)."""
        u = units_cfg[ut]
        for cl in u.pref:
            best = None
            tied = []
            for s in P.cls_members[cl]:
                if not s_alive[s]:
                    continue
                key = (D2[cell][s_cell[s]], -s_value[s])
                if best is None or key < best:
                    best, tied = key, [s]
                elif key == best:
                    tied.append(s)
            if tied:
                if len(tied) > 1:
                    return canon_pick(tied, lambda s: ({s: (6,)}, {i: (5, rem)}, {i: cell})), False
                return tied[0], True
        return -1, True

    # ---- Mender helpers (target point = centroid of living non-Mender allies, snapshot positions)
    def mender_sources(m, sr, sc):
        """Passable cells within 1.5 of the centroid (sr/m, sc/m); only the 4x4 box around it can qualify."""
        wb = P.wall_bit_of_cell
        out = []
        r0, r1 = max(0, (sr - 2 * m) // m), min(N - 1, (sr + 2 * m) // m + 1)
        q0, q1 = max(0, (sc - 2 * m) // m), min(N - 1, (sc + 2 * m) // m + 1)
        for r_ in range(r0, r1 + 1):
            for q_ in range(q0, q1 + 1):
                c = r_ * N + q_
                if wallmask & wb[c]:
                    continue
                T = m * m * (r_ * r_ + q_ * q_) - 2 * m * (r_ * sr + q_ * sc) + sr * sr + sc * sc
                if 100 * T <= 225 * m * m:
                    out.append(c)
        return out

    def mender_T(c, m, sr, sc):
        r_, q_ = ROW[c], COL[c]
        return m * m * (r_ * r_ + q_ * q_) - 2 * m * (r_ * sr + q_ * sc) + sr * sr + sc * sc

    def mender_field(p, srcs):
        """Reverse BFS from `srcs`, stopped once p is labelled. Returns dict cell->steps or None if p unreachable."""
        wb = P.wall_bit_of_cell
        f = {c: 0 for c in srcs}
        if p in f:
            return f
        frontier = list(srcs)
        d = 0
        while frontier:
            d += 1
            nxt = []
            for c in frontier:
                for n, req in MOVES[c]:
                    if n in f or (wallmask & wb[n]):
                        continue
                    if req is not None and ((wallmask & wb[req[0]]) or (wallmask & wb[req[1]])):
                        continue
                    f[n] = d
                    nxt.append(n)
            if p in f:
                return f
            frontier = nxt
        return None

    if _no_attackers_exit and nu and not n_attackers:
        reason = "no_attackers"    # nothing in this army can ever damage a structure; ends before tick 1
        max_ticks = 0
    while tick < max_ticks:
        # ---------------- (a)/(b)/(c): intents, damage and heals from the start-of-tick snapshot
        dmg_s: dict = {}
        dmg_u: dict = {}
        heal_u: dict = {}
        acted = set()
        moving = []
        pend.clear()
        wall_this_tick = False
        for i in alive_units:
            ut = u_type[i]
            c_alive[ut] += 1
            cands = P.cand(ut, u_cell[i])
            first = -1
            for k, (rk, d2, s) in enumerate(cands):
                if s_alive[s]:
                    first = k
                    break
            if first < 0:
                moving.append(i)
                continue
            rk0, d20, _ = cands[first]
            tied = []
            for k in range(first, len(cands)):
                rk, d2, s = cands[k]
                if rk != rk0 or d2 != d20:
                    break
                if s_alive[s]:
                    tied.append(s)
            if len(tied) > 1:
                bk = None
                keep = []
                for s in tied:
                    key = (_hpfrac(s_hp[s], s_max[s]), -s_value[s])
                    if bk is None or key < bk:
                        bk, keep = key, [s]
                    elif key == bk:
                        keep.append(s)
                tied = keep
            u = units_cfg[ut]
            acted.add(i)
            if u.splash_pct:
                if len(tied) > 1:   # a splash volley never splits: resolve to one centre
                    tied = [canon_pick(tied, lambda s, i=i: ({s: (2,)}, {i: (1,)}, {}))]
                s = tied[0]
                dmg_s[s] = dmg_s.get(s, 0) + u.damage
                share = u.damage * u.splash_pct // 100
                for t in P.splash_nb(s, u.splash_r_x10):
                    if s_alive[t]:
                        dmg_s[t] = dmg_s.get(t, 0) + share
            else:
                hit = u.damage // len(tied)
                for s in tied:
                    dmg_s[s] = dmg_s.get(s, 0) + hit
            # B5 counters (read-only): the primary target class of this volley, after any splash-centre resolution
            c_att[ut] += 1
            if rk0 == u.class_rank[CLASS_WALL]:
                c_wall[ut] += 1
                wall_this_tick = True
                for s_ in tied:
                    c_walls_hit[ut].add(s_)

        wall_any += wall_this_tick
        # Menders: heal the nearest damaged ally in range (key chain), ties split exactly
        for m in (alive_units if has_mender else ()):
            if u_type[m] != MENDER:
                continue
            mu_ = units_cfg[MENDER]
            lim = mu_.heal_range_x10 * mu_.heal_range_x10
            best, tiedh = None, []
            for j in alive_units:
                if j == m or u_hp[j] >= u_max[j]:
                    continue
                d2 = D2[u_cell[m]][u_cell[j]]
                if 100 * d2 > lim:
                    continue
                key = (d2, _hpfrac(u_hp[j], u_max[j]), -units_cfg[u_type[j]].cost)
                if best is None or key < best:
                    best, tiedh = key, [j]
                elif key == best:
                    tiedh.append(j)
            if tiedh:
                acted.add(m)  # healing is the Mender's action (progress for the idle invariant); it still moves
                share = mu_.heal // len(tiedh)
                for j in tiedh:
                    heal_u[j] = heal_u.get(j, 0) + share

        # defenders fire at units
        if alive_units and P.n_def:
            occ: dict = {}
            for i in alive_units:
                occ.setdefault(u_cell[i], []).append(i)
            near: dict = {}
            area: dict = {}
            for c, ul in occ.items():
                for did, d2 in P.cover[c]:
                    if not d_alive[did]:
                        continue
                    ds = P.def_stats[did]
                    if ds.period > 1 and tick % ds.period:
                        continue
                    if ds.area:
                        area.setdefault(did, []).append((c, d2))
                        continue
                    b = near.get(did)
                    if b is None or d2 < b[0]:
                        near[did] = [d2, [c]]
                    elif d2 == b[0]:
                        b[1].append(c)
            for did, (d2, cl) in near.items():
                ds = P.def_stats[did]
                cand_u = [i for c in cl for i in occ[c]]
                if len(cand_u) > 1:
                    bk = None
                    keep = []
                    for i in cand_u:
                        key = (_hpfrac(u_hp[i], u_max[i]), -units_cfg[u_type[i]].cost)
                        if bk is None or key < bk:
                            bk, keep = key, [i]
                        elif key == bk:
                            keep.append(i)
                    cand_u = keep
                k = len(cand_u)
                hit = ds.damage // k
                floor_ = cfg.min_hit // k
                for i in cand_u:
                    dmg_u[i] = dmg_u.get(i, 0) + max(hit - units_cfg[u_type[i]].armor, floor_)
            for did, cl in area.items():
                ds = P.def_stats[did]
                best = None
                tiedc = []
                for c, d2 in cl:
                    cnt = len(occ[c])
                    cost = sum(units_cfg[u_type[i]].cost for i in occ[c])
                    for n in disc_others(c, ds.splash_r_x10):
                        o = occ.get(n)
                        if o:
                            cnt += len(o)
                            cost += sum(units_cfg[u_type[i]].cost for i in o)
                    key = (-cnt, -cost, d2)
                    if best is None or key < best:
                        best, tiedc = key, [c]
                    elif key == best:
                        tiedc.append(c)
                if len(tiedc) > 1:   # a burst has one centre: resolve by canonical form
                    dsid = P.def_sids[did]
                    tiedc = [canon_pick(tiedc, lambda c, dsid=dsid: ({dsid: (3,)}, {i: (4,) for i in occ[c]}, {}))]
                c0 = tiedc[0]
                ul = occ[c0]
                # the centre is the lowest-(hp fraction, -cost) unit in the cell; exact ties share centre + neighbour damage
                ck = min((_hpfrac(u_hp[i], u_max[i]), -units_cfg[u_type[i]].cost) for i in ul)
                ctie = [i for i in ul if (_hpfrac(u_hp[i], u_max[i]), -units_cfg[u_type[i]].cost) == ck]
                kc = len(ctie)
                centre_share = (ds.damage + (kc - 1) * ds.damage_neighbour) // kc
                for i in ul:
                    dm = centre_share if i in ctie else ds.damage_neighbour
                    dmg_u[i] = dmg_u.get(i, 0) + max(dm - units_cfg[u_type[i]].armor, cfg.min_hit)
                for n in disc_others(c0, ds.splash_r_x10):
                    for i in occ.get(n, ()):
                        dmg_u[i] = dmg_u.get(i, 0) + max(ds.damage_neighbour - units_cfg[u_type[i]].armor, cfg.min_hit)

        # ---------------- (d)/(e): apply damage, then heals to survivors, remove dead
        changed = any(d > 0 for d in dmg_s.values()) or any(d > 0 for d in dmg_u.values())
        structure_died = False
        for s, d in dmg_s.items():
            s_hp[s] -= d
            if s_hp[s] <= 0:
                s_hp[s] = 0
                s_alive[s] = False
                structure_died = True
                if s in live_value:
                    live_value.discard(s)
                    destroyed_order.append(s)
                if s_class[s] == CLASS_TURRET:
                    d_alive[s - P.def_sids[0]] = False
                elif s_class[s] == CLASS_WALL:
                    wallmask &= ~(1 << (s - wall_base))
        if structure_died:
            version += 1
            mt_cache.clear()
            cen = centroid()
        died_units = False
        for i, d in dmg_u.items():
            u_hp[i] -= d
            if u_hp[i] <= 0:
                u_hp[i] = 0
                u_alive[i] = False
                died_units = True
        for j, h in heal_u.items():   # a unit killed this tick receives nothing; no overheal
            if u_alive[j]:
                nh = min(u_hp[j] + h, u_max[j])
                if nh != u_hp[j]:
                    u_hp[j] = nh
                    changed = True
        if died_units:
            alive_units = [i for i in alive_units if u_alive[i]]
            n_attackers = sum(1 for i in alive_units if attackers[i])
        # ---------------- (f): movement of survivors that did not attack (snapshot positions)
        status = {} if record else None
        for i in alive_units:
            if i in acted and u_type[i] != MENDER:
                u_idle[i] = 0
                if record:
                    status[i] = 2
        movers = [i for i in moving if u_alive[i]]
        for i in movers:   # speed accumulators advance for ALL movers before any decision: the snapshot is order-free
            u_acc[i] += units_cfg[u_type[i]].speed_x10
            pend[i] = u_acc[i] // 10
            u_acc[i] -= pend[i] * 10
        # Mender target point: centroid of living non-Mender allies at the snapshot
        am = asr = asc = 0
        if has_mender:
            for j in alive_units:
                if u_type[j] != MENDER:
                    am += 1; asr += ROW[u_cell[j]]; asc += COL[u_cell[j]]
        msrc = None
        new_cell: dict = {}
        for i in movers:
            ut = u_type[i]
            u = units_cfg[ut]
            steps = pend[i]
            moved = False
            p = u_cell[i]
            if ut == MENDER:
                wants = False
                if am:
                    if msrc is None:
                        msrc = mender_sources(am, asr, asc)
                    for si in range(steps):
                        rem = steps - si
                        if 100 * mender_T(p, am, asr, asc) <= 225 * am * am:
                            break
                        f = mender_field(p, msrc)
                        if f is None:   # walled off from its allies: holds (Menders cannot breach, it has no target)
                            break
                        wants = True
                        dp = f[p]
                        best, tiedn = None, []
                        for n, req in MOVES[p]:
                            if f.get(n, -9) != dp - 1:
                                continue
                            key = (mender_T(n, am, asr, asc),
                                   sum(1 for d in P.cover_ids[n] if d_alive[d]),
                                   cen[0] * (ROW[n] ** 2 + COL[n] ** 2) - 2 * (ROW[n] * cen[1] + COL[n] * cen[2])
                                   + cen[1] ** 2 + cen[2] ** 2)
                            if best is None or key < best:
                                best, tiedn = key, [n]
                            elif key == best:
                                tiedn.append(n)
                        if not tiedn:
                            break
                        if len(tiedn) > 1:
                            tiedn = [canon_pick(tiedn, lambda n, i=i, rem=rem: ({}, {i: (9, rem)}, {i: n}))]
                        p = tiedn[0]
                        moved = True
                if moved:
                    new_cell[i] = p
                    changed = True
                    u_idle[i] = 0
                    if record:
                        status[i] = 1
                else:
                    # idle only counts when the Mender wanted to move and could not; holding is not stalling
                    if wants and i not in acted:
                        u_idle[i] += 1
                        if u_idle[i] > max_idle:
                            max_idle = u_idle[i]
                    else:
                        u_idle[i] = 0
                    if record:
                        status[i] = 2 if i in acted else 0
                continue
            for si in range(steps):
                rem = steps - si
                mk = ut * NCELL + p
                t = mt_cache.get(mk)
                if t is None:
                    t, cacheable = move_target(i, ut, p, rem)
                    if cacheable:
                        mt_cache[mk] = t
                if t < 0:
                    break
                f = P.field(t, u.range_x10, wallmask)
                tgt = t
                if f[p] == 255:  # blocked: pick the wall on the cheapest shortest path
                    rid, reg = P.region(p, wallmask)
                    bkey = (t, u.range_x10, wallmask, rid)
                    tiedw = P._blocked.get(bkey)
                    if tiedw is None:
                        ghost = P.field(t, u.range_x10, 0)
                        best, tiedw = None, []
                        for wi in range(P.n_walls):
                            ws = wall_base + wi
                            if not s_alive[ws]:
                                continue
                            if not any(q in reg for q in cells_in_range(s_cell[ws], u.range_x10)):
                                continue
                            key = (ghost[s_cell[ws]], D2[s_cell[ws]][s_cell[t]])
                            if best is None or key < best:
                                best, tiedw = key, [ws]
                            elif key == best:
                                tiedw.append(ws)
                        tiedw = tuple(tiedw)
                        P._blocked[bkey] = tiedw
                    if not tiedw:
                        break
                    if len(tiedw) > 1:
                        w = canon_pick(list(tiedw), lambda w, i=i, rem=rem: ({w: (8,)}, {i: (7, rem)}, {i: p}))
                    else:
                        w = tiedw[0]
                    tgt = w
                    f = P.field(w, u.range_x10, wallmask)
                    if f[p] == 255:
                        break
                dp = f[p]
                if dp == 0:
                    break
                tcell = s_cell[tgt]
                best, tiedn = None, []
                for n, req in MOVES[p]:
                    if f[n] != dp - 1:
                        continue
                    key = (D2[n][tcell],
                           sum(1 for d in P.cover_ids[n] if d_alive[d]),
                           cen[0] * (ROW[n] ** 2 + COL[n] ** 2) - 2 * (ROW[n] * cen[1] + COL[n] * cen[2])
                           + cen[1] ** 2 + cen[2] ** 2)
                    if best is None or key < best:
                        best, tiedn = key, [n]
                    elif key == best:
                        tiedn.append(n)
                if not tiedn:
                    break
                if len(tiedn) > 1:   # rule 7 key 4
                    tiedn = [canon_pick(tiedn, lambda n, i=i, rem=rem, tgt=tgt: ({tgt: (10,)}, {i: (9, rem)}, {i: n}))]
                p = tiedn[0]
                moved = True
                if f[p] == 0:
                    break
            if moved:
                new_cell[i] = p
                changed = True
                u_idle[i] = 0
                if record:
                    status[i] = 1
            else:
                u_idle[i] += 1
                if u_idle[i] > max_idle:
                    max_idle = u_idle[i]
                if record:
                    status[i] = 0
        for i, c in new_cell.items():
            u_cell[i] = c
        pend.clear()
        if record:
            for i in alive_units:
                if i in acted and u_type[i] == MENDER and i not in status:
                    status[i] = 2

        if record:
            trace.append([(u_cell[i], u_hp[i], status.get(i, 0) if u_alive[i] else -1) for i in range(nu)])
        tick += 1
        # ---------------- (8) early exit
        if not alive_units:
            reason = "all_dead"
            break
        if not live_value:
            reason = "cleared"
            break
        if _no_attackers_exit and not n_attackers:
            reason = "no_attackers"   # only Menders remain: no structure can lose hp any more; score and structures are final
            break
        if changed:
            idle_ticks = 0
        else:
            idle_ticks += 1
            if idle_ticks >= cfg.stalemate_ticks:
                reason = "stalemate"
                break

    core_dead = not s_alive[P.core_sid]
    d_num = sum(s_value[s] for s in range(P.n_buildings) if not s_alive[s])
    den = P.d_den
    # score = 0.6 D + 0.4 C, reported from exact integers
    score = ((6 * d_num * 1) / (10 * den) if den else 0.0) + (0.4 if core_dead else 0.0)
    return BattleResult(
        ticks=tick, reason=reason, core_destroyed=core_dead, d_num=d_num, d_den=den, score=score,
        destroyed=tuple(destroyed_order), units_alive=len(alive_units), fallback_calls=fallback_calls,
        walls_destroyed=sum(1 for w in range(P.n_walls) if not s_alive[wall_base + w]),
        defenders_destroyed=sum(1 for d in d_alive if not d), max_idle_run=max_idle,
        struct_hp=tuple(s_hp), unit_hp=tuple(u_hp), trace=trace, canon_equal_picks=equal_picks,
        attack_ticks=tuple(c_att), wall_attack_ticks=tuple(c_wall), alive_ticks=tuple(c_alive),
        walls_hit=tuple(len(x) for x in c_walls_hit), wall_attack_any_ticks=wall_any)
