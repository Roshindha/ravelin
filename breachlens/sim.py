"""BreachLens battle simulator, B2 scope (docs/SPEC_V2.md 3.1, 3.1b, 3.3, 3.4 rule 7 keys 1-3).

In scope: Scout / Sniper / Bulwark / Blaster, Gun / Lancer / Flak defenders, walls, buildings, core,
fixed-point hp (scale S), tick order, nearest-in-preference targeting, BFS movement with the rule-7
key chain (keys 1-3), early exit. NOT in scope (B3): Mender (rejected), rule-7 key 4 canonical-form
fallback. Where keys are exhausted a deterministic STOPGAP (smallest raw (row, col)) is used and
`fallback_calls` is counted; B3 replaces it. All outcome-deciding arithmetic is integer.

Layout dict (cells are (row, col)):
  {"buildings": [{"cell": (r, c), "value": 1..3, "hp": human_hp (optional)}],
   "core": {"cell": (r, c), "hp": human_hp (optional)},
   "walls": [(r, c), ...],
   "defenders": [{"cell": (r, c), "type": "gun" | "lancer" | "flak"}]}
Army: list of (unit_type_idx, border_tile_idx). A (row, col) tuple in place of the tile index is a test hook
for interior start cells.
"""
from __future__ import annotations

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
    reason: str                 # cleared | all_dead | stalemate | horizon
    core_destroyed: bool
    d_num: int                  # value-weighted destroyed (ints)
    d_den: int
    score: float                # reporting only; derived from the ints above
    destroyed: tuple            # destroyed building sids (value buildings and core)
    units_alive: int
    fallback_calls: int
    walls_destroyed: int = 0
    defenders_destroyed: int = 0
    max_idle_run: int = 0       # longest run of ticks any unit made no progress (move or attack)
    struct_hp: tuple = ()       # final fixed-point hp per structure sid
    unit_hp: tuple = ()         # final fixed-point hp per unit (army order)
    trace: list = field(default_factory=list)  # record mode: per tick [(cell, hp, status)] per unit

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


def simulate(layout, army, record: bool = False, max_ticks: int = MAX_BATTLE_TICKS) -> BattleResult:
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
        if ut == MENDER:
            raise NotImplementedError("Mender is B3 scope")
        rc = tile if isinstance(tile, tuple) else BORDER[tile]  # tuple = test hook: interior start cell
        c = cell_of(rc)
        if P.wall_bit_of_cell[c]:
            raise ValueError(f"unit deployed on a wall cell {rc}")
        u_type.append(ut); u_cell.append(c)
        u_hp.append(units_cfg[ut].hp); u_max.append(units_cfg[ut].hp); u_acc.append(0)
    nu = len(u_type)
    u_alive = [True] * nu
    alive_units = list(range(nu))
    u_idle = [0] * nu
    max_idle = 0

    d_alive = [True] * P.n_def
    fallback_calls = 0
    version = 0          # bumped whenever any structure dies (invalidates movement-target cache)
    mt_cache: dict = {}
    cen = None           # centroid sums (m, sx, sy) of surviving value buildings + core
    live_value = set(P.value_sids)
    trace = []
    destroyed_order = []
    idle_ticks = 0
    reason = "horizon"
    tick = 0

    def centroid():
        m = sx = sy = 0
        for s in live_value:
            m += 1; sx += ROW[s_cell[s]]; sy += COL[s_cell[s]]
        return m, sx, sy

    cen = centroid()

    def move_target(ut, cell):
        """Nearest alive structure of the most-preferred non-empty class (movement target)."""
        u = units_cfg[ut]
        nonlocal fallback_calls
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
                    fallback_calls += 1
                    tied.sort(key=lambda s: s_cell[s])  # STOPGAP (B3: canonical fallback)
                return tied[0]
        return -1

    while tick < max_ticks:
        # ---------------- (a)/(b)/(c): intents and damage from the start-of-tick snapshot
        dmg_s: dict = {}
        dmg_u: dict = {}
        acted = set()
        moving = []
        for i in alive_units:
            ut = u_type[i]
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
                if len(tied) > 1:
                    fallback_calls += 1
                    tied.sort(key=lambda s: s_cell[s])  # STOPGAP (B3)
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
                if len(tiedc) > 1:
                    fallback_calls += 1
                    tiedc.sort(key=lambda c: (ROW[c], COL[c]))  # STOPGAP (B3)
                c0 = tiedc[0]
                ul = occ[c0]
                centre = min(ul, key=lambda i: (_hpfrac(u_hp[i], u_max[i]), -units_cfg[u_type[i]].cost))
                for i in ul:
                    dm = ds.damage if i == centre else ds.damage_neighbour
                    dmg_u[i] = dmg_u.get(i, 0) + max(dm - units_cfg[u_type[i]].armor, cfg.min_hit)
                for n in disc_others(c0, ds.splash_r_x10):
                    for i in occ.get(n, ()):
                        dmg_u[i] = dmg_u.get(i, 0) + max(ds.damage_neighbour - units_cfg[u_type[i]].armor, cfg.min_hit)

        # ---------------- (d)/(e): apply damage, remove dead
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
        if died_units:
            alive_units = [i for i in alive_units if u_alive[i]]
        # ---------------- (f): movement of survivors that did not attack
        status = {} if record else None
        for i in alive_units:
            if i in acted:
                u_idle[i] = 0
                if record:
                    status[i] = 2
        for i in moving:
            if not u_alive[i]:
                continue
            ut = u_type[i]
            u = units_cfg[ut]
            u_acc[i] += u.speed_x10
            steps = u_acc[i] // 10
            u_acc[i] -= steps * 10
            moved = False
            p = u_cell[i]
            for _ in range(steps):
                mk = ut * NCELL + p
                t = mt_cache.get(mk)
                if t is None:
                    t = move_target(ut, p)
                    mt_cache[mk] = t
                if t < 0:
                    break
                f = P.field(t, u.range_x10, wallmask)
                tgt = t
                if f[p] == 255:  # blocked: pick the wall on the cheapest shortest path
                    rid, reg = P.region(p, wallmask)
                    bkey = (t, u.range_x10, wallmask, rid)
                    w = P._blocked.get(bkey)
                    if w is None:
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
                        if tiedw:
                            if len(tiedw) > 1:
                                fallback_calls += 1
                                tiedw.sort(key=lambda s: s_cell[s])  # STOPGAP (B3)
                            w = tiedw[0]
                        else:
                            w = -1
                        P._blocked[bkey] = w
                    if w < 0:
                        break
                    tgt = w
                    f = P.field(w, u.range_x10, wallmask)
                    if f[p] == 255:
                        break
                dp = f[p]
                if dp == 0:
                    break
                tcell = s_cell[tgt]
                best, tiedn = None, []
                wb = P.wall_bit_of_cell
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
                if len(tiedn) > 1:
                    fallback_calls += 1
                    tiedn.sort()  # STOPGAP (B3: canonical-form fallback, rule 7 key 4)
                p = tiedn[0]
                moved = True
                if f[p] == 0:
                    break
            if moved:
                u_cell[i] = p
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
        struct_hp=tuple(s_hp), unit_hp=tuple(u_hp), trace=trace)
