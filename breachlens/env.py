"""BreachEnv (B7): the Gymnasium environment of BreachLens v2 (docs/SPEC_V2.md 4, 6, 7.4).

* Action space Discrete(267): 0-4 buy, 5..264 deploy (5 + type*52 + border tile), 265 END_BUY, 266 WAIT. `action_masks()` is the
  source of truth; an illegal action RAISES `IllegalActionError` (a ValueError). `action_space.sample()` draws from the CURRENT mask (so
  gymnasium's `check_env` only ever steps legal actions).
* Observation: Dict(grid float32 (9, 14, 14) in [0, 1], state float32 (11,) in [0, 1]); channels as in spec 4.2 (docstring of `_Static`).
* Episode: BUY (buy / END_BUY) -> DEPLOY (deploy / WAIT) -> BATTLE. The battle is resolved inside the step that ends deployment (last owned
  unit deployed, or WAIT) and that step returns terminated=True. `prebought="random"` starts the episode in DEPLOY with a fresh random
  legal army (spec 6, phase A); a count list is also accepted.
* Reward: terminal only (default): the canonical score (breachlens.score) at the last step, 0 before. `shaping=True` (default OFF) switches
  to the potential-based option of spec 7.4: r_t = Phi(s_{t+1}) - Phi(s_t), Phi = 0 in BUY and while nothing is deployed, else
  0.5 * score(preview battle of the units deployed so far, tick cap 120), Phi(terminal) = 0, so the episode return still equals the true
  score (up to float summation). `info["score"]` is always the true score at the end; evaluation must use it, never the shaped reward.
* Max steps per episode: 2 * unit_cap + 1 = 41 (<= 20 buys + END_BUY + <= 20 deploys; the last deploy triggers the battle).
* Layout access ONLY through breachlens.pool loaders with role guards: role in train / validation / tuning may not load FINAL or transfer
  layouts (PermissionError); role "eval" needs unlock=True and writes an 'eval' ledger entry per FINAL load. `layouts=` (a list of layout
  dicts) is the synthetic path used by tests and symmetry checks; it never touches the pool.
* Movement-field caches live in `Prepared` (bounded LRU, see breachlens.sim); the env keeps a bounded LRU of `Prepared` objects
  (`layout_cache`, default 64).
"""
from __future__ import annotations

from collections import OrderedDict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import layouts as Y
from . import pool as P
from .actions import BORDER
from .armies import random_types
from .config import CFG
from .constants import (BUDGET, BUY_BASE, DEPLOY_BASE, END_BUY, GRID, MAX_BATTLE_TICKS, N_ACTIONS, N_BORDER, N_GRID_CHANNELS, N_STATE,
                        N_UNIT_TYPES, PHASE_BATTLE, PHASE_BUY, PHASE_DEPLOY, WAIT)
from .score import result_score
from .sim import FIELD_CACHE_SIZE, SMALL_CACHE_SIZE, Prepared, cells_in_range, simulate

MAX_STEPS = 2 * CFG.unit_cap + 1          # 41
PREVIEW_TICKS = 120                       # spec 7.4
UNIT_COST = tuple(u.cost for u in CFG.units)
BORDER_CELL = tuple(r * GRID + c for r, c in BORDER)
ROLES = ("train", "validation", "tuning", "eval")
SOURCES = {"DEV-train", "DEV-validation", "TUNE-train", "TUNE-val", "LAB-DESIGN", "FINAL-familiar", "FINAL-transfer"}


class IllegalActionError(ValueError):
    """Raised by `BreachEnv.step` for an action whose mask bit is False."""


class _MaskedDiscrete(spaces.Discrete):
    """Discrete whose `sample()` draws uniformly from the owning env's current legal mask unless a mask is passed."""
    mask_fn = None

    def sample(self, mask=None, probability=None):
        if mask is None and probability is None and self.mask_fn is not None:
            mask = self.mask_fn().astype(np.int8)
        return super().sample(mask=mask, probability=probability)


# ------------------------------------------------------------------ D4 helpers (conventions of breachlens.layouts.transform)
def d4_cell_perm(k: int) -> np.ndarray:
    """perm[cell] = flat index of the image of `cell` under symmetry k (the one `layouts.transform` applies)."""
    return np.array([r2 * GRID + c2 for r2, c2 in (Y._tr(k, (c // GRID, c % GRID)) for c in range(GRID * GRID))])


def d4_border_perm(k: int) -> np.ndarray:
    """perm[tile] = border index of the image of border tile `tile` under symmetry k (a permutation of the 52 tiles)."""
    idx = {rc: i for i, rc in enumerate(BORDER)}
    return np.array([idx[tuple(Y._tr(k, rc))] for rc in BORDER])


def d4_action_perm(k: int) -> np.ndarray:
    """perm[a] = the action that does, in the transformed world, what action `a` does in the original (buy / END_BUY / WAIT fixed)."""
    bp = d4_border_perm(k)
    perm = np.arange(N_ACTIONS)
    for t in range(N_UNIT_TYPES):
        for i in range(N_BORDER):
            perm[DEPLOY_BASE + t * N_BORDER + i] = DEPLOY_BASE + t * N_BORDER + bp[i]
    return perm


def d4_transform_grid(grid: np.ndarray, k: int) -> np.ndarray:
    """Grid of the transformed world: out[ch, T(r, c)] = grid[ch, r, c]."""
    out = np.empty_like(grid)
    out.reshape(grid.shape[0], -1)[:, d4_cell_perm(k)] = grid.reshape(grid.shape[0], -1)
    return out


def d4_transform_mask(mask: np.ndarray, k: int) -> np.ndarray:
    out = np.empty_like(mask)
    out[d4_action_perm(k)] = mask
    return out


# ------------------------------------------------------------------ per-layout static data
class _Static:
    """Everything about one layout that does not change during an episode: Prepared (sim caches), border legality and the initial grid.

    Grid channels (spec 4.2): 0 deployed-unit occupancy (count / 4) | 1 building hp fraction | 2 turret (defender) hp fraction | 3 wall hp
    fraction | 4 building value / 3 (core = 1) | 5 core mask | 6 defender threat (sum over living defenders covering the cell of
    damage per tick / period, / 20, clipped; range min-max respected) | 7 free legal-deploy border tiles (DEPLOY phase, units left) |
    8 repair reach (deployed Menders whose range-2 disc covers the cell / 3, clipped; 0 in BUY phase).
    Channels 1-6 describe the CURRENT board: after the battle they show the final structure state (destroyed = 0)."""

    def __init__(self, layout: dict, rid, field_cache, small_cache):
        self.id = rid
        self.layout = layout
        self.P = P_ = Prepared(layout, field_cache=field_cache, small_cache=small_cache)
        wall_cells = {P_.s_cell[P_.wall_base + k] for k in range(P_.n_walls)}
        self.border_free = np.array([BORDER_CELL[i] not in wall_cells for i in range(N_BORDER)])
        S = CFG.S
        self.def_cells = [np.array(cells_cov, dtype=np.int64) for cells_cov in
                          ([c for c in range(GRID * GRID) if any(d == did for d, _ in P_.cover[c])] for did in range(P_.n_def))]
        self.def_w = [ds.damage / S / ds.period / 20.0 for ds in P_.def_stats]
        mr = CFG.units[4].heal_range_x10
        self.mender_disc = [np.array(cells_in_range(BORDER_CELL[i], mr), dtype=np.int64) for i in range(N_BORDER)]
        self.board0 = self.board(list(P_.s_hp0))

    def board(self, hp) -> np.ndarray:
        """Channels 1-6 (flat (6, 196) float32) for a structure-hp vector."""
        P_ = self.P
        b = np.zeros((6, GRID * GRID), dtype=np.float32)
        alive = [h > 0 for h in hp]
        for s in range(P_.n_buildings):
            if alive[s]:
                c = P_.s_cell[s]
                b[0, c] = hp[s] / P_.s_hp0[s]
                b[3, c] = P_.s_value[s] / 3.0
        cs = P_.core_sid
        if alive[cs]:
            c = P_.s_cell[cs]
            b[0, c] = hp[cs] / P_.s_hp0[cs]
            b[3, c] = 1.0
            b[4, c] = 1.0
        for did, s in enumerate(P_.def_sids):
            if alive[s]:
                b[1, P_.s_cell[s]] = hp[s] / P_.s_hp0[s]
                b[5, self.def_cells[did]] += self.def_w[did]
        for k in range(P_.n_walls):
            s = P_.wall_base + k
            if alive[s]:
                b[2, P_.s_cell[s]] = hp[s] / P_.s_hp0[s]
        np.clip(b[5], 0.0, 1.0, out=b[5])
        return b


class BreachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, records=None, *, source: str | None = None, layouts=None, role: str = "train", unlock: bool = False,
                 prebought=None, shaping: bool = False, layout_cache: int = 64, field_cache: int | None = FIELD_CACHE_SIZE,
                 small_cache: int | None = SMALL_CACHE_SIZE, entry_point: str = "BreachEnv") -> None:
        super().__init__()
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        if sum(x is not None for x in (records, source, layouts)) != 1:
            raise ValueError("give exactly one of records=, source=, layouts=")
        self.role, self.unlock, self.entry_point = role, unlock, entry_point
        self.shaping = bool(shaping)
        self._caches = (field_cache, small_cache)
        self._layout_cache_size = layout_cache
        self._static: OrderedDict = OrderedDict()
        self._synthetic = None
        self.records = None
        if layouts is not None:
            self._synthetic = list(layouts)
            if not self._synthetic:
                raise ValueError("empty layouts")
        else:
            self.records = self._resolve_records(records, source)
            if not self.records:
                raise ValueError("no layout records")
        self.prebought = None
        self.set_prebought(prebought)
        self.action_space = _MaskedDiscrete(N_ACTIONS)
        self.action_space.mask_fn = self.action_masks
        self.observation_space = spaces.Dict({"grid": spaces.Box(0.0, 1.0, (N_GRID_CHANNELS, GRID, GRID), np.float32),
                                              "state": spaces.Box(0.0, 1.0, (N_STATE,), np.float32)})
        self._st = None
        self._st_idx = None
        self._init_episode_state()

    # ---------------------------------------------------------------- pickling (B10: bit-identical resume of a training run)
    def __getstate__(self) -> dict:
        """Episode state + RNG only. The layout caches (up to ~4 MB per layout, results identical with caches on / off) are dropped and rebuilt
        on load through the same role-guarded loaders; the current layout is re-fetched by its index."""
        d = self.__dict__.copy()
        d["_static"] = OrderedDict()
        d["_st"] = None
        return d

    def __setstate__(self, d: dict) -> None:
        self.__dict__.update(d)
        if self._st_idx is not None:
            self._st = self._get_static(self._st_idx)

    # ---------------------------------------------------------------- layout access (pool loaders, role guards)
    def _resolve_records(self, records, source):
        if records is not None:
            recs = [dict(r) for r in records]
            P.assert_dev_only([r["id"] for r in recs], self.role)          # train / validation / tuning: FINAL ids raise here
            return recs
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}")
        if source.startswith("FINAL"):
            if self.role != "eval":
                raise PermissionError(f"role {self.role!r} may not load {source}")
            return P.load_pool_records(source, "eval", purpose="eval", unlock=self.unlock, entry_point=self.entry_point)
        if self.role == "eval":
            raise PermissionError("role 'eval' loads FINAL splits only")
        if source.startswith("DEV"):
            return P.load_pool_records(source, self.role, entry_point=self.entry_point)
        return P.design_records(source, self.role)

    def _get_static(self, key) -> _Static:
        st = self._static.get(key)
        if st is not None:
            self._static.move_to_end(key)
            return st
        if self._synthetic is not None:
            layout, rid = self._synthetic[key], f"synthetic-{key}"
        else:
            rec = self.records[key]
            layout = P.load_layout(rec, self.role, purpose="eval" if self.role == "eval" else None, unlock=self.unlock,
                                   entry_point=self.entry_point)
            rid = rec["id"]
        st = _Static(layout, rid, *self._caches)
        if self._layout_cache_size:
            self._static[key] = st
            while len(self._static) > self._layout_cache_size:
                self._static.popitem(last=False)
        return st

    @property
    def n_layouts(self) -> int:
        return len(self._synthetic) if self._synthetic is not None else len(self.records)

    # ---------------------------------------------------------------- state
    def _init_episode_state(self) -> None:
        self.phase = PHASE_BUY
        self.gold = BUDGET
        self.owned = [0] * N_UNIT_TYPES            # bought, not yet deployed
        self.bought = 0
        self.army: list = []                       # [(type, border tile)] in deployment order
        self.occupied = np.zeros(N_BORDER, dtype=bool)
        self.repair = np.zeros(GRID * GRID, dtype=np.float32)
        self.tick = 0
        self.result = None
        self.steps = 0
        self.done = False
        self._phi = 0.0

    def set_prebought(self, prebought) -> None:
        """None = normal BUY->DEPLOY game; "random" = fresh random legal army per episode (starts in DEPLOY); or a list of 5 counts."""
        if isinstance(prebought, str):
            if prebought != "random":
                raise ValueError(prebought)
        elif prebought is not None:
            counts = [int(c) for c in prebought]
            if len(counts) != N_UNIT_TYPES or min(counts) < 0 or sum(counts) < 1 or sum(counts) > CFG.unit_cap or \
                    sum(c * k for c, k in zip(counts, UNIT_COST)) > BUDGET:
                raise ValueError(f"illegal prebought army {counts}")
            prebought = counts
        self.prebought = prebought

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """options: `layout_index` (int, into records / layouts) pins the layout, `prebought` overrides the env setting for this episode."""
        super().reset(seed=seed)
        self._init_episode_state()
        options = options or {}
        idx = options.get("layout_index")
        if idx is None:
            idx = int(self.np_random.integers(self.n_layouts))
        self._st = self._get_static(int(idx))
        self._st_idx = int(idx)
        pre = options.get("prebought", self.prebought)
        if isinstance(pre, str):
            pre = [0] * N_UNIT_TYPES
            for t in random_types(self.np_random):
                pre[t] += 1
        if pre:
            self.owned = list(pre)
            self.bought = sum(pre)
            self.gold = BUDGET - sum(c * k for c, k in zip(pre, UNIT_COST))
            self.phase = PHASE_DEPLOY
        return self._obs(), {"layout_id": self._st.id, "layout_index": int(idx)}

    # ---------------------------------------------------------------- masks
    def action_masks(self) -> np.ndarray:
        m = np.zeros(N_ACTIONS, dtype=bool)
        if self.done:
            m[WAIT] = True                         # terminal state: a non-empty mask for masked policies; stepping raises anyway
            return m
        if self.phase == PHASE_BUY:
            if self.bought < CFG.unit_cap:
                for t in range(N_UNIT_TYPES):
                    m[BUY_BASE + t] = UNIT_COST[t] <= self.gold
            m[END_BUY] = self.bought >= 1
        elif self.phase == PHASE_DEPLOY:
            free = self._st.border_free & ~self.occupied
            for t in range(N_UNIT_TYPES):
                if self.owned[t] > 0:
                    m[DEPLOY_BASE + t * N_BORDER: DEPLOY_BASE + (t + 1) * N_BORDER] = free
            m[WAIT] = True                         # ends deployment (undeployed units are forfeited), as in the prototype
        return m

    # ---------------------------------------------------------------- dynamics
    def step(self, action):
        if self.done:
            raise RuntimeError("step() on a terminated episode; call reset()")
        a = int(action)
        if not 0 <= a < N_ACTIONS:
            raise IllegalActionError(f"action {a} out of range")
        if not self.action_masks()[a]:
            raise IllegalActionError(f"action {a} is illegal in phase {self.phase}")
        self.steps += 1
        battle = False
        if a < DEPLOY_BASE:
            self.gold -= UNIT_COST[a]
            self.owned[a] += 1
            self.bought += 1
        elif a == END_BUY:
            self.phase = PHASE_DEPLOY
        elif a == WAIT:
            battle = True
        else:
            t, tile = divmod(a - DEPLOY_BASE, N_BORDER)
            self.owned[t] -= 1
            self.occupied[tile] = True
            self.army.append((t, tile))
            if t == 4:
                self.repair[self._st.mender_disc[tile]] += 1.0
            battle = sum(self.owned) == 0
        info = {"layout_id": self._st.id}
        reward = 0.0
        if battle:
            self.result = simulate(self._st.P, self.army)
            self.phase, self.done = PHASE_BATTLE, True
            self.tick = self.result.ticks
            score = result_score(self.result)
            reward = score - self._phi if self.shaping else score
            info.update(score=score, d_num=self.result.d_num, d_den=self.result.d_den, core_destroyed=bool(self.result.core_destroyed),
                        ticks=self.result.ticks, reason=self.result.reason, n_units=len(self.army), steps=self.steps)
        elif self.shaping:
            phi = self._potential()
            reward, self._phi = phi - self._phi, phi
        return self._obs(), float(reward), self.done, False, info

    def _potential(self) -> float:
        """Phi(s): 0 in BUY and while nothing is deployed; else 0.5 * score(preview battle of the units deployed so far, 120 ticks)."""
        if self.phase != PHASE_DEPLOY or not self.army:
            return 0.0
        return 0.5 * result_score(simulate(self._st.P, self.army, max_ticks=PREVIEW_TICKS))

    # ---------------------------------------------------------------- observation
    def _obs(self) -> dict:
        st = self._st
        grid = np.zeros((N_GRID_CHANNELS, GRID * GRID), dtype=np.float32)
        board = st.board0 if self.result is None else st.board(list(self.result.struct_hp))
        grid[1:7] = board
        if self.army:
            occ = np.bincount([BORDER_CELL[tl] for _, tl in self.army], minlength=GRID * GRID)
            grid[0] = np.minimum(occ / 4.0, 1.0)
        if self.phase == PHASE_DEPLOY and sum(self.owned) > 0:
            free = st.border_free & ~self.occupied
            grid[7, [BORDER_CELL[i] for i in np.flatnonzero(free)]] = 1.0
        if self.phase != PHASE_BUY:
            grid[8] = np.minimum(self.repair / 3.0, 1.0)
        state = np.zeros(N_STATE, dtype=np.float32)
        state[0] = self.gold / BUDGET
        state[1:6] = np.minimum(np.array(self.owned, dtype=np.float32) / CFG.unit_cap, 1.0)
        state[6 + self.phase] = 1.0
        state[9] = self.bought / CFG.unit_cap
        state[10] = self.tick / MAX_BATTLE_TICKS
        return {"grid": grid.reshape(N_GRID_CHANNELS, GRID, GRID), "state": state}
