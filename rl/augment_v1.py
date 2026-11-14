"""PROTOTYPE copy of the pre-B9 rl/augment.py (246 actions, siege env), kept for rl/train_ppo.py until the v2 trainer (B10). v2: rl/augment.py.

D4 (dihedral group of order 8) augmentation for SiegeEnv.

A symmetry is an integer ``s`` in ``0..7``: ``s = 4 * flip + k`` means "mirror the columns first if
``flip``, then rotate ``k`` quarter turns clockwise". Cells are ``(row, col)`` == the simulator's
``(x, y)``, the same indexing as ``grid[channel, x, y]`` and ``siege.env.BORDER``.

Three things are transformed consistently:
  * the grid observation (spatially; the 9-d ``state`` vector is symmetry-invariant),
  * the 246-way action space, via an explicit permutation table (buy / END_BUY / WAIT are fixed;
    a deploy action keeps its troop type and moves its border-tile index), and
  * the action mask (a pure re-indexing of the original mask through that table).

``D4Wrapper`` picks a random symmetry on every reset, shows the agent the transformed view, and maps
each chosen action back to original simulator coordinates before stepping the wrapped env.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from siege.env import BORDER, DEPLOY_BASE, GRID, N_ACTIONS, N_BORDER, N_TROOPS

N_SYMMETRIES = 8
IDENTITY = 0
LAST = GRID - 1


def decompose(sym: int) -> tuple[bool, int]:
    if not 0 <= sym < N_SYMMETRIES:
        raise ValueError(f"symmetry out of range: {sym}")
    return bool(sym // 4), sym % 4


def transform_cell(sym: int, r: int, c: int) -> tuple[int, int]:
    """Where cell (r, c) lands under symmetry ``sym``."""
    flip, k = decompose(sym)
    if flip:
        c = LAST - c
    for _ in range(k):  # clockwise quarter turn: (r, c) -> (c, LAST - r)
        r, c = c, LAST - r
    return r, c


def inverse_symmetry(sym: int) -> int:
    flip, k = decompose(sym)
    if not flip:
        return (-k) % 4  # rotation only
    return sym  # every reflection-type element of D4 is an involution


# -- cell / grid tables ---------------------------------------------------------------------------
def _build_cell_dest() -> np.ndarray:
    dest = np.zeros((N_SYMMETRIES, GRID * GRID), dtype=np.int64)
    for s in range(N_SYMMETRIES):
        for r in range(GRID):
            for c in range(GRID):
                nr, nc = transform_cell(s, r, c)
                dest[s, r * GRID + c] = nr * GRID + nc
    return dest


CELL_DEST = _build_cell_dest()  # CELL_DEST[s][i] = flat index that flat cell i moves to


def transform_grid(grid: np.ndarray, sym: int) -> np.ndarray:
    """Apply ``sym`` to the spatial axes (-2, -1) of a (..., 16, 16) array; returns a new array."""
    flat = grid.reshape(*grid.shape[:-2], GRID * GRID)
    out = np.empty_like(flat)
    out[..., CELL_DEST[sym]] = flat
    return out.reshape(grid.shape)


def transform_obs(obs: dict, sym: int) -> dict:
    """Grid is transformed spatially; the state vector is invariant and passed through unchanged."""
    return {"grid": transform_grid(obs["grid"], sym), "state": obs["state"].copy()}


# -- action permutation tables --------------------------------------------------------------------
_BORDER_INDEX = {cell: i for i, cell in enumerate(BORDER)}


def _build_tile_perm() -> np.ndarray:
    perm = np.zeros((N_SYMMETRIES, N_BORDER), dtype=np.int64)
    for s in range(N_SYMMETRIES):
        for t, (r, c) in enumerate(BORDER):
            perm[s, t] = _BORDER_INDEX[transform_cell(s, r, c)]  # KeyError == border not preserved
    return perm


TILE_PERM = _build_tile_perm()


def _build_action_perm() -> np.ndarray:
    """ACTION_PERM[s][a] = the action, in the transformed view, that corresponds to original ``a``."""
    perm = np.tile(np.arange(N_ACTIONS, dtype=np.int64), (N_SYMMETRIES, 1))  # buy/END_BUY/WAIT fixed
    for s in range(N_SYMMETRIES):
        for k in range(N_TROOPS):
            for t in range(N_BORDER):
                perm[s, DEPLOY_BASE + k * N_BORDER + t] = DEPLOY_BASE + k * N_BORDER + TILE_PERM[s, t]
    return perm


ACTION_PERM = _build_action_perm()
ACTION_PERM_INV = np.argsort(ACTION_PERM, axis=1)  # ACTION_PERM_INV[s][a'] = original action


def transform_action(action: int, sym: int) -> int:
    """Original-coordinate action -> action as seen in the transformed view."""
    return int(ACTION_PERM[sym, int(action)])


def untransform_action(action: int, sym: int) -> int:
    """Action chosen in the transformed view -> original-coordinate action."""
    return int(ACTION_PERM_INV[sym, int(action)])


def transform_mask(mask: np.ndarray, sym: int) -> np.ndarray:
    out = np.empty_like(mask)
    out[ACTION_PERM[sym]] = mask  # out[P[a]] = mask[a]
    return out


# -- layouts / armies (for simulator-level equivariance checks) -----------------------------------
def transform_layout(layout: dict, sym: int) -> dict:
    """Return a copy of a layout dict with every building/turret/wall moved by ``sym``.

    List / dict insertion order is preserved so object labels (ids) stay attached to the same object.
    """

    def move(d: dict) -> dict:
        nd = dict(d)
        nd["x"], nd["y"] = transform_cell(sym, d["x"], d["y"])
        return nd

    out = dict(layout)
    out["buildings"] = [move(b) for b in layout["buildings"]]
    out["turrets"] = [move(t) for t in layout["turrets"]]
    out["walls"] = {transform_cell(sym, x, y): hp for (x, y), hp in layout["walls"].items()}
    return out


def transform_army(army: list[dict], sym: int) -> list[dict]:
    out = []
    for u in army:
        nu = dict(u)
        nu["x"], nu["y"] = transform_cell(sym, u["x"], u["y"])
        out.append(nu)
    return out


# -- wrapper --------------------------------------------------------------------------------------
class D4Wrapper(gym.Wrapper):
    """Presents a randomly D4-transformed view of a SiegeEnv to the agent.

    ``symmetry`` pins a fixed symmetry (useful for tests/eval); ``None`` samples uniformly per reset.
    The wrapped env always runs in original coordinates; only what the agent sees/does is mapped.
    """

    def __init__(self, env: gym.Env, symmetry: int | None = None, seed: int | None = None) -> None:
        super().__init__(env)
        self._fixed = symmetry
        self._rng = np.random.default_rng(seed)
        self.sym = IDENTITY if symmetry is None else int(symmetry)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng([seed, 0xD4])
        self.sym = int(self._rng.integers(N_SYMMETRIES)) if self._fixed is None else self._fixed
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info, symmetry=self.sym)
        return transform_obs(obs, self.sym), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(untransform_action(action, self.sym))
        return transform_obs(obs, self.sym), reward, terminated, truncated, dict(info, symmetry=self.sym)

    def action_masks(self) -> np.ndarray:
        return transform_mask(self.env.unwrapped.action_masks(), self.sym)
