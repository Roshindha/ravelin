"""D4 augmentation for BreachEnv (v2, B9; docs/SPEC_V2.md 4.1, 4.2, 7, 9).

A symmetry is an integer ``s`` in 0..7: ``s = 4 * flip + k`` = mirror the columns first if ``flip``, then rotate ``k`` quarter turns
clockwise (the convention of ``breachlens.layouts._tr``). Cells are (row, col), the same indexing as ``grid[channel, row, col]`` and
``breachlens.actions.BORDER``. The tables here are written out independently of ``breachlens.env.d4_*`` and ``breachlens.sim``; the tests
cross-check them.

Transformed consistently: the 9-channel grid (spatially; channel 8 "repair reach" is a count of range-2 discs, hence covariant like the rest),
the 267-way action space (explicit permutation: buy / END_BUY / WAIT fixed, a deploy keeps its unit type and moves its border-tile index),
the action mask (a re-indexing through that table) and the 11-d state vector (invariant).

``D4Wrapper`` shows the agent a transformed view (random symmetry per reset, or a pinned one) and maps chosen actions back to original
coordinates before stepping the wrapped env. Evaluation of D4 models uses the canonical (identity) view; TTA lives in ``rl/tta.py``.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from breachlens.actions import BORDER
from breachlens.constants import DEPLOY_BASE, GRID, N_ACTIONS, N_BORDER, N_UNIT_TYPES

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
    return sym if flip else (-k) % 4   # every reflection-type element of D4 is an involution


def compose(a: int, b: int) -> int:
    """The symmetry 'apply b, then a' (found by its action on cells; the group is closed, so it exists)."""
    probe = [(0, 1), (1, 3), (5, 2)]            # three non-collinear-orbit cells pin down a D4 element
    want = [transform_cell(a, *transform_cell(b, r, c)) for r, c in probe]
    for s in range(N_SYMMETRIES):
        if [transform_cell(s, r, c) for r, c in probe] == want:
            return s
    raise AssertionError("D4 is not closed")    # unreachable


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
    """Apply ``sym`` to the spatial axes (-2, -1) of a (..., 14, 14) array (all 9 channels incl. repair reach); returns a new array."""
    flat = grid.reshape(*grid.shape[:-2], GRID * GRID)
    out = np.empty_like(flat)
    out[..., CELL_DEST[sym]] = flat
    return out.reshape(grid.shape)


def transform_obs(obs: dict, sym: int) -> dict:
    """Grid is transformed spatially; the state vector is invariant and passed through (copied)."""
    return {"grid": transform_grid(obs["grid"], sym), "state": obs["state"].copy()}


# -- action permutation tables --------------------------------------------------------------------
_BORDER_INDEX = {cell: i for i, cell in enumerate(BORDER)}


def _build_tile_perm() -> np.ndarray:
    perm = np.zeros((N_SYMMETRIES, N_BORDER), dtype=np.int64)
    for s in range(N_SYMMETRIES):
        for t, (r, c) in enumerate(BORDER):
            perm[s, t] = _BORDER_INDEX[transform_cell(s, r, c)]  # KeyError == the border is not preserved
    return perm


TILE_PERM = _build_tile_perm()


def _build_action_perm() -> np.ndarray:
    """ACTION_PERM[s][a] = the action, in the transformed view, that corresponds to original ``a`` (267 entries)."""
    perm = np.tile(np.arange(N_ACTIONS, dtype=np.int64), (N_SYMMETRIES, 1))  # buy / END_BUY / WAIT fixed
    for s in range(N_SYMMETRIES):
        for k in range(N_UNIT_TYPES):
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
    out[..., ACTION_PERM[sym]] = mask  # out[P[a]] = mask[a]
    return out


def untransform_probs(probs: np.ndarray, sym: int) -> np.ndarray:
    """Probabilities / Q-values over the transformed view's actions -> over original actions: result[a] = probs[ACTION_PERM[s][a]]."""
    return probs[..., ACTION_PERM[sym]]


# -- armies (for simulator-level checks) ----------------------------------------------------------
def transform_army(army: list, sym: int) -> list:
    """[(unit type, border tile)] -> the same army in the transformed world."""
    return [(int(t), int(TILE_PERM[sym, tile])) for t, tile in army]


# -- wrapper --------------------------------------------------------------------------------------
class D4Wrapper(gym.Wrapper):
    """Presents a randomly D4-transformed view of a BreachEnv to the agent.

    ``symmetry`` pins a fixed symmetry (tests / evaluation: ``symmetry=0`` is the canonical view); ``None`` samples uniformly per reset.
    The wrapped env always runs in original coordinates; only what the agent sees and does is mapped. The symmetry stream is seeded from
    ``reset(seed=...)`` (or the constructor ``seed``) and is independent of the env's own RNG.
    """

    def __init__(self, env: gym.Env, symmetry: int | None = None, seed: int | None = None) -> None:
        super().__init__(env)
        if symmetry is not None:
            decompose(int(symmetry))
        self._fixed = symmetry
        self._rng = np.random.default_rng(seed)
        self.sym = IDENTITY if symmetry is None else int(symmetry)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng([seed, 0xD4])
        self.sym = int(self._rng.integers(N_SYMMETRIES)) if self._fixed is None else int(self._fixed)
        obs, info = self.env.reset(seed=seed, options=options)
        return transform_obs(obs, self.sym), dict(info, symmetry=self.sym)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(untransform_action(action, self.sym))
        return transform_obs(obs, self.sym), reward, terminated, truncated, dict(info, symmetry=self.sym)

    def action_masks(self) -> np.ndarray:
        return transform_mask(self.env.unwrapped.action_masks(), self.sym)
