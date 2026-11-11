"""Minimal SiegeEnv skeleton (STEP 1).

Only the interface is real here: spaces, phases, action layout, masks and
seeding. Combat is a placeholder (see `_run_battle_stub`) and will be replaced
by the deterministic simulator in a later step.

Action layout (Discrete(246)):
    0..3        buy troop type k
    4..243      deploy: 4 + troop_type * 60 + border_tile_index
    244         END_BUY
    245         WAIT
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from siege.troops import TROOP_COST

GRID = 16
N_TROOPS = 4
N_BORDER = 4 * GRID - 4  # 60
BUDGET = 200
N_CHANNELS = 8
STATE_DIM = 9

BUY_BASE = 0
DEPLOY_BASE = N_TROOPS  # 4
END_BUY = DEPLOY_BASE + N_TROOPS * N_BORDER  # 244
WAIT = END_BUY + 1  # 245
N_ACTIONS = WAIT + 1  # 246

PHASE_BUY, PHASE_DEPLOY, PHASE_BATTLE = 0, 1, 2

MAX_BATTLE_TICKS = 50


def border_tiles() -> list[tuple[int, int]]:
    """The 60 border cells (row, col), clockwise from the top-left corner."""
    top = [(0, c) for c in range(GRID)]
    right = [(r, GRID - 1) for r in range(1, GRID)]
    bottom = [(GRID - 1, c) for c in range(GRID - 2, -1, -1)]
    left = [(r, 0) for r in range(GRID - 2, 0, -1)]
    tiles = top + right + bottom + left
    assert len(tiles) == N_BORDER and len(set(tiles)) == N_BORDER
    return tiles


BORDER = border_tiles()


class SiegeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0.0, 1.0, (N_CHANNELS, GRID, GRID), np.float32),
                "state": spaces.Box(0.0, 1.0, (STATE_DIM,), np.float32),
            }
        )
        self._reset_state()

    # -- state -----------------------------------------------------------
    def _reset_state(self) -> None:
        self.phase = PHASE_BUY
        self.gold = BUDGET
        self.owned = np.zeros(N_TROOPS, dtype=np.int32)  # bought, not yet deployed
        self.deployed = np.zeros(N_TROOPS, dtype=np.int32)
        self.occupied = np.zeros(N_BORDER, dtype=bool)
        self.tick = 0
        self.destruction = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # seeds self.np_random
        self._reset_state()
        return self._obs(), {}

    # -- masks -----------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        m = np.zeros(N_ACTIONS, dtype=bool)
        if self.phase == PHASE_BUY:
            m[BUY_BASE:BUY_BASE + N_TROOPS] = TROOP_COST <= self.gold
            m[END_BUY] = True
        elif self.phase == PHASE_DEPLOY:
            free = ~self.occupied
            for k in range(N_TROOPS):
                if self.owned[k] > 0:
                    s = DEPLOY_BASE + k * N_BORDER
                    m[s:s + N_BORDER] = free
            m[WAIT] = True
        else:  # BATTLE: only waiting is legal
            m[WAIT] = True
        return m

    # -- dynamics --------------------------------------------------------
    def step(self, action: int):
        action = int(action)
        if not self.action_masks()[action]:
            raise ValueError(f"illegal action {action} in phase {self.phase}")
        reward = 0.0
        if action < DEPLOY_BASE:
            self.gold -= int(TROOP_COST[action])
            self.owned[action] += 1
        elif action == END_BUY:
            self.phase = PHASE_DEPLOY
        elif action == WAIT:
            if self.phase == PHASE_DEPLOY:
                self.phase = PHASE_BATTLE
            else:
                self.tick += 1
        else:
            k, tile = divmod(action - DEPLOY_BASE, N_BORDER)
            self.owned[k] -= 1
            self.deployed[k] += 1
            self.occupied[tile] = True
            if self.owned.sum() == 0:
                self.phase = PHASE_BATTLE
        if self.phase == PHASE_BATTLE:
            reward = self._run_battle_stub()
        terminated = self.phase == PHASE_BATTLE and self.tick >= MAX_BATTLE_TICKS
        return self._obs(), float(reward), terminated, False, {}

    def _run_battle_stub(self) -> float:
        """PLACEHOLDER: ends the battle immediately with 0 destruction."""
        self.tick = MAX_BATTLE_TICKS
        return 0.0

    # -- observation -----------------------------------------------------
    def _obs(self) -> dict:
        grid = np.zeros((N_CHANNELS, GRID, GRID), dtype=np.float32)
        for tile in np.flatnonzero(self.occupied):
            r, c = BORDER[tile]
            grid[0, r, c] = 1.0  # placeholder channel: deployed-troop occupancy
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0] = self.gold / BUDGET
        state[1:5] = np.minimum(self.owned / 10.0, 1.0)
        state[5 + self.phase] = 1.0  # one-hot phase in 5..7
        state[8] = self.tick / MAX_BATTLE_TICKS
        return {"grid": grid, "state": state}
