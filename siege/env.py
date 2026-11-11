"""SiegeEnv: buy/deploy against a randomized base, then resolve one deterministic battle.

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

from siege.bases import generate_base_layout
from siege.combat import simulate_battle
from siege.troops import TROOP_COST, TROOP_TYPES

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
        self.deployed_troops: list[dict] = []
        self.battle_result = None
        self.tick = 0
        self.destruction = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # seeds self.np_random
        self._reset_state()
        self.layout = generate_base_layout(self.np_random)
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
            r, c = BORDER[tile]
            self.deployed_troops.append({"type": TROOP_TYPES[k].name, "x": r, "y": c})
            if self.owned.sum() == 0:
                self.phase = PHASE_BATTLE
        if self.phase == PHASE_BATTLE and self.battle_result is None:
            reward = self._run_battle()
        terminated = self.phase == PHASE_BATTLE and self.tick >= MAX_BATTLE_TICKS
        return self._obs(), float(reward), terminated, False, {}

    def _run_battle(self) -> float:
        """Resolve the whole battle deterministically in one step."""
        self.battle_result = simulate_battle(
            self.deployed_troops,
            self.layout["buildings"],
            self.layout["walls"],
            self.layout["turrets"],
            max_ticks=MAX_BATTLE_TICKS,
        )
        self.destruction = self.battle_result.destruction
        self.tick = MAX_BATTLE_TICKS
        return self.battle_result.reward

    # -- observation -----------------------------------------------------
    def _board_state(self) -> tuple[list[dict], list[dict], list[dict]]:
        if self.battle_result is not None:
            last = self.battle_result.frames[-1]
            return last["buildings"], last["turrets"], last["walls"]
        walls = [{"x": x, "y": y, "hp": hp} for (x, y), hp in self.layout["walls"].items()]
        return self.layout["buildings"], self.layout["turrets"], walls

    def _obs(self) -> dict:
        grid = np.zeros((N_CHANNELS, GRID, GRID), dtype=np.float32)
        for tile in np.flatnonzero(self.occupied):
            r, c = BORDER[tile]
            grid[0, r, c] = 1.0  # deployed-troop occupancy (pre-battle positions)
        buildings, turrets, walls = self._board_state()
        for b in buildings:
            grid[1, int(b["x"]), int(b["y"])] = min(1.0, b["hp"] / 80.0)
        for t in turrets:
            grid[2, int(t["x"]), int(t["y"])] = min(1.0, t["hp"] / 50.0)
        for w in walls:
            grid[3, int(w["x"]), int(w["y"])] = min(1.0, w["hp"] / 80.0)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0] = self.gold / BUDGET
        state[1:5] = np.minimum(self.owned / 10.0, 1.0)
        state[5 + self.phase] = 1.0  # one-hot phase in 5..7
        state[8] = self.tick / MAX_BATTLE_TICKS
        return {"grid": grid, "state": state}
