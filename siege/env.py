"""SiegeEnv: buy/deploy against a randomized base, then resolve one deterministic battle.

Observation layout:
    grid: (8, 16, 16), partially observed board state. Channels are:
        0: friendly troop occupancy on the deployment border / pre-battle positions
        1: building hp as a normalized per-tile value in [0, 1] (hp / 80)
        2: turret hp as a normalized per-tile value in [0, 1] (hp / 50)
        3: wall hp as a normalized per-tile value in [0, 1] (hp / 80)
        4: building value / priority heatmap, normalized by the maximum building value in the current layout
        5: turret range heatmap, normalized by the maximum turret range in the layout
        6: border deployment mask (legal entry tiles set to 1)
        7: reserved / battle-progress channel, currently 0 until the battle resolves
    state: (9,) values are:
        0: current gold / BUDGET
        1..4: owned troop counts for the 4 troop types, normalized to 0..1 by /10
        5..7: one-hot phase encoding (BUY, DEPLOY, BATTLE)
        8: elapsed tick fraction (tick / MAX_BATTLE_TICKS)

Action layout (Discrete(246)):
    0..3        buy troop type k
    4..243      deploy: 4 + troop_type * 60 + border_tile_index
    244         END_BUY
    245         WAIT

The action mask is the source of truth for legal actions. A random invalid action is treated as
an empty no-op so Gymnasium's checker can exercise the environment without crashing; the legal
mask still rejects illegal actions in normal gameplay.
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


def encode_deploy(troop_idx: int, border_idx: int) -> int:
    """Turn a troop type and border index into the Discrete(246) deploy action."""
    troop_idx = int(troop_idx)
    border_idx = int(border_idx)
    if troop_idx < 0 or troop_idx >= N_TROOPS:
        raise ValueError(f"troop_idx out of range: {troop_idx}")
    if border_idx < 0 or border_idx >= N_BORDER:
        raise ValueError(f"border_idx out of range: {border_idx}")
    return DEPLOY_BASE + troop_idx * N_BORDER + border_idx


def decode_action(action: int) -> tuple[str, int | None, int | None]:
    """Return a semantic tuple for the given action index."""
    action = int(action)
    if action < 0 or action >= N_ACTIONS:
        raise ValueError(f"invalid action index {action}; expected 0 <= action < {N_ACTIONS}")
    if action < BUY_BASE + N_TROOPS:
        return ("buy", action, None)
    if action < DEPLOY_BASE + N_TROOPS * N_BORDER:
        troop_idx, border_idx = divmod(action - DEPLOY_BASE, N_BORDER)
        return ("deploy", troop_idx, border_idx)
    if action == END_BUY:
        return ("end_buy", None, None)
    if action == WAIT:
        return ("wait", None, None)
    raise ValueError(f"unknown action {action}")


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
    """
    `layout_pool`, when given, is a list of layout records (as saved by `siege.layouts`:
    {"family": ..., "seed": ...}, extra keys like "id"/"pool"/"tag" ignored). Each reset samples
    one record uniformly via `self.np_random` and deterministically regenerates that exact layout
    from its (family, seed) rather than drawing a fresh random one. Pass a DEV-pool slice for
    training and a FINAL-pool slice for held-out evaluation -- see `siege.layouts.load_layout_pool`.
    Omit it (the default) to fall back to a fresh random layout every reset, as before.
    """

    metadata = {"render_modes": []}

    def __init__(self, layout_pool: list[dict] | None = None, prebought: list[int] | str | None = None) -> None:
        super().__init__()
        self.layout_pool = layout_pool
        # Curriculum hook: troop counts [swarm, ranger, tank, splash] bought for the agent at reset.
        # None = normal game (agent shops). Switched mid-training via `set_prebought`.
        self.prebought: list[int] | None = None
        self.set_prebought(prebought)
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

    def _random_army(self) -> list[int]:
        """A random legal army: buy a uniformly random affordable type until broke, stopping early w.p. 0.1 per buy."""
        counts, gold = [0] * N_TROOPS, BUDGET
        while True:
            afford = [k for k in range(N_TROOPS) if TROOP_COST[k] <= gold]
            if not afford or (sum(counts) > 0 and self.np_random.random() < 0.1):
                return counts
            k = afford[int(self.np_random.integers(len(afford)))]
            counts[k] += 1
            gold -= int(TROOP_COST[k])

    def set_prebought(self, counts: list[int] | str | None) -> None:
        """Enable (counts, or "random" = fresh random legal army each episode) or disable (None) the deploy-only curriculum."""
        if isinstance(counts, str):
            if counts != "random":
                raise ValueError(counts)
        elif counts is not None:
            counts = [int(c) for c in counts]
            if len(counts) != N_TROOPS or min(counts) < 0 or int(np.dot(counts, TROOP_COST)) > BUDGET:
                raise ValueError(f"invalid prebought army {counts}")
        self.prebought = counts

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """options (all optional): ``layout_record`` pins the layout; ``prebought`` overrides the
        curriculum army for this episode (a count list, or ``False`` to force normal shopping)."""
        super().reset(seed=seed)  # seeds self.np_random
        self._reset_state()
        options = options or {}
        prebought = options.get("prebought", self.prebought)
        if isinstance(prebought, str):  # "random"
            prebought = self._random_army()
        if prebought:  # deployment-first curriculum: skip BUY, start in DEPLOY with the army owned
            self.owned = np.array(prebought, dtype=np.int32)
            self.gold = BUDGET - int(np.dot(prebought, TROOP_COST))
            self.phase = PHASE_DEPLOY
        if options.get("layout_record") is not None:
            record = options["layout_record"]
            self.layout_record = record
            self.layout = generate_base_layout(np.random.default_rng(record["seed"]), family=record["family"])
        elif self.layout_pool:
            record = self.layout_pool[int(self.np_random.integers(len(self.layout_pool)))]
            self.layout_record = record
            self.layout = generate_base_layout(np.random.default_rng(record["seed"]), family=record["family"])
        else:
            self.layout_record = None
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
        if action < 0 or action >= N_ACTIONS:
            raise ValueError(f"action index out of range: {action}")
        legal = self.action_masks()
        if not legal[action]:
            # Keep the Gymnasium contract happy while preserving the legal-mask interface used by
            # policy code. Invalid actions are treated as safe no-ops rather than crashing the env.
            return self._obs(), 0.0, False, False, {}
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
            grid[0, r, c] = 1.0
        buildings, turrets, walls = self._board_state()
        for b in buildings:
            grid[1, int(b["x"]), int(b["y"])] = min(1.0, b["hp"] / 80.0)
        for t in turrets:
            grid[2, int(t["x"]), int(t["y"])] = min(1.0, t["hp"] / 50.0)
        for w in walls:
            grid[3, int(w["x"]), int(w["y"])] = min(1.0, w["hp"] / 80.0)
        max_building_value = max((b["value"] for b in buildings), default=1)
        for b in buildings:
            grid[4, int(b["x"]), int(b["y"])] = min(1.0, b["value"] / max_building_value)
        max_turret_range = max((t["range"] for t in turrets), default=1)
        for t in turrets:
            grid[5, int(t["x"]), int(t["y"])] = min(1.0, t["range"] / max_turret_range)
        for r, c in BORDER:
            grid[6, r, c] = 1.0
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0] = self.gold / BUDGET
        state[1:5] = np.minimum(self.owned / 10.0, 1.0)
        state[5 + self.phase] = 1.0  # one-hot phase in 5..7
        state[8] = self.tick / MAX_BATTLE_TICKS
        return {"grid": grid, "state": state}


__all__ = [
    "BORDER",
    "BUDGET",
    "BUY_BASE",
    "DEPLOY_BASE",
    "END_BUY",
    "GRID",
    "MAX_BATTLE_TICKS",
    "N_ACTIONS",
    "N_BORDER",
    "N_TROOPS",
    "PHASE_BATTLE",
    "PHASE_BUY",
    "PHASE_DEPLOY",
    "STATE_DIM",
    "SiegeEnv",
    "decode_action",
    "encode_deploy",
]
