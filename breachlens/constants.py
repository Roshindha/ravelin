"""BreachLens (v2) world constants. Source of truth: docs/SPEC_V2.md section 2 and 4.1.

B1 scope only: grid, budget, horizon, cap, unit-type list, action-space layout. Unit stats are
NOT defined here (they are proposed in the spec and pinned in B5 after the design gates).
"""

GRID = 14
BUDGET = 240
MAX_BATTLE_TICKS = 240
MAX_UNITS = 20
N_BORDER = 4 * GRID - 4  # 52

UNIT_TYPES = ("scout", "sniper", "bulwark", "blaster", "mender")
N_UNIT_TYPES = len(UNIT_TYPES)

PHASE_BUY, PHASE_DEPLOY, PHASE_BATTLE = 0, 1, 2
N_PHASES = 3

# Action layout: 0-4 buy | 5..264 deploy (5 + type*52 + tile) | 265 END_BUY | 266 WAIT
BUY_BASE = 0
DEPLOY_BASE = BUY_BASE + N_UNIT_TYPES
END_BUY = DEPLOY_BASE + N_UNIT_TYPES * N_BORDER
WAIT = END_BUY + 1
N_ACTIONS = WAIT + 1  # 267

# Observation shapes (contents are implemented in B7)
N_GRID_CHANNELS = 9
N_STATE = 11
