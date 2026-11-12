# Observation specification

The environment returns a dictionary with exactly two keys:

- `grid`: a `(8, 16, 16)` float32 tensor
- `state`: a `(9,)` float32 vector

## `grid` channels

1. Channel 0: friendly troop occupancy on the deployment border and pre-battle positions. Value is 1.0 on occupied cells and 0.0 elsewhere.
2. Channel 1: building HP heatmap. Each building cell stores `hp / 80.0`, clipped to `[0, 1]`.
3. Channel 2: turret HP heatmap. Each turret cell stores `hp / 50.0`, clipped to `[0, 1]`.
4. Channel 3: wall HP heatmap. Each wall cell stores `hp / 80.0`, clipped to `[0, 1]`.
5. Channel 4: building value priority map. For each building cell, stores `value / max_building_value` in the current layout, clipped to `[0, 1]`.
6. Channel 5: turret range heatmap. For each turret cell, stores `range / max_turret_range`, clipped to `[0, 1]`.
7. Channel 6: border deployment mask. All legal border tiles are marked 1.0, and all interior cells are 0.0.
8. Channel 7: reserved / battle-progress channel. Currently zero while the battle is unresolved; it is kept for future battle-state features and remains partially observed to preserve the canonical 8-channel interface.

## `state` values

The state vector has 9 values in this order:

1. `state[0]`: remaining gold normalized by the 200-resource budget, i.e. `gold / 200`
2. `state[1]`: owned count for troop type 0 (`swarm`) normalized by 10
3. `state[2]`: owned count for troop type 1 (`ranger`) normalized by 10
4. `state[3]`: owned count for troop type 2 (`tank`) normalized by 10
5. `state[4]`: owned count for troop type 3 (`splash`) normalized by 10
6. `state[5]`: BUY phase indicator (`1` if `phase == BUY`, else `0`)
7. `state[6]`: DEPLOY phase indicator (`1` if `phase == DEPLOY`, else `0`)
8. `state[7]`: BATTLE phase indicator (`1` if `phase == BATTLE`, else `0`)
9. `state[8]`: elapsed tick fraction, `tick / MAX_BATTLE_TICKS`

This remains deliberately partially observed: the agent sees board structure and the vital threat/resource state, but not all internal combat details or the full hidden battle state.
