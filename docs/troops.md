# Troop stats

The battlefield uses four troop archetypes with distinct combat roles. Costs are paired so a 200-resource budget encourages meaningful trade-offs rather than a single dominant pick.

## Current (v2) stats

| Name | Cost | HP | Damage | Speed | Range | Role | Target preference |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| swarm | 10 | 18 | 8 | 1.5 | 1.2 | cheap skirmisher | building > turret > wall |
| ranger | 25 | 30 | 12 | 1.0 | 4.0 | ranged attacker | turret > building > wall |
| tank | 30 | 150 | 24 | 0.7 | 1.4 | frontliner | wall > building > turret |
| splash | 40 | 75 | 34 | 0.6 | 2.5 | wall-breaker | wall > building > turret |

Base/turret knobs that changed alongside troops (`siege/bases.py`):

| Knob | v1 | v2 |
| --- | --- | --- |
| turret HP | 30-50 | 70-110 |
| turret damage | 8-15 | 10-18 |
| turret range | 3-6 | 4-6 |
| wall HP | 80 | 70 |
| buildings per layout | 3-5 | 6-12 |
| turrets per layout | 1-2 | 2-5 |

## Why v1 was broken

The v1 stats (swarm 10/ranger 20/tank 30/splash 40, turret HP 30-50/damage 8-15/range 3-6, a single
`ring_defence`-style layout) produced a rangers-only army that cleared almost every generated base
(mean destruction 0.96 over a 200k-timestep PPO run; see PROGRESS.md). Two compounding causes, found
by running fixed single-type armies through the real `simulate_battle` and watching per-tick frames:

1. **Ranger range (5.0) exceeded turret range (3-6, often <5)**, and `combat._select_target` has no
   line-of-sight check -- a ranger standing still can snipe through walls with no retaliation chance
   as long as it's within its own range of a building. Rangers never needed to enter turret range to
   fight, only to kill buildings.
2. **Turret HP (30-50) was far too low relative to combined ranger DPS.** A ranger's `target_preference`
   is `turret > building > wall`, so 8-10 rangers (floor(200/20)) focus-fire the nearest turret first.
   At 12 damage each, 8 rangers deal 96 damage/tick -- enough to kill any turret in that HP range in a
   single tick, before it gets a single shot off. Once turrets die, nothing else in the game can hurt
   a ranger, so building destruction is just a matter of ticks.

## Iteration to v2

Fixed with `scripts_balance.py` (fixed single-type and mixed armies, clustered at one border push
point, run through the real simulator across 200 generated layouts per family), not by guessing:

1. Raised ranger's cost (20->25, fewer rangers per budget) and lowered its range (5.0->4.0) so it can
   no longer reliably out-range every turret.
2. Raised turret HP substantially (30-50 -> 70-110) so even a full ranger squad's alpha strike takes
   multiple ticks to kill one, during which the turret (and any other turrets in range) shoot back.
   Overshot once to 170-220 while iterating -- that made every layout nearly unbeatable by anything
   (mean destruction <0.2 across the board) -- then settled at 70-110.
3. Raised turret damage (8-15 -> 10-18) so turret fire is a real threat to low-HP units (ranger 30 HP,
   swarm 18 HP), not raised so high it one-shots everything.
4. Buffed tank HP a lot (90 -> 150) and splash HP somewhat (42 -> 75, after an initial pass at 55 that
   still left it useless) so melee/breacher units can survive long enough under turret fire to actually
   reach and break a wall -- previously they just died in the open before contributing anything (mean
   destruction ~0.000 for tank/splash on `ring_defence`/`clustered_core` even after the turret-HP fix).
5. **Changed how test armies are deployed for the balance check** from "spread evenly across all 60
   border tiles" to "clustered at one shared push point" (`siege.armies.place_on_border(..., start_idx)`).
   Spreading sends each unit through its own isolated gap, so a mixed army of 4 types never actually
   fights alongside itself -- there's no way for a "mixed army beats single-type" result to emerge from
   that setup regardless of stats. Clustering lets tanks/turrets draw fire while splash/rangers follow
   through the same breach, which is the actual combined-arms thesis of having 4 troop types at all.

## Result (200 layouts/family, v2 stats, see `scripts_balance.py`)

| family | swarm | ranger | tank | splash | mixed_balanced | mixed_breach |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ring_defence | 0.000 | 0.082 | **0.278** | 0.023 | 0.255 | 0.238 |
| clustered_core | 0.086 | 0.165 | 0.176 | 0.003 | **0.291** | 0.075 |
| scattered_outposts | 0.763 | **0.901** | 0.810 | 0.321 | 0.831 | 0.641 |

`mixed_balanced` = 5 swarm + 2 ranger + 3 tank (190/200 gold). `mixed_breach` = 2 splash + 2 tank + 2
ranger (190/200 gold).

No single troop type wins across all three families: tank wins `ring_defence` (one heavy wall, high HP
needed to survive the approach), a mixed army wins `clustered_core` (two separately-defended clusters
reward a force that can split and still hit hard at each), and ranger wins `scattered_outposts` (thin,
spread-out defenses with lots of open sight lines reward raw range/DPS over survivability). Splash
remains the weakest archetype overall -- a real candidate for another iteration once there's a UI to
actually watch why it's dying (my hypothesis: its speed of 0.6, the slowest in the game, means it
spends more ticks under turret fire before reaching the wall than its HP buff fully compensates for).
