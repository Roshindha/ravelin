# Troop stats

The battlefield uses four troop archetypes with distinct combat roles. Costs are paired so a 200-resource budget encourages meaningful trade-offs rather than a single dominant pick.

## Stats (v2 base; v3 adds splash area damage -- see the STEP 5c section at the end)

| Name | Cost | HP | Damage | Speed | Range | Role | Target preference |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| swarm | 10 | 18 | 8 | 1.5 | 1.2 | cheap skirmisher | building > turret > wall |
| ranger | 25 | 30 | 12 | 1.0 | 4.0 | ranged attacker | turret > building > wall |
| tank | 30 | 150 | 24 | 0.7 | 1.4 | frontliner | wall > building > turret |
| splash | 40 | 75 | 34 | 0.6 | 2.5 | wall-breaker (area damage r=1.5, 30% falloff, added in v3) | wall > building > turret |

Base/turret knobs that changed alongside troops (`siege/bases.py`):

| Knob | v1 | v2 |
| --- | --- | --- |
| turret HP | 30-50 | 70-110 |
| turret damage | 8-15 | 10-18 |
| turret range | 3-6 | 4-6 |
| wall HP | 80 | 80 (effective; see note) |
| buildings per layout | 3-5 | 6-12 |
| turrets per layout | 1-2 | 2-5 |

Note on wall HP: v2 documented 70, and the generator constant was 70, but the simulator ignored the hp in
the wall dict and used its default of 80 for every wall. All balance numbers and the frozen pools were
measured at 80, so the constant and docs now say 80 (simulator behaviour unchanged). Wall HP is **80**.

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

## Layout pool v2 (STEP 5b)
`results/layouts_v1.json` / `layouts_v1.sha256` are the original frozen pool, kept for the record but
**superseded**: they were accepted under the old simulator, whose position-ordered tie-breaks were not
D4-equivariant. After the equivariance fix (see PROGRESS.md, STEP 5), 23 of its 241 layouts no longer
passed the acceptance rules. v2 (`results/layouts.json`, build log in `results/layouts_build.json`) uses the
same generators, acceptance rules and split logic with fresh seeds (seed_base 5,000,000).


## Final balance pass (STEP 5c, v3 stats) -- every change and why

Tuned **only** on a separate TUNING seed set (300 raw layouts per family; seeds 9,000,000 / 19,000,000 /
29,000,000 + 0..299; recorded in `results/layouts_build.json`; disjoint from the v1, v2 and v3 pools and
never touching FINAL). Two tuning iterations used, then frozen. Harness: `scripts_tuning.py` (fixed armies,
acceptance rates, oracle search with per-type ban test); iteration-2 splash sweep: `scripts_sweep_splash.py`.
Raw outputs: `results/tuning_iter0_baseline.json`, `tuning_iter1.json`, `tuning_iter2.json`.

| Change | Before | After | Reason |
| --- | --- | --- | --- |
| splash area damage (new mechanic) | none (single target) | radius 1.5, falloff 0.3 (10 dmg to everything within 1.5 of the struck tile) | splash had no area effect despite its name, so it was just a slow 40-gold single-target unit (mean 0.00-0.02 on walled families; banning it cost the oracle <=0.03). Area damage makes it good against dense building cores and wall rings. |
| splash stats | 40 g / 75 hp / 34 dmg / range 2.5 | unchanged | |
| turret fire | sequential in id order | simultaneous (targets chosen from start-of-phase state, damage applied together) | removes the last label (id) dependence in the simulator; allows overkill. Residual D4 id-shuffle mismatches went from 4/7000 to 0/7000. |
| scattered_outposts turrets | 2-5 total, shared round-robin across 3-5 outposts | 1-2 per outpost (`SCATTER_TURRETS_PER_OUTPOST = (1, 3)`) | most outposts had no turret, so a rangers-only army averaged 0.88 (raw) and 83% of generated layouts were `ranger_trivial`. |
| GreedyAgent | buy by damage/cost | buy by Lanchester value hp*damage/cost^2, same deploy rule as best-single | see below |

Iteration log (tuning set, fixed armies, mean destruction; splash alone / mixed_breach / ranger):

| iteration | splash falloff | ring_defence | clustered_core | scattered_outposts | verdict |
| --- | --- | --- | --- | --- | --- |
| 0 (v2 stats) | no AoE | 0.014 / 0.168 / 0.044 | 0.003 / 0.080 / 0.177 | 0.266 / 0.560 / 0.881 | splash dead, outposts trivial |
| 1 | 0.6 | 0.622 / 0.557 / 0.045 | 0.561 / 0.486 / 0.182 | 0.363 / 0.501 / 0.442 | splash overshoots (strictly best army on 77% / 67% of ring / clustered layouts); outposts goal met |
| 2 | 0.3 (picked by a 4-point sweep 0.2-0.5 on the same tuning seeds) | 0.274 / 0.353 / 0.045 | 0.215 / 0.241 / 0.182 | 0.120 / 0.259 / 0.442 | accepted and frozen |

### Why GreedyAgent scored below Random
The old Greedy ranked troops by damage per gold. splash (34/40 = 0.85) came first, so it bought
**five splash and nothing else** (visible in `_greedy_army`: `['splash'] * 5`) -- the weakest unit in practice
(slowest, walls-first targeting, no area damage). It was not stacking tiles; it was a bad composition. The new
Greedy ranks by hp*damage/cost^2 (swarm 1.44, ranger 0.58, tank 4.00, splash 1.59), giving 6 tank + 2 swarm.

### Layout version caveat
`layouts_v1.json` / `layouts_v2.json` are archived records (seed + family only). They regenerate to the layouts
they described only under the generator/simulator of their own era; the scattered_outposts generator changed in
v3 (and the simulator in Step 5 / 5c), so do not evaluate them with the current code.
