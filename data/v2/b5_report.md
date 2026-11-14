# B5 report (BreachLens v2)

Revision 0 | commit `e6f4a04893` (dirty=True) | sim_version `5962b969fe445f4b...` | layouts: {'TUNE-train': 180, 'TUNE-val': 60, 'LAB-DESIGN': 40} | bootstrap: 10000 resamples, seed 8080

**All gates pass: False**  (failed: G1, G2, G3, G4, G5, G6, G9).  generator_hash recorded: False

## 1. G1-G9 summary

| Gate | Result | Key numbers |
|---|---|---|
| G1 | **FAIL** | max best-single share 0.75 (limit 0.60); units over ban-loss 0.25 everywhere: none |
| G2 | **FAIL** | units passing in: scout: NONE; sniper: NONE; bulwark: NONE; blaster: NONE; mender: NONE |
| G3 | **FAIL** | niche-layout share (need >= 8%, lo95 > 0): scout 2.9% [1.1%]; sniper 47.9% [42.1%]; bulwark 6.1% [3.6%]; blaster 1.1% [0.0%]; mender 24.6% [19.6%] |
| G4 | **FAIL** | Mender gain bunker_nest -0.001 [-0.004, 0.002], LAB 0.008 [0.000, 0.023]; Mender-only TUNE 0.000 / LAB 0.000 |
| G5 | **FAIL** | spine_pods: best-single sniper 0.694, Oracle breach 1.00; archipelago: best-single sniper 0.727, Oracle breach 1.00; bunker_nest: best-single sniper 0.526, Oracle breach 1.00 |
| G6 | **FAIL** | Oracle >12 units on 0.01 of layouts (need >=0.30); layouts with 20 Scouts >= Oracle: 0 |
| G8 | **PASS** | spine_pods: army-search +0.237 (84% layouts), Oracle +0.304; archipelago: army-search +0.230 (90% layouts), Oracle +0.270; bunker_nest: army-search +0.371 (82% layouts), Oracle +0.473 |
| G9 | **FAIL** | Oracle 1.000 [1.000, 1.000], breach 1.00; Greedy 0.636 [0.581, 0.690], 100% of layouts >= 0.05; stalemate with core alive: 0 |

G7 is not evaluated at B5 (pilot, B13).

## 2. Gate details

### G1
best-single-unit share per layout (ties split), pooled TUNE: scout 0.04, sniper 0.75, bulwark 0.14, blaster 0.07, mender 0.00

| family | spine_pods | archipelago | bunker_nest |
|---|---|---|---|
| best-single share scout | 0.00 | 0.00 | 0.12 |
| best-single share sniper | 0.85 | 0.63 | 0.75 |
| best-single share bulwark | 0.03 | 0.34 | 0.05 |
| best-single share blaster | 0.11 | 0.03 | 0.07 |
| best-single share mender | 0.00 | 0.00 | 0.00 |
| Oracle ban-loss scout | 0.002 | -0.001 | -0.002 |
| Oracle ban-loss sniper | 0.005 | 0.007 | 0.007 |
| Oracle ban-loss bulwark | 0.009 | 0.000 | 0.000 |
| Oracle ban-loss blaster | 0.000 | -0.003 | -0.002 |
| Oracle ban-loss mender | 0.001 | -0.003 | -0.001 |

### G2 paired Oracle ban-loss (mean [95% CI]; one-sided 98.75% lower bound; need mean >= 0.02, lo95 > 0, lo_1s > 0)

| unit | family | ban-loss | lo(98.75% 1-sided) | pass |
|---|---|---|---|---|
| scout | spine_pods | 0.002 [0.000, 0.004] | 0.000 | no |
| scout | archipelago | -0.001 [-0.003, 0.000] | -0.004 | no |
| scout | bunker_nest | -0.002 [-0.004, 0.000] | -0.006 | no |
| sniper | spine_pods | 0.005 [0.000, 0.014] | 0.000 | no |
| sniper | archipelago | 0.007 [0.001, 0.015] | 0.000 | no |
| sniper | bunker_nest | 0.007 [-0.002, 0.020] | -0.003 | no |
| bulwark | spine_pods | 0.009 [0.000, 0.021] | 0.000 | no |
| bulwark | archipelago | 0.000 [-0.002, 0.003] | -0.002 | no |
| bulwark | bunker_nest | 0.000 [0.000, 0.000] | 0.000 | no |
| blaster | spine_pods | 0.000 [-0.006, 0.007] | -0.006 | no |
| blaster | archipelago | -0.003 [-0.009, 0.000] | -0.011 | no |
| blaster | bunker_nest | -0.002 [-0.004, 0.000] | -0.006 | no |
| mender | spine_pods | 0.001 [-0.006, 0.009] | -0.006 | no |
| mender | archipelago | -0.003 [-0.009, 0.000] | -0.011 | no |
| mender | bunker_nest | -0.001 [-0.004, 0.002] | -0.005 | no |

### G3 per-layout niches (B5-R1: pooled TUNE-train + TUNE-val + LAB-DESIGN, margin >= 0.03 over the runner-up template; need share >= 8% and lo95 > 0)

Mean template scores: scout 0.146, sniper 0.657, bulwark 0.429, blaster 0.436, bulwark_mender 0.539, mender_only 0.000

| unit | template | niche layouts | share [95% CI] | pass |
|---|---|---|---|---|
| scout | scout | 8 / 280 | 0.029 [0.011, 0.050] | no |
| sniper | sniper | 134 / 280 | 0.479 [0.421, 0.536] | yes |
| bulwark | bulwark | 17 / 280 | 0.061 [0.036, 0.089] | no |
| blaster | blaster | 3 / 280 | 0.011 [0.000, 0.025] | no |
| mender | bulwark_mender | 69 / 280 | 0.246 [0.196, 0.296] | yes |

Descriptive only (per family, share of layouts where the unit's template wins by >= 0.03): spine_pods: scout 0.00, sniper 0.62, bulwark 0.00, blaster 0.01, mender 0.19, no clear winner 0.17; archipelago: scout 0.00, sniper 0.47, bulwark 0.17, blaster 0.00, mender 0.16, no clear winner 0.19; bunker_nest: scout 0.10, sniper 0.42, bulwark 0.01, blaster 0.03, mender 0.35, no clear winner 0.09; labyrinth: scout 0.00, sniper 0.30, bulwark 0.05, blaster 0.00, mender 0.33, no clear winner 0.33

### G4
- Mender gain on bunker_nest: -0.001 [-0.004, 0.002] -> no
- Mender gain on LAB-DESIGN: 0.008 [0.000, 0.023] -> no

- Mender-only army on TUNE: mean 0.000 [0.000, 0.000] (upper95 0.000) -> ok
- Mender-only army on LAB-DESIGN: mean 0.000 [0.000, 0.000] (upper95 0.000) -> ok

### G5 / G6 / G8

| family | best single | mean | Oracle breach | G8a army-search gain [95% CI] | better on | G8b Oracle gain [95% CI] |
|---|---|---|---|---|---|---|
| spine_pods | sniper | 0.694 | 1.00 | 0.237 [0.198, 0.276] | 84% | 0.304 [0.278, 0.330] |
| archipelago | sniper | 0.727 | 1.00 | 0.230 [0.198, 0.260] | 90% | 0.270 [0.243, 0.297] |
| bunker_nest | sniper | 0.526 | 1.00 | 0.371 [0.310, 0.434] | 82% | 0.473 [0.421, 0.528] |

G6: Oracle armies >12 units: 0.01 (by family spine_pods 0.00, archipelago 0.03, bunker_nest 0.00); 20 Scouts >= Oracle on 0 layouts; Oracle army exactly 20 Scouts on 0.

### G9 (LAB-DESIGN, 40)
checks: {'oracle_mean': True, 'oracle_breach': True, 'greedy_mean_in_range': False, 'greedy_layout_share': True, 'no_stalemate_core_alive': True}; exit reasons over all battles: {'all_dead': 584, 'no_attackers': 216, 'cleared': 36, 'horizon': 4}

## 3. Per-unit statics (effective hp per gold incl. armor, dps per gold, threatening defenders)

| unit | cost | hp | armor | EHP/gold Gun | Lancer | Flak (centre) | Flak (neighbour hit) | mean of 3 | DPS/gold | threatened by (<=3 hits & range) | outranged by | can hit from inside blind spot of |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scout | 8 | 16 | 0 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 0.750 | gun (2 hits), lancer (1 hits), flak (2 hits) | gun, lancer, flak | lancer, flak |
| sniper | 24 | 28 | 0 | 1.17 | 1.17 | 1.17 | 1.17 | 1.17 | 0.500 | lancer (2 hits), flak (2 hits) | lancer | none |
| bulwark | 30 | 170 | 4 | 8.50 | 7.29 | 7.93 | 13.22 | 7.91 | 0.533 | none | gun, lancer, flak | lancer, flak |
| blaster | 36 | 60 | 0 | 1.67 | 1.67 | 1.67 | 1.67 | 1.67 | 0.722 (3-bldg cluster 1.444) | none | gun, lancer, flak | none |
| mender | 28 | 40 | 0 | 1.43 | 1.43 | 1.43 | 1.43 | 1.43 | 0.000 (heal 0.286) | lancer (3 hits), flak (3 hits) | gun, lancer, flak | none |

## 4. Single-unit-army score table (mean score [95% CI] | D | breach | units / gold; * = best per family)

| unit | spine_pods (n=80) | archipelago (n=80) | bunker_nest (n=80) | LAB-DESIGN (n=40) |
|---|---|---|---|---|
| scout | 0.136 [0.122, 0.150] / D 0.218 / breach 0.01 / 20.0u 160g | 0.140 [0.115, 0.168] / D 0.200 / breach 0.05 / 20.0u 160g | 0.169 [0.140, 0.198] / D 0.281 / breach 0.00 / 20.0u 160g | 0.129 [0.077, 0.186] / D 0.032 / breach 0.28 / 20.0u 160g |
| sniper | 0.694 [0.667, 0.721]* / D 0.490 / breach 1.00 / 10.0u 240g | 0.727 [0.701, 0.753]* / D 0.545 / breach 1.00 / 10.0u 240g | 0.526 [0.471, 0.578]* / D 0.359 / breach 0.78 / 10.0u 240g | 0.705 [0.662, 0.746]* / D 0.508 / breach 1.00 / 10.0u 240g |
| bulwark | 0.438 [0.426, 0.450] / D 0.063 / breach 1.00 / 8.0u 240g | 0.638 [0.595, 0.678] / D 0.497 / breach 0.85 / 8.0u 240g | 0.109 [0.082, 0.141] / D 0.165 / breach 0.03 / 8.0u 240g | 0.636 [0.581, 0.690] / D 0.410 / breach 0.97 / 8.0u 240g |
| blaster | 0.521 [0.500, 0.544] / D 0.202 / breach 1.00 / 6.0u 216g | 0.532 [0.496, 0.566] / D 0.303 / breach 0.88 / 6.0u 216g | 0.172 [0.130, 0.218] / D 0.195 / breach 0.14 / 6.0u 216g | 0.605 [0.556, 0.654] / D 0.358 / breach 0.97 / 6.0u 216g |
| mender | 0.000 [0.000, 0.000] / D 0.000 / breach 0.00 / 8.0u 224g | 0.000 [0.000, 0.000] / D 0.000 / breach 0.00 / 8.0u 224g | 0.000 [0.000, 0.000] / D 0.000 / breach 0.00 / 8.0u 224g | 0.000 [0.000, 0.000] / D 0.000 / breach 0.00 / 8.0u 224g |

## 5. Wall behaviour (walls attacked per battle per unit type; share of time spent attacking walls)

| family | unit | walls attacked / battle (single army) | wall share of alive unit-ticks (single) | walls attacked / battle (Oracle army) | wall share (Oracle army) |
|---|---|---|---|---|---|
| spine_pods | scout | 0.00 | 0.000 | 0.00 | 0.000 |
| spine_pods | sniper | 3.85 | 0.181 | 3.35 | 0.173 |
| spine_pods | bulwark | 6.01 | 0.198 | 3.44 | 0.169 |
| spine_pods | blaster | 0.20 | 0.010 | 0.44 | 0.053 |
| archipelago | scout | 0.66 | 0.053 | 0.15 | 0.044 |
| archipelago | sniper | 3.00 | 0.148 | 1.98 | 0.106 |
| archipelago | bulwark | 4.62 | 0.126 | 3.27 | 0.123 |
| archipelago | blaster | 1.15 | 0.073 | 0.76 | 0.097 |
| bunker_nest | scout | 9.99 | 0.495 | 0.66 | 0.293 |
| bunker_nest | sniper | 27.25 | 0.696 | 15.43 | 0.618 |
| bunker_nest | bulwark | 29.44 | 0.548 | 21.76 | 0.465 |
| bunker_nest | blaster | 13.07 | 0.620 | 4.01 | 0.458 |
| LAB-DESIGN | scout | 17.38 | 0.835 | 2.88 | 0.765 |
| LAB-DESIGN | sniper | 38.17 | 0.769 | 14.35 | 0.722 |
| LAB-DESIGN | bulwark | 41.75 | 0.576 | 31.30 | 0.582 |
| LAB-DESIGN | blaster | 15.32 | 0.472 | 8.72 | 0.599 |

| family | single-unit armies: wall-attack share of attacker unit-ticks | ticks with any wall attack / battle ticks | Oracle armies: share of attacker unit-ticks | ticks with any wall attack |
|---|---|---|---|---|
| spine_pods | 0.133 | 0.312 | 0.160 | 0.268 |
| archipelago | 0.115 | 0.255 | 0.112 | 0.193 |
| bunker_nest | 0.603 | 0.824 | 0.517 | 0.645 |
| LAB-DESIGN | 0.683 | 0.844 | 0.629 | 0.814 |

## 6. Oracle / baselines per family

| set | Oracle mean | Oracle breach | Oracle units | Oracle counts s/sn/b/bl/m | army-search mean | Greedy mean |
|---|---|---|---|---|---|---|
| spine_pods | 0.998 | 1.00 | 9.3 | 1.7, 2.0, 2.9, 0.9, 1.7 | 0.931 | 0.438 |
| archipelago | 0.997 | 1.00 | 9.6 | 2.0, 2.3, 2.8, 1.1, 1.5 | 0.957 | 0.638 |
| bunker_nest | 0.998 | 1.00 | 8.9 | 1.3, 1.6, 3.2, 0.9, 1.8 | 0.897 | 0.109 |
| LAB-DESIGN | 1.000 | 1.00 | 8.9 | 1.6, 1.5, 2.9, 1.0, 2.0 | 0.979 | 0.636 |

## 7. Defender mix per family and generator-rule checks (spec 3.1b)

| family | layouts | mean defenders | gun / lancer / flak total | share with Lancer | share with Flak | gun majority |
|---|---|---|---|---|---|---|
| spine_pods | 80 | 8.6 | 515 / 140 / 30 | 1.00 | 0.26 | 1.00 |
| archipelago | 80 | 9.3 | 517 / 125 / 104 | 0.90 | 0.91 | 1.00 |
| bunker_nest | 80 | 9.0 | 577 / 112 / 29 | 1.00 | 0.24 | 1.00 |
| labyrinth | 40 | 5.2 | 76 / 69 / 64 | 1.00 | 1.00 | 0.17 |

Rule checks: {'spine_pods_lancer>=70%': True, 'bunker_nest_lancer>=70%': True, 'archipelago_flak>=50%': True, 'labyrinth_uses_all_three': True, 'spine_pods_gun_majority_every_layout': True, 'archipelago_gun_majority_every_layout': True, 'bunker_nest_gun_majority_every_layout': True}

## 8. Stat-dependent acceptance (too_hard, single_type_trivial) on TUNE / LAB-DESIGN streams

| family | structurally accepted & evaluated | single_type_trivial | too_hard | rejected (either) | rejection rate | flag >50% |
|---|---|---|---|---|---|---|
| spine_pods | 912 | 832 | 0 | 832 | 91.2% | **FLAG** |
| archipelago | 13334 | 13254 | 0 | 13254 | 99.4% | **FLAG** |
| bunker_nest | 200 | 120 | 0 | 120 | 60.0% | **FLAG** |
| labyrinth | 155 | 115 | 0 | 115 | 74.2% | **FLAG** |
