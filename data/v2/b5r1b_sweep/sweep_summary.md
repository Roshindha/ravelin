# B5-R1b exploration sweep (not a revision)

**Oracle score / breach**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0: R0 (revision 0) | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 |
| R1a: B + Sniper 40% vs bldg/wall/core | 0.886 / 1.00 | 0.931 / 1.00 | 0.899 / 1.00 | 0.945 / 1.00 |
| R1b: R1a + sealed cores 50% + core hp x3 | 0.893 / 1.00 | 0.926 / 1.00 | 0.835 / 0.95 | 0.953 / 1.00 |
| R1c: R1b + cap 14 | 0.893 / 1.00 | 0.926 / 1.00 | 0.835 / 0.95 | 0.953 / 1.00 |
| R1d: R1b + Mender heal 14, hp 60 | 0.969 / 1.00 | 0.989 / 1.00 | 0.934 / 1.00 | 0.990 / 1.00 |

**Best single-unit army (type, mean score)**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0: R0 (revision 0) | sniper 0.663 | sniper 0.748 | sniper 0.547 | sniper 0.716 |
| R1a: B + Sniper 40% vs bldg/wall/core | sniper 0.538 | sniper 0.581 | sniper 0.384 | sniper 0.656 |
| R1b: R1a + sealed cores 50% + core hp x3 | sniper 0.499 | sniper 0.581 | sniper 0.343 | sniper 0.641 |
| R1c: R1b + cap 14 | sniper 0.499 | sniper 0.581 | sniper 0.343 | sniper 0.641 |
| R1d: R1b + Mender heal 14, hp 60 | sniper 0.499 | sniper 0.581 | sniper 0.343 | sniper 0.641 |

**Sniper share of best-single (per layout, ties split)**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0: R0 (revision 0) | 0.84 | 0.62 | 0.88 | 0.50 |
| R1a: B + Sniper 40% vs bldg/wall/core | 0.82 | 0.62 | 0.65 | 0.37 |
| R1b: R1a + sealed cores 50% + core hp x3 | 0.73 | 0.77 | 0.63 | 0.51 |
| R1c: R1b + cap 14 | 0.73 | 0.77 | 0.67 | 0.51 |
| R1d: R1b + Mender heal 14, hp 60 | 0.73 | 0.77 | 0.63 | 0.51 |

**Single-type rejection / stat rejection rate (trivial / too_hard / evaluated)**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0: R0 (revision 0) | 91.9% (228/0/248) | 99.4% (3119/0/3139) | 53.5% (23/0/43) | 74.7% (59/0/79) |
| R1a: B + Sniper 40% vs bldg/wall/core | 37.5% (12/0/32) | 79.6% (78/0/98) | 16.7% (4/0/24) | 48.7% (19/0/39) |
| R1b: R1a + sealed cores 50% + core hp x3 | 9.1% (2/0/22) | 58.3% (28/0/48) | 16.7% (3/1/24) | 47.4% (18/0/38) |
| R1c: R1b + cap 14 | 9.1% (2/0/22) | 58.3% (28/0/48) | 16.7% (3/1/24) | 47.4% (18/0/38) |
| R1d: R1b + Mender heal 14, hp 60 | 9.1% (2/0/22) | 58.3% (28/0/48) | 16.7% (3/1/24) | 47.4% (18/0/38) |

**Greedy score**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0: R0 (revision 0) | 0.445 | 0.674 | 0.082 | 0.651 |
| R1a: B + Sniper 40% vs bldg/wall/core | 0.403 | 0.470 | 0.026 | 0.613 |
| R1b: R1a + sealed cores 50% + core hp x3 | 0.392 | 0.483 | 0.051 | 0.604 |
| R1c: R1b + cap 14 | 0.392 | 0.483 | 0.051 | 0.604 |
| R1d: R1b + Mender heal 14, hp 60 | 0.392 | 0.483 | 0.051 | 0.604 |

**Share of Oracle armies using > 12 units**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0: R0 (revision 0) | 0.00 | 0.00 | 0.00 | 0.00 |
| R1a: B + Sniper 40% vs bldg/wall/core | 0.00 | 0.00 | 0.00 | 0.00 |
| R1b: R1a + sealed cores 50% + core hp x3 | 0.00 | 0.05 | 0.00 | 0.00 |
| R1c: R1b + cap 14 | 0.00 | 0.05 | 0.00 | 0.00 |
| R1d: R1b + Mender heal 14, hp 60 | 0.00 | 0.00 | 0.00 | 0.00 |

**Sealed-core share / mean walls**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0: R0 (revision 0) | 0.00 / 6.0 | 0.00 / 5.0 | 0.00 / 45.1 | 0.00 / 69.0 |
| R1a: B + Sniper 40% vs bldg/wall/core | 0.00 / 5.8 | 0.00 / 4.4 | 0.00 / 45.0 | 0.00 / 67.5 |
| R1b: R1a + sealed cores 50% + core hp x3 | 0.50 / 9.6 | 0.70 / 8.8 | 0.35 / 47.8 | 0.65 / 68.6 |
| R1c: R1b + cap 14 | 0.50 / 9.6 | 0.70 / 8.8 | 0.35 / 47.8 | 0.65 / 68.6 |
| R1d: R1b + Mender heal 14, hp 60 | 0.50 / 9.6 | 0.70 / 8.8 | 0.35 / 47.8 | 0.65 / 68.6 |

**Monotone ban-loss, scout (mean [95% CI])**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0: R0 (revision 0) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| R1a: B + Sniper 40% vs bldg/wall/core | 0.014 [0.000, 0.041] | 0.011 [0.000, 0.030] | 0.024 [0.000, 0.058] | 0.000 [0.000, 0.000] |
| R1b: R1a + sealed cores 50% + core hp x3 | 0.021 [0.000, 0.057] | 0.000 [0.000, 0.000] | 0.008 [0.000, 0.024] | 0.007 [0.000, 0.018] |
| R1c: R1b + cap 14 | 0.021 [0.000, 0.057] | 0.000 [0.000, 0.000] | 0.008 [0.000, 0.024] | 0.007 [0.000, 0.018] |
| R1d: R1b + Mender heal 14, hp 60 | 0.008 [0.000, 0.023] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

**Monotone ban-loss, sniper (mean [95% CI])**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0: R0 (revision 0) | 0.000 [0.000, 0.000] | 0.013 [0.000, 0.040] | 0.005 [0.000, 0.015] | 0.009 [0.000, 0.026] |
| R1a: B + Sniper 40% vs bldg/wall/core | 0.025 [0.000, 0.055] | 0.012 [0.000, 0.034] | 0.137 [0.061, 0.220] | 0.013 [0.000, 0.037] |
| R1b: R1a + sealed cores 50% + core hp x3 | 0.059 [0.012, 0.117] | 0.046 [0.000, 0.106] | 0.090 [0.034, 0.154] | 0.018 [0.000, 0.043] |
| R1c: R1b + cap 14 | 0.059 [0.012, 0.117] | 0.046 [0.000, 0.106] | 0.090 [0.034, 0.154] | 0.018 [0.000, 0.043] |
| R1d: R1b + Mender heal 14, hp 60 | 0.023 [0.000, 0.070] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

**Monotone ban-loss, bulwark (mean [95% CI])**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0: R0 (revision 0) | 0.000 [0.000, 0.000] | 0.004 [0.000, 0.012] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| R1a: B + Sniper 40% vs bldg/wall/core | 0.078 [0.034, 0.124] | 0.035 [0.007, 0.069] | 0.059 [0.024, 0.098] | 0.037 [0.000, 0.082] |
| R1b: R1a + sealed cores 50% + core hp x3 | 0.107 [0.053, 0.167] | 0.045 [0.012, 0.083] | 0.049 [0.005, 0.102] | 0.024 [0.000, 0.057] |
| R1c: R1b + cap 14 | 0.107 [0.053, 0.167] | 0.045 [0.012, 0.083] | 0.049 [0.005, 0.102] | 0.024 [0.000, 0.057] |
| R1d: R1b + Mender heal 14, hp 60 | 0.114 [0.051, 0.181] | 0.046 [0.009, 0.094] | 0.065 [0.018, 0.126] | 0.017 [0.000, 0.047] |

**Monotone ban-loss, blaster (mean [95% CI])**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0: R0 (revision 0) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| R1a: B + Sniper 40% vs bldg/wall/core | 0.009 [0.000, 0.026] | 0.000 [0.000, 0.000] | 0.012 [0.000, 0.035] | 0.000 [0.000, 0.000] |
| R1b: R1a + sealed cores 50% + core hp x3 | 0.014 [0.000, 0.041] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| R1c: R1b + cap 14 | 0.014 [0.000, 0.041] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| R1d: R1b + Mender heal 14, hp 60 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

**Monotone ban-loss, mender (mean [95% CI])**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0: R0 (revision 0) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.015 [0.000, 0.045] |
| R1a: B + Sniper 40% vs bldg/wall/core | 0.021 [0.000, 0.050] | 0.012 [0.000, 0.033] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| R1b: R1a + sealed cores 50% + core hp x3 | 0.036 [0.002, 0.076] | 0.004 [0.000, 0.011] | 0.009 [0.000, 0.027] | 0.009 [0.000, 0.027] |
| R1c: R1b + cap 14 | 0.036 [0.002, 0.076] | 0.004 [0.000, 0.011] | 0.009 [0.000, 0.027] | 0.009 [0.000, 0.027] |
| R1d: R1b + Mender heal 14, hp 60 | 0.073 [0.024, 0.127] | 0.040 [0.007, 0.080] | 0.058 [0.024, 0.100] | 0.059 [0.017, 0.110] |

**Ban-loss estimators side by side, R0: R0 (revision 0)** (independent = two independent searches; monotone = full warm-started from the ablated best; mean [95% CI]; 'neg' = layouts with a negative independent ban-loss)

| unit / family | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| scout independent | 0.000 [0.000, 0.000] neg 0 | 0.000 [0.000, 0.000] neg 0 | 0.000 [0.000, 0.000] neg 0 | 0.000 [0.000, 0.000] neg 0 |
| scout monotone | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| sniper independent | 0.000 [0.000, 0.000] neg 0 | 0.013 [0.000, 0.040] neg 0 | 0.005 [0.000, 0.015] neg 0 | 0.012 [0.000, 0.032] neg 0 |
| sniper monotone | 0.000 [0.000, 0.000] | 0.013 [0.000, 0.040] | 0.005 [0.000, 0.015] | 0.009 [0.000, 0.026] |
| bulwark independent | 0.000 [0.000, 0.000] neg 0 | 0.004 [0.000, 0.012] neg 0 | 0.000 [0.000, 0.000] neg 0 | 0.009 [0.000, 0.026] neg 0 |
| bulwark monotone | 0.000 [0.000, 0.000] | 0.004 [0.000, 0.012] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| blaster independent | 0.000 [0.000, 0.000] neg 0 | 0.000 [0.000, 0.000] neg 0 | 0.000 [0.000, 0.000] neg 0 | 0.000 [0.000, 0.000] neg 0 |
| blaster monotone | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| mender independent | 0.000 [0.000, 0.000] neg 0 | 0.000 [0.000, 0.000] neg 0 | 0.000 [0.000, 0.000] neg 0 | 0.015 [0.000, 0.045] neg 0 |
| mender monotone | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.015 [0.000, 0.045] |

## Selection rule (4 families)

| arm | i_sniper_best_single_share_le_0.60 | ii_best_single_in_0.25_0.65 | iii_rejection_le_0.40 | iv_oracle_score_in_0.80_0.97 | v_mender_monotone_ban_loss | met |
|---|---|---|---|---|---|---|
| R0: R0 (revision 0) | no | no | no | no | no | 0/5 |
| R1a: B + Sniper 40% vs bldg/wall/core | no | no | no | yes | no | 1/5 |
| R1b: R1a + sealed cores 50% + core hp x3 | no | yes | no | yes | yes | 3/5 |
| R1c: R1b + cap 14 | no | yes | no | yes | yes | 3/5 |
| R1d: R1b + Mender heal 14, hp 60 | no | yes | no | no | yes | 2/5 |

Ranking: ['R1b', 'R1c', 'R1d', 'R1a']; chosen: **None** (best meets 3/5). Sensitivity, 3 familiar families only: chosen None (3/5), ranking ['R1b', 'R1c', 'R1a', 'R1d'].
