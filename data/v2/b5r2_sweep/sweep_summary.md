# B5-R2 exploration sweep (not a revision)

**Oracle score / breach**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R2a: R1b + Sniper cost 28 | 0.856 / 1.00 | 0.928 / 1.00 | 0.804 / 0.90 | 0.938 / 1.00 |
| R2b: R1b + Sniper cost 30 | 0.878 / 1.00 | 0.916 / 1.00 | 0.828 / 1.00 | 0.949 / 1.00 |
| R2c: R2a + island gap >= 6 | 0.856 / 1.00 | 1.000 / 1.00 | 0.804 / 0.90 | 0.938 / 1.00 |
| R2d: R2a + Mender heal 10, hp 50 | 0.901 / 1.00 | 0.952 / 1.00 | 0.855 / 0.95 | 0.946 / 1.00 |

**Best single-unit army (type, mean score)**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R2a: R1b + Sniper cost 28 | sniper 0.454 | bulwark 0.483 | sniper 0.288 | sniper 0.669 |
| R2b: R1b + Sniper cost 30 | sniper 0.454 | bulwark 0.483 | sniper 0.288 | sniper 0.669 |
| R2c: R2a + island gap >= 6 | sniper 0.454 | bulwark 0.615 | sniper 0.288 | sniper 0.669 |
| R2d: R2a + Mender heal 10, hp 50 | sniper 0.454 | bulwark 0.483 | sniper 0.280 | sniper 0.669 |

**Sniper share of best-single (per layout, ties split)**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R2a: R1b + Sniper cost 28 | 0.64 | 0.52 | 0.59 | 0.57 |
| R2b: R1b + Sniper cost 30 | 0.64 | 0.52 | 0.58 | 0.57 |
| R2c: R2a + island gap >= 6 | 0.64 | 0.33 | 0.59 | 0.57 |
| R2d: R2a + Mender heal 10, hp 50 | 0.64 | 0.52 | 0.54 | 0.57 |

**Single-type rejection / stat rejection rate (trivial / too_hard / evaluated)**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R2a: R1b + Sniper cost 28 | 9.1% (2/0/22) | 58.3% (28/0/48) | 20.0% (3/2/25) | 33.3% (10/0/30) |
| R2b: R1b + Sniper cost 30 | 9.1% (2/0/22) | 58.3% (28/0/48) | 20.0% (3/2/25) | 33.3% (10/0/30) |
| R2c: R2a + island gap >= 6 | 9.1% (2/0/22) | 89.8% (177/0/197) | 20.0% (3/2/25) | 33.3% (10/0/30) |
| R2d: R2a + Mender heal 10, hp 50 | 9.1% (2/0/22) | 58.3% (28/0/48) | 13.0% (3/0/23) | 33.3% (10/0/30) |

**Greedy score**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R2a: R1b + Sniper cost 28 | 0.392 | 0.483 | 0.061 | 0.633 |
| R2b: R1b + Sniper cost 30 | 0.392 | 0.483 | 0.061 | 0.633 |
| R2c: R2a + island gap >= 6 | 0.392 | 0.615 | 0.061 | 0.633 |
| R2d: R2a + Mender heal 10, hp 50 | 0.392 | 0.483 | 0.053 | 0.633 |

**Share of Oracle armies using > 12 units**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R2a: R1b + Sniper cost 28 | 0.00 | 0.00 | 0.00 | 0.00 |
| R2b: R1b + Sniper cost 30 | 0.05 | 0.00 | 0.05 | 0.00 |
| R2c: R2a + island gap >= 6 | 0.00 | 0.00 | 0.00 | 0.00 |
| R2d: R2a + Mender heal 10, hp 50 | 0.00 | 0.00 | 0.00 | 0.00 |

**Sealed-core share / mean walls**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R2a: R1b + Sniper cost 28 | 0.50 / 9.6 | 0.70 / 8.8 | 0.45 / 48.6 | 0.70 / 68.4 |
| R2b: R1b + Sniper cost 30 | 0.50 / 9.6 | 0.70 / 8.8 | 0.45 / 48.6 | 0.70 / 68.4 |
| R2c: R2a + island gap >= 6 | 0.50 / 9.6 | 0.50 / 7.3 | 0.45 / 48.6 | 0.70 / 68.4 |
| R2d: R2a + Mender heal 10, hp 50 | 0.50 / 9.6 | 0.70 / 8.8 | 0.35 / 47.9 | 0.70 / 68.4 |

**Monotone ban-loss, scout (mean [95% CI])**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R2a: R1b + Sniper cost 28 | 0.041 [0.000, 0.089] | 0.000 [0.000, 0.000] | 0.004 [0.000, 0.013] | 0.000 [0.000, 0.000] |
| R2b: R1b + Sniper cost 30 | 0.052 [0.007, 0.105] | 0.012 [0.000, 0.031] | 0.026 [0.000, 0.058] | 0.016 [0.000, 0.036] |
| R2c: R2a + island gap >= 6 | 0.041 [0.000, 0.089] | 0.000 [0.000, 0.000] | 0.004 [0.000, 0.013] | 0.000 [0.000, 0.000] |
| R2d: R2a + Mender heal 10, hp 50 | 0.030 [0.000, 0.073] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

**Monotone ban-loss, sniper (mean [95% CI])**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R2a: R1b + Sniper cost 28 | 0.049 [0.006, 0.104] | 0.037 [0.000, 0.086] | 0.075 [0.021, 0.143] | 0.000 [0.000, 0.000] |
| R2b: R1b + Sniper cost 30 | 0.049 [0.002, 0.107] | 0.041 [0.000, 0.094] | 0.090 [0.032, 0.160] | 0.000 [0.000, 0.000] |
| R2c: R2a + island gap >= 6 | 0.049 [0.006, 0.104] | 0.000 [0.000, 0.000] | 0.075 [0.021, 0.143] | 0.000 [0.000, 0.000] |
| R2d: R2a + Mender heal 10, hp 50 | 0.054 [0.001, 0.115] | 0.000 [0.000, 0.000] | 0.043 [0.000, 0.098] | 0.000 [0.000, 0.000] |

**Monotone ban-loss, bulwark (mean [95% CI])**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R2a: R1b + Sniper cost 28 | 0.106 [0.037, 0.188] | 0.071 [0.031, 0.118] | 0.067 [0.024, 0.118] | 0.029 [0.000, 0.074] |
| R2b: R1b + Sniper cost 30 | 0.085 [0.028, 0.157] | 0.087 [0.044, 0.137] | 0.069 [0.023, 0.121] | 0.021 [0.003, 0.042] |
| R2c: R2a + island gap >= 6 | 0.106 [0.037, 0.188] | 0.014 [0.000, 0.041] | 0.067 [0.024, 0.118] | 0.029 [0.000, 0.074] |
| R2d: R2a + Mender heal 10, hp 50 | 0.063 [0.012, 0.132] | 0.072 [0.024, 0.130] | 0.060 [0.020, 0.107] | 0.031 [0.005, 0.064] |

**Monotone ban-loss, blaster (mean [95% CI])**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R2a: R1b + Sniper cost 28 | 0.016 [0.000, 0.049] | 0.006 [0.000, 0.016] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| R2b: R1b + Sniper cost 30 | 0.000 [0.000, 0.000] | 0.004 [0.000, 0.013] | 0.032 [0.006, 0.063] | 0.009 [0.000, 0.026] |
| R2c: R2a + island gap >= 6 | 0.016 [0.000, 0.049] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| R2d: R2a + Mender heal 10, hp 50 | 0.001 [0.000, 0.003] | 0.003 [0.000, 0.008] | 0.006 [0.000, 0.017] | 0.013 [0.000, 0.039] |

**Monotone ban-loss, mender (mean [95% CI])**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R2a: R1b + Sniper cost 28 | 0.000 [0.000, 0.000] | 0.038 [0.009, 0.074] | 0.016 [0.000, 0.049] | 0.000 [0.000, 0.000] |
| R2b: R1b + Sniper cost 30 | 0.043 [0.000, 0.095] | 0.017 [0.000, 0.044] | 0.000 [0.000, 0.000] | 0.013 [0.000, 0.034] |
| R2c: R2a + island gap >= 6 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.016 [0.000, 0.049] | 0.000 [0.000, 0.000] |
| R2d: R2a + Mender heal 10, hp 50 | 0.015 [0.000, 0.044] | 0.045 [0.014, 0.082] | 0.034 [0.005, 0.073] | 0.016 [0.000, 0.045] |

**Sniper share of gold in Oracle armies (pooled gold / mean per-layout)**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R2a: R1b + Sniper cost 28 | 0.23 / 0.23 | 0.21 / 0.21 | 0.37 / 0.38 | 0.28 / 0.28 |
| R2b: R1b + Sniper cost 30 | 0.18 / 0.18 | 0.21 / 0.21 | 0.39 / 0.38 | 0.25 / 0.24 |
| R2c: R2a + island gap >= 6 | 0.23 / 0.23 | 0.13 / 0.14 | 0.37 / 0.38 | 0.28 / 0.28 |
| R2d: R2a + Mender heal 10, hp 50 | 0.25 / 0.25 | 0.17 / 0.17 | 0.30 / 0.30 | 0.22 / 0.23 |

**Pooled Sniper best-single share (familiar)**: R2a 0.58, R2b 0.58, R2c 0.52, R2d 0.56


## Selection rule

| arm | i_sniper_best_single_share_le_0.60 | ii_best_single_in_0.25_0.65 | iii_rejection_le_0.65 | iv_oracle_score_in_0.80_0.97 | v_mender_monotone_ban_loss | met |
|---|---|---|---|---|---|---|
| R2a: R1b + Sniper cost 28 | yes | no | yes | yes | yes | 4/5 |
| R2b: R1b + Sniper cost 30 | yes | no | yes | yes | no | 3/5 |
| R2c: R2a + island gap >= 6 | yes | no | no | no | no | 1/5 |
| R2d: R2a + Mender heal 10, hp 50 | yes | no | yes | yes | yes | 4/5 |

Ranking: ['R2a', 'R2d', 'R2b', 'R2c']; best: R2a; chosen (needs (ii), (iv), >= 4/5): **None**
