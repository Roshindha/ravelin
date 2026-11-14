# B5-R3 exploration sweep (not a revision)

**Oracle score / breach**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R3a: Blaster cost 30, hp 75, splash 70% | 0.892 / 1.00 | 0.961 / 1.00 | 0.910 / 1.00 | 0.976 / 1.00 |
| R3b: R3a + splash radius 2.0 | 0.925 / 1.00 | 0.975 / 1.00 | 0.933 / 1.00 | 0.993 / 1.00 |
| R3c: R3a + Blaster range 3.0 | 0.938 / 1.00 | 0.955 / 1.00 | 0.941 / 1.00 | 0.987 / 1.00 |
| R3d: R3b + Scout hp 20 | 0.949 / 1.00 | 0.987 / 1.00 | 0.933 / 1.00 | 0.989 / 1.00 |

**Best single-unit army (type, mean score)**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R3a: Blaster cost 30, hp 75, splash 70% | blaster 0.507 | bulwark 0.464 | sniper 0.281 | blaster 0.689 |
| R3b: R3a + splash radius 2.0 | blaster 0.607 | bulwark 0.479 | blaster 0.304 | blaster 0.705 |
| R3c: R3a + Blaster range 3.0 | blaster 0.560 | blaster 0.586 | blaster 0.498 | blaster 0.688 |
| R3d: R3b + Scout hp 20 | blaster 0.607 | bulwark 0.479 | sniper 0.281 | blaster 0.705 |

**Sniper share of best-single (per layout, ties split)**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R3a: Blaster cost 30, hp 75, splash 70% | 0.23 | 0.38 | 0.42 | 0.23 |
| R3b: R3a + splash radius 2.0 | 0.05 | 0.34 | 0.38 | 0.25 |
| R3c: R3a + Blaster range 3.0 | 0.03 | 0.23 | 0.20 | 0.12 |
| R3d: R3b + Scout hp 20 | 0.05 | 0.34 | 0.37 | 0.25 |

**Single-type rejection / stat rejection rate (trivial / too_hard / evaluated)**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R3a: Blaster cost 30, hp 75, splash 70% | 9.1% (2/0/22) | 62.3% (33/0/53) | 16.7% (4/0/24) | 51.2% (21/0/41) |
| R3b: R3a + splash radius 2.0 | 9.1% (2/0/22) | 63.0% (34/0/54) | 20.0% (4/1/25) | 56.5% (26/0/46) |
| R3c: R3a + Blaster range 3.0 | 9.1% (2/0/22) | 62.3% (33/0/53) | 13.0% (3/0/23) | 60.0% (30/0/50) |
| R3d: R3b + Scout hp 20 | 9.1% (2/0/22) | 63.0% (34/0/54) | 16.7% (4/0/24) | 56.5% (26/0/46) |

**Greedy score**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R3a: Blaster cost 30, hp 75, splash 70% | 0.392 | 0.464 | 0.055 | 0.571 |
| R3b: R3a + splash radius 2.0 | 0.392 | 0.479 | 0.061 | 0.614 |
| R3c: R3a + Blaster range 3.0 | 0.392 | 0.464 | 0.053 | 0.577 |
| R3d: R3b + Scout hp 20 | 0.392 | 0.479 | 0.055 | 0.614 |

**Share of Oracle armies using > 12 units**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R3a: Blaster cost 30, hp 75, splash 70% | 0.00 | 0.00 | 0.00 | 0.00 |
| R3b: R3a + splash radius 2.0 | 0.00 | 0.00 | 0.00 | 0.00 |
| R3c: R3a + Blaster range 3.0 | 0.00 | 0.00 | 0.00 | 0.00 |
| R3d: R3b + Scout hp 20 | 0.05 | 0.00 | 0.00 | 0.00 |

**Sealed-core share / mean walls**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R3a: Blaster cost 30, hp 75, splash 70% | 0.50 / 9.6 | 0.70 / 8.7 | 0.40 / 48.2 | 0.70 / 67.3 |
| R3b: R3a + splash radius 2.0 | 0.50 / 9.6 | 0.70 / 8.8 | 0.45 / 48.6 | 0.60 / 68.2 |
| R3c: R3a + Blaster range 3.0 | 0.50 / 9.6 | 0.70 / 8.7 | 0.35 / 47.9 | 0.60 / 68.0 |
| R3d: R3b + Scout hp 20 | 0.50 / 9.6 | 0.70 / 8.8 | 0.40 / 48.2 | 0.60 / 68.2 |

**Monotone ban-loss, scout (mean [95% CI])**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R3a: Blaster cost 30, hp 75, splash 70% | 0.010 [0.000, 0.031] | 0.000 [0.000, 0.000] | 0.014 [0.000, 0.042] | 0.000 [0.000, 0.000] |
| R3b: R3a + splash radius 2.0 | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.007 [0.000, 0.021] | 0.003 [0.000, 0.010] |
| R3c: R3a + Blaster range 3.0 | 0.010 [0.000, 0.031] | 0.004 [0.000, 0.013] | 0.014 [0.000, 0.042] | 0.000 [0.000, 0.000] |
| R3d: R3b + Scout hp 20 | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.007 [0.000, 0.021] | 0.000 [0.000, 0.000] |

**Monotone ban-loss, sniper (mean [95% CI])**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R3a: Blaster cost 30, hp 75, splash 70% | 0.000 [0.000, 0.000] | 0.031 [0.000, 0.087] | 0.075 [0.019, 0.143] | 0.000 [0.000, 0.000] |
| R3b: R3a + splash radius 2.0 | 0.000 [0.000, 0.000] | 0.013 [0.000, 0.039] | 0.065 [0.016, 0.125] | 0.000 [0.000, 0.000] |
| R3c: R3a + Blaster range 3.0 | 0.013 [0.000, 0.037] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| R3d: R3b + Scout hp 20 | 0.000 [0.000, 0.000] | 0.013 [0.000, 0.039] | 0.062 [0.014, 0.121] | 0.000 [0.000, 0.000] |

**Monotone ban-loss, bulwark (mean [95% CI])**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R3a: Blaster cost 30, hp 75, splash 70% | 0.068 [0.021, 0.124] | 0.039 [0.003, 0.085] | 0.066 [0.028, 0.111] | 0.017 [0.000, 0.047] |
| R3b: R3a + splash radius 2.0 | 0.077 [0.022, 0.140] | 0.029 [0.003, 0.071] | 0.060 [0.019, 0.113] | 0.008 [0.000, 0.020] |
| R3c: R3a + Blaster range 3.0 | 0.087 [0.034, 0.146] | 0.025 [0.000, 0.067] | 0.077 [0.020, 0.147] | 0.010 [0.000, 0.030] |
| R3d: R3b + Scout hp 20 | 0.095 [0.040, 0.158] | 0.019 [0.000, 0.056] | 0.079 [0.024, 0.148] | 0.004 [0.000, 0.013] |

**Monotone ban-loss, blaster (mean [95% CI])**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R3a: Blaster cost 30, hp 75, splash 70% | 0.030 [0.000, 0.077] | 0.020 [0.000, 0.051] | 0.048 [0.013, 0.092] | 0.030 [0.004, 0.065] |
| R3b: R3a + splash radius 2.0 | 0.037 [0.000, 0.082] | 0.015 [0.000, 0.042] | 0.055 [0.014, 0.109] | 0.046 [0.000, 0.100] |
| R3c: R3a + Blaster range 3.0 | 0.046 [0.002, 0.095] | 0.023 [0.000, 0.054] | 0.078 [0.031, 0.133] | 0.060 [0.017, 0.113] |
| R3d: R3b + Scout hp 20 | 0.045 [0.007, 0.090] | 0.010 [0.000, 0.028] | 0.065 [0.023, 0.120] | 0.043 [0.000, 0.096] |

**Monotone ban-loss, mender (mean [95% CI])**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R3a: Blaster cost 30, hp 75, splash 70% | 0.014 [0.000, 0.041] | 0.014 [0.000, 0.040] | 0.025 [0.000, 0.074] | 0.012 [0.000, 0.032] |
| R3b: R3a + splash radius 2.0 | 0.012 [0.000, 0.036] | 0.009 [0.000, 0.026] | 0.020 [0.000, 0.060] | 0.010 [0.000, 0.030] |
| R3c: R3a + Blaster range 3.0 | 0.008 [0.000, 0.024] | 0.013 [0.000, 0.037] | 0.009 [0.000, 0.027] | 0.000 [0.000, 0.000] |
| R3d: R3b + Scout hp 20 | 0.008 [0.000, 0.024] | 0.023 [0.000, 0.051] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

**Niche share (G3 definition, margin >= 0.03; pooled over the 80-layout sample / per family)**

| arm | unit | pooled | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|---|---|
| R3a | scout | 1.2% | 0% | 0% | 5% | 0% |
| R3a | sniper | 15.0% | 5% | 20% | 30% | 5% |
| R3a | bulwark | 5.0% | 0% | 15% | 0% | 5% |
| R3a | blaster | 31.2% | 40% | 15% | 40% | 30% |
| R3a | mender | 10.0% | 5% | 5% | 10% | 20% |
| R3b | scout | 2.5% | 0% | 0% | 10% | 0% |
| R3b | sniper | 12.5% | 0% | 20% | 25% | 5% |
| R3b | bulwark | 6.2% | 0% | 15% | 0% | 10% |
| R3b | blaster | 43.8% | 75% | 30% | 40% | 30% |
| R3b | mender | 6.2% | 5% | 0% | 10% | 10% |
| R3c | scout | 1.2% | 0% | 0% | 5% | 0% |
| R3c | sniper | 5.0% | 0% | 10% | 5% | 5% |
| R3c | bulwark | 3.8% | 0% | 5% | 0% | 10% |
| R3c | blaster | 55.0% | 75% | 55% | 50% | 40% |
| R3c | mender | 6.2% | 5% | 0% | 10% | 10% |
| R3d | scout | 2.5% | 0% | 0% | 10% | 0% |
| R3d | sniper | 12.5% | 0% | 20% | 25% | 5% |
| R3d | bulwark | 6.2% | 0% | 15% | 0% | 10% |
| R3d | blaster | 43.8% | 75% | 30% | 40% | 30% |
| R3d | mender | 6.2% | 5% | 0% | 10% | 10% |

**Gold share in Oracle armies (scout / sniper / bulwark / blaster / mender)**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R3a | 0.07 / 0.13 / 0.34 / 0.28 / 0.17 | 0.07 / 0.31 / 0.21 / 0.19 / 0.22 | 0.06 / 0.34 / 0.26 / 0.22 / 0.13 | 0.05 / 0.10 / 0.30 / 0.27 / 0.27 |
| R3b | 0.08 / 0.10 / 0.37 / 0.31 / 0.15 | 0.07 / 0.16 / 0.31 / 0.29 / 0.17 | 0.06 / 0.21 / 0.29 / 0.25 / 0.19 | 0.05 / 0.17 / 0.27 / 0.29 / 0.21 |
| R3c | 0.07 / 0.10 / 0.31 / 0.38 / 0.15 | 0.09 / 0.23 / 0.25 / 0.21 / 0.22 | 0.06 / 0.19 / 0.26 / 0.35 / 0.14 | 0.06 / 0.16 / 0.29 / 0.31 / 0.19 |
| R3d | 0.08 / 0.10 / 0.35 / 0.32 / 0.15 | 0.07 / 0.15 / 0.29 / 0.25 / 0.23 | 0.07 / 0.22 / 0.28 / 0.27 / 0.15 | 0.05 / 0.18 / 0.27 / 0.30 / 0.20 |

**Pooled familiar Sniper best-single share**: R3a 0.35, R3b 0.26, R3c 0.16, R3d 0.25

## Selection (SPEC 3.4 item 10)

| arm | R1_blaster_ban_loss | R2_oracle_in_0.80_0.97 | R3_best_single_in_0.25_0.65_and_sniper_share_le_0.60 | P1_blaster_niche_ge_8% | P2_scout_niche_ge_8% | P3_mender_bulwark_ban_loss_kept |
|---|---|---|---|---|---|---|
| R3a | yes | no | no | yes | no | no |
| R3b | yes | no | no | yes | no | no |
| R3c | yes | no | no | yes | no | no |
| R3d | yes | no | no | yes | no | no |

Arms meeting all three required items: []; ranked []; **chosen: None**
