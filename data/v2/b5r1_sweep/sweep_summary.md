# B5-R1 exploration sweep (not a revision)

**Oracle score / breach**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0 (reference) | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 |
| A: building/core/defender hp x1.4 | 1.000 / 1.00 | 0.988 / 1.00 | 1.000 / 1.00 | 0.987 / 1.00 |
| B: A + Lancer share ~40 % | 0.876 / 1.00 | 0.919 / 1.00 | 0.917 / 1.00 | 0.950 / 1.00 |
| C: B + Sniper cost 28 | 0.881 / 1.00 | 0.914 / 1.00 | 0.887 / 1.00 | 0.950 / 1.00 |
| D: defender count x1.25 + Lancer share ~40 % | 0.872 / 1.00 | 0.950 / 1.00 | 0.901 / 1.00 | 0.990 / 1.00 |
| E: building/core/defender hp x1.8 + Lancer share ~40 % | 0.874 / 1.00 | 0.903 / 1.00 | 0.899 / 1.00 | 0.976 / 1.00 |

**Best single-unit army (type, mean score)**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0 (reference) | sniper 0.663 | sniper 0.748 | sniper 0.547 | sniper 0.716 |
| A: building/core/defender hp x1.4 | sniper 0.665 | sniper 0.724 | sniper 0.486 | sniper 0.680 |
| B: A + Lancer share ~40 % | sniper 0.566 | sniper 0.620 | sniper 0.384 | sniper 0.682 |
| C: B + Sniper cost 28 | sniper 0.471 | sniper 0.498 | sniper 0.299 | sniper 0.674 |
| D: defender count x1.25 + Lancer share ~40 % | sniper 0.559 | sniper 0.569 | sniper 0.292 | sniper 0.746 |
| E: building/core/defender hp x1.8 + Lancer share ~40 % | sniper 0.524 | sniper 0.562 | sniper 0.323 | sniper 0.643 |

**Sniper share of best-single (per layout, ties split)**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0 (reference) | 0.84 | 0.62 | 0.88 | 0.50 |
| A: building/core/defender hp x1.4 | 0.88 | 0.52 | 0.72 | 0.38 |
| B: A + Lancer share ~40 % | 0.84 | 0.70 | 0.65 | 0.54 |
| C: B + Sniper cost 28 | 0.62 | 0.47 | 0.51 | 0.42 |
| D: defender count x1.25 + Lancer share ~40 % | 0.84 | 0.47 | 0.52 | 0.57 |
| E: building/core/defender hp x1.8 + Lancer share ~40 % | 0.78 | 0.63 | 0.60 | 0.43 |

**Mender Oracle ban-loss (paired mean [95% CI])**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0 (reference) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.015 [0.000, 0.045] |
| A: building/core/defender hp x1.4 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.013 [0.000, 0.034] |
| B: A + Lancer share ~40 % | -0.022 [-0.082, 0.036] | -0.026 [-0.067, 0.011] | -0.046 [-0.107, 0.009] | -0.002 [-0.033, 0.033] |
| C: B + Sniper cost 28 | 0.015 [-0.067, 0.095] | -0.031 [-0.068, -0.001] | -0.009 [-0.058, 0.043] | -0.001 [-0.031, 0.034] |
| D: defender count x1.25 + Lancer share ~40 % | -0.035 [-0.098, 0.027] | 0.025 [-0.007, 0.059] | -0.004 [-0.056, 0.049] | 0.015 [0.000, 0.043] |
| E: building/core/defender hp x1.8 + Lancer share ~40 % | -0.032 [-0.095, 0.026] | -0.036 [-0.088, 0.015] | -0.015 [-0.052, 0.026] | 0.035 [0.000, 0.076] |

**single_type_trivial rejection rate (any stat rejection; trivial / too_hard / evaluated)**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0 (reference) | 91.9% (228/0/248) | 99.4% (3119/0/3139) | 53.5% (23/0/43) | 74.7% (59/0/79) |
| A: building/core/defender hp x1.4 | 78.5% (73/0/93) | 96.7% (594/0/614) | 45.9% (17/0/37) | 65.5% (38/0/58) |
| B: A + Lancer share ~40 % | 37.5% (12/0/32) | 79.6% (78/0/98) | 16.7% (4/0/24) | 57.4% (27/0/47) |
| C: B + Sniper cost 28 | 37.5% (12/0/32) | 79.6% (78/0/98) | 16.7% (4/0/24) | 48.7% (19/0/39) |
| D: defender count x1.25 + Lancer share ~40 % | 4.8% (1/0/21) | 66.1% (39/0/59) | 4.8% (1/0/21) | 53.5% (23/0/43) |
| E: building/core/defender hp x1.8 + Lancer share ~40 % | 25.9% (7/0/27) | 69.7% (46/0/66) | 16.7% (3/1/24) | 50.0% (20/0/40) |

**Greedy score**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0 (reference) | 0.445 | 0.674 | 0.082 | 0.651 |
| A: building/core/defender hp x1.4 | 0.448 | 0.664 | 0.048 | 0.642 |
| B: A + Lancer share ~40 % | 0.403 | 0.470 | 0.026 | 0.582 |
| C: B + Sniper cost 28 | 0.403 | 0.470 | 0.026 | 0.631 |
| D: defender count x1.25 + Lancer share ~40 % | 0.413 | 0.523 | 0.031 | 0.636 |
| E: building/core/defender hp x1.8 + Lancer share ~40 % | 0.418 | 0.452 | 0.026 | 0.620 |

**Oracle armies using > 12 units**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0 (reference) | 0.00 | 0.00 | 0.00 | 0.00 |
| A: building/core/defender hp x1.4 | 0.00 | 0.05 | 0.00 | 0.00 |
| B: A + Lancer share ~40 % | 0.00 | 0.00 | 0.00 | 0.00 |
| C: B + Sniper cost 28 | 0.00 | 0.00 | 0.00 | 0.00 |
| D: defender count x1.25 + Lancer share ~40 % | 0.00 | 0.05 | 0.05 | 0.00 |
| E: building/core/defender hp x1.8 + Lancer share ~40 % | 0.00 | 0.05 | 0.00 | 0.00 |

**Defenders per layout / Lancer share**

| arm | spine_pods | archipelago | bunker_nest | labyrinth |
|---|---|---|---|---|
| R0 (reference) | 8.7 / 21% | 9.6 / 16% | 8.9 / 16% | 5.3 / 31% |
| A: building/core/defender hp x1.4 | 8.7 / 19% | 9.3 / 12% | 9.1 / 16% | 5.3 / 33% |
| B: A + Lancer share ~40 % | 8.3 / 39% | 9.1 / 34% | 9.1 / 38% | 5.2 / 46% |
| C: B + Sniper cost 28 | 8.3 / 39% | 9.1 / 34% | 9.1 / 38% | 5.1 / 43% |
| D: defender count x1.25 + Lancer share ~40 % | 10.2 / 39% | 10.8 / 37% | 10.9 / 39% | 6.3 / 43% |
| E: building/core/defender hp x1.8 + Lancer share ~40 % | 8.3 / 38% | 9.0 / 36% | 9.0 / 38% | 5.1 / 44% |

Sniper best-single share pooled over the 3 familiar families: R0 (reference) 0.78, A: building/core/defender hp x1.4 0.71, B: A + Lancer share ~40 % 0.73, C: B + Sniper cost 28 0.53, D: defender count x1.25 + Lancer share ~40 % 0.61, E: building/core/defender hp x1.8 + Lancer share ~40 % 0.67

## Selection rule (4 families)

| arm | 1_oracle_not_saturated | 2_best_single_in_0.25_0.65 | 3_sniper_best_single_share_le_0.60 | 4_rejection_rate_le_0.40 | 5_greedy_labyrinth_le_0.45 | met |
|---|---|---|---|---|---|---|
| R0 (reference) | no | no | no | no | no | 0/5 |
| A: building/core/defender hp x1.4 | no | no | no | no | no | 0/5 |
| B: A + Lancer share ~40 % | no | no | no | no | no | 0/5 |
| C: B + Sniper cost 28 | no | no | yes | no | no | 1/5 |
| D: defender count x1.25 + Lancer share ~40 % | no | no | no | no | no | 0/5 |
| E: building/core/defender hp x1.8 + Lancer share ~40 % | no | yes | no | no | no | 1/5 |

Ranking: ['E', 'C', 'A', 'B', 'D']; chosen: **None** (best meets 1/5). Sensitivity, 3 familiar families only: chosen None (2/5), ranking ['C', 'B', 'D', 'E', 'A'].
