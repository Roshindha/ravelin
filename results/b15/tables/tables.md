# B15 tables (FINAL, single session)

> session 3ccef86e0e2349d4af22e210d4b46a07; commits 7e3aa42

Primary interval: hierarchical bootstrap (seeds, then layouts; 10 000 resamples; percentile 95 %; rng 8080 per interval). Learner-only interval: SECONDARY, never decides 'holds'. A primary comparison holds only if its paired-difference primary CI excludes 0. Everything outside P1-P5 is exploratory (no multiplicity correction).

## Primary comparisons (P1-P5): 16 x 2 groups x 2 metrics

| id | comparison | group | metric | A | B | A - B | lo95 | hi95 | learner-only lo95 | learner-only hi95 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | PPO plain beats BestSingle | FINAL-familiar | breach | 0.3400 | 0.7000 | -0.3600 | -0.4800 | -0.2417 | -0.3967 | -0.3267 | not established |
| P1 | PPO plain beats BestSingle | FINAL-familiar | score | 0.4370 | 0.3810 | 0.0560 | -0.0202 | 0.1335 | 0.0370 | 0.0735 | not established |
| P1 | PPO plain beats BestSingle | FINAL-transfer | breach | 0.5967 | 1.0000 | -0.4033 | -0.4967 | -0.3100 | -0.4417 | -0.3633 | not established |
| P1 | PPO plain beats BestSingle | FINAL-transfer | score | 0.5166 | 0.6659 | -0.1493 | -0.2066 | -0.0918 | -0.1760 | -0.1242 | not established |
| P1 | PPO+D4 beats BestSingle | FINAL-familiar | breach | 0.3483 | 0.7000 | -0.3517 | -0.4767 | -0.2250 | -0.3883 | -0.3133 | not established |
| P1 | PPO+D4 beats BestSingle | FINAL-familiar | score | 0.4545 | 0.3810 | 0.0735 | -0.0059 | 0.1538 | 0.0543 | 0.0931 | not established |
| P1 | PPO+D4 beats BestSingle | FINAL-transfer | breach | 0.6350 | 1.0000 | -0.3650 | -0.4683 | -0.2717 | -0.4117 | -0.3233 | not established |
| P1 | PPO+D4 beats BestSingle | FINAL-transfer | score | 0.5178 | 0.6659 | -0.1481 | -0.2139 | -0.0829 | -0.1874 | -0.1132 | not established |
| P1 | DDQN beats BestSingle | FINAL-familiar | breach | 0.3900 | 0.7000 | -0.3100 | -0.4317 | -0.1867 | -0.3417 | -0.2783 | not established |
| P1 | DDQN beats BestSingle | FINAL-familiar | score | 0.4335 | 0.3810 | 0.0525 | -0.0273 | 0.1341 | 0.0378 | 0.0693 | not established |
| P1 | DDQN beats BestSingle | FINAL-transfer | breach | 0.6517 | 1.0000 | -0.3483 | -0.4383 | -0.2617 | -0.3800 | -0.3167 | not established |
| P1 | DDQN beats BestSingle | FINAL-transfer | score | 0.5628 | 0.6659 | -0.1031 | -0.1585 | -0.0489 | -0.1244 | -0.0828 | not established |
| P2 | PPO plain beats Greedy | FINAL-familiar | breach | 0.3400 | 0.5833 | -0.2433 | -0.3600 | -0.1283 | -0.2800 | -0.2100 | not established |
| P2 | PPO plain beats Greedy | FINAL-familiar | score | 0.4370 | 0.3380 | 0.0990 | 0.0201 | 0.1766 | 0.0801 | 0.1165 | holds |
| P2 | PPO plain beats Greedy | FINAL-transfer | breach | 0.5967 | 0.9500 | -0.3533 | -0.4617 | -0.2417 | -0.3917 | -0.3133 | not established |
| P2 | PPO plain beats Greedy | FINAL-transfer | score | 0.5166 | 0.6002 | -0.0836 | -0.1455 | -0.0222 | -0.1103 | -0.0585 | not established |
| P2 | PPO+D4 beats Greedy | FINAL-familiar | breach | 0.3483 | 0.5833 | -0.2350 | -0.3583 | -0.1117 | -0.2717 | -0.1967 | not established |
| P2 | PPO+D4 beats Greedy | FINAL-familiar | score | 0.4545 | 0.3380 | 0.1165 | 0.0381 | 0.1939 | 0.0973 | 0.1361 | holds |
| P2 | PPO+D4 beats Greedy | FINAL-transfer | breach | 0.6350 | 0.9500 | -0.3150 | -0.4267 | -0.2050 | -0.3617 | -0.2733 | not established |
| P2 | PPO+D4 beats Greedy | FINAL-transfer | score | 0.5178 | 0.6002 | -0.0824 | -0.1550 | -0.0121 | -0.1217 | -0.0475 | not established |
| P2 | DDQN beats Greedy | FINAL-familiar | breach | 0.3900 | 0.5833 | -0.1933 | -0.3133 | -0.0717 | -0.2250 | -0.1617 | not established |
| P2 | DDQN beats Greedy | FINAL-familiar | score | 0.4335 | 0.3380 | 0.0955 | 0.0134 | 0.1762 | 0.0808 | 0.1123 | holds |
| P2 | DDQN beats Greedy | FINAL-transfer | breach | 0.6517 | 0.9500 | -0.2983 | -0.4033 | -0.1917 | -0.3300 | -0.2667 | not established |
| P2 | DDQN beats Greedy | FINAL-transfer | score | 0.5628 | 0.6002 | -0.0374 | -0.1014 | 0.0248 | -0.0586 | -0.0171 | not established |
| P3 | PPO+D4 beats PPO plain | FINAL-familiar | breach | 0.3483 | 0.3400 | 0.0083 | -0.0733 | 0.0917 | -0.0417 | 0.0617 | not established |
| P3 | PPO+D4 beats PPO plain | FINAL-familiar | score | 0.4545 | 0.4370 | 0.0175 | -0.0277 | 0.0622 | -0.0089 | 0.0446 | not established |
| P3 | PPO+D4 beats PPO plain | FINAL-transfer | breach | 0.6350 | 0.5967 | 0.0383 | -0.0517 | 0.1283 | -0.0233 | 0.0967 | not established |
| P3 | PPO+D4 beats PPO plain | FINAL-transfer | score | 0.5178 | 0.5166 | 0.0012 | -0.0627 | 0.0637 | -0.0455 | 0.0456 | not established |
| P4 | Planner on PPO plain N=8 beats the strongest equal-N control (RPS-full-8) | FINAL-familiar | breach | 0.6133 | 0.4533 | 0.1600 | 0.0850 | 0.2367 | 0.1233 | 0.1967 | holds |
| P4 | Planner on PPO plain N=8 beats the strongest equal-N control (RPS-full-8) | FINAL-familiar | score | 0.6591 | 0.4960 | 0.1631 | 0.1286 | 0.1980 | 0.1436 | 0.1822 | holds |
| P4 | Planner on PPO plain N=8 beats the strongest equal-N control (RAS-8) | FINAL-transfer | breach | 0.8733 | 0.9767 | -0.1033 | -0.1700 | -0.0417 | -0.1300 | -0.0733 | not established |
| P4 | Planner on PPO plain N=8 beats the strongest equal-N control (RAS-8) | FINAL-transfer | score | 0.7683 | 0.7175 | 0.0508 | 0.0035 | 0.0958 | 0.0266 | 0.0748 | holds |
| P4 | Planner on PPO plain N=32 beats the strongest equal-N control (RPS-full-32) | FINAL-familiar | breach | 0.7533 | 0.6500 | 0.1033 | 0.0250 | 0.1833 | 0.0633 | 0.1417 | holds |
| P4 | Planner on PPO plain N=32 beats the strongest equal-N control (RPS-full-32) | FINAL-familiar | score | 0.7538 | 0.6390 | 0.1148 | 0.0852 | 0.1432 | 0.0980 | 0.1298 | holds |
| P4 | Planner on PPO plain N=32 beats the strongest equal-N control (RAS-32) | FINAL-transfer | breach | 0.9567 | 0.9950 | -0.0383 | -0.0833 | -0.0033 | -0.0500 | -0.0250 | not established |
| P4 | Planner on PPO plain N=32 beats the strongest equal-N control (RAS-32) | FINAL-transfer | score | 0.8683 | 0.8306 | 0.0377 | 0.0050 | 0.0714 | 0.0234 | 0.0520 | holds |
| P4 | Planner on PPO plain N=128 beats the strongest equal-N control (RPS-full-128) | FINAL-familiar | breach | 0.8583 | 0.7900 | 0.0683 | 0.0133 | 0.1300 | 0.0483 | 0.0867 | holds |
| P4 | Planner on PPO plain N=128 beats the strongest equal-N control (RPS-full-128) | FINAL-familiar | score | 0.8321 | 0.7457 | 0.0863 | 0.0616 | 0.1105 | 0.0766 | 0.0957 | holds |
| P4 | Planner on PPO plain N=128 beats the strongest equal-N control (RAS-128) | FINAL-transfer | breach | 0.9850 | 0.9967 | -0.0117 | -0.0383 | 0.0083 | -0.0217 | -0.0017 | not established |
| P4 | Planner on PPO plain N=128 beats the strongest equal-N control (RAS-128) | FINAL-transfer | score | 0.9267 | 0.9104 | 0.0163 | -0.0105 | 0.0457 | 0.0070 | 0.0253 | not established |
| P4 | Planner on PPO+D4 N=8 beats the strongest equal-N control (RPS-full-8) | FINAL-familiar | breach | 0.5500 | 0.4533 | 0.0967 | 0.0283 | 0.1667 | 0.0733 | 0.1217 | holds |
| P4 | Planner on PPO+D4 N=8 beats the strongest equal-N control (RPS-full-8) | FINAL-familiar | score | 0.6155 | 0.4960 | 0.1196 | 0.0913 | 0.1488 | 0.1075 | 0.1319 | holds |
| P4 | Planner on PPO+D4 N=8 beats the strongest equal-N control (RAS-8) | FINAL-transfer | breach | 0.8350 | 0.9767 | -0.1417 | -0.2250 | -0.0650 | -0.1733 | -0.1133 | not established |
| P4 | Planner on PPO+D4 N=8 beats the strongest equal-N control (RAS-8) | FINAL-transfer | score | 0.7169 | 0.7175 | -0.0006 | -0.0557 | 0.0528 | -0.0311 | 0.0281 | not established |
| P4 | Planner on PPO+D4 N=32 beats the strongest equal-N control (RPS-full-32) | FINAL-familiar | breach | 0.6883 | 0.6500 | 0.0383 | -0.0333 | 0.1133 | 0.0083 | 0.0667 | not established |
| P4 | Planner on PPO+D4 N=32 beats the strongest equal-N control (RPS-full-32) | FINAL-familiar | score | 0.7151 | 0.6390 | 0.0761 | 0.0481 | 0.1047 | 0.0653 | 0.0872 | holds |
| P4 | Planner on PPO+D4 N=32 beats the strongest equal-N control (RAS-32) | FINAL-transfer | breach | 0.9267 | 0.9950 | -0.0683 | -0.1283 | -0.0200 | -0.0883 | -0.0483 | not established |
| P4 | Planner on PPO+D4 N=32 beats the strongest equal-N control (RAS-32) | FINAL-transfer | score | 0.8192 | 0.8306 | -0.0114 | -0.0544 | 0.0304 | -0.0350 | 0.0112 | not established |
| P4 | Planner on PPO+D4 N=128 beats the strongest equal-N control (RPS-full-128) | FINAL-familiar | breach | 0.7717 | 0.7900 | -0.0183 | -0.0767 | 0.0433 | -0.0467 | 0.0083 | not established |
| P4 | Planner on PPO+D4 N=128 beats the strongest equal-N control (RPS-full-128) | FINAL-familiar | score | 0.7831 | 0.7457 | 0.0374 | 0.0113 | 0.0631 | 0.0257 | 0.0490 | holds |
| P4 | Planner on PPO+D4 N=128 beats the strongest equal-N control (RAS-128) | FINAL-transfer | breach | 0.9533 | 0.9967 | -0.0433 | -0.0917 | -0.0050 | -0.0583 | -0.0300 | not established |
| P4 | Planner on PPO+D4 N=128 beats the strongest equal-N control (RAS-128) | FINAL-transfer | score | 0.8785 | 0.9104 | -0.0320 | -0.0664 | 0.0030 | -0.0492 | -0.0138 | not established |
| P5 | PPO plain + TTA beats PPO plain | FINAL-familiar | breach | 0.4633 | 0.3400 | 0.1233 | 0.0567 | 0.1917 | 0.1050 | 0.1417 | holds |
| P5 | PPO plain + TTA beats PPO plain | FINAL-familiar | score | 0.5299 | 0.4370 | 0.0929 | 0.0579 | 0.1295 | 0.0819 | 0.1042 | holds |
| P5 | PPO plain + TTA beats PPO plain | FINAL-transfer | breach | 0.7583 | 0.5967 | 0.1617 | 0.0867 | 0.2367 | 0.1217 | 0.1950 | holds |
| P5 | PPO plain + TTA beats PPO plain | FINAL-transfer | score | 0.6136 | 0.5166 | 0.0970 | 0.0490 | 0.1429 | 0.0702 | 0.1199 | holds |
| P5 | PPO+D4 + TTA beats PPO+D4 | FINAL-familiar | breach | 0.4317 | 0.3483 | 0.0833 | 0.0183 | 0.1517 | 0.0483 | 0.1117 | holds |
| P5 | PPO+D4 + TTA beats PPO+D4 | FINAL-familiar | score | 0.5213 | 0.4545 | 0.0668 | 0.0281 | 0.1054 | 0.0412 | 0.0904 | holds |
| P5 | PPO+D4 + TTA beats PPO+D4 | FINAL-transfer | breach | 0.7467 | 0.6350 | 0.1117 | 0.0567 | 0.1700 | 0.0883 | 0.1317 | holds |
| P5 | PPO+D4 + TTA beats PPO+D4 | FINAL-transfer | score | 0.5958 | 0.5178 | 0.0780 | 0.0409 | 0.1162 | 0.0573 | 0.0971 | holds |
| P5 | DDQN + TTA beats DDQN | FINAL-familiar | breach | 0.5050 | 0.3900 | 0.1150 | 0.0483 | 0.1867 | 0.0800 | 0.1500 | holds |
| P5 | DDQN + TTA beats DDQN | FINAL-familiar | score | 0.5198 | 0.4335 | 0.0863 | 0.0452 | 0.1277 | 0.0607 | 0.1120 | holds |
| P5 | DDQN + TTA beats DDQN | FINAL-transfer | breach | 0.7917 | 0.6517 | 0.1400 | 0.0700 | 0.2100 | 0.0983 | 0.1783 | holds |
| P5 | DDQN + TTA beats DDQN | FINAL-transfer | score | 0.6470 | 0.5628 | 0.0841 | 0.0464 | 0.1235 | 0.0655 | 0.1045 | holds |

## Strongest equal-N control (rule: highest mean true score at that group and N; all three shown)

| group | N | ras_score | rps_score | rpsfull_score | ras_breach | rps_breach | rpsfull_breach | strongest |
|---|---|---|---|---|---|---|---|---|
| FINAL-familiar | 8 | 0.4631 | 0.3478 | 0.4960 | 0.7650 | 0.2600 | 0.4533 | rpsfull |
| FINAL-familiar | 32 | 0.5543 | 0.5124 | 0.6390 | 0.8450 | 0.5083 | 0.6500 | rpsfull |
| FINAL-familiar | 128 | 0.6390 | 0.6363 | 0.7457 | 0.8883 | 0.6717 | 0.7900 | rpsfull |
| FINAL-transfer | 8 | 0.7175 | 0.4621 | 0.6277 | 0.9767 | 0.6000 | 0.7717 | ras |
| FINAL-transfer | 32 | 0.8306 | 0.6595 | 0.7762 | 0.9950 | 0.8583 | 0.9250 | ras |
| FINAL-transfer | 128 | 0.9104 | 0.7907 | 0.8768 | 0.9967 | 0.9400 | 0.9700 | ras |

## Conditions: FINAL-familiar (60 layouts)

| condition | seeds | breach rate [95% CI] | breach learner-only CI | mean score [95% CI] | score learner-only CI | mean D | bought | deployed |
|---|---|---|---|---|---|---|---|---|
| Random | 1 | 0.060 [0.033, 0.090] | [0.060, 0.060] | 0.107 [0.088, 0.126] | [0.107, 0.107] | 0.138 | 4.7 | 4.5 |
| Greedy | 1 | 0.583 [0.450, 0.700] | [0.583, 0.583] | 0.338 [0.275, 0.403] | [0.338, 0.338] | 0.174 | 8.0 | 8.0 |
| BestSingle | 1 | 0.700 [0.583, 0.817] | [0.700, 0.700] | 0.381 [0.312, 0.447] | [0.381, 0.381] | 0.168 | 8.0 | 8.0 |
| BestSingle-perlayout (diagnostic) | 1 | 0.767 [0.650, 0.867] | [0.767, 0.767] | 0.461 [0.403, 0.518] | [0.461, 0.461] | 0.258 | 10.2 | 10.2 |
| BestSingle-hindsight (diagnostic) | 1 | 0.683 [0.567, 0.800] | [0.683, 0.683] | 0.396 [0.326, 0.464] | [0.396, 0.396] | 0.205 | 8.0 | 8.0 |
| Oracle (upper reference, ~400 sims) | 1 | 1.000 [1.000, 1.000] | [1.000, 1.000] | 0.892 [0.857, 0.924] | [0.892, 0.892] | 0.820 | 9.4 | 9.4 |
| PPO plain | 10 | 0.340 [0.252, 0.432] | [0.303, 0.373] | 0.437 [0.390, 0.486] | [0.418, 0.455] | 0.502 | 8.7 | 8.7 |
| PPO+D4 | 10 | 0.348 [0.255, 0.447] | [0.312, 0.387] | 0.454 [0.406, 0.505] | [0.435, 0.474] | 0.525 | 9.4 | 9.4 |
| DDQN | 10 | 0.390 [0.295, 0.490] | [0.358, 0.422] | 0.434 [0.381, 0.486] | [0.419, 0.450] | 0.463 | 10.2 | 10.2 |
| PPO plain + TTA | 10 | 0.463 [0.363, 0.565] | [0.438, 0.485] | 0.530 [0.481, 0.581] | [0.514, 0.549] | 0.574 | 8.4 | 8.4 |
| PPO+D4 + TTA | 10 | 0.432 [0.323, 0.542] | [0.412, 0.452] | 0.521 [0.469, 0.574] | [0.510, 0.534] | 0.581 | 9.1 | 9.1 |
| DDQN + TTA | 10 | 0.505 [0.402, 0.610] | [0.472, 0.533] | 0.520 [0.465, 0.575] | [0.496, 0.540] | 0.530 | 9.9 | 9.9 |
| RAS-8 | 10 | 0.765 [0.667, 0.857] | [0.752, 0.778] | 0.463 [0.401, 0.524] | [0.452, 0.475] | 0.262 | 8.5 | 8.5 |
| RAS-32 | 10 | 0.845 [0.757, 0.927] | [0.835, 0.853] | 0.554 [0.490, 0.616] | [0.549, 0.559] | 0.360 | 8.9 | 8.9 |
| RAS-128 | 10 | 0.888 [0.808, 0.960] | [0.880, 0.897] | 0.639 [0.571, 0.705] | [0.632, 0.645] | 0.473 | 9.3 | 9.3 |
| RPS-8 | 10 | 0.260 [0.178, 0.345] | [0.238, 0.280] | 0.348 [0.317, 0.379] | [0.342, 0.353] | 0.406 | 7.3 | 7.2 |
| RPS-32 | 10 | 0.508 [0.403, 0.612] | [0.488, 0.532] | 0.512 [0.477, 0.548] | [0.505, 0.521] | 0.515 | 7.9 | 7.9 |
| RPS-128 | 10 | 0.672 [0.565, 0.773] | [0.652, 0.690] | 0.636 [0.598, 0.674] | [0.631, 0.642] | 0.613 | 8.4 | 8.4 |
| RPS-full-8 | 10 | 0.453 [0.357, 0.552] | [0.435, 0.470] | 0.496 [0.463, 0.531] | [0.487, 0.505] | 0.524 | 10.0 | 10.0 |
| RPS-full-32 | 10 | 0.650 [0.542, 0.753] | [0.628, 0.673] | 0.639 [0.601, 0.677] | [0.635, 0.644] | 0.632 | 10.0 | 10.0 |
| RPS-full-128 | 10 | 0.790 [0.702, 0.873] | [0.777, 0.805] | 0.746 [0.707, 0.785] | [0.740, 0.752] | 0.716 | 10.0 | 10.0 |
| Planner on PPO plain, N=8 | 10 | 0.613 [0.512, 0.710] | [0.582, 0.647] | 0.659 [0.609, 0.708] | [0.642, 0.676] | 0.690 | 8.7 | 8.7 |
| Planner on PPO plain, N=32 | 10 | 0.753 [0.657, 0.840] | [0.720, 0.785] | 0.754 [0.706, 0.799] | [0.737, 0.768] | 0.754 | 8.7 | 8.7 |
| Planner on PPO plain, N=128 | 10 | 0.858 [0.782, 0.925] | [0.843, 0.872] | 0.832 [0.790, 0.871] | [0.825, 0.839] | 0.815 | 8.7 | 8.7 |
| Planner on PPO+D4, N=8 | 10 | 0.550 [0.448, 0.652] | [0.533, 0.567] | 0.616 [0.572, 0.658] | [0.608, 0.624] | 0.659 | 9.3 | 9.3 |
| Planner on PPO+D4, N=32 | 10 | 0.688 [0.590, 0.785] | [0.668, 0.708] | 0.715 [0.670, 0.760] | [0.705, 0.725] | 0.733 | 9.2 | 9.2 |
| Planner on PPO+D4, N=128 | 10 | 0.772 [0.680, 0.857] | [0.747, 0.795] | 0.783 [0.740, 0.824] | [0.774, 0.793] | 0.791 | 9.3 | 9.3 |

## Conditions: FINAL-transfer (60 layouts)

| condition | seeds | breach rate [95% CI] | breach learner-only CI | mean score [95% CI] | score learner-only CI | mean D | bought | deployed |
|---|---|---|---|---|---|---|---|---|
| Random | 1 | 0.197 [0.140, 0.253] | [0.197, 0.197] | 0.158 [0.128, 0.188] | [0.158, 0.158] | 0.133 | 5.2 | 5.1 |
| Greedy | 1 | 0.950 [0.883, 1.000] | [0.950, 0.950] | 0.600 [0.561, 0.640] | [0.600, 0.600] | 0.367 | 8.0 | 8.0 |
| BestSingle | 1 | 1.000 [1.000, 1.000] | [1.000, 1.000] | 0.666 [0.626, 0.704] | [0.666, 0.666] | 0.443 | 8.0 | 8.0 |
| BestSingle-perlayout (diagnostic) | 1 | 1.000 [1.000, 1.000] | [1.000, 1.000] | 0.708 [0.671, 0.744] | [0.708, 0.708] | 0.513 | 8.2 | 8.2 |
| BestSingle-hindsight (diagnostic) | 1 | 1.000 [1.000, 1.000] | [1.000, 1.000] | 0.666 [0.626, 0.704] | [0.666, 0.666] | 0.443 | 8.0 | 8.0 |
| Oracle (upper reference, ~400 sims) | 1 | 1.000 [1.000, 1.000] | [1.000, 1.000] | 0.964 [0.938, 0.984] | [0.964, 0.964] | 0.939 | 8.9 | 8.9 |
| PPO plain | 10 | 0.597 [0.503, 0.690] | [0.558, 0.637] | 0.517 [0.462, 0.571] | [0.490, 0.542] | 0.463 | 8.7 | 8.7 |
| PPO+D4 | 10 | 0.635 [0.532, 0.728] | [0.588, 0.677] | 0.518 [0.455, 0.577] | [0.478, 0.553] | 0.440 | 9.2 | 9.2 |
| DDQN | 10 | 0.652 [0.562, 0.738] | [0.620, 0.683] | 0.563 [0.512, 0.611] | [0.542, 0.583] | 0.504 | 10.0 | 9.9 |
| PPO plain + TTA | 10 | 0.758 [0.675, 0.838] | [0.727, 0.790] | 0.614 [0.563, 0.663] | [0.595, 0.634] | 0.517 | 8.3 | 8.3 |
| PPO+D4 + TTA | 10 | 0.747 [0.643, 0.840] | [0.698, 0.793] | 0.596 [0.526, 0.663] | [0.550, 0.638] | 0.495 | 9.0 | 9.0 |
| DDQN + TTA | 10 | 0.792 [0.693, 0.878] | [0.738, 0.840] | 0.647 [0.594, 0.699] | [0.615, 0.677] | 0.550 | 9.5 | 9.5 |
| RAS-8 | 10 | 0.977 [0.942, 0.998] | [0.968, 0.985] | 0.717 [0.673, 0.760] | [0.704, 0.732] | 0.545 | 8.7 | 8.7 |
| RAS-32 | 10 | 0.995 [0.980, 1.000] | [0.990, 1.000] | 0.831 [0.792, 0.867] | [0.824, 0.837] | 0.721 | 9.0 | 9.0 |
| RAS-128 | 10 | 0.997 [0.985, 1.000] | [0.992, 1.000] | 0.910 [0.879, 0.939] | [0.906, 0.915] | 0.853 | 9.2 | 9.2 |
| RPS-8 | 10 | 0.600 [0.505, 0.692] | [0.562, 0.640] | 0.462 [0.414, 0.510] | [0.442, 0.481] | 0.370 | 7.2 | 7.1 |
| RPS-32 | 10 | 0.858 [0.788, 0.918] | [0.832, 0.885] | 0.659 [0.621, 0.698] | [0.648, 0.670] | 0.527 | 7.8 | 7.8 |
| RPS-128 | 10 | 0.940 [0.888, 0.982] | [0.925, 0.953] | 0.791 [0.754, 0.826] | [0.783, 0.799] | 0.691 | 8.3 | 8.2 |
| RPS-full-8 | 10 | 0.772 [0.695, 0.843] | [0.755, 0.788] | 0.628 [0.581, 0.672] | [0.614, 0.642] | 0.532 | 10.1 | 10.1 |
| RPS-full-32 | 10 | 0.925 [0.872, 0.968] | [0.912, 0.940] | 0.776 [0.738, 0.813] | [0.771, 0.781] | 0.677 | 10.0 | 10.0 |
| RPS-full-128 | 10 | 0.970 [0.928, 0.997] | [0.962, 0.977] | 0.877 [0.845, 0.906] | [0.871, 0.882] | 0.815 | 10.0 | 10.0 |
| Planner on PPO plain, N=8 | 10 | 0.873 [0.810, 0.928] | [0.848, 0.902] | 0.768 [0.726, 0.807] | [0.749, 0.788] | 0.698 | 8.7 | 8.7 |
| Planner on PPO plain, N=32 | 10 | 0.957 [0.913, 0.988] | [0.947, 0.968] | 0.868 [0.835, 0.899] | [0.855, 0.881] | 0.809 | 8.6 | 8.6 |
| Planner on PPO plain, N=128 | 10 | 0.985 [0.960, 1.000] | [0.977, 0.993] | 0.927 [0.904, 0.947] | [0.919, 0.935] | 0.888 | 8.7 | 8.7 |
| Planner on PPO+D4, N=8 | 10 | 0.835 [0.757, 0.903] | [0.805, 0.862] | 0.717 [0.667, 0.763] | [0.691, 0.742] | 0.638 | 9.1 | 9.1 |
| Planner on PPO+D4, N=32 | 10 | 0.927 [0.870, 0.972] | [0.907, 0.945] | 0.819 [0.778, 0.858] | [0.797, 0.841] | 0.748 | 9.1 | 9.1 |
| Planner on PPO+D4, N=128 | 10 | 0.953 [0.907, 0.988] | [0.940, 0.967] | 0.878 [0.844, 0.910] | [0.862, 0.897] | 0.829 | 9.1 | 9.1 |

## Conditions: POOLED (descriptive) (120 layouts)

| condition | seeds | breach rate [95% CI] | breach learner-only CI | mean score [95% CI] | score learner-only CI | mean D | bought | deployed |
|---|---|---|---|---|---|---|---|---|
| Random | 1 | 0.128 [0.095, 0.163] | [0.128, 0.128] | 0.132 [0.114, 0.151] | [0.132, 0.132] | 0.135 | 4.9 | 4.8 |
| Greedy | 1 | 0.767 [0.692, 0.842] | [0.767, 0.767] | 0.469 [0.424, 0.512] | [0.469, 0.469] | 0.271 | 8.0 | 8.0 |
| BestSingle | 1 | 0.850 [0.783, 0.908] | [0.850, 0.850] | 0.523 [0.476, 0.568] | [0.523, 0.523] | 0.306 | 8.0 | 8.0 |
| BestSingle-perlayout (diagnostic) | 1 | 0.883 [0.825, 0.942] | [0.883, 0.883] | 0.585 [0.544, 0.623] | [0.585, 0.585] | 0.385 | 9.2 | 9.2 |
| BestSingle-hindsight (diagnostic) | 1 | 0.842 [0.775, 0.900] | [0.842, 0.842] | 0.531 [0.484, 0.575] | [0.531, 0.531] | 0.324 | 8.0 | 8.0 |
| Oracle (upper reference, ~400 sims) | 1 | 1.000 [1.000, 1.000] | [1.000, 1.000] | 0.928 [0.906, 0.949] | [0.928, 0.928] | 0.880 | 9.2 | 9.2 |
| PPO plain | 10 | 0.468 [0.402, 0.534] | [0.447, 0.489] | 0.477 [0.441, 0.512] | [0.463, 0.490] | 0.482 | 8.7 | 8.7 |
| PPO+D4 | 10 | 0.492 [0.416, 0.567] | [0.457, 0.526] | 0.486 [0.446, 0.526] | [0.461, 0.510] | 0.482 | 9.3 | 9.3 |
| DDQN | 10 | 0.521 [0.451, 0.590] | [0.496, 0.544] | 0.498 [0.460, 0.536] | [0.484, 0.512] | 0.483 | 10.1 | 10.1 |
| PPO plain + TTA | 10 | 0.611 [0.541, 0.682] | [0.591, 0.632] | 0.572 [0.535, 0.609] | [0.558, 0.588] | 0.546 | 8.4 | 8.4 |
| PPO+D4 + TTA | 10 | 0.589 [0.512, 0.667] | [0.562, 0.616] | 0.559 [0.515, 0.602] | [0.535, 0.582] | 0.538 | 9.1 | 9.1 |
| DDQN + TTA | 10 | 0.648 [0.573, 0.724] | [0.612, 0.680] | 0.583 [0.542, 0.625] | [0.558, 0.605] | 0.540 | 9.7 | 9.7 |
| RAS-8 | 10 | 0.871 [0.815, 0.922] | [0.864, 0.877] | 0.590 [0.545, 0.633] | [0.583, 0.599] | 0.403 | 8.6 | 8.6 |
| RAS-32 | 10 | 0.920 [0.873, 0.963] | [0.914, 0.926] | 0.692 [0.647, 0.736] | [0.688, 0.697] | 0.541 | 8.9 | 8.9 |
| RAS-128 | 10 | 0.943 [0.900, 0.978] | [0.938, 0.947] | 0.775 [0.729, 0.818] | [0.771, 0.778] | 0.663 | 9.2 | 9.2 |
| RPS-8 | 10 | 0.430 [0.359, 0.501] | [0.402, 0.458] | 0.405 [0.374, 0.436] | [0.393, 0.416] | 0.388 | 7.2 | 7.1 |
| RPS-32 | 10 | 0.683 [0.613, 0.751] | [0.663, 0.704] | 0.586 [0.556, 0.615] | [0.578, 0.594] | 0.521 | 7.9 | 7.8 |
| RPS-128 | 10 | 0.806 [0.743, 0.866] | [0.792, 0.817] | 0.713 [0.683, 0.742] | [0.708, 0.720] | 0.652 | 8.3 | 8.3 |
| RPS-full-8 | 10 | 0.613 [0.545, 0.680] | [0.603, 0.623] | 0.562 [0.532, 0.593] | [0.554, 0.571] | 0.528 | 10.1 | 10.1 |
| RPS-full-32 | 10 | 0.787 [0.725, 0.848] | [0.778, 0.797] | 0.708 [0.678, 0.736] | [0.704, 0.711] | 0.654 | 10.0 | 10.0 |
| RPS-full-128 | 10 | 0.880 [0.829, 0.927] | [0.871, 0.889] | 0.811 [0.783, 0.838] | [0.806, 0.816] | 0.765 | 10.0 | 10.0 |
| Planner on PPO plain, N=8 | 10 | 0.743 [0.681, 0.803] | [0.726, 0.761] | 0.714 [0.680, 0.746] | [0.701, 0.726] | 0.694 | 8.7 | 8.7 |
| Planner on PPO plain, N=32 | 10 | 0.855 [0.799, 0.905] | [0.837, 0.872] | 0.811 [0.779, 0.841] | [0.799, 0.822] | 0.782 | 8.7 | 8.7 |
| Planner on PPO plain, N=128 | 10 | 0.922 [0.879, 0.958] | [0.912, 0.930] | 0.879 [0.854, 0.903] | [0.874, 0.885] | 0.851 | 8.7 | 8.7 |
| Planner on PPO+D4, N=8 | 10 | 0.693 [0.625, 0.761] | [0.673, 0.710] | 0.666 [0.633, 0.699] | [0.652, 0.680] | 0.649 | 9.2 | 9.2 |
| Planner on PPO+D4, N=32 | 10 | 0.807 [0.746, 0.864] | [0.794, 0.822] | 0.767 [0.734, 0.798] | [0.752, 0.782] | 0.740 | 9.2 | 9.2 |
| Planner on PPO+D4, N=128 | 10 | 0.863 [0.807, 0.912] | [0.847, 0.878] | 0.831 [0.802, 0.858] | [0.820, 0.843] | 0.810 | 9.2 | 9.2 |

## Per family (every cell is also reported per family)

### FINAL-familiar / spine_pods (20 layouts)

| condition | breach rate [95% CI] | mean score [95% CI] |
|---|---|---|
| Random | 0.080 [0.040, 0.120] | 0.079 [0.055, 0.103] |
| Greedy | 0.900 [0.750, 1.000] | 0.363 [0.303, 0.406] |
| BestSingle | 0.900 [0.750, 1.000] | 0.426 [0.347, 0.494] |
| BestSingle-perlayout (diagnostic) | 1.000 [1.000, 1.000] | 0.466 [0.429, 0.509] |
| BestSingle-hindsight (diagnostic) | 0.900 [0.750, 1.000] | 0.426 [0.347, 0.494] |
| Oracle (upper reference, ~400 sims) | 1.000 [1.000, 1.000] | 0.826 [0.768, 0.884] |
| PPO plain | 0.570 [0.405, 0.725] | 0.504 [0.416, 0.592] |
| PPO+D4 | 0.605 [0.455, 0.750] | 0.519 [0.433, 0.603] |
| DDQN | 0.700 [0.555, 0.835] | 0.524 [0.435, 0.614] |
| PPO plain + TTA | 0.790 [0.645, 0.915] | 0.634 [0.563, 0.700] |
| PPO+D4 + TTA | 0.780 [0.620, 0.915] | 0.616 [0.533, 0.691] |
| DDQN + TTA | 0.830 [0.695, 0.940] | 0.600 [0.516, 0.679] |
| RAS-8 | 0.980 [0.935, 1.000] | 0.519 [0.470, 0.571] |
| RAS-32 | 1.000 [1.000, 1.000] | 0.594 [0.536, 0.658] |
| RAS-128 | 1.000 [1.000, 1.000] | 0.681 [0.608, 0.760] |
| RPS-8 | 0.515 [0.370, 0.660] | 0.379 [0.324, 0.431] |
| RPS-32 | 0.850 [0.740, 0.940] | 0.563 [0.513, 0.613] |
| RPS-128 | 0.990 [0.955, 1.000] | 0.698 [0.658, 0.739] |
| RPS-full-8 | 0.765 [0.630, 0.885] | 0.545 [0.493, 0.595] |
| RPS-full-32 | 0.960 [0.895, 1.000] | 0.699 [0.656, 0.743] |
| RPS-full-128 | 0.995 [0.970, 1.000] | 0.785 [0.745, 0.827] |
| Planner on PPO plain, N=8 | 0.865 [0.745, 0.955] | 0.732 [0.666, 0.795] |
| Planner on PPO plain, N=32 | 0.950 [0.880, 1.000] | 0.805 [0.752, 0.855] |
| Planner on PPO plain, N=128 | 0.975 [0.925, 1.000] | 0.864 [0.817, 0.907] |
| Planner on PPO+D4, N=8 | 0.855 [0.735, 0.955] | 0.684 [0.622, 0.743] |
| Planner on PPO+D4, N=32 | 0.930 [0.845, 0.990] | 0.777 [0.724, 0.827] |
| Planner on PPO+D4, N=128 | 0.970 [0.915, 1.000] | 0.838 [0.792, 0.880] |

### FINAL-familiar / archipelago (20 layouts)

| condition | breach rate [95% CI] | mean score [95% CI] |
|---|---|---|
| Random | 0.100 [0.040, 0.180] | 0.117 [0.085, 0.152] |
| Greedy | 0.850 [0.700, 1.000] | 0.590 [0.502, 0.673] |
| BestSingle | 0.900 [0.750, 1.000] | 0.544 [0.445, 0.631] |
| BestSingle-perlayout (diagnostic) | 1.000 [1.000, 1.000] | 0.658 [0.603, 0.713] |
| BestSingle-hindsight (diagnostic) | 0.850 [0.700, 1.000] | 0.590 [0.502, 0.673] |
| Oracle (upper reference, ~400 sims) | 1.000 [1.000, 1.000] | 0.984 [0.964, 1.000] |
| PPO plain | 0.360 [0.230, 0.500] | 0.424 [0.349, 0.502] |
| PPO+D4 | 0.350 [0.195, 0.510] | 0.442 [0.357, 0.533] |
| DDQN | 0.340 [0.190, 0.500] | 0.400 [0.313, 0.491] |
| PPO plain + TTA | 0.455 [0.290, 0.620] | 0.515 [0.431, 0.604] |
| PPO+D4 + TTA | 0.405 [0.225, 0.590] | 0.506 [0.411, 0.607] |
| DDQN + TTA | 0.410 [0.235, 0.585] | 0.463 [0.368, 0.557] |
| RAS-8 | 0.885 [0.765, 0.975] | 0.587 [0.501, 0.665] |
| RAS-32 | 0.965 [0.905, 1.000] | 0.695 [0.619, 0.770] |
| RAS-128 | 1.000 [1.000, 1.000] | 0.798 [0.719, 0.872] |
| RPS-8 | 0.255 [0.135, 0.395] | 0.351 [0.296, 0.413] |
| RPS-32 | 0.605 [0.450, 0.750] | 0.548 [0.487, 0.612] |
| RPS-128 | 0.835 [0.710, 0.940] | 0.695 [0.637, 0.753] |
| RPS-full-8 | 0.550 [0.415, 0.680] | 0.540 [0.477, 0.609] |
| RPS-full-32 | 0.855 [0.745, 0.950] | 0.718 [0.659, 0.777] |
| RPS-full-128 | 0.975 [0.930, 1.000] | 0.837 [0.785, 0.888] |
| Planner on PPO plain, N=8 | 0.715 [0.575, 0.845] | 0.702 [0.622, 0.780] |
| Planner on PPO plain, N=32 | 0.850 [0.715, 0.950] | 0.807 [0.738, 0.873] |
| Planner on PPO plain, N=128 | 0.980 [0.940, 1.000] | 0.901 [0.858, 0.940] |
| Planner on PPO+D4, N=8 | 0.640 [0.475, 0.790] | 0.656 [0.581, 0.734] |
| Planner on PPO+D4, N=32 | 0.835 [0.700, 0.945] | 0.778 [0.713, 0.842] |
| Planner on PPO+D4, N=128 | 0.905 [0.805, 0.980] | 0.846 [0.792, 0.897] |

### FINAL-familiar / bunker_nest (20 layouts)

| condition | breach rate [95% CI] | mean score [95% CI] |
|---|---|---|
| Random | 0.000 [0.000, 0.000] | 0.124 [0.087, 0.160] |
| Greedy | 0.000 [0.000, 0.000] | 0.062 [0.036, 0.088] |
| BestSingle | 0.300 [0.100, 0.500] | 0.173 [0.065, 0.293] |
| BestSingle-perlayout (diagnostic) | 0.300 [0.100, 0.500] | 0.261 [0.166, 0.361] |
| BestSingle-hindsight (diagnostic) | 0.300 [0.100, 0.500] | 0.173 [0.065, 0.293] |
| Oracle (upper reference, ~400 sims) | 1.000 [1.000, 1.000] | 0.867 [0.804, 0.925] |
| PPO plain | 0.090 [0.030, 0.165] | 0.382 [0.315, 0.455] |
| PPO+D4 | 0.090 [0.020, 0.185] | 0.403 [0.341, 0.469] |
| DDQN | 0.130 [0.045, 0.240] | 0.377 [0.293, 0.468] |
| PPO plain + TTA | 0.145 [0.040, 0.270] | 0.441 [0.363, 0.529] |
| PPO+D4 + TTA | 0.110 [0.020, 0.235] | 0.442 [0.364, 0.529] |
| DDQN + TTA | 0.275 [0.135, 0.440] | 0.497 [0.405, 0.597] |
| RAS-8 | 0.430 [0.245, 0.625] | 0.283 [0.164, 0.412] |
| RAS-32 | 0.570 [0.365, 0.765] | 0.374 [0.245, 0.509] |
| RAS-128 | 0.665 [0.460, 0.850] | 0.438 [0.310, 0.567] |
| RPS-8 | 0.010 [0.000, 0.045] | 0.314 [0.269, 0.361] |
| RPS-32 | 0.070 [0.015, 0.140] | 0.426 [0.374, 0.482] |
| RPS-128 | 0.190 [0.075, 0.320] | 0.516 [0.455, 0.580] |
| RPS-full-8 | 0.045 [0.005, 0.100] | 0.403 [0.362, 0.446] |
| RPS-full-32 | 0.135 [0.055, 0.240] | 0.500 [0.451, 0.553] |
| RPS-full-128 | 0.400 [0.255, 0.555] | 0.614 [0.551, 0.679] |
| Planner on PPO plain, N=8 | 0.260 [0.135, 0.405] | 0.543 [0.468, 0.624] |
| Planner on PPO plain, N=32 | 0.460 [0.290, 0.635] | 0.649 [0.560, 0.740] |
| Planner on PPO plain, N=128 | 0.620 [0.445, 0.785] | 0.731 [0.641, 0.818] |
| Planner on PPO+D4, N=8 | 0.155 [0.060, 0.270] | 0.507 [0.442, 0.576] |
| Planner on PPO+D4, N=32 | 0.300 [0.155, 0.465] | 0.591 [0.513, 0.673] |
| Planner on PPO+D4, N=128 | 0.440 [0.275, 0.610] | 0.666 [0.584, 0.751] |

### FINAL-transfer / labyrinth (60 layouts)

| condition | breach rate [95% CI] | mean score [95% CI] |
|---|---|---|
| Random | 0.197 [0.140, 0.253] | 0.158 [0.128, 0.188] |
| Greedy | 0.950 [0.883, 1.000] | 0.600 [0.561, 0.640] |
| BestSingle | 1.000 [1.000, 1.000] | 0.666 [0.626, 0.704] |
| BestSingle-perlayout (diagnostic) | 1.000 [1.000, 1.000] | 0.708 [0.671, 0.744] |
| BestSingle-hindsight (diagnostic) | 1.000 [1.000, 1.000] | 0.666 [0.626, 0.704] |
| Oracle (upper reference, ~400 sims) | 1.000 [1.000, 1.000] | 0.964 [0.938, 0.984] |
| PPO plain | 0.597 [0.503, 0.690] | 0.517 [0.462, 0.571] |
| PPO+D4 | 0.635 [0.532, 0.728] | 0.518 [0.455, 0.577] |
| DDQN | 0.652 [0.562, 0.738] | 0.563 [0.512, 0.611] |
| PPO plain + TTA | 0.758 [0.675, 0.838] | 0.614 [0.563, 0.663] |
| PPO+D4 + TTA | 0.747 [0.643, 0.840] | 0.596 [0.526, 0.663] |
| DDQN + TTA | 0.792 [0.693, 0.878] | 0.647 [0.594, 0.699] |
| RAS-8 | 0.977 [0.942, 0.998] | 0.717 [0.673, 0.760] |
| RAS-32 | 0.995 [0.980, 1.000] | 0.831 [0.792, 0.867] |
| RAS-128 | 0.997 [0.985, 1.000] | 0.910 [0.879, 0.939] |
| RPS-8 | 0.600 [0.505, 0.692] | 0.462 [0.414, 0.510] |
| RPS-32 | 0.858 [0.788, 0.918] | 0.659 [0.621, 0.698] |
| RPS-128 | 0.940 [0.888, 0.982] | 0.791 [0.754, 0.826] |
| RPS-full-8 | 0.772 [0.695, 0.843] | 0.628 [0.581, 0.672] |
| RPS-full-32 | 0.925 [0.872, 0.968] | 0.776 [0.738, 0.813] |
| RPS-full-128 | 0.970 [0.928, 0.997] | 0.877 [0.845, 0.906] |
| Planner on PPO plain, N=8 | 0.873 [0.810, 0.928] | 0.768 [0.726, 0.807] |
| Planner on PPO plain, N=32 | 0.957 [0.913, 0.988] | 0.868 [0.835, 0.899] |
| Planner on PPO plain, N=128 | 0.985 [0.960, 1.000] | 0.927 [0.904, 0.947] |
| Planner on PPO+D4, N=8 | 0.835 [0.757, 0.903] | 0.717 [0.667, 0.763] |
| Planner on PPO+D4, N=32 | 0.927 [0.870, 0.972] | 0.819 [0.778, 0.858] |
| Planner on PPO+D4, N=128 | 0.953 [0.907, 0.988] | 0.878 [0.844, 0.910] |

## Generalisation gap (descriptive, not a test): DEV-train in-sample score minus FINAL-familiar score

| kind | label | dev_train_score | final_familiar_score | gap | lo95 | hi95 | learner_only_lo95 | learner_only_hi95 |
|---|---|---|---|---|---|---|---|---|
| learner | PPO plain | 0.7054 | 0.4370 | 0.2683 | 0.2131 | 0.3229 | 0.2518 | 0.2850 |
| learner | PPO+D4 | 0.4864 | 0.4545 | 0.0319 | -0.0242 | 0.0866 | 0.0152 | 0.0506 |
| learner | DDQN | 0.7268 | 0.4335 | 0.2933 | 0.2335 | 0.3516 | 0.2806 | 0.3068 |
| baseline (difficulty reference) | Random | 0.1173 | 0.1065 | 0.0108 | -0.0138 | 0.0356 |  |  |
| baseline (difficulty reference) | Greedy | 0.3416 | 0.3380 | 0.0035 | -0.0706 | 0.0784 |  |  |
| baseline (difficulty reference) | BestSingle | 0.4435 | 0.3810 | 0.0625 | -0.0144 | 0.1397 |  |  |
| baseline (difficulty reference) | Oracle (upper reference, ~400 sims) | 0.9077 | 0.8923 | 0.0154 | -0.0228 | 0.0554 |  |  |

## Exploratory comparisons (no multiplicity correction)

| id | comparison | group | metric | diff | lo95 | hi95 | learner_only_lo95 | learner_only_hi95 |
|---|---|---|---|---|---|---|---|---|
| E1 | Planner on PPO plain N=8 vs its own learner | FINAL-familiar | breach | 0.2733 | 0.2017 | 0.3467 | 0.2350 | 0.3133 |
| E1 | Planner on PPO plain N=8 vs its own learner | FINAL-familiar | score | 0.2221 | 0.1848 | 0.2601 | 0.2018 | 0.2414 |
| E1 | Planner on PPO plain N=8 vs its own learner | FINAL-transfer | breach | 0.2767 | 0.2033 | 0.3500 | 0.2350 | 0.3150 |
| E1 | Planner on PPO plain N=8 vs its own learner | FINAL-transfer | score | 0.2517 | 0.2113 | 0.2933 | 0.2243 | 0.2795 |
| E1 | Planner on PPO plain N=32 vs its own learner | FINAL-familiar | breach | 0.4133 | 0.3283 | 0.4983 | 0.3767 | 0.4550 |
| E1 | Planner on PPO plain N=32 vs its own learner | FINAL-familiar | score | 0.3168 | 0.2778 | 0.3559 | 0.2990 | 0.3335 |
| E1 | Planner on PPO plain N=32 vs its own learner | FINAL-transfer | breach | 0.3600 | 0.2783 | 0.4450 | 0.3217 | 0.3950 |
| E1 | Planner on PPO plain N=32 vs its own learner | FINAL-transfer | score | 0.3517 | 0.3079 | 0.3961 | 0.3292 | 0.3759 |
| E1 | Planner on PPO plain N=128 vs its own learner | FINAL-familiar | breach | 0.5183 | 0.4317 | 0.6067 | 0.4850 | 0.5567 |
| E1 | Planner on PPO plain N=128 vs its own learner | FINAL-familiar | score | 0.3950 | 0.3539 | 0.4364 | 0.3806 | 0.4102 |
| E1 | Planner on PPO plain N=128 vs its own learner | FINAL-transfer | breach | 0.3883 | 0.3000 | 0.4784 | 0.3483 | 0.4267 |
| E1 | Planner on PPO plain N=128 vs its own learner | FINAL-transfer | score | 0.4101 | 0.3627 | 0.4562 | 0.3888 | 0.4336 |
| E2 | Planner on PPO plain N=32 vs N=8 | FINAL-familiar | breach | 0.1400 | 0.0933 | 0.1933 | 0.1250 | 0.1567 |
| E2 | Planner on PPO plain N=32 vs N=8 | FINAL-familiar | score | 0.0948 | 0.0758 | 0.1154 | 0.0878 | 0.1013 |
| E2 | Planner on PPO plain N=32 vs N=8 | FINAL-transfer | breach | 0.0833 | 0.0483 | 0.1267 | 0.0633 | 0.1033 |
| E2 | Planner on PPO plain N=32 vs N=8 | FINAL-transfer | score | 0.1000 | 0.0794 | 0.1237 | 0.0892 | 0.1112 |
| E2 | Planner on PPO plain N=128 vs N=32 | FINAL-familiar | breach | 0.1050 | 0.0583 | 0.1617 | 0.0817 | 0.1300 |
| E2 | Planner on PPO plain N=128 vs N=32 | FINAL-familiar | score | 0.0782 | 0.0600 | 0.0997 | 0.0686 | 0.0892 |
| E2 | Planner on PPO plain N=128 vs N=32 | FINAL-transfer | breach | 0.0283 | 0.0050 | 0.0583 | 0.0183 | 0.0367 |
| E2 | Planner on PPO plain N=128 vs N=32 | FINAL-transfer | score | 0.0584 | 0.0430 | 0.0755 | 0.0513 | 0.0672 |
| E1 | Planner on PPO+D4 N=8 vs its own learner | FINAL-familiar | breach | 0.2017 | 0.1300 | 0.2750 | 0.1617 | 0.2417 |
| E1 | Planner on PPO+D4 N=8 vs its own learner | FINAL-familiar | score | 0.1611 | 0.1285 | 0.1951 | 0.1433 | 0.1796 |
| E1 | Planner on PPO+D4 N=8 vs its own learner | FINAL-transfer | breach | 0.2000 | 0.1367 | 0.2733 | 0.1633 | 0.2433 |
| E1 | Planner on PPO+D4 N=8 vs its own learner | FINAL-transfer | score | 0.1991 | 0.1621 | 0.2381 | 0.1743 | 0.2233 |
| E1 | Planner on PPO+D4 N=32 vs its own learner | FINAL-familiar | breach | 0.3400 | 0.2617 | 0.4217 | 0.3133 | 0.3650 |
| E1 | Planner on PPO+D4 N=32 vs its own learner | FINAL-familiar | score | 0.2606 | 0.2243 | 0.2985 | 0.2491 | 0.2718 |
| E1 | Planner on PPO+D4 N=32 vs its own learner | FINAL-transfer | breach | 0.2917 | 0.2150 | 0.3750 | 0.2583 | 0.3267 |
| E1 | Planner on PPO+D4 N=32 vs its own learner | FINAL-transfer | score | 0.3014 | 0.2635 | 0.3400 | 0.2831 | 0.3207 |
| E1 | Planner on PPO+D4 N=128 vs its own learner | FINAL-familiar | breach | 0.4233 | 0.3367 | 0.5100 | 0.3917 | 0.4500 |
| E1 | Planner on PPO+D4 N=128 vs its own learner | FINAL-familiar | score | 0.3286 | 0.2894 | 0.3686 | 0.3134 | 0.3433 |
| E1 | Planner on PPO+D4 N=128 vs its own learner | FINAL-transfer | breach | 0.3183 | 0.2367 | 0.4067 | 0.2817 | 0.3583 |
| E1 | Planner on PPO+D4 N=128 vs its own learner | FINAL-transfer | score | 0.3607 | 0.3168 | 0.4063 | 0.3375 | 0.3864 |
| E2 | Planner on PPO+D4 N=32 vs N=8 | FINAL-familiar | breach | 0.1383 | 0.0900 | 0.1933 | 0.1133 | 0.1583 |
| E2 | Planner on PPO+D4 N=32 vs N=8 | FINAL-familiar | score | 0.0995 | 0.0806 | 0.1213 | 0.0892 | 0.1092 |
| E2 | Planner on PPO+D4 N=32 vs N=8 | FINAL-transfer | breach | 0.0917 | 0.0500 | 0.1400 | 0.0683 | 0.1133 |
| E2 | Planner on PPO+D4 N=32 vs N=8 | FINAL-transfer | score | 0.1023 | 0.0809 | 0.1264 | 0.0888 | 0.1158 |
| E2 | Planner on PPO+D4 N=128 vs N=32 | FINAL-familiar | breach | 0.0833 | 0.0467 | 0.1250 | 0.0683 | 0.0983 |
| E2 | Planner on PPO+D4 N=128 vs N=32 | FINAL-familiar | score | 0.0680 | 0.0535 | 0.0845 | 0.0610 | 0.0758 |
| E2 | Planner on PPO+D4 N=128 vs N=32 | FINAL-transfer | breach | 0.0267 | 0.0083 | 0.0517 | 0.0183 | 0.0350 |
| E2 | Planner on PPO+D4 N=128 vs N=32 | FINAL-transfer | score | 0.0593 | 0.0436 | 0.0766 | 0.0485 | 0.0700 |
| E3 | DDQN vs PPO plain | FINAL-familiar | breach | 0.0500 | -0.0283 | 0.1283 | 0.0050 | 0.0983 |
| E3 | DDQN vs PPO plain | FINAL-familiar | score | -0.0035 | -0.0457 | 0.0372 | -0.0262 | 0.0211 |
| E3 | DDQN vs PPO plain | FINAL-transfer | breach | 0.0550 | -0.0283 | 0.1400 | 0.0033 | 0.1033 |
| E3 | DDQN vs PPO plain | FINAL-transfer | score | 0.0462 | -0.0034 | 0.0985 | 0.0134 | 0.0803 |
| E3 | DDQN vs PPO+D4 | FINAL-familiar | breach | 0.0417 | -0.0350 | 0.1183 | -0.0083 | 0.0883 |
| E3 | DDQN vs PPO+D4 | FINAL-familiar | score | -0.0210 | -0.0619 | 0.0208 | -0.0455 | 0.0043 |
| E3 | DDQN vs PPO+D4 | FINAL-transfer | breach | 0.0167 | -0.0617 | 0.1017 | -0.0367 | 0.0733 |
| E3 | DDQN vs PPO+D4 | FINAL-transfer | score | 0.0450 | -0.0091 | 0.1041 | 0.0039 | 0.0897 |

### E4: FINAL-transfer minus FINAL-familiar

| label | metric | transfer_minus_familiar | lo95 | hi95 | learner_only_lo95 | learner_only_hi95 |
|---|---|---|---|---|---|---|
| PPO plain | breach | 0.2567 | 0.1200 | 0.3850 | 0.1950 | 0.3183 |
| PPO plain | score | 0.0796 | 0.0039 | 0.1524 | 0.0436 | 0.1140 |
| PPO+D4 | breach | 0.2867 | 0.1533 | 0.4150 | 0.2400 | 0.3300 |
| PPO+D4 | score | 0.0633 | -0.0099 | 0.1379 | 0.0274 | 0.0982 |
| DDQN | breach | 0.2617 | 0.1267 | 0.3933 | 0.2217 | 0.3000 |
| DDQN | score | 0.1293 | 0.0575 | 0.2008 | 0.1061 | 0.1529 |
| Greedy | breach | 0.3667 | 0.2333 | 0.5000 | 0.3667 | 0.3667 |
| Greedy | score | 0.2622 | 0.1842 | 0.3379 | 0.2622 | 0.2622 |
| BestSingle | breach | 0.3000 | 0.1833 | 0.4167 | 0.3000 | 0.3000 |
| BestSingle | score | 0.2849 | 0.2086 | 0.3637 | 0.2849 | 0.2849 |
| Random | breach | 0.1367 | 0.0733 | 0.2000 | 0.1367 | 0.1367 |
| Random | score | 0.0518 | 0.0159 | 0.0879 | 0.0518 | 0.0518 |
| Oracle (upper reference, ~400 sims) | breach | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Oracle (upper reference, ~400 sims) | score | 0.0714 | 0.0311 | 0.1123 | 0.0714 | 0.0714 |

## Army composition (mean units bought per type; Random and Oracle carry no army)

| group | label | army_scout | army_sniper | army_bulwark | army_blaster | army_mender | n_bought | n_deployed |
|---|---|---|---|---|---|---|---|---|
| FINAL-familiar | Greedy | 0.0000 | 0.0000 | 8.0000 | 0.0000 | 0.0000 | 8.0000 | 8.0000 |
| FINAL-familiar | BestSingle | 0.0000 | 8.0000 | 0.0000 | 0.0000 | 0.0000 | 8.0000 | 8.0000 |
| FINAL-familiar | BestSingle-perlayout (diagnostic) | 3.6667 | 4.8000 | 1.7333 | 0.0000 | 0.0000 | 10.2000 | 10.2000 |
| FINAL-familiar | BestSingle-hindsight (diagnostic) | 0.0000 | 5.3333 | 2.6667 | 0.0000 | 0.0000 | 8.0000 | 8.0000 |
| FINAL-familiar | PPO plain | 1.3483 | 3.7367 | 2.5000 | 1.1283 | 0.0000 | 8.7133 | 8.7133 |
| FINAL-familiar | PPO+D4 | 2.4633 | 3.5817 | 1.8633 | 1.4267 | 0.0983 | 9.4333 | 9.4300 |
| FINAL-familiar | DDQN | 2.7533 | 6.8467 | 0.3350 | 0.2750 | 0.0083 | 10.2183 | 10.2000 |
| FINAL-familiar | PPO plain + TTA | 0.7400 | 4.5167 | 2.5700 | 0.6217 | 0.0000 | 8.4483 | 8.4483 |
| FINAL-familiar | PPO+D4 + TTA | 1.9000 | 4.6450 | 1.2800 | 1.3000 | 0.0183 | 9.1433 | 9.1433 |
| FINAL-familiar | DDQN + TTA | 2.2483 | 7.1650 | 0.2717 | 0.2333 | 0.0033 | 9.9217 | 9.9167 |
| FINAL-familiar | RAS-8 | 2.1667 | 1.8800 | 1.6300 | 1.4150 | 1.4000 | 8.4917 | 8.4917 |
| FINAL-familiar | RAS-32 | 2.3317 | 1.9883 | 1.7950 | 1.3133 | 1.4500 | 8.8783 | 8.8783 |
| FINAL-familiar | RAS-128 | 2.4283 | 2.0750 | 1.9950 | 1.2233 | 1.5550 | 9.2767 | 9.2767 |
| FINAL-familiar | RPS-8 | 1.5250 | 1.6100 | 1.4917 | 1.5083 | 1.1233 | 7.2583 | 7.2017 |
| FINAL-familiar | RPS-32 | 1.6650 | 2.0000 | 1.5533 | 1.5767 | 1.1350 | 7.9300 | 7.8717 |
| FINAL-familiar | RPS-128 | 1.7633 | 2.3367 | 1.6350 | 1.6250 | 1.0417 | 8.4017 | 8.3650 |
| FINAL-familiar | RPS-full-8 | 3.0033 | 2.1933 | 1.7867 | 1.6600 | 1.3867 | 10.0300 | 10.0300 |
| FINAL-familiar | RPS-full-32 | 2.9850 | 2.3883 | 1.7883 | 1.6567 | 1.1983 | 10.0167 | 10.0167 |
| FINAL-familiar | RPS-full-128 | 2.8983 | 2.5333 | 1.9117 | 1.5550 | 1.0817 | 9.9800 | 9.9800 |
| FINAL-familiar | Planner on PPO plain, N=8 | 1.3583 | 3.6583 | 2.5117 | 1.1517 | 0.0333 | 8.7133 | 8.7117 |
| FINAL-familiar | Planner on PPO plain, N=32 | 1.3400 | 3.7350 | 2.5133 | 1.0917 | 0.0350 | 8.7150 | 8.7133 |
| FINAL-familiar | Planner on PPO plain, N=128 | 1.2733 | 3.7400 | 2.5417 | 1.1000 | 0.0233 | 8.6783 | 8.6783 |
| FINAL-familiar | Planner on PPO+D4, N=8 | 2.2000 | 3.4950 | 1.8700 | 1.4383 | 0.2650 | 9.2683 | 9.2633 |
| FINAL-familiar | Planner on PPO+D4, N=32 | 2.1067 | 3.5583 | 1.9667 | 1.4067 | 0.1933 | 9.2317 | 9.2267 |
| FINAL-familiar | Planner on PPO+D4, N=128 | 2.1017 | 3.6233 | 1.9650 | 1.3533 | 0.2167 | 9.2600 | 9.2533 |
| FINAL-transfer | Greedy | 0.0000 | 0.0000 | 8.0000 | 0.0000 | 0.0000 | 8.0000 | 8.0000 |
| FINAL-transfer | BestSingle | 0.0000 | 8.0000 | 0.0000 | 0.0000 | 0.0000 | 8.0000 | 8.0000 |
| FINAL-transfer | BestSingle-perlayout (diagnostic) | 0.3333 | 5.7333 | 2.1333 | 0.0000 | 0.0000 | 8.2000 | 8.2000 |
| FINAL-transfer | BestSingle-hindsight (diagnostic) | 0.0000 | 8.0000 | 0.0000 | 0.0000 | 0.0000 | 8.0000 | 8.0000 |
| FINAL-transfer | PPO plain | 1.2833 | 3.8083 | 2.3833 | 1.2033 | 0.0033 | 8.6817 | 8.6817 |
| FINAL-transfer | PPO+D4 | 2.2767 | 3.5333 | 1.6767 | 1.6400 | 0.1183 | 9.2450 | 9.2450 |
| FINAL-transfer | DDQN | 2.3950 | 6.8817 | 0.2800 | 0.3950 | 0.0117 | 9.9633 | 9.9367 |
| FINAL-transfer | PPO plain + TTA | 0.5250 | 4.3433 | 2.6217 | 0.7700 | 0.0000 | 8.2600 | 8.2600 |
| FINAL-transfer | PPO+D4 + TTA | 1.9483 | 4.0367 | 1.2583 | 1.7400 | 0.0467 | 9.0300 | 9.0300 |
| FINAL-transfer | DDQN + TTA | 1.6750 | 7.2317 | 0.2250 | 0.3933 | 0.0000 | 9.5250 | 9.5117 |
| FINAL-transfer | RAS-8 | 2.0833 | 1.5883 | 1.8000 | 1.4967 | 1.6917 | 8.6600 | 8.6600 |
| FINAL-transfer | RAS-32 | 2.1200 | 1.5233 | 1.9817 | 1.4717 | 1.9133 | 9.0100 | 9.0100 |
| FINAL-transfer | RAS-128 | 2.1000 | 1.4067 | 2.2200 | 1.3567 | 2.0933 | 9.1767 | 9.1767 |
| FINAL-transfer | RPS-8 | 1.5283 | 1.6483 | 1.2867 | 1.4567 | 1.2350 | 7.1550 | 7.0767 |
| FINAL-transfer | RPS-32 | 1.5900 | 2.0333 | 1.4750 | 1.5183 | 1.1867 | 7.8033 | 7.7517 |
| FINAL-transfer | RPS-128 | 1.6717 | 2.2683 | 1.6233 | 1.5117 | 1.1917 | 8.2667 | 8.2100 |
| FINAL-transfer | RPS-full-8 | 3.0450 | 2.1817 | 1.7567 | 1.5983 | 1.4900 | 10.0717 | 10.0717 |
| FINAL-transfer | RPS-full-32 | 2.9017 | 2.4783 | 1.8400 | 1.4883 | 1.2917 | 10.0000 | 10.0000 |
| FINAL-transfer | RPS-full-128 | 2.8517 | 2.6267 | 1.9133 | 1.4450 | 1.1367 | 9.9733 | 9.9733 |
| FINAL-transfer | Planner on PPO plain, N=8 | 1.3700 | 3.7183 | 2.2500 | 1.3067 | 0.0200 | 8.6650 | 8.6533 |
| FINAL-transfer | Planner on PPO plain, N=32 | 1.2650 | 3.8017 | 2.3200 | 1.2217 | 0.0200 | 8.6283 | 8.6217 |
| FINAL-transfer | Planner on PPO plain, N=128 | 1.2650 | 3.8033 | 2.4200 | 1.1583 | 0.0283 | 8.6750 | 8.6683 |
| FINAL-transfer | Planner on PPO+D4, N=8 | 2.1033 | 3.5583 | 1.6583 | 1.6467 | 0.1600 | 9.1267 | 9.1183 |
| FINAL-transfer | Planner on PPO+D4, N=32 | 2.0783 | 3.6350 | 1.6400 | 1.5950 | 0.1900 | 9.1383 | 9.1333 |
| FINAL-transfer | Planner on PPO+D4, N=128 | 2.0450 | 3.6267 | 1.6867 | 1.5900 | 0.1733 | 9.1217 | 9.1167 |
| POOLED (descriptive) | Greedy | 0.0000 | 0.0000 | 8.0000 | 0.0000 | 0.0000 | 8.0000 | 8.0000 |
| POOLED (descriptive) | BestSingle | 0.0000 | 8.0000 | 0.0000 | 0.0000 | 0.0000 | 8.0000 | 8.0000 |
| POOLED (descriptive) | BestSingle-perlayout (diagnostic) | 2.0000 | 5.2667 | 1.9333 | 0.0000 | 0.0000 | 9.2000 | 9.2000 |
| POOLED (descriptive) | BestSingle-hindsight (diagnostic) | 0.0000 | 6.6667 | 1.3333 | 0.0000 | 0.0000 | 8.0000 | 8.0000 |
| POOLED (descriptive) | PPO plain | 1.3158 | 3.7725 | 2.4417 | 1.1658 | 0.0017 | 8.6975 | 8.6975 |
| POOLED (descriptive) | PPO+D4 | 2.3700 | 3.5575 | 1.7700 | 1.5333 | 0.1083 | 9.3392 | 9.3375 |
| POOLED (descriptive) | DDQN | 2.5742 | 6.8642 | 0.3075 | 0.3350 | 0.0100 | 10.0908 | 10.0683 |
| POOLED (descriptive) | PPO plain + TTA | 0.6325 | 4.4300 | 2.5958 | 0.6958 | 0.0000 | 8.3542 | 8.3542 |
| POOLED (descriptive) | PPO+D4 + TTA | 1.9242 | 4.3408 | 1.2692 | 1.5200 | 0.0325 | 9.0867 | 9.0867 |
| POOLED (descriptive) | DDQN + TTA | 1.9617 | 7.1983 | 0.2483 | 0.3133 | 0.0017 | 9.7233 | 9.7142 |
| POOLED (descriptive) | RAS-8 | 2.1250 | 1.7342 | 1.7150 | 1.4558 | 1.5458 | 8.5758 | 8.5758 |
| POOLED (descriptive) | RAS-32 | 2.2258 | 1.7558 | 1.8883 | 1.3925 | 1.6817 | 8.9442 | 8.9442 |
| POOLED (descriptive) | RAS-128 | 2.2642 | 1.7408 | 2.1075 | 1.2900 | 1.8242 | 9.2267 | 9.2267 |
| POOLED (descriptive) | RPS-8 | 1.5267 | 1.6292 | 1.3892 | 1.4825 | 1.1792 | 7.2067 | 7.1392 |
| POOLED (descriptive) | RPS-32 | 1.6275 | 2.0167 | 1.5142 | 1.5475 | 1.1608 | 7.8667 | 7.8117 |
| POOLED (descriptive) | RPS-128 | 1.7175 | 2.3025 | 1.6292 | 1.5683 | 1.1167 | 8.3342 | 8.2875 |
| POOLED (descriptive) | RPS-full-8 | 3.0242 | 2.1875 | 1.7717 | 1.6292 | 1.4383 | 10.0508 | 10.0508 |
| POOLED (descriptive) | RPS-full-32 | 2.9433 | 2.4333 | 1.8142 | 1.5725 | 1.2450 | 10.0083 | 10.0083 |
| POOLED (descriptive) | RPS-full-128 | 2.8750 | 2.5800 | 1.9125 | 1.5000 | 1.1092 | 9.9767 | 9.9767 |
| POOLED (descriptive) | Planner on PPO plain, N=8 | 1.3642 | 3.6883 | 2.3808 | 1.2292 | 0.0267 | 8.6892 | 8.6825 |
| POOLED (descriptive) | Planner on PPO plain, N=32 | 1.3025 | 3.7683 | 2.4167 | 1.1567 | 0.0275 | 8.6717 | 8.6675 |
| POOLED (descriptive) | Planner on PPO plain, N=128 | 1.2692 | 3.7717 | 2.4808 | 1.1292 | 0.0258 | 8.6767 | 8.6733 |
| POOLED (descriptive) | Planner on PPO+D4, N=8 | 2.1517 | 3.5267 | 1.7642 | 1.5425 | 0.2125 | 9.1975 | 9.1908 |
| POOLED (descriptive) | Planner on PPO+D4, N=32 | 2.0925 | 3.5967 | 1.8033 | 1.5008 | 0.1917 | 9.1850 | 9.1800 |
| POOLED (descriptive) | Planner on PPO+D4, N=128 | 2.0733 | 3.6250 | 1.8258 | 1.4717 | 0.1950 | 9.1908 | 9.1850 |

