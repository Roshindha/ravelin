# B14c item 2: SEEN-data summary (DEV only; seen, never headline)

Monitoring only: nothing is selected, and no number here changes any later choice. Final models at T = 1.5M, deterministic masked argmax, canonical view, no TTA. Learner CIs: hierarchical bootstrap (resample the 10 seeds, then layouts within each drawn seed), 10 000 resamples, rng seed 8080. Baseline CIs are the B8 ones (bootstrap over layouts). BestSingle = Sniper in every family (chosen on DEV-train; DEV-train row is in-sample).

Control: per-family DEV-validation means recomputed here vs the trainers' logged t = 1.5M validation eval: max abs diff 0.00e+00.

### DEV-train (150 layouts; in-sample, trained on)

| row | breach rate [95% CI] | mean score [95% CI] | mean D |
|---|---|---|---|
| PPO plain (ALL) | 0.718 [0.691, 0.745] | 0.7054 [0.6900, 0.7206] | 0.697 |
| PPO+D4 (ALL) | 0.423 [0.389, 0.458] | 0.4864 [0.4689, 0.5039] | 0.529 |
| DDQN (ALL) | 0.758 [0.725, 0.791] | 0.7268 [0.7085, 0.7465] | 0.706 |
| Random (ALL) | 0.075 [0.051, 0.101] | 0.1173 [0.1023, 0.1328] | 0.146 |
| Greedy (ALL) | 0.627 [0.547, 0.707] | 0.3416 [0.3043, 0.3790] | 0.151 |
| BestSingle (ALL) | 0.787 [0.720, 0.853] | 0.4435 [0.4055, 0.4809] | 0.215 |
| BestSingle-perlayout (ALL) | 0.813 [0.747, 0.873] | 0.4861 [0.4523, 0.5195] | 0.268 |
| Oracle (ALL) | 1.000 [1.000, 1.000] | 0.9077 [0.8868, 0.9277] | 0.846 |
| *spine_pods* | | | |
| PPO plain (spine_pods) | 0.884 [0.850, 0.916] | 0.7624 [0.7416, 0.7833] | 0.681 |
| PPO+D4 (spine_pods) | 0.632 [0.572, 0.694] | 0.5698 [0.5422, 0.5970] | 0.528 |
| DDQN (spine_pods) | 0.882 [0.840, 0.922] | 0.7582 [0.7336, 0.7809] | 0.676 |
| Random (spine_pods) | 0.132 [0.084, 0.180] | 0.1317 [0.1062, 0.1578] | 0.132 |
| Greedy (spine_pods) | 0.960 [0.900, 1.000] | 0.3895 [0.3637, 0.4079] | 0.009 |
| BestSingle (spine_pods) | 1.000 [1.000, 1.000] | 0.4921 [0.4621, 0.5245] | 0.153 |
| BestSingle-perlayout (spine_pods) | 1.000 [1.000, 1.000] | 0.4921 [0.4621, 0.5245] | 0.153 |
| Oracle (spine_pods) | 1.000 [1.000, 1.000] | 0.8701 [0.8268, 0.9105] | 0.783 |
| *archipelago* | | | |
| PPO plain (archipelago) | 0.840 [0.798, 0.880] | 0.7350 [0.7077, 0.7607] | 0.665 |
| PPO+D4 (archipelago) | 0.544 [0.490, 0.598] | 0.5034 [0.4759, 0.5313] | 0.476 |
| DDQN (archipelago) | 0.882 [0.844, 0.916] | 0.7729 [0.7488, 0.7962] | 0.700 |
| Random (archipelago) | 0.092 [0.044, 0.148] | 0.1178 [0.0893, 0.1503] | 0.135 |
| Greedy (archipelago) | 0.860 [0.760, 0.940] | 0.5312 [0.4817, 0.5771] | 0.312 |
| BestSingle (archipelago) | 0.900 [0.820, 0.980] | 0.5461 [0.4964, 0.5937] | 0.310 |
| BestSingle-perlayout (archipelago) | 0.960 [0.900, 1.000] | 0.6138 [0.5826, 0.6450] | 0.383 |
| Oracle (archipelago) | 1.000 [1.000, 1.000] | 0.9560 [0.9255, 0.9811] | 0.927 |
| *bunker_nest* | | | |
| PPO plain (bunker_nest) | 0.430 [0.370, 0.488] | 0.6187 [0.5874, 0.6505] | 0.745 |
| PPO+D4 (bunker_nest) | 0.092 [0.060, 0.128] | 0.3858 [0.3630, 0.4101] | 0.582 |
| DDQN (bunker_nest) | 0.510 [0.442, 0.578] | 0.6494 [0.6167, 0.6832] | 0.742 |
| Random (bunker_nest) | 0.000 [0.000, 0.000] | 0.1025 [0.0820, 0.1245] | 0.171 |
| Greedy (bunker_nest) | 0.060 [0.000, 0.140] | 0.1040 [0.0604, 0.1593] | 0.133 |
| BestSingle (bunker_nest) | 0.460 [0.320, 0.600] | 0.2924 [0.2121, 0.3743] | 0.181 |
| BestSingle-perlayout (bunker_nest) | 0.480 [0.340, 0.620] | 0.3525 [0.2775, 0.4269] | 0.268 |
| Oracle (bunker_nest) | 1.000 [1.000, 1.000] | 0.8969 [0.8662, 0.9264] | 0.828 |

### DEV-validation (30 layouts; monitored every 25k, never used to select)

| row | breach rate [95% CI] | mean score [95% CI] | mean D |
|---|---|---|---|
| PPO plain (ALL) | 0.243 [0.183, 0.307] | 0.3845 [0.3535, 0.4168] | 0.479 |
| PPO+D4 (ALL) | 0.323 [0.260, 0.390] | 0.4324 [0.3980, 0.4677] | 0.505 |
| DDQN (ALL) | 0.357 [0.290, 0.430] | 0.4001 [0.3658, 0.4375] | 0.429 |
| Random (ALL) | 0.020 [0.000, 0.047] | 0.0796 [0.0590, 0.1018] | 0.119 |
| Greedy (ALL) | 0.533 [0.367, 0.700] | 0.2810 [0.2028, 0.3631] | 0.113 |
| BestSingle (ALL) | 0.733 [0.567, 0.900] | 0.4010 [0.3146, 0.4865] | 0.179 |
| BestSingle-perlayout (ALL) | 0.800 [0.667, 0.933] | 0.4589 [0.3846, 0.5370] | 0.231 |
| Oracle (ALL) | 0.967 [0.900, 1.000] | 0.8718 [0.8109, 0.9304] | 0.809 |
| *spine_pods* | | | |
| PPO plain (spine_pods) | 0.440 [0.320, 0.560] | 0.4562 [0.4037, 0.5105] | 0.467 |
| PPO+D4 (spine_pods) | 0.550 [0.420, 0.680] | 0.5242 [0.4694, 0.5799] | 0.507 |
| DDQN (spine_pods) | 0.590 [0.470, 0.700] | 0.4766 [0.4269, 0.5250] | 0.401 |
| Random (spine_pods) | 0.040 [0.000, 0.100] | 0.0780 [0.0561, 0.0984] | 0.103 |
| Greedy (spine_pods) | 1.000 [1.000, 1.000] | 0.4030 [0.4000, 0.4090] | 0.005 |
| BestSingle (spine_pods) | 1.000 [1.000, 1.000] | 0.4630 [0.4072, 0.5432] | 0.105 |
| BestSingle-perlayout (spine_pods) | 1.000 [1.000, 1.000] | 0.4630 [0.4072, 0.5432] | 0.105 |
| Oracle (spine_pods) | 0.900 [0.700, 1.000] | 0.9087 [0.8018, 1.0000] | 0.915 |
| *archipelago* | | | |
| PPO plain (archipelago) | 0.270 [0.170, 0.380] | 0.3102 [0.2557, 0.3672] | 0.337 |
| PPO+D4 (archipelago) | 0.360 [0.260, 0.460] | 0.3722 [0.3146, 0.4323] | 0.380 |
| DDQN (archipelago) | 0.290 [0.180, 0.410] | 0.3149 [0.2577, 0.3773] | 0.331 |
| Random (archipelago) | 0.020 [0.000, 0.060] | 0.0533 [0.0170, 0.1048] | 0.075 |
| Greedy (archipelago) | 0.500 [0.200, 0.800] | 0.3660 [0.2095, 0.5403] | 0.277 |
| BestSingle (archipelago) | 0.700 [0.400, 1.000] | 0.4551 [0.2801, 0.6147] | 0.292 |
| BestSingle-perlayout (archipelago) | 0.800 [0.500, 1.000] | 0.5568 [0.4003, 0.6962] | 0.395 |
| Oracle (archipelago) | 1.000 [1.000, 1.000] | 0.9021 [0.8053, 0.9925] | 0.837 |
| *bunker_nest* | | | |
| PPO plain (bunker_nest) | 0.020 [0.000, 0.060] | 0.3871 [0.3515, 0.4256] | 0.632 |
| PPO+D4 (bunker_nest) | 0.060 [0.010, 0.130] | 0.4008 [0.3680, 0.4370] | 0.628 |
| DDQN (bunker_nest) | 0.190 [0.100, 0.300] | 0.4088 [0.3542, 0.4676] | 0.555 |
| Random (bunker_nest) | 0.000 [0.000, 0.000] | 0.1077 [0.0755, 0.1410] | 0.179 |
| Greedy (bunker_nest) | 0.100 [0.000, 0.300] | 0.0741 [0.0120, 0.1560] | 0.057 |
| BestSingle (bunker_nest) | 0.500 [0.200, 0.800] | 0.2848 [0.1279, 0.4460] | 0.141 |
| BestSingle-perlayout (bunker_nest) | 0.600 [0.300, 0.900] | 0.3568 [0.2309, 0.4834] | 0.195 |
| Oracle (bunker_nest) | 1.000 [1.000, 1.000] | 0.8046 [0.6994, 0.9079] | 0.674 |
