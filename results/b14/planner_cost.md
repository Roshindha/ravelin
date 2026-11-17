# B14c item 3: planner cost with real grid models (DEV-validation, 20 layouts, 2 PPO models)

Per layout, seconds (single process; N = 8 / 32 / 128 are nested prefixes of one 128-plan batch):

| cell | N=8 | N=32 | N=128 | per plan (mean) | per plan p95 | mean units | mean env steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| planner (PPO plain s0) | 0.046 | 0.176 | 0.678 | 5.30 ms | 6.86 ms | 8.0 | 17.0 |
| planner (PPO+D4 s0) | 0.053 | 0.198 | 0.773 | 6.04 ms | 8.10 ms | 9.5 | 20.0 |
| RAS | 0.009 | 0.031 | 0.115 | 0.90 ms | 3.32 ms | 6.5 | 6.5 |
| RPS | 0.005 | 0.019 | 0.071 | 0.55 ms | 1.50 ms | 4.7 | 10.6 |
| RPSfull | 0.012 | 0.041 | 0.150 | 1.17 ms | 2.20 ms | 10.1 | 21.2 |

7-worker contention factor (per-plan under load / alone): 1.90.

Deterministic / TTA episode seconds (mean over the 20 layouts): grid_ppo_plain_s0 deterministic 6.2 ms; grid_ppo_plain_s0 + TTA 37.1 ms; grid_ppo_d4_s0 deterministic 7.1 ms; grid_ppo_d4_s0 + TTA 43.4 ms; grid_ddqn_s0 deterministic 6.6 ms; grid_ddqn_s0 + TTA 34.9 ms

B15 planner session = 768,000 plans (planner plain + D4, RAS, RPS, RPS-full; 10 learners x 120 layouts x 128 each). Wall at 7 workers: ideal 0.085 h; with contention 0.162 h; worst model (6.04 ms/plan) with contention 0.170 h. Deterministic + TTA evaluation of the 30 models: 0.012 h wall.
Fallback selector on the measured times: step 0 (full), fits 3 h: True.
