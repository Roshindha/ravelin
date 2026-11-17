# B15 results: FINAL, one session

Source: `results/b15/tables/` only (`tables.md` and the CSVs beside it). No number here was computed outside those tables; the P1-P5 table below is those rows reformatted. Session `3ccef86e0e2349d4af22e210d4b46a07`, run at commit `7e3aa42`, evaluated once; the tables were rebuilt from the raw files into a temporary directory and match byte for byte. Protocol: SPEC_V2 rows (gg) and (hh).

Setup in brief: 120 FINAL layouts, reported as **FINAL-familiar** (60: spine_pods, archipelago, bunker_nest, 20 each) and **FINAL-transfer** (60, all labyrinth). Learners are 10 seeds each of PPO plain, PPO+D4 and DDQN (deterministic masked argmax, final checkpoint). Intervals are 95 % hierarchical bootstrap (seeds, then layouts; 10 000 resamples). A primary comparison "holds" only if its paired-difference interval excludes 0 in the direction claimed. "Breach rate" is the headline metric; "score" is the true (unshaped) mean score.

## Plain-language summary

1. **On breach rate no learned policy beats BestSingle.** BestSingle (8 Snipers) breaches 0.700 of FINAL-familiar and 1.000 of FINAL-transfer; PPO plain, PPO+D4 and DDQN breach 0.34-0.39 and 0.60-0.65. P1 on breach is "not established" in all six cells, and every interval lies wholly below zero (the learners breach *less* often than BestSingle).
2. **On score the learners are no better than BestSingle on familiar layouts, and worse on transfer.** P1 on score on FINAL-familiar: not established (intervals include 0 for all three). On FINAL-transfer, BestSingle's score (0.666) is above all three learners' (0.517 / 0.518 / 0.563), and the paired intervals exclude 0 in BestSingle's favour. Against Greedy, the learners hold on score on FINAL-familiar (3 of 3) and nowhere else.
3. **The planner beats the strongest equal-N controls on FINAL-familiar** (6 of 6 comparisons hold on score; on breach 4 of 6 hold: PPO plain N = 8, 32, 128 and PPO+D4 N = 8). On FINAL-transfer it holds on score only for PPO plain at N = 8 and 32 (2 of 6), and on breach it holds in none; there the strongest control (RAS-N) breaches 0.977-0.997 of layouts and is *ahead* of the planner on breach. The planner has privileged simulator access (SPEC 7, row r), so this is a claim about a search method, not about the policy alone.
4. **TTA holds 12 of 12** (3 learners x 2 groups x 2 metrics): test-time augmentation over the 8 D4 views raises both breach rate and score every time.
5. **D4 training versus plain is not established** (P3: 0 of 4). All four differences are small, with intervals that include 0.
6. **Labyrinth is easier in absolute terms.** Every baseline, including Random, scores higher on FINAL-transfer than on FINAL-familiar. The learners gain less than the baselines do, so they fall further behind BestSingle and Greedy there.
7. **Multiplicity.** There are 64 primary checks (16 comparisons x 2 groups x 2 metrics) with no correction. At a 5 % level about 3 of 64 could hold by chance alone (a rough guide: the checks are correlated). 27 of 64 hold, which is well above that, but the holds cluster in three places (TTA 12, planner vs control 12, learners vs Greedy 3); the other 37 do not hold.

## P1-P5: primary comparisons (27 of 64 hold)

Each cell is the difference A - B with its primary 95 % interval, in score or breach-rate units. "Holds" is marked. Where a cell is not marked, the comparison is "not established", whatever the sign of the point estimate. For breach rate a *negative* A - B means A breached less often, so it counts against "A beats B".

\* For P4 the control is the strongest equal-N control by mean true score in that group (rule pinned in (gg)): **RPS-full-N on FINAL-familiar and RAS-N on FINAL-transfer** at every N (table below).

| ID | Comparison (A beats B) | Familiar: breach | Familiar: score | Transfer: breach | Transfer: score |
|---|---|---|---|---|---|
| P1 | PPO plain beats BestSingle | -0.360 [-0.480, -0.242] | +0.056 [-0.020, +0.133] | -0.403 [-0.497, -0.310] | -0.149 [-0.207, -0.092] |
| P1 | PPO+D4 beats BestSingle | -0.352 [-0.477, -0.225] | +0.073 [-0.006, +0.154] | -0.365 [-0.468, -0.272] | -0.148 [-0.214, -0.083] |
| P1 | DDQN beats BestSingle | -0.310 [-0.432, -0.187] | +0.052 [-0.027, +0.134] | -0.348 [-0.438, -0.262] | -0.103 [-0.159, -0.049] |
| P2 | PPO plain beats Greedy | -0.243 [-0.360, -0.128] | +0.099 [+0.020, +0.177] **holds** | -0.353 [-0.462, -0.242] | -0.084 [-0.146, -0.022] |
| P2 | PPO+D4 beats Greedy | -0.235 [-0.358, -0.112] | +0.116 [+0.038, +0.194] **holds** | -0.315 [-0.427, -0.205] | -0.082 [-0.155, -0.012] |
| P2 | DDQN beats Greedy | -0.193 [-0.313, -0.072] | +0.095 [+0.013, +0.176] **holds** | -0.298 [-0.403, -0.192] | -0.037 [-0.101, +0.025] |
| P3 | PPO+D4 beats PPO plain | +0.008 [-0.073, +0.092] | +0.017 [-0.028, +0.062] | +0.038 [-0.052, +0.128] | +0.001 [-0.063, +0.064] |
| P4 | Planner on PPO plain N=8 beats the strongest equal-N control* | +0.160 [+0.085, +0.237] **holds** | +0.163 [+0.129, +0.198] **holds** | -0.103 [-0.170, -0.042] | +0.051 [+0.004, +0.096] **holds** |
| P4 | Planner on PPO plain N=32 beats the strongest equal-N control* | +0.103 [+0.025, +0.183] **holds** | +0.115 [+0.085, +0.143] **holds** | -0.038 [-0.083, -0.003] | +0.038 [+0.005, +0.071] **holds** |
| P4 | Planner on PPO plain N=128 beats the strongest equal-N control* | +0.068 [+0.013, +0.130] **holds** | +0.086 [+0.062, +0.111] **holds** | -0.012 [-0.038, +0.008] | +0.016 [-0.011, +0.046] |
| P4 | Planner on PPO+D4 N=8 beats the strongest equal-N control* | +0.097 [+0.028, +0.167] **holds** | +0.120 [+0.091, +0.149] **holds** | -0.142 [-0.225, -0.065] | -0.001 [-0.056, +0.053] |
| P4 | Planner on PPO+D4 N=32 beats the strongest equal-N control* | +0.038 [-0.033, +0.113] | +0.076 [+0.048, +0.105] **holds** | -0.068 [-0.128, -0.020] | -0.011 [-0.054, +0.030] |
| P4 | Planner on PPO+D4 N=128 beats the strongest equal-N control* | -0.018 [-0.077, +0.043] | +0.037 [+0.011, +0.063] **holds** | -0.043 [-0.092, -0.005] | -0.032 [-0.066, +0.003] |
| P5 | PPO plain + TTA beats PPO plain | +0.123 [+0.057, +0.192] **holds** | +0.093 [+0.058, +0.129] **holds** | +0.162 [+0.087, +0.237] **holds** | +0.097 [+0.049, +0.143] **holds** |
| P5 | PPO+D4 + TTA beats PPO+D4 | +0.083 [+0.018, +0.152] **holds** | +0.067 [+0.028, +0.105] **holds** | +0.112 [+0.057, +0.170] **holds** | +0.078 [+0.041, +0.116] **holds** |
| P5 | DDQN + TTA beats DDQN | +0.115 [+0.048, +0.187] **holds** | +0.086 [+0.045, +0.128] **holds** | +0.140 [+0.070, +0.210] **holds** | +0.084 [+0.046, +0.124] **holds** |

Holds by comparison: P1 0 of 12; P2 3 of 12 (all FINAL-familiar score); P3 0 of 4; P4 12 of 24 (familiar breach 4, familiar score 6, transfer score 2, transfer breach 0); P5 12 of 12. Total 27 of 64.

Reading the sign on the cells that did not hold (descriptive; the pinned rule only tests "A beats B"):

- P1 breach: A - B is -0.31 to -0.40 with intervals wholly below 0 in all six cells: the learners breach *less* often than BestSingle.
- P1 score, FINAL-transfer: all three intervals are wholly below 0 (PPO plain [-0.207, -0.092], PPO+D4 [-0.214, -0.083], DDQN [-0.159, -0.049]): BestSingle scores higher than each learner.
- P2 breach: wholly below 0 for all three learners on both groups: Greedy breaches more often. P2 score on FINAL-transfer: PPO plain and PPO+D4 are wholly below 0 (Greedy higher); DDQN's interval includes 0.
- P4 FINAL-transfer breach: five of six intervals are wholly below 0 (the exception is PPO plain N = 128, [-0.038, +0.008]): RAS-N breaches at least as often as the planner. RAS-N is near the ceiling there (0.977 / 0.995 / 0.997).

## Strongest equal-N control

Rule: highest mean true score at that group and N among RAS-N, RPS-N and RPS-full-N; all three are shown. (From `strongest_control.csv`.)

| Group | N | RAS score | RPS score | RPS-full score | RAS breach | RPS breach | RPS-full breach | Strongest |
|---|---|---|---|---|---|---|---|---|
| FINAL-familiar | 8 | 0.4631 | 0.3478 | 0.4960 | 0.7650 | 0.2600 | 0.4533 | RPS-full |
| FINAL-familiar | 32 | 0.5543 | 0.5124 | 0.6390 | 0.8450 | 0.5083 | 0.6500 | RPS-full |
| FINAL-familiar | 128 | 0.6390 | 0.6363 | 0.7457 | 0.8883 | 0.6717 | 0.7900 | RPS-full |
| FINAL-transfer | 8 | 0.7175 | 0.4621 | 0.6277 | 0.9767 | 0.6000 | 0.7717 | RAS |
| FINAL-transfer | 32 | 0.8306 | 0.6595 | 0.7762 | 0.9950 | 0.8583 | 0.9250 | RAS |
| FINAL-transfer | 128 | 0.9104 | 0.7907 | 0.8768 | 0.9967 | 0.9400 | 0.9700 | RAS |

The strongest control is chosen by score, not by breach. On FINAL-familiar the RAS controls have the highest breach rate (0.765 / 0.845 / 0.888) but low score, because they spend little on damage. A planner claim that is stated on breach should be read against RAS-N, not against the strongest-by-score control. Planner cells for reference (breach / score):

| Group | Learner | N = 8 | N = 32 | N = 128 |
|---|---|---|---|---|
| FINAL-familiar | PPO plain | 0.613 / 0.659 | 0.753 / 0.754 | 0.858 / 0.832 |
| FINAL-familiar | PPO+D4 | 0.550 / 0.616 | 0.688 / 0.715 | 0.772 / 0.783 |
| FINAL-transfer | PPO plain | 0.873 / 0.768 | 0.957 / 0.868 | 0.985 / 0.927 |
| FINAL-transfer | PPO+D4 | 0.835 / 0.717 | 0.927 / 0.819 | 0.953 / 0.878 |

## Headline conditions (breach rate / mean score, 95 % primary intervals)

| Condition | FINAL-familiar breach | FINAL-familiar score | FINAL-transfer breach | FINAL-transfer score |
|---|---|---|---|---|
| Random | 0.060 [0.033, 0.090] | 0.107 [0.088, 0.126] | 0.197 [0.140, 0.253] | 0.158 [0.128, 0.188] |
| Greedy | 0.583 [0.450, 0.700] | 0.338 [0.275, 0.403] | 0.950 [0.883, 1.000] | 0.600 [0.561, 0.640] |
| BestSingle | 0.700 [0.583, 0.817] | 0.381 [0.312, 0.447] | 1.000 [1.000, 1.000] | 0.666 [0.626, 0.704] |
| Oracle (upper reference, about 400 simulations) | 1.000 | 0.892 [0.857, 0.924] | 1.000 | 0.964 [0.938, 0.984] |
| PPO plain | 0.340 [0.252, 0.432] | 0.437 [0.390, 0.486] | 0.597 [0.503, 0.690] | 0.517 [0.462, 0.571] |
| PPO+D4 | 0.348 [0.255, 0.447] | 0.454 [0.406, 0.505] | 0.635 [0.532, 0.728] | 0.518 [0.455, 0.577] |
| DDQN | 0.390 [0.295, 0.490] | 0.434 [0.381, 0.486] | 0.652 [0.562, 0.738] | 0.563 [0.512, 0.611] |
| PPO plain + TTA | 0.463 [0.363, 0.565] | 0.530 [0.481, 0.581] | 0.758 [0.675, 0.838] | 0.614 [0.563, 0.663] |
| PPO+D4 + TTA | 0.432 [0.323, 0.542] | 0.521 [0.469, 0.574] | 0.747 [0.643, 0.840] | 0.596 [0.526, 0.663] |
| DDQN + TTA | 0.505 [0.402, 0.610] | 0.520 [0.465, 0.575] | 0.792 [0.693, 0.878] | 0.647 [0.594, 0.699] |

All other conditions (RAS / RPS / RPS-full at N = 8, 32, 128, planners, diagnostics, per-family and pooled rows, the learner-only secondary intervals) are in `results/b15/tables/tables.md`. BestSingle-perlayout and BestSingle-hindsight are diagnostics, never comparison baselines. Oracle is a fixed-budget search and a lower bound on the optimum. Learners spend the budget: mean units bought is 8.7 (PPO plain), 9.4 (PPO+D4) and 10.2 (DDQN) on FINAL-familiar, against 8 for the fixed baselines.

## Generalisation gap (descriptive, not a test)

DEV-train in-sample score (150 layouts, final models, seen data) minus FINAL-familiar score; the two layout sets are resampled independently in the interval. (`generalisation_gap.csv`.)

| Kind | Condition | DEV-train | FINAL-familiar | Gap | 95 % interval |
|---|---|---|---|---|---|
| learner | PPO plain | 0.7054 | 0.4370 | **0.2683** | [0.2131, 0.3229] |
| learner | PPO+D4 | 0.4864 | 0.4545 | **0.0319** | [-0.0242, 0.0866] |
| learner | DDQN | 0.7268 | 0.4335 | **0.2933** | [0.2335, 0.3516] |
| baseline (difficulty reference) | Random | 0.1173 | 0.1065 | 0.0108 | [-0.0138, 0.0356] |
| baseline (difficulty reference) | Greedy | 0.3416 | 0.3380 | 0.0035 | [-0.0706, 0.0784] |
| baseline (difficulty reference) | BestSingle | 0.4435 | 0.3810 | 0.0625 | [-0.0144, 0.1397] |
| baseline (difficulty reference) | Oracle | 0.9077 | 0.8923 | 0.0154 | [-0.0228, 0.0554] |

The baselines change by 0.00-0.06 between the two sets, so the FINAL-familiar layouts are not materially harder than DEV-train. PPO plain and DDQN fall by about 0.27-0.29, far more than that. PPO+D4 falls by 0.03 (interval includes 0) but starts from a much lower in-sample score (0.486 against 0.705), so D4 mostly removes the in-sample advantage rather than adding out-of-sample score: on FINAL-familiar PPO+D4 and PPO plain are indistinguishable (P3). That reading is exploratory. The in-sample scores come from seen layouts and are not a performance claim.

## Per-family notes (exploratory; 20 layouts per familiar family, intervals are wide)

Breach rate / mean score; full tables in `tables.md`.

- **spine_pods:** BestSingle 0.900 / 0.426, Greedy 0.900 / 0.363, PPO plain 0.570 / 0.504, PPO+D4 0.605 / 0.519, DDQN 0.700 / 0.524; with TTA 0.78-0.83 breach and 0.60-0.63 score. Oracle 1.000 / 0.826. Planner PPO plain N = 8: 0.865 / 0.732.
- **archipelago:** BestSingle 0.900 / 0.544, Greedy 0.850 / 0.590 (Greedy's score is above every learner row, with or without TTA: PPO plain 0.424, PPO+D4 0.442, DDQN 0.400). Learners breach 0.34-0.36. Oracle 1.000 / 0.984. Planner PPO plain N = 8: 0.715 / 0.702.
- **bunker_nest:** the hardest family. Greedy 0.000 / 0.062, Random 0.000 / 0.124, BestSingle 0.300 / 0.173; learners 0.09-0.13 breach with 0.377-0.403 score (DDQN + TTA 0.275 / 0.497). RAS-N breaches more often (0.43-0.665) at a score of 0.28-0.44, comparable to the learners; RPS-full-128 reaches 0.400 / 0.614. Planner PPO plain: 0.260 / 0.543 at N = 8 up to 0.620 / 0.731 at N = 128. Oracle breaches all 20 (0.867 score). Here the learners score higher than BestSingle (0.377-0.403 vs 0.173) but breach less often (0.09-0.13 vs 0.30).
- **labyrinth (FINAL-transfer, 60 layouts):** BestSingle breaches 1.000 and Greedy 0.950; learners 0.60-0.65 (TTA 0.75-0.79). Oracle score 0.964. RAS-128 breaches 0.997, the planner on PPO plain N = 128 0.985.

## Exploratory comparisons (no multiplicity correction; none of these is a "holds")

- **E1 planner vs its own learner:** the planner is above its learner in all 24 cells (6 planner cells x 2 groups x 2 metrics). Examples: PPO plain N = 8 on FINAL-familiar breach +0.273 [0.202, 0.347], N = 128 +0.518 [0.432, 0.607]; the smallest score difference is +0.161 (PPO+D4 N = 8, familiar). The planner also uses simulator access the learner does not have.
- **E2 N vs N:** more samples help in every cell, with diminishing returns: for the planner on PPO plain, N = 32 vs 8 gives +0.140 breach / +0.095 score on FINAL-familiar, and N = 128 vs 32 gives +0.105 / +0.078. On FINAL-transfer, N = 128 vs 32 gives +0.028 [0.005, 0.058] breach, near the ceiling.
- **E3 DDQN vs PPO:** no difference is distinguishable except possibly on FINAL-transfer score (DDQN +0.046 [-0.003, +0.099] vs PPO plain; +0.045 [-0.009, +0.104] vs PPO+D4), and those include 0.
- **E4 transfer minus familiar** (two independent layout sets):

| Condition | breach | score |
|---|---|---|
| Random | +0.137 [0.073, 0.200] | +0.052 [0.016, 0.088] |
| Greedy | +0.367 [0.233, 0.500] | +0.262 [0.184, 0.338] |
| BestSingle | +0.300 [0.183, 0.417] | +0.285 [0.209, 0.364] |
| Oracle | +0.000 | +0.071 [0.031, 0.112] |
| PPO plain | +0.257 [0.120, 0.385] | +0.080 [0.004, 0.152] |
| PPO+D4 | +0.287 [0.153, 0.415] | +0.063 [-0.010, 0.138] |
| DDQN | +0.262 [0.127, 0.393] | +0.129 [0.058, 0.201] |

  Every condition is higher on FINAL-transfer, including Random and Oracle, so labyrinth is easier in absolute terms (Oracle score 0.964 against 0.892). The baselines' score rises by 0.26-0.29 and the learners' by 0.06-0.13, which is why the learners fall behind BestSingle and Greedy on transfer score (P1, P2). This is a comparison across different layout families, not a controlled difficulty measure.

## What this does and does not show

- It shows that the learned policies are not better breachers than a fixed single-unit army on these layouts, and that their value, where it exists, is in score (damage dealt) on familiar layouts. It does not show that they generalise to labyrinth.
- It shows the planner improves on equal-N random controls on familiar layouts. On transfer it improves score over RAS-N at N = 8 and 32 only, and not breach.
- It shows TTA helps every learner, in both groups and on both metrics.
- It does not show a benefit of D4 training over plain training.
- Breach rate saturates for several baselines (BestSingle, RAS-N, Oracle on transfer), which limits what breach can say there; score is the better discriminator on FINAL-transfer.
- 10 seeds per learner, 60 layouts per group, one session. The 37 primary checks that did not hold are "not established", not evidence of no effect.
