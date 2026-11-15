# B13c diagnostics (TUNE-val only, role validation; final.zip at t = 300k of pilot_plain_s9001 / 9002 / 9003; no training, no FINAL access)

## 1a. armies x deployers decomposition (SPEC 7.3, quoted)

> ### 7.3 Pilot diagnostic: armies × deployers decomposition (DEV only)
>
> Part of B13, reusing the prototype's `diagnose_buy_deploy` design. Layouts: **TUNE-val (60) only** — FINAL and LAB-DESIGN are never loaded (test). Applied to the 3 pilot PPO plain seeds:
>
> | army \ deployer | pilot PPO deploy | scripted deploy | Oracle deploy (reference) |
> |---|---|---|---|
> | own buying (policy buys) | ✓ | ✓ | – |
> | forced Greedy army | ✓ | ✓ | – |
> | forced best-single army | ✓ | ✓ | – |
> | forced random army (phase-A generator) | ✓ | ✓ | – |
> | forced Oracle army (army-search result) | ✓ | ✓ | ✓ |
>
> Armies are forced through real buy actions + END_BUY; control checks as in step 9b (own-buy × PPO reproduces the validation eval exactly; scripted × Greedy reproduces the baseline file). Purpose: when G7 fails, say whether buying or deployment is the bottleneck, and give the redesign (or shaped-reward ablation) something specific to target. It is a **diagnostic, not a tuning loop**: it can trigger a revision only through the iteration rule above; no hyperparameter is changed because of it.

Pinned choices (SPEC silent, none depends on a result): forced random army = `random_types(default_rng([crc32(layout id), 7313]))` (one army per layout, the same for all seeds and deployers); forced best-single = per-family type chosen on TUNE-train (B8, Gate changelog (k)): archipelago=bulwark, bunker_nest=sniper, spine_pods=sniper; forced Oracle army = `army_search` result (128 random + 128 hill-climb, scripted deployment, rng [layout seed, 4343], SPEC 3.4 G8(a)); scripted deployer = `place_on_border` in Greedy ranking order from the default start tile; PPO deployer = deterministic masked argmax after the forced buys + END_BUY (a unit left undeployed when the policy plays WAIT is forfeited).

**Oracle-deploy column NOT computed (SPEC ambiguity, owner decision needed):** the table row says `forced Oracle army (army-search result)` with an Oracle-deploy reference cell, but the SPEC defines no deployment-only search for a fixed army. See the options in the report message.

Reading caveats: the forced Oracle army is SELECTED by searching armies under the scripted deployer, so the scripted column is optimistic for that row by construction (winner's curse) and the PPO column is not; the forced armies other than `own` are not armies the pilot policy chose, so its deployment on them is partly off-distribution. Differences between rows are armies (buying), differences inside a row are deployers.

### Controls

| control | max abs diff |
|---|---|
| own_x_ppo_vs_val_eval_s9001 (n=60) | score 0, D 0, breach 0 |
| own_x_ppo_vs_val_eval_s9002 (n=60) | score 0, D 0, breach 0 |
| own_x_ppo_vs_val_eval_s9003 (n=60) | score 0, D 0, breach 0 |
| greedy_x_scripted_vs_b8_Greedy (n=60) | score 0, D 0 |
| best_single_x_scripted_vs_b8_BestSingle (n=60) | score 0, D 0 |

### Mean score / D / breach, spine_pods, pooled over the 3 seeds (per-seed numbers in decomposition.csv)

| army \ deployer | PPO deploy | scripted deploy | PPO - scripted |
|---|---|---|---|
| own | 0.577 / 0.528 / 0.650 | 0.432 / 0.098 / 0.933 | +0.145 |
| greedy | 0.295 / 0.447 / 0.067 | 0.380 / 0.000 / 0.950 | -0.085 |
| best_single | 0.545 / 0.464 / 0.667 | 0.441 / 0.069 / 1.000 | +0.104 |
| random | 0.267 / 0.266 / 0.267 | 0.316 / 0.060 / 0.700 | -0.050 |
| oracle_army | 0.394 / 0.401 / 0.383 | 0.773 / 0.622 / 1.000 | -0.379 |

### Mean score / D / breach, archipelago, pooled over the 3 seeds (per-seed numbers in decomposition.csv)

| army \ deployer | PPO deploy | scripted deploy | PPO - scripted |
|---|---|---|---|
| own | 0.416 / 0.405 / 0.433 | 0.531 / 0.330 / 0.833 | -0.115 |
| greedy | 0.425 / 0.542 / 0.250 | 0.516 / 0.360 / 0.750 | -0.091 |
| best_single | 0.425 / 0.542 / 0.250 | 0.516 / 0.360 / 0.750 | -0.091 |
| random | 0.214 / 0.256 / 0.150 | 0.213 / 0.154 / 0.300 | +0.001 |
| oracle_army | 0.245 / 0.286 / 0.183 | 0.834 / 0.724 / 1.000 | -0.589 |

### Mean score / D / breach, bunker_nest, pooled over the 3 seeds (per-seed numbers in decomposition.csv)

| army \ deployer | PPO deploy | scripted deploy | PPO - scripted |
|---|---|---|---|
| own | 0.331 / 0.497 / 0.083 | 0.234 / 0.190 / 0.300 | +0.098 |
| greedy | 0.213 / 0.355 / 0.000 | 0.085 / 0.109 / 0.050 | +0.128 |
| best_single | 0.404 / 0.529 / 0.217 | 0.185 / 0.176 / 0.200 | +0.219 |
| random | 0.157 / 0.261 / 0.000 | 0.055 / 0.058 / 0.050 | +0.102 |
| oracle_army | 0.257 / 0.429 / 0.000 | 0.407 / 0.312 / 0.550 | -0.150 |

### Mean score / D / breach, ALL, pooled over the 3 seeds (per-seed numbers in decomposition.csv)

| army \ deployer | PPO deploy | scripted deploy | PPO - scripted |
|---|---|---|---|
| own | 0.441 / 0.476 / 0.389 | 0.399 / 0.206 / 0.689 | +0.042 |
| greedy | 0.311 / 0.448 / 0.106 | 0.327 / 0.156 / 0.583 | -0.016 |
| best_single | 0.458 / 0.512 / 0.378 | 0.381 / 0.201 / 0.650 | +0.077 |
| random | 0.212 / 0.261 / 0.139 | 0.195 / 0.091 / 0.350 | +0.018 |
| oracle_army | 0.299 / 0.372 / 0.189 | 0.672 / 0.553 / 0.850 | -0.373 |

### Per seed, all families (score)

| army \ deployer | 9001 ppo | 9001 scripted | 9002 ppo | 9002 scripted | 9003 ppo | 9003 scripted |
|---|---|---|---|---|---|---|
| own | 0.444 | 0.396 | 0.407 | 0.392 | 0.473 | 0.409 |
| greedy | 0.323 | 0.327 | 0.296 | 0.327 | 0.315 | 0.327 |
| best_single | 0.492 | 0.381 | 0.445 | 0.381 | 0.437 | 0.381 |
| random | 0.222 | 0.195 | 0.208 | 0.195 | 0.206 | 0.195 |
| oracle_army | 0.322 | 0.672 | 0.273 | 0.672 | 0.301 | 0.672 |

## 1b. breach vs destruction (same 60 layouts; PPO = own buying + PPO deployment = the G7 eval)

| seed | family | policy | breach | D | score |
|---|---|---|---|---|---|
| 9001 | spine_pods | PPO | 0.700 | 0.498 | 0.579 |
| 9001 | spine_pods | BestSingle | 1.000 | 0.069 | 0.441 |
| 9001 | spine_pods | Greedy | 0.950 | 0.000 | 0.380 |
| 9001 | spine_pods | Oracle | 1.000 | 0.781 | 0.869 |
| 9001 | archipelago | PPO | 0.450 | 0.402 | 0.421 |
| 9001 | archipelago | BestSingle | 0.750 | 0.360 | 0.516 |
| 9001 | archipelago | Greedy | 0.750 | 0.360 | 0.516 |
| 9001 | archipelago | Oracle | 1.000 | 0.916 | 0.950 |
| 9001 | bunker_nest | PPO | 0.050 | 0.518 | 0.331 |
| 9001 | bunker_nest | BestSingle | 0.200 | 0.176 | 0.185 |
| 9001 | bunker_nest | Greedy | 0.050 | 0.109 | 0.085 |
| 9001 | bunker_nest | Oracle | 1.000 | 0.761 | 0.857 |
| 9001 | ALL | PPO | 0.400 | 0.473 | 0.444 |
| 9001 | ALL | BestSingle | 0.650 | 0.201 | 0.381 |
| 9001 | ALL | Greedy | 0.583 | 0.156 | 0.327 |
| 9001 | ALL | Oracle | 1.000 | 0.820 | 0.892 |
| 9002 | spine_pods | PPO | 0.550 | 0.557 | 0.554 |
| 9002 | spine_pods | BestSingle | 1.000 | 0.069 | 0.441 |
| 9002 | spine_pods | Greedy | 0.950 | 0.000 | 0.380 |
| 9002 | spine_pods | Oracle | 1.000 | 0.781 | 0.869 |
| 9002 | archipelago | PPO | 0.350 | 0.413 | 0.388 |
| 9002 | archipelago | BestSingle | 0.750 | 0.360 | 0.516 |
| 9002 | archipelago | Greedy | 0.750 | 0.360 | 0.516 |
| 9002 | archipelago | Oracle | 1.000 | 0.916 | 0.950 |
| 9002 | bunker_nest | PPO | 0.050 | 0.434 | 0.280 |
| 9002 | bunker_nest | BestSingle | 0.200 | 0.176 | 0.185 |
| 9002 | bunker_nest | Greedy | 0.050 | 0.109 | 0.085 |
| 9002 | bunker_nest | Oracle | 1.000 | 0.761 | 0.857 |
| 9002 | ALL | PPO | 0.317 | 0.468 | 0.407 |
| 9002 | ALL | BestSingle | 0.650 | 0.201 | 0.381 |
| 9002 | ALL | Greedy | 0.583 | 0.156 | 0.327 |
| 9002 | ALL | Oracle | 1.000 | 0.820 | 0.892 |
| 9003 | spine_pods | PPO | 0.700 | 0.529 | 0.597 |
| 9003 | spine_pods | BestSingle | 1.000 | 0.069 | 0.441 |
| 9003 | spine_pods | Greedy | 0.950 | 0.000 | 0.380 |
| 9003 | spine_pods | Oracle | 1.000 | 0.781 | 0.869 |
| 9003 | archipelago | PPO | 0.500 | 0.399 | 0.439 |
| 9003 | archipelago | BestSingle | 0.750 | 0.360 | 0.516 |
| 9003 | archipelago | Greedy | 0.750 | 0.360 | 0.516 |
| 9003 | archipelago | Oracle | 1.000 | 0.916 | 0.950 |
| 9003 | bunker_nest | PPO | 0.150 | 0.538 | 0.383 |
| 9003 | bunker_nest | BestSingle | 0.200 | 0.176 | 0.185 |
| 9003 | bunker_nest | Greedy | 0.050 | 0.109 | 0.085 |
| 9003 | bunker_nest | Oracle | 1.000 | 0.761 | 0.857 |
| 9003 | ALL | PPO | 0.450 | 0.488 | 0.473 |
| 9003 | ALL | BestSingle | 0.650 | 0.201 | 0.381 |
| 9003 | ALL | Greedy | 0.583 | 0.156 | 0.327 |
| 9003 | ALL | Oracle | 1.000 | 0.820 | 0.892 |
| ALL | spine_pods | PPO | 0.650 | 0.528 | 0.577 |
| ALL | spine_pods | BestSingle | 1.000 | 0.069 | 0.441 |
| ALL | spine_pods | Greedy | 0.950 | 0.000 | 0.380 |
| ALL | spine_pods | Oracle | 1.000 | 0.781 | 0.869 |
| ALL | archipelago | PPO | 0.433 | 0.405 | 0.416 |
| ALL | archipelago | BestSingle | 0.750 | 0.360 | 0.516 |
| ALL | archipelago | Greedy | 0.750 | 0.360 | 0.516 |
| ALL | archipelago | Oracle | 1.000 | 0.916 | 0.950 |
| ALL | bunker_nest | PPO | 0.083 | 0.497 | 0.331 |
| ALL | bunker_nest | BestSingle | 0.200 | 0.176 | 0.185 |
| ALL | bunker_nest | Greedy | 0.050 | 0.109 | 0.085 |
| ALL | bunker_nest | Oracle | 1.000 | 0.761 | 0.857 |
| ALL | ALL | PPO | 0.389 | 0.476 | 0.441 |
| ALL | ALL | BestSingle | 0.650 | 0.201 | 0.381 |
| ALL | ALL | Greedy | 0.583 | 0.156 | 0.327 |
| ALL | ALL | Oracle | 1.000 | 0.820 | 0.892 |

PPO vs BestSingle layout by layout (fractions of layouts; BestSingle = train-chosen type, B8 rows):

| seed | family | n | both breach | only PPO | only BestSingle | neither | PPO higher D, no breach, BS breaches | PPO higher D | PPO higher score |
|---|---|---|---|---|---|---|---|---|---|
| 9001 | spine_pods | 20 | 0.700 | 0.000 | 0.300 | 0.000 | 0.300 | 1.000 | 0.700 |
| 9001 | archipelago | 20 | 0.300 | 0.150 | 0.450 | 0.100 | 0.300 | 0.550 | 0.350 |
| 9001 | bunker_nest | 20 | 0.000 | 0.050 | 0.200 | 0.750 | 0.100 | 0.850 | 0.750 |
| 9001 | ALL | 60 | 0.333 | 0.067 | 0.317 | 0.283 | 0.233 | 0.800 | 0.600 |
| 9002 | spine_pods | 20 | 0.550 | 0.000 | 0.450 | 0.000 | 0.400 | 0.950 | 0.600 |
| 9002 | archipelago | 20 | 0.150 | 0.200 | 0.600 | 0.050 | 0.300 | 0.500 | 0.400 |
| 9002 | bunker_nest | 20 | 0.000 | 0.050 | 0.200 | 0.750 | 0.100 | 0.900 | 0.800 |
| 9002 | ALL | 60 | 0.233 | 0.083 | 0.417 | 0.267 | 0.267 | 0.783 | 0.600 |
| 9003 | spine_pods | 20 | 0.700 | 0.000 | 0.300 | 0.000 | 0.250 | 0.900 | 0.800 |
| 9003 | archipelago | 20 | 0.350 | 0.150 | 0.400 | 0.100 | 0.250 | 0.600 | 0.500 |
| 9003 | bunker_nest | 20 | 0.000 | 0.150 | 0.200 | 0.650 | 0.100 | 0.850 | 0.750 |
| 9003 | ALL | 60 | 0.350 | 0.100 | 0.300 | 0.250 | 0.200 | 0.783 | 0.683 |
| ALL | spine_pods | 60 | 0.650 | 0.000 | 0.350 | 0.000 | 0.317 | 0.950 | 0.700 |
| ALL | archipelago | 60 | 0.267 | 0.167 | 0.483 | 0.083 | 0.283 | 0.550 | 0.417 |
| ALL | bunker_nest | 60 | 0.000 | 0.083 | 0.200 | 0.717 | 0.100 | 0.867 | 0.767 |
| ALL | ALL | 180 | 0.306 | 0.083 | 0.344 | 0.267 | 0.233 | 0.789 | 0.628 |

## 1c. units bought / deployed and type mix (mean per layout)

| seed | family | policy | bought | deployed | gold | n scout | n sniper | n bulwark | n blaster | n mender | gold% scout | gold% sniper | gold% bulwark | gold% blaster | gold% mender |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9001 | spine_pods | PPO | 8.65 | 8.65 | 234 | 0.95 | 5.60 | 1.05 | 1.05 | 0.00 | 3 | 67 | 13 | 16 | 0 |
| 9001 | spine_pods | BestSingle | 8.00 | 8.00 | 224 | 0.00 | 8.00 | 0.00 | 0.00 | 0.00 | 0 | 100 | 0 | 0 | 0 |
| 9001 | spine_pods | Greedy | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| 9001 | archipelago | PPO | 8.70 | 8.70 | 233 | 1.15 | 5.45 | 0.75 | 1.35 | 0.00 | 4 | 66 | 10 | 21 | 0 |
| 9001 | archipelago | BestSingle | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| 9001 | archipelago | Greedy | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| 9001 | bunker_nest | PPO | 8.95 | 8.95 | 232 | 2.00 | 3.10 | 1.65 | 2.20 | 0.00 | 7 | 37 | 21 | 34 | 0 |
| 9001 | bunker_nest | BestSingle | 8.00 | 8.00 | 224 | 0.00 | 8.00 | 0.00 | 0.00 | 0.00 | 0 | 100 | 0 | 0 | 0 |
| 9001 | bunker_nest | Greedy | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| 9001 | ALL | PPO | 8.77 | 8.77 | 233 | 1.37 | 4.72 | 1.15 | 1.53 | 0.00 | 5 | 57 | 15 | 24 | 0 |
| 9001 | ALL | BestSingle | 8.00 | 8.00 | 229 | 0.00 | 5.33 | 2.67 | 0.00 | 0.00 | 0 | 65 | 35 | 0 | 0 |
| 9001 | ALL | Greedy | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| 9002 | spine_pods | PPO | 8.25 | 8.25 | 232 | 0.60 | 3.85 | 2.90 | 0.85 | 0.05 | 2 | 47 | 38 | 13 | 1 |
| 9002 | spine_pods | BestSingle | 8.00 | 8.00 | 224 | 0.00 | 8.00 | 0.00 | 0.00 | 0.00 | 0 | 100 | 0 | 0 | 0 |
| 9002 | spine_pods | Greedy | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| 9002 | archipelago | PPO | 8.60 | 8.60 | 230 | 0.85 | 5.20 | 2.35 | 0.20 | 0.00 | 3 | 63 | 31 | 3 | 0 |
| 9002 | archipelago | BestSingle | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| 9002 | archipelago | Greedy | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| 9002 | bunker_nest | PPO | 7.90 | 7.90 | 231 | 0.55 | 1.95 | 3.55 | 1.70 | 0.15 | 2 | 24 | 46 | 27 | 2 |
| 9002 | bunker_nest | BestSingle | 8.00 | 8.00 | 224 | 0.00 | 8.00 | 0.00 | 0.00 | 0.00 | 0 | 100 | 0 | 0 | 0 |
| 9002 | bunker_nest | Greedy | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| 9002 | ALL | PPO | 8.25 | 8.25 | 231 | 0.67 | 3.67 | 2.93 | 0.92 | 0.07 | 2 | 44 | 38 | 14 | 1 |
| 9002 | ALL | BestSingle | 8.00 | 8.00 | 229 | 0.00 | 5.33 | 2.67 | 0.00 | 0.00 | 0 | 65 | 35 | 0 | 0 |
| 9002 | ALL | Greedy | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| 9003 | spine_pods | PPO | 8.30 | 8.30 | 225 | 0.60 | 6.25 | 1.15 | 0.30 | 0.00 | 2 | 78 | 15 | 5 | 0 |
| 9003 | spine_pods | BestSingle | 8.00 | 8.00 | 224 | 0.00 | 8.00 | 0.00 | 0.00 | 0.00 | 0 | 100 | 0 | 0 | 0 |
| 9003 | spine_pods | Greedy | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| 9003 | archipelago | PPO | 7.70 | 7.70 | 226 | 0.00 | 5.35 | 1.35 | 1.00 | 0.00 | 0 | 66 | 18 | 16 | 0 |
| 9003 | archipelago | BestSingle | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| 9003 | archipelago | Greedy | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| 9003 | bunker_nest | PPO | 8.55 | 8.55 | 230 | 0.85 | 5.05 | 2.20 | 0.45 | 0.00 | 3 | 61 | 29 | 7 | 0 |
| 9003 | bunker_nest | BestSingle | 8.00 | 8.00 | 224 | 0.00 | 8.00 | 0.00 | 0.00 | 0.00 | 0 | 100 | 0 | 0 | 0 |
| 9003 | bunker_nest | Greedy | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| 9003 | ALL | PPO | 8.18 | 8.18 | 227 | 0.48 | 5.55 | 1.57 | 0.58 | 0.00 | 2 | 68 | 21 | 9 | 0 |
| 9003 | ALL | BestSingle | 8.00 | 8.00 | 229 | 0.00 | 5.33 | 2.67 | 0.00 | 0.00 | 0 | 65 | 35 | 0 | 0 |
| 9003 | ALL | Greedy | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| ALL | spine_pods | PPO | 8.40 | 8.40 | 230 | 0.72 | 5.23 | 1.70 | 0.73 | 0.02 | 2 | 64 | 22 | 11 | 0 |
| ALL | spine_pods | BestSingle | 8.00 | 8.00 | 224 | 0.00 | 8.00 | 0.00 | 0.00 | 0.00 | 0 | 100 | 0 | 0 | 0 |
| ALL | spine_pods | Greedy | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| ALL | archipelago | PPO | 8.33 | 8.33 | 230 | 0.67 | 5.33 | 1.48 | 0.85 | 0.00 | 2 | 65 | 19 | 13 | 0 |
| ALL | archipelago | BestSingle | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| ALL | archipelago | Greedy | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| ALL | bunker_nest | PPO | 8.47 | 8.47 | 231 | 1.13 | 3.37 | 2.47 | 1.45 | 0.05 | 4 | 41 | 32 | 23 | 1 |
| ALL | bunker_nest | BestSingle | 8.00 | 8.00 | 224 | 0.00 | 8.00 | 0.00 | 0.00 | 0.00 | 0 | 100 | 0 | 0 | 0 |
| ALL | bunker_nest | Greedy | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
| ALL | ALL | PPO | 8.40 | 8.40 | 230 | 0.84 | 4.64 | 1.88 | 1.01 | 0.02 | 3 | 56 | 25 | 16 | 0 |
| ALL | ALL | BestSingle | 8.00 | 8.00 | 229 | 0.00 | 5.33 | 2.67 | 0.00 | 0.00 | 0 | 65 | 35 | 0 | 0 |
| ALL | ALL | Greedy | 8.00 | 8.00 | 240 | 0.00 | 0.00 | 8.00 | 0.00 | 0.00 | 0 | 0 | 100 | 0 | 0 |
