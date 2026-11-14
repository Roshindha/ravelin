# B5 report (BreachLens v2)

Revision 1 | commit `21f71ff577` (dirty=False) | sim_version `3048422b62c94dcd...` | layouts: {'TUNE-train': 180, 'TUNE-val': 60, 'LAB-DESIGN': 40} | bootstrap: 10000 resamples, seed 8080

**All gates pass: False**  (failed: G2, G3, G4, G8).  generator_hash recorded: False

## 1. G1-G9 summary

| Gate | Result | Key numbers |
|---|---|---|
| G1 | **PASS** | max best-single share 0.54 (limit 0.60); units over ban-loss 0.25 everywhere: none |
| G2 | **FAIL** | units passing in: scout: spine_pods; sniper: spine_pods,archipelago,bunker_nest; bulwark: spine_pods,archipelago,bunker_nest; blaster: NONE; mender: spine_pods,archipelago |
| G3 | **FAIL** | niche-layout share (need >= 8%, lo95 > 0): scout 7.9% [5.0%]; sniper 33.6% [28.2%]; bulwark 9.3% [6.1%]; blaster 1.1% [0.0%]; mender 15.0% [11.1%] |
| G4 | **FAIL** | Mender gain bunker_nest 0.017 [0.006, 0.030], LAB 0.000 [0.000, 0.000]; Mender-only TUNE 0.000 / LAB 0.000 |
| G5 | **PASS** | spine_pods: best-single sniper 0.456, Oracle 0.885 / breach 1.00; archipelago: best-single bulwark 0.503, Oracle 0.928 / breach 1.00; bunker_nest: best-single sniper 0.277, Oracle 0.835 / breach 0.96 |
| G6 (diagnostic) | **FAIL** | Oracle >12 units on 0.00 of layouts (diagnostic, was >=0.30); layouts with 20 Scouts >= Oracle: 0 |
| G8 | **FAIL** | spine_pods: army-search +0.273 (89% layouts), Oracle +0.429; archipelago: army-search +0.275 (88% layouts), Oracle +0.425; bunker_nest: army-search +0.207 (56% layouts), Oracle +0.558 |
| G9 | **PASS** | Oracle 0.956 [0.924, 0.982], breach 1.00; Greedy 0.603 [0.540, 0.662], 95% of layouts >= 0.05; stalemate with core alive: 0 |

G7 is not evaluated at B5 (pilot, B13).

## 2. Gate details

### G1
best-single-unit share per layout (ties split), pooled TUNE: scout 0.11, sniper 0.54, bulwark 0.26, blaster 0.08, mender 0.01

| family | spine_pods | archipelago | bunker_nest |
|---|---|---|---|
| best-single share scout | 0.00 | 0.02 | 0.31 |
| best-single share sniper | 0.66 | 0.48 | 0.49 |
| best-single share bulwark | 0.23 | 0.44 | 0.12 |
| best-single share blaster | 0.11 | 0.07 | 0.07 |
| best-single share mender | 0.00 | 0.00 | 0.01 |
| Oracle ban-loss scout | 0.003 | -0.023 | -0.036 |
| Oracle ban-loss sniper | 0.012 | -0.000 | 0.014 |
| Oracle ban-loss bulwark | 0.110 | 0.037 | 0.051 |
| Oracle ban-loss blaster | -0.034 | -0.031 | -0.058 |
| Oracle ban-loss mender | 0.015 | 0.016 | -0.037 |

### G2 paired Oracle ban-loss (mean [95% CI]; one-sided 98.75% lower bound; need mean >= 0.02, lo95 > 0, lo_1s > 0)

| unit | family | ban-loss | lo(98.75% 1-sided) | pass |
|---|---|---|---|---|
| scout | spine_pods | 0.030 [0.011, 0.052] | 0.009 | yes |
| scout | archipelago | 0.001 [0.000, 0.004] | 0.000 | no |
| scout | bunker_nest | 0.012 [0.003, 0.024] | 0.002 | no |
| sniper | spine_pods | 0.050 [0.027, 0.077] | 0.024 | yes |
| sniper | archipelago | 0.028 [0.012, 0.047] | 0.010 | yes |
| sniper | bunker_nest | 0.080 [0.047, 0.115] | 0.043 | yes |
| bulwark | spine_pods | 0.088 [0.058, 0.119] | 0.054 | yes |
| bulwark | archipelago | 0.043 [0.025, 0.063] | 0.023 | yes |
| bulwark | bunker_nest | 0.067 [0.043, 0.093] | 0.040 | yes |
| blaster | spine_pods | 0.015 [0.003, 0.031] | 0.001 | no |
| blaster | archipelago | 0.002 [0.000, 0.005] | 0.000 | no |
| blaster | bunker_nest | 0.004 [0.000, 0.009] | 0.000 | no |
| mender | spine_pods | 0.029 [0.013, 0.046] | 0.012 | yes |
| mender | archipelago | 0.031 [0.016, 0.048] | 0.014 | yes |
| mender | bunker_nest | 0.017 [0.006, 0.030] | 0.005 | no |

### G3 per-layout niches (B5-R1: pooled TUNE-train + TUNE-val + LAB-DESIGN, margin >= 0.03 over the runner-up template; need share >= 8% and lo95 > 0)

Mean template scores: scout 0.098, sniper 0.439, bulwark 0.367, blaster 0.242, bulwark_mender 0.392, mender_only 0.000

| unit | template | niche layouts | share [95% CI] | pass |
|---|---|---|---|---|
| scout | scout | 22 / 280 | 0.079 [0.050, 0.111] | no |
| sniper | sniper | 94 / 280 | 0.336 [0.282, 0.393] | yes |
| bulwark | bulwark | 26 / 280 | 0.093 [0.061, 0.129] | yes |
| blaster | blaster | 3 / 280 | 0.011 [0.000, 0.025] | no |
| mender | bulwark_mender | 42 / 280 | 0.150 [0.111, 0.193] | yes |

Descriptive only (per family, share of layouts where the unit's template wins by >= 0.03): spine_pods: scout 0.00, sniper 0.34, bulwark 0.01, blaster 0.00, mender 0.14, no clear winner 0.51; archipelago: scout 0.01, sniper 0.31, bulwark 0.23, blaster 0.01, mender 0.14, no clear winner 0.30; bunker_nest: scout 0.26, sniper 0.38, bulwark 0.04, blaster 0.01, mender 0.12, no clear winner 0.19; labyrinth: scout 0.00, sniper 0.30, bulwark 0.10, blaster 0.03, mender 0.25, no clear winner 0.33

### G4
- Mender gain on bunker_nest: 0.017 [0.006, 0.030] -> no
- Mender gain on LAB-DESIGN: 0.000 [0.000, 0.000] -> no

- Mender-only army on TUNE: mean 0.000 [0.000, 0.000] (upper95 0.000) -> ok
- Mender-only army on LAB-DESIGN: mean 0.000 [0.000, 0.000] (upper95 0.000) -> ok

### G5 / G6 / G8

| family | best single | mean | Oracle breach | G8a army-search gain [95% CI] | better on | G8b Oracle gain [95% CI] |
|---|---|---|---|---|---|---|
| spine_pods | sniper | 0.456 | 1.00 | 0.273 [0.229, 0.319] | 89% | 0.429 [0.391, 0.467] |
| archipelago | bulwark | 0.503 | 1.00 | 0.275 [0.228, 0.323] | 88% | 0.425 [0.388, 0.463] |
| bunker_nest | sniper | 0.277 | 0.96 | 0.207 [0.151, 0.264] | 56% | 0.558 [0.487, 0.628] |

G6: Oracle armies >12 units: 0.00 (by family spine_pods 0.00, archipelago 0.00, bunker_nest 0.01); 20 Scouts >= Oracle on 0 layouts; Oracle army exactly 20 Scouts on 0.

### G9 (LAB-DESIGN, 40)
checks: {'oracle_mean': True, 'oracle_breach': True, 'greedy_mean_in_range': True, 'greedy_layout_share': True, 'no_stalemate_core_alive': True}; exit reasons over all battles: {'all_dead': 533, 'no_attackers': 212, 'horizon': 74, 'cleared': 21}

## 3. Per-unit statics (effective hp per gold incl. armor, dps per gold, threatening defenders)

| unit | cost | hp | armor | EHP/gold Gun | Lancer | Flak (centre) | Flak (neighbour hit) | mean of 3 | DPS/gold | threatened by (<=3 hits & range) | outranged by | can hit from inside blind spot of |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scout | 8 | 16 | 0 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 0.750 | gun (2 hits), lancer (1 hits), flak (2 hits) | gun, lancer, flak | lancer, flak |
| sniper | 28 | 28 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.429 | lancer (2 hits), flak (2 hits) | lancer | none |
| bulwark | 30 | 170 | 4 | 8.50 | 7.29 | 7.93 | 13.22 | 7.91 | 0.533 | none | gun, lancer, flak | lancer, flak |
| blaster | 36 | 60 | 0 | 1.67 | 1.67 | 1.67 | 1.67 | 1.67 | 0.722 (3-bldg cluster 1.444) | none | gun, lancer, flak | none |
| mender | 28 | 40 | 0 | 1.43 | 1.43 | 1.43 | 1.43 | 1.43 | 0.000 (heal 0.286) | lancer (3 hits), flak (3 hits) | gun, lancer, flak | none |

## 4. Single-unit-army score table (mean score [95% CI] | D | breach | units / gold; * = best per family)

| unit | spine_pods (n=80) | archipelago (n=80) | bunker_nest (n=80) | LAB-DESIGN (n=40) |
|---|---|---|---|---|
| scout | 0.053 [0.043, 0.063] / D 0.088 / breach 0.00 / 20.0u 160g | 0.094 [0.080, 0.108] / D 0.156 / breach 0.00 / 20.0u 160g | 0.131 [0.106, 0.156] / D 0.218 / breach 0.00 / 20.0u 160g | 0.128 [0.073, 0.186] / D 0.030 / breach 0.28 / 20.0u 160g |
| sniper | 0.456 [0.431, 0.479]* / D 0.110 / breach 0.97 / 8.0u 224g | 0.478 [0.422, 0.531] / D 0.288 / breach 0.76 / 8.0u 224g | 0.277 [0.218, 0.338]* / D 0.186 / breach 0.41 / 8.0u 224g | 0.653 [0.593, 0.712]* / D 0.438 / breach 0.97 / 8.0u 224g |
| bulwark | 0.399 [0.382, 0.415] / D 0.016 / breach 0.97 / 8.0u 240g | 0.503 [0.461, 0.543]* / D 0.313 / breach 0.79 / 8.0u 240g | 0.080 [0.057, 0.106] / D 0.109 / breach 0.04 / 8.0u 240g | 0.603 [0.540, 0.662] / D 0.388 / breach 0.93 / 8.0u 240g |
| blaster | 0.255 [0.209, 0.301] / D 0.026 / breach 0.60 / 6.0u 216g | 0.266 [0.221, 0.311] / D 0.210 / breach 0.35 / 6.0u 216g | 0.070 [0.050, 0.091] / D 0.116 / breach 0.00 / 6.0u 216g | 0.512 [0.437, 0.584] / D 0.287 / breach 0.85 / 6.0u 216g |
| mender | 0.000 [0.000, 0.000] / D 0.000 / breach 0.00 / 8.0u 224g | 0.000 [0.000, 0.000] / D 0.000 / breach 0.00 / 8.0u 224g | 0.000 [0.000, 0.000] / D 0.000 / breach 0.00 / 8.0u 224g | 0.000 [0.000, 0.000] / D 0.000 / breach 0.00 / 8.0u 224g |

## 5. Wall behaviour (walls attacked per battle per unit type; share of time spent attacking walls)

| family | unit | walls attacked / battle (single army) | wall share of alive unit-ticks (single) | walls attacked / battle (Oracle army) | wall share (Oracle army) |
|---|---|---|---|---|---|
| spine_pods | scout | 1.46 | 0.131 | 0.20 | 0.190 |
| spine_pods | sniper | 4.29 | 0.237 | 3.25 | 0.182 |
| spine_pods | bulwark | 6.64 | 0.210 | 4.03 | 0.160 |
| spine_pods | blaster | 1.44 | 0.073 | 0.41 | 0.060 |
| archipelago | scout | 1.44 | 0.104 | 0.30 | 0.138 |
| archipelago | sniper | 4.45 | 0.248 | 2.83 | 0.178 |
| archipelago | bulwark | 7.47 | 0.195 | 3.95 | 0.139 |
| archipelago | blaster | 1.68 | 0.084 | 0.81 | 0.091 |
| bunker_nest | scout | 9.05 | 0.452 | 1.25 | 0.259 |
| bunker_nest | sniper | 17.66 | 0.683 | 16.49 | 0.517 |
| bunker_nest | bulwark | 24.20 | 0.533 | 12.80 | 0.472 |
| bunker_nest | blaster | 10.95 | 0.606 | 4.55 | 0.464 |
| LAB-DESIGN | scout | 17.43 | 0.718 | 3.23 | 0.588 |
| LAB-DESIGN | sniper | 30.23 | 0.677 | 9.50 | 0.589 |
| LAB-DESIGN | bulwark | 38.48 | 0.529 | 28.07 | 0.532 |
| LAB-DESIGN | blaster | 11.72 | 0.329 | 6.80 | 0.440 |

| family | single-unit armies: wall-attack share of attacker unit-ticks | ticks with any wall attack / battle ticks | Oracle armies: share of attacker unit-ticks | ticks with any wall attack |
|---|---|---|---|---|
| spine_pods | 0.200 | 0.364 | 0.171 | 0.241 |
| archipelago | 0.198 | 0.353 | 0.160 | 0.226 |
| bunker_nest | 0.606 | 0.802 | 0.498 | 0.626 |
| LAB-DESIGN | 0.622 | 0.746 | 0.550 | 0.729 |

### Monotone ban-loss of every unit in every family (mean [95% CI]; one-sided 98.75% lower bound; reported, only the 3 TUNE families gate G2)

| unit | spine_pods | archipelago | bunker_nest | LAB-DESIGN |
|---|---|---|---|---|
| scout | 0.030 [0.011, 0.052] (1s lo 0.009) | 0.001 [0.000, 0.004] (1s lo 0.000) | 0.012 [0.003, 0.024] (1s lo 0.002) | 0.000 [0.000, 0.000] (1s lo 0.000) |
| sniper | 0.050 [0.027, 0.077] (1s lo 0.024) | 0.028 [0.012, 0.047] (1s lo 0.010) | 0.080 [0.047, 0.115] (1s lo 0.043) | 0.004 [0.000, 0.011] (1s lo 0.000) |
| bulwark | 0.088 [0.058, 0.119] (1s lo 0.054) | 0.043 [0.025, 0.063] (1s lo 0.023) | 0.067 [0.043, 0.093] (1s lo 0.040) | 0.041 [0.011, 0.077] (1s lo 0.008) |
| blaster | 0.015 [0.003, 0.031] (1s lo 0.001) | 0.002 [0.000, 0.005] (1s lo 0.000) | 0.004 [0.000, 0.009] (1s lo 0.000) | 0.000 [0.000, 0.000] (1s lo 0.000) |
| mender | 0.029 [0.013, 0.046] (1s lo 0.012) | 0.031 [0.016, 0.048] (1s lo 0.014) | 0.017 [0.006, 0.030] (1s lo 0.005) | 0.000 [0.000, 0.000] (1s lo 0.000) |

## 6. Oracle / baselines per family

| set | Oracle mean | Oracle breach | Oracle units | Oracle counts s/sn/b/bl/m | army-search mean | Greedy mean |
|---|---|---|---|---|---|---|
| spine_pods | 0.885 | 1.00 | 9.0 | 1.8, 2.0, 2.5, 1.1, 1.7 | 0.729 | 0.399 |
| archipelago | 0.928 | 1.00 | 9.1 | 1.7, 2.0, 2.4, 1.1, 1.9 | 0.778 | 0.503 |
| bunker_nest | 0.835 | 0.96 | 9.3 | 2.1, 2.9, 2.2, 1.0, 1.2 | 0.483 | 0.080 |
| LAB-DESIGN | 0.956 | 1.00 | 9.2 | 1.9, 1.9, 3.1, 1.0, 1.3 | 0.858 | 0.603 |

**Gold share of each unit in Oracle armies** (pooled gold / mean of per-layout shares)

| set | scout | sniper | bulwark | blaster | mender |
|---|---|---|---|---|---|
| spine_pods | 0.061 / 0.061 | 0.241 / 0.242 | 0.330 / 0.326 | 0.164 / 0.167 | 0.205 / 0.204 |
| archipelago | 0.057 / 0.057 | 0.243 / 0.244 | 0.304 / 0.304 | 0.163 / 0.163 | 0.233 / 0.231 |
| bunker_nest | 0.073 / 0.071 | 0.345 / 0.349 | 0.285 / 0.282 | 0.150 / 0.149 | 0.148 / 0.148 |
| LAB-DESIGN | 0.064 / 0.064 | 0.225 / 0.230 | 0.396 / 0.393 | 0.158 / 0.158 | 0.156 / 0.155 |

## 7. Defender mix per family and generator-rule checks (spec 3.1b)

| family | layouts | mean defenders | gun / lancer / flak total | share with Lancer | share with Flak | gun majority |
|---|---|---|---|---|---|---|
| spine_pods | 80 | 8.3 | 391 / 255 / 17 | 1.00 | 0.16 | 1.00 |
| archipelago | 80 | 8.4 | 372 / 225 / 78 | 1.00 | 0.75 | 1.00 |
| bunker_nest | 80 | 8.7 | 411 / 269 / 12 | 1.00 | 0.12 | 1.00 |
| labyrinth | 40 | 5.1 | 72 / 82 / 50 | 1.00 | 1.00 | 0.17 |

Rule checks: {'spine_pods_lancer>=70%': True, 'bunker_nest_lancer>=70%': True, 'archipelago_flak>=50%': True, 'labyrinth_uses_all_three': True, 'spine_pods_gun_majority_every_layout': True, 'archipelago_gun_majority_every_layout': True, 'bunker_nest_gun_majority_every_layout': True}

## 8. Stat-dependent acceptance (too_hard, single_type_trivial) on TUNE / LAB-DESIGN streams

| family | structurally accepted & evaluated | single_type_trivial | too_hard | rejected (either) | rejection rate | flag >50% |
|---|---|---|---|---|---|---|
| spine_pods | 86 | 6 | 0 | 6 | 7.0% | no |
| archipelago | 202 | 122 | 0 | 122 | 60.4% | **FLAG** |
| bunker_nest | 96 | 13 | 3 | 16 | 16.7% | no |
| labyrinth | 58 | 18 | 0 | 18 | 31.0% | no |
