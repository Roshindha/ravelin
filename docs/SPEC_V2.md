# BreachLens (v2) — design spec

Status: **design only** (step 10a; amended in step 10b: gates G7–G9, movement tie rule, planner control, speed gate, step-budget rule, shaped-reward fallback, siege/ removal; amended in step 10c: paired-bootstrap gates, G7 BestSingle definition and 600k extension, revision budget 3+1, defender types, B5 report, README placeholders; amended in step B5-R1: per-layout niche G3, sweep selection rule; amended in step B5: `no_attackers` exit, one-sided 98.75 % bound for G2/G3, pre-registered B5 gate-script interpretations, extra wall metrics). No code has been written for v2. The prototype is frozen at git tag `prototype-v1`; all v2 work lives on branch `v2-redesign`.
Supersedes: [SPEC.md](SPEC.md) (Base Siege). Where this file is silent, SPEC.md's rules still apply (PROGRESS.md after every step, pytest per module, fixed seeds, one step at a time, report in ≤ 5 lines).

Every number marked **(proposed)** is a starting value to be tested by the balance gates in §3.4, not a result. Every timing marked **(estimate)** is extrapolated from prototype measurements in §9 and must be re-measured in build step B10 before any grid is launched.

---

## 1. Concept

A drone fleet breaches an automated citadel. The agent buys a fleet under a budget, then deploys it on the border; the deterministic simulator runs the battle. The agent never steers units.

- **Score** = `0.6 · D + 0.4 · C`
  - `D` = value-weighted destruction = `Σ v_b·[b destroyed] / Σ v_b` over **non-core** buildings (`v_b ∈ {1,2,3}` assigned by the generator; turrets and walls have no value, they are obstacles).
  - `C` = 1 if the core is destroyed, else 0.
- **Success metric = breach rate** (mean `C`). Mean score and `D` are reported alongside.
- Reward is terminal: the score of the finished battle, nothing else (no step penalties, no shaping). The prototype's "full clear" is replaced by breach.

### What the prototype taught us (and what v2 changes)

| Prototype finding (PROGRESS.md) | v2 response |
|---|---|
| Splash was dead; tank was weakest after tuning | Every unit must pass explicit non-dominance *and* non-deadness gates (§3.4), tuned on a separate seed set, 2 iterations max |
| D4 hurt at 1M steps (underfit, deployment worst) | D4 kept as a condition, but also tested as test-time ensembling (§7.1); step budget raised; Mender tie-breaks designed to be equivariant from the start (§3.3) |
| Phase-A used a fixed Greedy army → PPO only ever saw one post-buy state before phase B | Phase A uses **random armies from step 0** (§6) |
| Wall targeting was a global rank; tanks/splash chewed every wall first | Target choice is nearest-within-preference, not global (§3.3) |
| `ring_defence` as transfer made "best single" meaningless (0.000) | BestSingle baseline is chosen per *family-agnostic DEV pool*, and transfer is reported against Oracle/Greedy/Random plus a hindsight reference only |
| Residual 4/7000 D4 mismatches came from id-ordered ties | **Ids never decide anything.** Exact ties are resolved by a symmetric action (split or abstain) — §3.3 |
| Battle had 50 ticks | 240 ticks, early exit when nothing can change |

---

## 2. World parameters (fixed)

| Parameter | Value |
|---|---|
| Grid | 14 × 14 |
| Budget | 240 gold |
| Battle horizon | 240 ticks (cap; battle exits early when no unit can act or the base is cleared) |
| Max units | 20 |
| Border tiles | 52 (`4·14 − 4`), indexed clockwise from top-left, as in the prototype |
| Unit types | 5 |
| Phases | BUY → DEPLOY → BATTLE (battle resolved inside one `simulate_battle` call, frames recorded for replay) |

---

## 3. Units, turrets, simulator rules

### 3.1 Units (proposed stats)

| Unit | Cost | HP | Damage/tick | Range | Speed | Armor | Target pref. | Special |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **Scout** | 8 | 16 | 6 | 1.2 | 1.5 | 0 | building > turret > wall | Cheap swarm body |
| **Sniper** | 24 | 28 | 12 | 5.0 | 1.0 | 0 | turret > building > wall | Outranges range-4 turrets |
| **Bulwark** | 30 | 170 | 16 | 1.4 | 0.7 | 4 | wall > turret > building | Flat −4 on each hit taken; draws turret fire (turrets shoot nearest unit) |
| **Blaster** | 36 | 60 | 26 | 2.5 | 0.7 | 0 | building > wall > turret | Splash r = 1.5, 50 % falloff, hits structures only |
| **Mender** | 28 | 40 | 0 | — | 1.0 | 0 | — (no attack) | Heals **nearest damaged ally within range 2 per tick**, 8 hp/tick |

**The 20-unit cap is a design lever**: 20 Scouts cost only 160, so a swarm leaves 80 gold unspendable — the agent is pushed to trade cheap bodies for expensive ones, and the cap makes "buy everything cheap" a deliberate choice, not a free lunch.

### 3.1b Defenders (proposed stats) and what carries score value

| Defender | HP | Damage | Range (min–max) | Rate | Targeting | Role |
|---|---:|---:|---|---|---|---|
| **Gun turret** | 90 | 12 | 0–4.0 | every tick | nearest unit | Baseline defender; cheap to place, many per layout |
| **Lancer** (long-range) | 70 | 18 | 2.0–6.0 | every tick | nearest unit in range | **Threatens range-5 units**: out-ranges Snipers (5.0), kills a 28-hp Sniper in 2 hits and a Scout in 1. Blind spot inside 2.0 is the counterplay (Scouts, Bulwarks that close in) |
| **Flak turret** (area) | 100 | 14 centre / 7 neighbours | 2.0–5.0 | every 2nd tick | the unit with the most units within 1.5 (ties: more total unit cost in the burst, then nearer) | **Area-damage defender that punishes clumped Scouts**: a 3×3 clump of Scouts (16 hp) takes 7–14 per burst, so 2 bursts erase it; barely scratches a lone Bulwark. Splash r = 1.5, hits units only |

- All volleys are simultaneous (tick order, §3.3). Target ties: key chain on D4-invariant properties (hp fraction, cost, Euclidean distance), then the canonical-form fallback of rule 7 (splash volleys never "split"; they resolve to one centre).
- **Generator requirements** (checked as acceptance/test rules, not tuned afterwards): `spine_pods` and `bunker_nest` layouts contain ≥ 1 Lancer in ≥ 70 % of layouts; `archipelago` layouts contain ≥ 1 Flak in ≥ 50 % of layouts; `labyrinth` uses all three types; every familiar family keeps Gun turrets as the majority (≥ 50 % of defenders). So Snipers always face a real counter somewhere and clumped Scouts are always punishable — the §3.2 "loses when" columns are structural, not hopes.
- Defender type is not a separate observation channel; it is recoverable from the footprint radius in the threat channel (Gun 4, Flak 5, Lancer 6, with a hollow centre of radius 2 for the latter two) — see §4.2 ch 6.
- Walls: HP 80, destructible. Cores: HP 200.
- **Score value (explicit):** *walls carry no score value, and neither do turrets.* Only non-core buildings enter `D` (weights `v_b ∈ {1,2,3}`) and the core enters `C`. Walls and defenders matter only instrumentally (they block and kill). This keeps Bulwark/Blaster from being paid for chewing walls and removes any "destroy every wall" incentive.

### 3.2 Why each unit should be worth buying (hypotheses to test, not claims)

| Unit | Wins when | Loses when |
|---|---|---|
| Scout | `archipelago`: many lightly guarded pods; Gun turrets can only kill so many bodies per volley, so mass beats HP | Against dense overlapping fire (`bunker_nest`) 16 hp dies in one volley, and **a clump is punished by Flak turrets** (7–14 per burst to every Scout in the 1.5 radius); Lancers one-shot it |
| Sniper | `spine_pods`: Gun turrets (range 4) guarding the spine are killed without return fire | **Lancers (range 6) out-range it** and kill it in 2 hits; Flak range 5 matches it; useless vs walls |
| Bulwark | `bunker_nest`: armor makes 10–16 dmg turret hits cost 6–12; nearest-unit targeting spares squishier allies; breaks walls | Open layouts: speed 0.7 means it arrives late, and 30 gold buys 3.75 Scouts |
| Blaster | Clustered pods (`spine_pods` pods, `bunker_nest` interiors): one shot hits 3+ buildings, wall lines | `archipelago` islands (nothing to splash); 60 hp is fragile |
| Mender | Sustained fire on a tanky target (Bulwark + Mender in `bunker_nest`, `labyrinth` corridors); +8 hp/tick offsets one turret | Burst deaths (Scouts); costs a unit slot and does zero damage on its own |

### 3.3 Simulator rules that matter for exactness

1. **Fixed-point arithmetic.** All hp, damage, heal values are integers scaled by `S = 840` (= lcm(1…8)), so splitting by any k ≤ 8 is exact and no float comparison can differ across symmetries.
2. **Tick order.** (a) snapshot start-of-tick state; (b) every turret and unit selects a target from the snapshot; (c) all damage and heals computed from the snapshot; (d) apply damage (armor first, then hp), then apply heals to units that survived; clamp at max hp; (e) remove dead; (f) movement (from the snapshot of survivors).
3. **Mender heal rule.** Candidates: allies alive at start of tick, `hp < max`, not itself, Euclidean distance ≤ 2.0. Choose by the key chain `(distance, hp_fraction, −cost)` — all D4-invariant. If still tied, **split the heal equally across all tied candidates** (exact, because of `S`). Heals from several Menders add. No overheal, no revive, a unit killed this tick receives nothing.
4. **Mender movement.** Uses the same movement rule as every unit (rule 7) with the target point = centroid of living non-Mender allies (D4-covariant); stops within 1.5 of it; with no allies left, holds.
5. **General tie principle:** tie-breaks go through D4-invariant properties (hp, value, Euclidean distance, summed distance to the rest of the base). **Object ids and raw coordinates never decide.** If an exact tie survives: attacks split damage and heals split (exact, because of `S`); movement uses the canonical-form fallback of rule 7. Movement never "abstains" on a tie, because abstaining can deadlock a unit on a symmetric approach.
6. **Target choice** = most-preferred class that has any target in range, then nearest within that class (not a global rank across the map). Fixes the prototype's wall-first behaviour.
7. **Movement and the exact tie-break for equal-length shortest paths.**
   - *Move set:* 8-connected, one step per tick, cost 1 per step (Chebyshev metric). A diagonal step is legal only if both orthogonally adjacent cells are passable (no corner cutting). Wall cells are impassable until destroyed; structures and other units never block (units may share a cell).
   - *Distance field:* BFS `d(c)` = steps from `c` to the nearest cell from which the unit's current target is within its range. Computed per target, cached per (layout, target) between wall destructions.
   - *Candidates:* `N(p) = { n passable neighbour of p : d(n) = d(p) − 1 }`. A unit with `d(p) > 0` and `N(p)` empty is blocked by a wall: it attacks the wall that lies on its shortest path (cheapest by `d` through the wall), by the same key chain below.
   - *Key chain (lexicographic minimum wins; every key is integer-valued and D4-invariant):*
     1. squared Euclidean distance from `n` to the target cell (this alone picks the diagonal on an exact-diagonal approach, the straight step on an axis approach);
     2. number of enemy turrets whose range covers `n` (fewer first);
     3. squared distance from `n` to the centroid of all surviving value buildings and the core, computed as `m·|n|² − 2·n·Σx + |Σx|²` in integers;
     4. **canonical-form fallback:** for each still-tied candidate form the world state after that single move, serialize it, take the lexicographically smallest serialization over the 8 D4 transforms, and pick the candidate with the smallest canonical string. If two canonical strings are equal, the two candidate worlds are D4 images of each other and either choice yields equal scores.
   - *Ties among several units* are resolved sequentially in the order of their own canonical keys (type, hp, then the same fallback), each decision applied to the working copy before the next, so simultaneous tied units cannot desynchronize between a world and its transform.
   - The fallback fires rarely (only after keys 1–3 are exhausted); a counter is exposed so tests can assert it was exercised.
8. **Early exit.** Battle ends when the core and all value buildings are destroyed (`cleared`), all units are dead (`all_dead`), **no live unit has damage > 0 (`no_attackers`, B5: only Menders remain, or the army never had an attacker; it ends the battle immediately, is counted separately from stalemate, and cannot change any structure or the score)**, or no unit changed hp/position for 12 consecutive ticks (`stalemate`); cap 240 (`horizon`). The `stalemate` exit is the only one G9's "no stalemate with a reachable core" refers to.
9. Splash is radius-based (D4-invariant) and never damages units.

### 3.4 Design gates G1–G9 (pre-registered; thresholds frozen in this document before any simulator exists)

All gates use only DEV-style data: the TUNE sets and LAB-DESIGN of §3.5 (and, for G7/B13 diagnostics, DEV validation never FINAL). "Score" is the game score `0.6·D + 0.4·C`.
"Best fixed single-unit army" = for one unit type, buy as many as fit in `min(budget, 20 units)`, deploy with the scripted deployer (`place_on_border`, Greedy ranking order, default start tile); the best of the 5 types **per family, chosen in hindsight on the evaluation layouts** (generous to the baseline).

| Gate | When | Layouts | Pass condition |
|---|---|---|---|
| **G1** no dominant unit | B5 | TUNE (3 families) | No unit type is the best single-unit army on > 60 % of layouts, and no unit's removal from the Oracle's allowed set costs > 0.25 mean score in *every* family |
| **G2** no dead unit | B5 | TUNE | Every unit has paired Oracle ban-loss (monotone estimator, Gate changelog (d)) ≥ 0.02 mean score in ≥ 1 family **and** the 95 % bootstrap lower bound of that paired ban-loss is > 0 **and** the one-sided 98.75 % lower bound is > 0 (B5 clarification below), all in the same family |
| **G3** niche exists | B5 | TUNE + LAB-DESIGN (pooled) | Every unit has a **per-layout niche** (B5-R1 clarification below): over the template set (5 single-focus armies + the `bulwark_mender` template = 6 Bulwark + 2 Mender), the unit's own template is the best template by a margin ≥ 0.03 over the runner-up template on **≥ 8 % of layouts** (pooled over TUNE-train, TUNE-val and LAB-DESIGN) **and** the 95 % bootstrap lower bound of that layout share is > 0 |
| **G4** Mender specifically | B5 | TUNE + LAB-DESIGN | Paired Oracle(with Menders) − Oracle(without) (monotone estimator, Gate changelog (d)) ≥ 0.02 in `bunker_nest` or on LAB-DESIGN **and** 95 % lower bound > 0 in that same set; a Mender-only army has mean score < 0.01 (point threshold; the 95 % upper bound must also be < 0.02) |
| **G5** not trivial / not hopeless | B5 | TUNE | Best single-unit army mean score ∈ [0.25, 0.65] per family; Oracle mean score ∈ [0.80, 0.97] per family; Oracle breach rate lower bound ≥ 0.6 per family (no ceiling; see Gate changelog, 2026-11-14 (a)) |
| **G6** cap pressure (**reported diagnostic, not a blocker** since 2026-11-14 (b)) | B5 | TUNE | Reported: share of layouts where Oracle armies use > 12 units (previous target ≥ 30 %), and the layouts where 20 Scouts scores ≥ the Oracle (previous target: none). Does not enter `all_pass` |
| **G7** learnability pilot | B13, before the grid | TUNE-train → TUNE-val | 3 PPO plain seeds × 300k steps. Metric per seed = mean score on TUNE-val averaged over the last 3 validation evals (t = 250k / 275k / 300k). **BestSingle** = per family, the single-unit army (type chosen on **TUNE-train**, scripted deployment) **scored on TUNE-val**; **Greedy** = the Greedy army and scripted deployment scored on TUNE-val; both are recorded in B8 before any pilot exists, and the layout-level baseline is `max(BestSingle, Greedy)` on each layout. **Pass iff** mean over the 3 seeds `≥ max(BestSingle, Greedy) + 0.05` **and** ≥ 2 of 3 seeds individually exceed that max (margin > 0) **and** no collapse (≥ 10 distinct army compositions over TUNE-val for every seed, and no single unit type > 90 % of bought units in every family). D and breach rate are reported next to the score. **Pre-registered extension (at most once):** if G7 fails at 300k but the mean over the 3 seeds of the TUNE-val slope over the last 100k steps (least-squares, t = 200k…300k, in score per 100k steps) is **≥ +0.02**, continue the same 3 runs once to **600k** and apply the *same* pass condition with the last 3 evals (550k / 575k / 600k); otherwise it is a fail. No second extension; the shaped pilot (§7.4) gets none |
| **G8** buying matters | B5 (re-checked B13) | TUNE, per familiar family | (a) **Army search** (128 random legal armies + 128 composition hill-climb steps, scripted deployment, 256 sims) beats the best single-unit army by **≥ 0.10 mean score** and on ≥ 60 % of layouts; (b) the **full Oracle** (army + deployment search, ~400 sims) beats the best single-unit army by **≥ 0.20 mean score**. (a) isolates buying from deployment skill |
| **G9** labyrinth floor | B5 | LAB-DESIGN (40 layouts) | Oracle mean score ≥ 0.60 and breach rate ≥ 0.60; Greedy mean score ≥ 0.10 with ≥ 50 % of layouts at score ≥ 0.05 (clearly above 0), and Greedy mean ≤ 0.65 (not trivial; Gate changelog (c)). Also: no battle ends by stalemate exit with the core alive and a reachable path (checked on all 40 × several armies) |

**Gate changelog (post-B5-R1 sweep; written and committed before the sweep it motivates is run; thresholds of other gates are unchanged).**

| Date | Change | Reason |
|---|---|---|
| 2026-11-14 | (a) **G5**: Oracle mean score must lie in [0.80, 0.97] per family and the Oracle breach rate lower bound must be ≥ 0.6 per family. The old ceiling "Oracle breach ≤ 0.95" is removed. ("Lower bound" = the lower end of the old [0.6, 0.95] range; the breach rate is the point rate over the family's TUNE layouts.) | The B5-R1 sweep showed that a breach ceiling measures an unreachable-core property, not headroom: breach stayed 1.00 while Oracle score fell from 1.00 to 0.87-0.95. Oracle score ≤ 0.97 measures headroom directly; ≥ 0.80 keeps the world solvable |
| 2026-11-14 | (b) **G6** becomes a reported diagnostic (value and both parts are still computed and reported), not a blocker; it no longer enters `all_pass` | The cap-pressure target depends on a unit cap and unit costs that are being swept (R1c); failing it says nothing about learnability or about any unit being dead |
| 2026-11-14 | (c) **G9**: Greedy mean band becomes [0.10, 0.65] (was [0.10, 0.45]); all other G9 conditions unchanged | Revision-0 and sweep Greedy means on labyrinth are 0.58-0.65 while the Oracle is far above them; 0.45 was a guess made before any simulator existed |
| 2026-11-14 | (d) **G2 / G4** use the **monotone ban-loss estimator**: the full Oracle is warm-started with the best army found by the Oracle without the unit, and the ablated Oracle is warm-started likewise, so full ≥ ablated holds by construction | The independent-search estimator can report negative ban-losses (full Oracle below the ablated one) that are search noise, not unit value. The old and new estimator are reported side by side on the R0 reference data. Thresholds (≥ 0.02, 95 % and 98.75 % bounds) are unchanged |

**Paired-difference rule (G2, G3, G4).** Every compared quantity is a *per-layout paired difference* on the **same TUNE layouts** with the **same search seeds**: the Oracle (with / without a unit) uses the identical per-layout seed stream and budget (e.g. `[layout_id, 4242]`), and competing templates use the same template-search seeds. The statistic is the mean of the per-layout differences; the CI is a bootstrap **over layouts** (10 000 resamples, seed 8080, percentile 95 % interval). A gate part passes only if the **point estimate meets the stated threshold AND the 95 % lower bound is > 0**. Because "in ≥ 1 family" tests up to 4 families, each family is tested separately at the same level (no pooling to rescue a failing family) and the number of families tried is reported. Thresholds and the bootstrap settings are read by `scripts_gates.py` from `data/v2/gate_thresholds.json`, and a test checks that file against this document.

**Pre-registered clarification (B5, written before the B5 gate run).**
1. *Multiplicity bound for G2 (and, before B5-R1, G3; see item 6).* "In ≥ 1 family" is a search over families, so in addition to the point threshold and the 95 % percentile lower bound, G2 and G3 require a **one-sided 98.75 % bootstrap lower bound > 0** (Bonferroni over 4 families: 0.05 / 4 = 0.0125; the bound is the 1.25th percentile of the same 10 000 bootstrap means, seed 8080). It is applied to every family tested (3 TUNE families; LAB-DESIGN is reported but not used to rescue G2/G3) and is stricter than the 95 % two-sided bound, so it can only turn a pass into a fail. G4 keeps the 95 % rules. Values are in `data/v2/gate_thresholds.json` (`bootstrap.one_sided_level` = 0.9875).
2. *Oracle (all Oracle-based gates).* `breachlens/oracle.py`: 200 random legal armies (phase-A generator, stop p = 0.1, random start tile) + 200 hill-climbing proposals (replace / add / remove / start shift ±7 / order swap, strict improvement), consecutive-free-tile deployment from the start tile, rng `[layout seed, 4242]`; **warm starts** (no rng use): the 5 single-unit armies, the Bulwark+Mender template and the Greedy army at the default start tile, so the Oracle is never below those baselines. A banned-type Oracle uses the same rng stream and skips warm starts containing the banned type. Army search (G8a): 128 random armies + 128 replace/add/remove proposals, scripted deployment (Greedy ranking order, default start tile), rng `[seed, 4343]`.
3. *Scripted deployer and baselines.* `place_on_border`: units in order on consecutive border tiles clockwise from the default start tile (the border tile nearest the core, ties by index). Greedy ranking = hp·damage/cost² descending (Mender last). Greedy army = walk that ranking buying as many of each as fit (no Mender pairing in B5). Single-unit army = `min(20, 240 // cost)` units. "Best single-unit army" per family = the type with the highest mean score (hindsight, as defined above).
4. *G1* counts exact ties in the per-layout best single-unit army fractionally. *G3* is superseded by the B5-R1 clarification (item 6): the per-family "winner" reading was arithmetically capped at 3 winners for 5 units. *G6* "20 Scouts is the best army" on a layout means: the 20-Scout scripted army scores ≥ the Oracle's best score there; the gate requires zero such layouts. *G9 stalemate check*: 21 armies per LAB-DESIGN layout (5 single-unit, scripted, Greedy, 14 random with random start), violation = `stalemate` exit with the core alive.
5. *Stat-dependent acceptance (§5.1), as run at B5 on TUNE / LAB-DESIGN.* `too_hard`: an acceptance Oracle (same budget and warm starts, rng `[seed, 5151]`) finds no breach. `single_type_trivial`: the best single-unit army scores > 0.99. Rejected seeds are skipped and the next seed of the same stream is drawn; rejection rates per family are reported and any family above 50 % is flagged. FINAL and stream F are never generated or loaded.

6. *B5-R1 clarification of G3 (pre-registered, written before the revision-1 sweep and gate run; thresholds `G3.min_template_margin` = 0.03 and `G3.min_layout_share` = 0.08 in `data/v2/gate_thresholds.json`).* The original G3 asked each unit to win a whole family on the family mean; with 5 units and 3 tested families that cannot be satisfied, and it ignores that a niche is a property of layouts, not of family means. G3 is therefore a **per-layout niche** test. **Template set** = the 5 single-focus armies (`min(20, 240 // cost)` units of one type, scripted deployment: Scout, Sniper, Bulwark, Blaster, Mender) + the `bulwark_mender` template (6 Bulwark + 2 Mender, scripted deployment): 6 templates. **Ownership:** the Scout, Sniper, Bulwark and Blaster own their own single-focus army; the Mender owns `bulwark_mender` (its single-focus army scores about 0 and is a competitor only). **A layout is a niche layout for unit `u`** iff `u`'s template is the strictly best of the 6 and beats the runner-up template on that layout by **≥ 0.03 score**. **Pass:** for every unit, the niche-layout share is **≥ 8 %**, pooled over TUNE-train, TUNE-val and LAB-DESIGN (280 layouts), **and** the 95 % percentile bootstrap lower bound of that share is > 0 (bootstrap over layouts, 10 000 resamples, seed 8080, as above). Because the test is pooled and does not search over families, the one-sided 98.75 % bound of item 1 does **not** apply to G3 (it still applies to G2); per-family shares are reported but never used to pass or fail. This clarification changes only the gate definition; it is applied identically to every revision (the revision-0 raw data are re-scored under it for reference only).
7. *B5-R1 exploration sweep and selection rule (pre-registered, fixed before any sweep result existed; the sweep is not a revision).* Arms A-E (hp scaling of buildings, core and defenders with walls unchanged; Lancer share ~40 % of defenders with Guns still >= 50 %; Sniper cost 28; defender count x1.25) are evaluated on TUNE-train only (the first 20 accepted layouts per familiar family of each arm's own seed walk through stream T) and the first 20 accepted LAB-DESIGN layouts (stream L), with the same seed streams and search seeds for every arm; TUNE-val is not touched. Criteria, over the four families (three familiar + labyrinth): (1) Oracle breach <= 0.95 and Oracle mean score <= 0.97 in each family; (2) best single-unit army mean in [0.25, 0.65] in each family; (3) Sniper's best-single share, pooled over the three familiar families (ties split), <= 60 %; (4) stat-acceptance rejection rate <= 40 % in each family; (5) Greedy mean on labyrinth <= 0.45. The arm meeting the most criteria is chosen; ties go to the arm with the fewest changed parameters (A = 1, B = 2, C = 3, D = 2, E = 2), then to the earlier letter. If no arm meets at least 4 of 5, nothing is applied.
8. *B5-R1b exploration sweep and selection rule (pre-registered, fixed before any R1b sweep result existed; the sweep is not a revision and spends none).* Sweep parameters (all off by default, all behind the override hooks `BREACHLENS_STATS_PATH` / `BREACHLENS_GEN_PARAMS`, which gate runs and tests assert unset): per-unit **class-damage multipliers** (`class_damage_pct` per unit in the stats file: percent of damage dealt to building (the core counts as a building), turret and wall; default 100 each); **sealed cores** (`GEN_PARAMS.sealed_fraction`: that fraction of layouts, decided by a separate rng stream `[family, seed, 7373]`, has the 8 cells around the core turned into walls, displacing any building / defender standing there to the nearest free interior cell; for those layouts only, the "core reachable without breaking a wall" rule is waived and the family wall-count maximum is raised by 8); **core hp multiplier** (`structures.core_hp`); **unit cap** (`rules.unit_cap`, default 20, used by every army builder and the Oracle); **Mender heal / hp**. Arms (cumulative on the B5-R1 arm B = building / core / defender hp x1.4 + Lancer share ~40 %): **R1a** = B + Sniper 40 % vs buildings / walls / core (100 % vs turrets); **R1b** = R1a + sealed cores (fraction 0.5) + core hp x3 (of the arm-B value, 280 -> 840); **R1c** = R1b + unit cap 14; **R1d** = R1b + Mender heal 14 and hp 60. A revision-0 arm R0 is re-run with the same evaluator for the side-by-side ban-loss comparison. Layout sets: TUNE-train first 20 accepted per familiar family of each arm's own seed walk through stream T + the first 20 accepted LAB-DESIGN (stream L), same seed streams and search seeds for every arm; TUNE-val untouched. **Criteria (count how many are met), over the four families (three familiar + labyrinth/LAB-DESIGN):** (i) Sniper's best-single share, pooled over the three familiar families (ties split), <= 0.60; (ii) best single-unit army mean in [0.25, 0.65] in every family; (iii) stat-acceptance rejection rate <= 40 % in every family; (iv) Oracle mean score in [0.80, 0.97] in every family; (v) Mender monotone ban-loss >= 0.02 with 95 % lower bound > 0 in at least one family (labyrinth = LAB-DESIGN). The arm meeting the most criteria is chosen; ties go to the smallest change (R1a 1, R1b 3, R1c 4, R1d 5 changed parameters), then to the earlier letter. **If no arm meets at least 4 of 5, nothing is applied, no revision is spent, and the work stops with the table.** Otherwise the chosen arm is applied as revision 1 of 3 (stats file + generator parameters), versions are bumped, and the full gate script is run once from the clean commit.

**Why these margins.**
- *G7 margin 0.05.* In the prototype, two seeds of the same config differed by 0.08 (SD ≈ 0.05–0.06), so a 3-seed mean has SE ≈ 0.03; 0.05 is ≈ 1.5 SE. The step-6b adoption bar of 0.03 was later judged to be inside noise, and the prototype sanity run beat best-single by only 0.03 while ending ~0.46 below Oracle. Requiring 0.05 *and* 2-of-3 seeds *and* no collapse rejects "PPO ≈ a fixed army" (what the prototype produced on `scattered_outposts`) while staying reachable at 300k steps. It is deliberately **not** "close to Oracle": it only certifies the task is learnable enough to be worth 30 grid runs.
- *G8 margins.* Prototype `scattered_outposts`: an Oracle-chosen army with scripted deployment (.178) was *worse* than a fixed swarm (.568) — buying did not matter, so no buying policy could be rewarded. A mean gap of 0.10 is ≈ 3 SE for 80 layouts at per-layout SD ≈ 0.25; the stricter 0.20 for the full Oracle reflects that it also optimizes deployment (prototype Oracle was +0.46 over best single).
- *G9.* Oracle ≥ 0.6 shows the labyrinth is solvable with the movement model; Greedy ≥ 0.10 shows scripted play makes some progress there, so transfer numbers for learned agents are measured against a non-degenerate floor (the prototype's transfer family had BestSingle = 0.000).

### 3.5 Design sets, separation from FINAL, and the iteration rule

| Set | Size | Families | Seed stream | Used for | Never used for |
|---|---:|---|---|---|---|
| TUNE-train | 180 (60/family) | spine_pods, archipelago, bunker_nest | `T` | G1–G6, G8; G7 pilot training; BestSingle choice | FINAL, grid training |
| TUNE-val | 60 (20/family) | same | `T` | G7 pilot validation; pilot diagnostics (§7.3); G8 re-check | training in the grid |
| LAB-DESIGN | 40 | labyrinth | `L` | G9; labyrinth tuning; D4/deadlock tests | the pool, any training |
| DEV / FINAL pools (§5.2) | 300 | all | `F` | the experiment | design |

- Seed streams `T`, `L`, `F` are disjoint by construction (distinct bases; ranges recorded in `data/v2/design_sets.json`).
- **Separation of labyrinth design layouts from the FINAL labyrinth pool:** (1) FINAL labyrinth layouts are generated only at B6, from stream `F`; no FINAL labyrinth layout exists during B2–B5. (2) The generator source and parameter file are hashed at the end of B5 (`generator_hash`); B6 asserts the hash is unchanged, so FINAL cannot influence the generator. (3) A test asserts ids, seeds **and layout-content hashes** of TUNE/LAB-DESIGN are disjoint from every pool record (a regenerated duplicate under a different seed is caught). (4) After B6 no generator or gate change is allowed; the only permitted later change is the post-G7 stats-or-reward revision, which re-accepts the pool from the same seed streams (new sha) while the FINAL-access ledger has zero `eval` entries.
- **Iteration (revision) rule.** Initial design = iteration 0 (this file). A *revision* is any change to unit/defender/wall stats, generator parameters, score weights, tick count or grid size (B2 speed fallback), or adoption of the shaped reward (§7.4). **Budget: up to 3 revisions in B5, plus 1 held back for after G7 — 4 in total — all measured on TUNE and LAB-DESIGN only.** After every revision, every gate that applies at that point is re-run with the thresholds above (thresholds never change), and the B5 report (§11) is regenerated. If revisions are exhausted and a gate still fails, the gate is reported as failed in PROGRESS.md and the ship / re-scope decision goes to the user — no silent extra iteration. Gate results go to `data/v2/gates.json` with git commit and seed-set ids.
- **Post-G7 revision is restricted:** it may change **unit stats or the reward only — never the generators** (generator source/parameters stay at the B5 `generator_hash`). Unit-stat changes can change layout *acceptance* (the too_hard / single-type-trivial rules run the simulator), so if the revision could change acceptance — which any unit-stat change is assumed to — the whole pool (DEV and FINAL) is **re-accepted from the same pre-registered seed streams** (`F`, same candidate order, same generators) under the new stats, giving a new layouts file and a new sha256, **before any agent evaluation exists**. A reward-only revision (shaped reward) leaves acceptance unchanged and the pool is not regenerated. TUNE and LAB-DESIGN are likewise re-accepted and gates G1–G6, G8, G9 and then G7 are re-run.
- **FINAL-access ledger.** Every load of FINAL or transfer layouts goes through one loader that appends `{time, git commit, entry point, purpose ∈ build | eval}` to `data/v2/final_access.log`. *Build* = generation, acceptance, hashing; *eval* = any rollout of any agent, baseline, Oracle, planner or control on a FINAL layout. Pool regeneration is permitted only while the ledger has zero *eval* entries. Required pytest: **no agent evaluation of FINAL exists before the final eval** (see B6/B13/B15 rows).
- **G7 failing means redesign, not launching the grid** (after the one-time 600k extension if its slope condition holds; §7.4 defines the single allowed shaped-reward ablation before a post-G7 revision is spent).

---

## 4. Interface

### 4.1 Action space — Discrete(267)

| Indices | Meaning |
|---|---|
| 0–4 | Buy Scout / Sniper / Bulwark / Blaster / Mender |
| 5 … 264 | Deploy: `5 + type·52 + border_tile` (5 types × 52 tiles = 260) |
| 265 | END_BUY |
| 266 | WAIT |

Total `5 + 260 + 1 + 1 = 267`. Legal-action masks (`action_masks()`): buys need affordability and unit count < 20; deploys need units of that type still in the barracks, a free border tile (one unit per tile per deploy step), and DEPLOY phase; END_BUY only in BUY phase (and ≥ 1 unit bought); WAIT only when legal as in the prototype (zero-army / battle). Illegal action raises, as in the prototype.

### 4.2 Observation — Dict

`grid` (9, 14, 14), float32, all in [0, 1]:

| Ch | Content |
|---|---|
| 0 | deployed-unit occupancy (count / 4, clipped) |
| 1 | building hp fraction (0 where none) |
| 2 | turret hp fraction |
| 3 | wall hp fraction |
| 4 | building value weight / 3 (core = 1) |
| 5 | core mask |
| 6 | defender threat: sum over defenders whose range covers the cell (min–max range respected) of their damage per tick (Flak counted at its centre damage ÷ 2) ÷ 20, clipped. The footprint radius (4 / 5 / 6, hollow centre for Lancer and Flak) reveals the defender type |
| 7 | free legal-deploy border tiles (1 = deployable now) |
| 8 | **repair reach** — for each cell, number of deployed Menders whose range-2 disc covers it / 3, clipped; zero in BUY phase |

`state` (11): gold/240, owned-not-yet-deployed counts ×5 (÷ 20), phase one-hot ×3, units bought/20 (cap headroom), tick/240.

All channels are built from rotation/reflection-covariant quantities (discs, counts, masks), so the D4 transform of the grid is the grid of the transformed world. A test pins this per channel.

---

## 5. Layouts and the frozen pool

### 5.1 Families (14 × 14)

| Family | Role | Shape |
|---|---|---|
| `spine_pods` | familiar | A central spine of turrets/walls; 3–4 pods of 3–4 buildings branch off; core at a spine end. Favors Snipers, Blasters |
| `archipelago` | familiar | 4–6 isolated islands of 1–3 buildings with 1–2 turrets each; core on one island. Favors Scouts |
| `bunker_nest` | familiar | Core inside two nested wall rings with dense turrets between them; satellite buildings outside. Favors Bulwark + Mender |
| `labyrinth` | **held-out transfer, never trained on** | Serpentine wall corridors with doors, turrets at junctions, core at the end of the path. Requires BFS pathing; not generated or loaded by any training/tuning code path |

Acceptance rules (carried from the prototype, re-tuned in the balance step): reject empty, unreachable core, too_hard (Oracle breach impossible), single-type-trivial (best single type scores > threshold).

### 5.2 Pool of 300

| Group | `spine_pods` | `archipelago` | `bunker_nest` | `labyrinth` | Total |
|---|---:|---:|---:|---:|---:|
| DEV-train | 50 | 50 | 50 | — | 150 |
| DEV-validation | 10 | 10 | 10 | — | 30 |
| FINAL-familiar | 20 | 20 | 20 | — | 60 |
| FINAL-transfer | — | — | — | 60 | 60 |
| **Total** | 80 | 80 | 80 | 60 | **300** |

DEV = 180, FINAL = 120. The design sets TUNE and LAB-DESIGN (§3.5) are **not** part of the 300 and are disjoint from it.

- Records are `{id, family, seed, pool, tag}`; geometry regenerated from `(family, seed)` as in the prototype. New seed streams (disjoint from the prototype's v1–v3 pools and from the tuning set).
- Frozen as `data/v2/layouts_v2.json` with a **new sha256** in `data/v2/layouts_v2.sha256`, recorded in every model's identity file.
- **FINAL/transfer are never used for training or tuning**, enforced by: loader refuses FINAL/transfer unless called from the evaluation entry point with an explicit unlock flag; a leak-guard test (`dev sampler never yields FINAL or labyrinth ids`); FINAL evaluated exactly once on a clean commit, logged. Hyperparameters, planner N values (8/32/128 are declared, not selected), curriculum, and stat tuning are all fixed using DEV/validation/tuning only.

---

## 6. Curriculum

Deployment-first, **with random armies from the first step**.

- Phase A (first 1/3 of timesteps): env starts in DEPLOY with a *fresh random legal army per episode* (random affordable types, stop with p = 0.1 per buy, ≤ 20 units, ≤ 240 gold, all five types possible including Menders).
- Phase B: full BUY → DEPLOY game.
- Switch by timestep, per-episode lazy (as in the prototype), logged to `curriculum.log` and the model identity.
- Open option (user decision, default **off**): a linear ramp of the pre-bought probability from 1 → 0 over the next 1/6 of training to soften the phase-switch dip seen in the prototype (0.16 → 0.05).

---

## 7. Learners and evaluation

| Condition | Description | Trained learners |
|---|---|---:|
| PPO plain | MaskablePPO, CNN(9×14×14) + MLP(11) features | 10 |
| PPO + D4 | same, random D4 symmetry per episode via `D4Wrapper` | 10 |
| Masked Double DQN | own PyTorch implementation, action masks in both argmax and target, replay buffer, same feature extractor | 10 |
| Planner N = 8 / 32 / 128 | **no new training**: for each PPO learner (plain and D4) sample N complete plans (buy + deploy) from the *stochastic* policy, simulate each with the deterministic simulator, play the best | 10 learners × 2 PPO types |

Planner details: plans are sampled with masks from the *stochastic* policy; one batch of **128 plans per learner per layout is sampled once and N = 8 and 32 are its nested prefixes** (so the three cells cost 128 plans, not 168, and best-of-N is non-decreasing in N by construction). The simulator is the exact one used in play, so the planner has privileged access to it — stated in every table. N is reported as a **compute budget** (simulator calls and wall-clock per layout).

**Equal-budget controls (same N, same layouts, same simulator calls, same number of "learners" for the bootstrap):**
- **RAS-N (headline control): random-army search + scripted deployment.** Sample N random legal armies from the phase-A generator, deploy each with the scripted deployer, simulate, play the best. Run with 10 independent seeds standing in for learners. Shows how much the *learned* policy adds over searching armies blindly with a fixed deployer.
- **RPS-N (secondary): random-plan best-of-N** (uniformly random legal buy and deploy actions).
A planner cell is reported next to RAS-N at the same N; "the planner beats RAS-N at equal N" is the claim, not "the planner beats PPO".

Evaluation protocol (inherits Step 8): deterministic actions (masked argmax) for PPO/DDQN; `final.zip`/final checkpoint only; D4 models evaluated on the canonical view; groups DEV-train / validation (seen, never headline) / FINAL-familiar (60) / FINAL-transfer (60); bootstrap over **learners** (10 000 resamples, percentile 95 % CI); paired differences; metrics: breach rate (headline), mean score, mean `D`, army composition per family; familiar vs transfer for every condition; baselines Random, Greedy, BestSingle (chosen on DEV), Oracle (search budget declared), random-plan best-of-N. A sampled-vs-deterministic probe is kept as exploratory.

### 7.1 Extras

- **D4 test-time ensembling (TTA):** for each of the 8 symmetries, transform obs and mask, get masked policy probabilities (Q-values for DDQN), map back through the inverse action permutation, average, then argmax. Reported as extra eval rows (`+TTA`) for PPO plain, PPO+D4 and DDQN; costs 8 forward passes, no retraining.
- **"Why this move?"** For every replay step the **Decision lens** shows: top-5 actions with probabilities, the value estimate, a deployment-probability heatmap over the 52 border tiles per unit type, and a simulator-backed counterfactual (planner rollout score for the top-3 alternatives, labelled with its N). Built on `policy_outputs` (kept structured for this).
- **Duel mode:** human vs agent vs oracle on the same layout and budget. The agent's and oracle's choices are hidden until the human submits; all three armies/deployments run through the same simulator and are compared on score and breach. Oracle = cached search with its declared budget (so it is a lower bound on optimum and labelled that way).

### 7.2 Planner evaluation cost and fallback (estimate; re-measured in B12)

Cost per plan = one simulation (B2 gate ≤ 60 ms mean, expected 40–60 ms) + ~28 policy forward passes (1.5 ms unbatched, ~0.3 ms batched across plans) ⇒ **≈ 0.05–0.10 s/plan**.

| Cell group | Plans | CPU time | Wall at 7 workers |
|---|---:|---:|---:|
| PPO plain + PPO+D4 planners: 20 learners × 120 FINAL layouts × 128 (nested N = 8/32/128) | 307 200 | 4.3–8.5 h | **0.6–1.2 h** |
| RAS-N: 10 seeds × 120 layouts × 128 | 153 600 | 2.1–4.3 h (no network) | 0.3–0.6 h |
| RPS-N (secondary) | 153 600 | 2.1–4.3 h | 0.3–0.6 h |
| **Total** | | | **≈ 1.2–2.4 h** |

**Trigger and fallback (pre-registered).** In B12 time the full pipeline on 10 TUNE-val layouts × 2 learners and extrapolate. If projected wall at 7 workers is **> 3 h**, apply in order until ≤ 3 h, and report what was applied:
1. N = 128 for only 5 of the 10 learners per PPO condition (the 5 lowest seed ids, fixed now); N = 8 and 32 keep all 10.
2. N = 128 for PPO plain only (D4 planner keeps N = 8, 32).
3. N = 128 on a 30-layout FINAL subset (15 familiar + 15 transfer) drawn by a seed fixed before FINAL is opened.
RAS-N and RPS-N are always run on exactly the same learners-count and layouts as the planner cell they are compared with.

### 7.3 Pilot diagnostic: armies × deployers decomposition (DEV only)

Part of B13, reusing the prototype's `diagnose_buy_deploy` design. Layouts: **TUNE-val (60) only** — FINAL and LAB-DESIGN are never loaded (test). Applied to the 3 pilot PPO plain seeds:

| army \ deployer | pilot PPO deploy | scripted deploy | Oracle deploy (reference) |
|---|---|---|---|
| own buying (policy buys) | ✓ | ✓ | – |
| forced Greedy army | ✓ | ✓ | – |
| forced best-single army | ✓ | ✓ | – |
| forced random army (phase-A generator) | ✓ | ✓ | – |
| forced Oracle army (army-search result) | ✓ | ✓ | ✓ |

Armies are forced through real buy actions + END_BUY; control checks as in step 9b (own-buy × PPO reproduces the validation eval exactly; scripted × Greedy reproduces the baseline file). Purpose: when G7 fails, say whether buying or deployment is the bottleneck, and give the redesign (or shaped-reward ablation) something specific to target. It is a **diagnostic, not a tuning loop**: it can trigger a revision only through the iteration rule above; no hyperparameter is changed because of it.

### 7.4 Pre-registered fallback: one shaped-reward ablation (used only if G7 fails)

Defined now so it cannot be invented after seeing results.

- **Shaping:** potential-based, `r_t = Φ(s_{t+1}) − Φ(s_t)` with γ = 1 and `Φ(terminal) = 0`, so the total return still equals the true score and the optimal policy is unchanged (Ng et al.). `Φ(s) = 0` in the BUY phase; in DEPLOY after k ≥ 1 deployments `Φ(s) = 0.5 · score(preview battle of the units deployed so far)`, where the preview is `simulate_battle` with the same rules, tick cap 120. The final step's reward is `score − Φ(s_last)`.
- **What it fixes and what it does not:** it densifies the *deployment* signal. It gives no signal for buying, so if the §7.3 decomposition shows buying is the bottleneck, the ablation is skipped and the revision goes to the design instead.
- **Protocol:** exactly the G7 pilot (3 PPO plain seeds, 300k steps, TUNE-train → TUNE-val), judged by the same G7 pass condition on the **true** score. Evaluation everywhere (pilot and grid) always uses the true score; shaping is a training-only signal.
- **Outcome:** shaped passes G7 → shaping is adopted for **all** learners in the grid (PPO plain, PPO+D4, DDQN) and the paper states the deviation; this consumes the post-G7 revision (reward only, so the pool is not regenerated). Shaped fails → the post-G7 revision is spent on a unit-stat change (informed by §7.3; generators untouched; pool re-accepted per §3.5), then G7 re-run once (with its own one-time 600k extension rule); failing again → stop and report to the user.
- **Cost (estimate):** one extra preview simulation per deploy step (avg ≤ ~14 per episode, smaller armies are cheaper) ⇒ training 3–5× slower per step; the pilot costs ≈ 1.5–3.5 h. If adopted, the §12 step budget is recomputed from the *shaped* pilot curves and the same cap formula (which will shrink it).

---

## 8. App

Streamlit shell, **no sidebar** (`initial_sidebar_state="collapsed"`, sidebar CSS removed). Dark blueprint theme (navy background, thin cyan grid lines, amber/white accents; palette as CSS variables, high-contrast text).

```
┌────────────────────────────────────────────────────────────────────┐
│ COMMAND BAR: layout ▾ | agent ▾ | mode: Play·Replay·Duel | budget │
│              [Buy] [Deploy] [Run] [Why?] [Heatmap ◐]  [Export]     │
├───────────────────────────────────────────────┬────────────────────┤
│                                               │  DECISION LENS     │
│          CENTRAL MAP  (14×14 SVG)             │  top-5 actions     │
│   buildings · turrets · walls · units         │  value estimate    │
│   heatmap toggle: deploy-prob / threat /      │  counterfactuals   │
│   repair reach                                │  army + gold       │
├───────────────────────────────────────────────┴────────────────────┤
│ TIMELINE SCRUBBER  ◀ ▶  |────●──────────────| tick 87 / 240         │
└────────────────────────────────────────────────────────────────────┘
```

The map is inline SVG rendered from precomputed battle frames so scrubbing does not re-run the simulator; the scrubber is the only widget that triggers a re-render. CPU inference with bounded model caches; JSON and standalone-HTML export as in SPEC.md.

---

## 9. Prototype → v2 module map

New package `breachlens/` is created next to `siege/` (copy-then-modify), so the prototype stays importable and testable until v2 is complete; `siege/` is deleted in build step B20, once every reused piece lives in `breachlens/` and nothing imports `siege` (it stays on the `prototype-v1` tag).

| Prototype module | Action | Notes |
|---|---|---|
| `siege/troops.py` | **Rewrite** | 5 types, armor, heal, fixed-point stats |
| `siege/combat.py` | **Rewrite** | Fixed-point, tick order, Mender, armor, BFS pathing, nearest-in-preference targeting, symmetric tie handling, early exit. Prototype `_strike`/frames record idea reused |
| `siege/bases.py` | **Rewrite** | Four new generators, value weights, new turret constants |
| `siege/layouts.py` | **Modify** | Keep record format, `(family, seed)` regeneration, acceptance framework, `stable_family_hash`, sha handling; new pool counts/groups, new rules, labyrinth guard |
| `siege/env.py` | **Modify (heavy)** | 14×14, 52-tile border, 267 actions, 9 ch + 11 state, unit cap, random pre-bought armies, new score |
| `siege/armies.py` | **Modify** | `place_on_border` for 52 tiles, 5 types |
| `siege/baselines.py` | **Modify** | Random: reuse. Greedy: rewrite (5 types, Mender pairing). BestSingle: modify. Oracle: modify (5 types, cap, new budget). New: random-plan best-of-N |
| `siege/render.py` | **Rewrite** | Blueprint theme, heatmaps, SVG frames |
| `rl/augment.py` | **Modify** | Re-derive 52-tile border permutation and 267-action table; add repair-channel transform; `D4Wrapper` logic reused |
| `rl/features.py` | **Modify** | Input 9×14×14 and 11-d state |
| `rl/policy_io.py` | **Modify** | 267 actions, per-type heatmaps, TTA |
| `rl/split.py` | **Modify** | 150/30 split, 10 per family validation, bound to new sha |
| `rl/evaluation.py` | **Reuse** | Rollout/helper functions, minor signature changes |
| `rl/train_ppo.py` | **Modify** | Random phase-A default, new ids, identity fields; resume logic reused as is |
| `rl/train.py`, `rl/eval.py` | **Drop** | Throwaway step-2 smoke code |
| `rl/eval_ppo.py` | **Modify** | Validation eval for new env |
| `rl/eval_all.py` | **Modify (heavy)** | New groups, DDQN/planner/TTA conditions, breach-rate headline, write-once FINAL guard |
| `rl/diagnose_buy_deploy.py` | **Modify, needed at B13** | Armies × deployers matrix (§7.3) on TUNE-val for the pilot, again after the grid |
| `rl/diagnose_grid.py` | **Reuse later / modify** | Group/curve/per-seed diagnostics after the grid (B16) |
| new `rl/ddqn.py`, `rl/planner.py`, `rl/tta.py`, `rl/explain.py`, `app/*` | **New** | |
| `scripts/run_ppo_grid.sh`, `run_one.sh`, `status.py` | **Modify (small)** | Condition list, step budget |
| `scripts_balance.py`, `scripts_tuning.py` | **Modify** | Become the G1–G6 gate script on the tuning set |
| `scripts_rebuild_pool.py`, `scripts_contact_sheet.py` | **Modify** | v2 pool builder, contact sheet incl. labyrinth |
| `scripts_smoke.py` | **Modify** | Trivial |
| `scripts_6b_analysis.py`, `scripts_diag_deploy.py`, `scripts_sweep_splash.py` | **Drop** | Prototype-specific; remain on the tag |
| `tests/test_augment.py` | **Modify** | Equivariance tests re-derived; keep structure |
| `tests/test_simulator.py`, `test_env.py`, `test_layouts.py`, `test_step4_contract.py`, `test_step6.py`, `test_eval_all.py`, `test_diagnose_*.py` | **Modify** | Same intent, new constants; most assertions rewritten |
| `docs/*.md` (actions, observation, reward, troops) | **Rewrite** | Regenerated from v2 constants |
| Prototype results (`results/`, `models/`, `logs_6b/`) | **Keep untouched** | Read-only history; v2 writes to `data/v2/`, `results_v2/`, `models_v2/` |

---

## 10. Risks

| # | Risk | Why it matters | Mitigation |
|---|---|---|---|
| R1 | **D4 exactness with the Mender** | Heal-target selection, Mender movement, and simultaneous heal+damage are new tie sources; id- or coordinate-ordered ties silently break equivariance (the prototype needed a dedicated step to reach 0 mismatches) | §3.3 rules 1–5: fixed-point integers; invariant key chain; exact ties → split heal / abstain move; ids never used; tick-snapshot ordering so heal order cannot matter. Tests: ≥ 10 000 random battles × 8 symmetries with Menders in ≥ 50 % of armies; adversarial **id and wall-order shuffles**; hand-built symmetric fixtures (Mender between two equally damaged allies, symmetric centroid ties) that must come out bit-identical. B3 does not close until mismatches are 0. Movement ties (equal-length shortest paths) are covered by the canonical-form fallback of §3.3 rule 7, tested in open ground and in labyrinth |
| R2 | Float/ordering drift | Equivariance can fail from 1e-16 differences | Integer fixed-point state; no float comparisons in decisions except distances compared via squared-integer coordinates |
| R3 | Balance of five units | Prototype took 2 tuning iterations and still left tank weak | Gates G1–G6 pre-registered; tuning seeds disjoint; hard cap on iterations; failed gates reported, not hidden |
| R4 | Simulator speed (240 ticks, 20 units, BFS) | Dominates both training throughput and Oracle/planner cost | Early exit; cache BFS fields per layout/target; B2 speed gate (§11): mean battle ≤ 60 ms, random-agent loop ≥ 300 steps/s; if missed: optimize, then 150 ticks, then 12×12 — before anything else is built |
| R5 | Labyrinth pathing | New movement model only needed for the transfer family, risk of stuck units or non-terminating stalemates | Stalemate exit rule; reachability acceptance rule; contact-sheet review; tests with constructed corridors |
| R6 | Planner leaks simulator knowledge | Looks like "the agent" is stronger than it is | Always labeled as search-augmented, reported with compute budget and compared to random-plan best-of-N at equal N |
| R7 | Larger action space (267) and 20-step buys → harder exploration | Prototype PPO plateaued far below Oracle | Random phase-A armies; larger step budget; one bounded pilot round (B13) on validation, then frozen |
| R8 | D4 may underfit again | Prototype D4 was still rising at 1M | Larger budget; TTA as a cheaper symmetry use; report either outcome — the question is the experiment |
| R9 | Evaluation leakage by convenience | Tempting to peek at FINAL for N, TTA, or stats | Write-once FINAL protocol + loader guard + tests (§5.2) |
| R10 | Streamlit scrubbing latency / no-sidebar styling | UX of the signature feature | Precomputed frames, SVG map, only scrubber triggers rerun; fall back to standalone HTML export |
| R12 | G7 fails after the revisions are spent | The grid would be 30 runs of an unlearnable task | G7 gates the grid; shaped-reward ablation (§7.4); revision reserved; otherwise stop and ask the user |
| R11 | Scope | Five conditions, three extras, an app | Strict step order (§11); app/extras only after the grid is evaluated |

---

## 11. Build steps (each ends with pytest green; one step at a time, only when told)

| Step | Deliverable | Pytest requirement |
|---|---|---|
| B1 | Scaffold `breachlens/`, constants, action codec, border tiles | Border has 52 unique clockwise tiles; codec is a bijection over 267 actions; constants match this doc (budget 240, cap 20, ticks 240) |
| B2 | Units/turrets/walls + core combat (no Mender), fixed-point, nearest-in-preference targeting, BFS movement with the §3.3 rule-7 key chain (keys 1–3), early exit; **simulator speed gate** (below) | Hand-built scenarios (range, armor, splash, wall block); determinism over 100 seeds; early-exit cases; **no-deadlock fixtures on exact-diagonal approaches** (below); the speed gate's measurements written to `data/v2/speed.json` and asserted against the gate |
| B3 | Mender, canonical-form fallback (rule 7 key 4), full tick order; **D4 equivariance of the simulator** | 0 mismatches across ≥ 10 000 random battles × 8 symmetries (Menders in ≥ 50 % of armies); adversarial id/order shuffles; symmetric-tie fixtures (Mender between two equally damaged allies; blocked-diagonal with two mirror gaps); fallback counter > 0 across the suite (the path is exercised); heal rules (no overheal, no revive, split) |
| B4 | Four layout generators + acceptance rules; TUNE (180/60) and LAB-DESIGN (40) sets generated from streams `T`, `L` | Determinism from `(family, seed)`; connectivity/reachability incl. labyrinth; building counts and value weights in range; contact sheet renders; **labyrinth D4-exactness and no-deadlock tests** (below) |
| B5 | Gate script `scripts_gates.py` for **G1–G6, G8, G9** on TUNE/LAB-DESIGN, paired bootstraps per §3.4; up to 3 design revisions; **B5 report** (below) regenerated after every revision; stats and generators frozen at the end, `generator_hash` recorded | Gate script deterministic; paired-bootstrap helper tested on synthetic data (known mean difference, lower bound sign); seed-stream disjointness (ids, seeds, content hashes) of TUNE/LAB-DESIGN vs `F`; gate results in `data/v2/gates.json`; every unit's and defender's stats pinned by test; thresholds in `gate_thresholds.json` equal this doc's (parsed by a test); B5 report exists with every required field for all 5 units × all families; defender-mix acceptance rules (§3.1b) checked on the generated sets |
| B6 | Build and freeze the 300-layout pool from stream `F`; new sha256 | **FINAL-access ledger test: no agent evaluation of FINAL before the final eval (zero `eval` entries)**; counts per group/family exactly as §5.2; sha matches; ids unique; `generator_hash` equals B5's; leak-guard (DEV sampler never yields FINAL/labyrinth); FINAL labyrinth records disjoint from LAB-DESIGN by id, seed and content hash; acceptance holds under the current simulator |
| B7 | `BreachEnv`: 267 actions, 9 ch + 11 state, masks, random pre-bought armies, terminal score (optional potential shaping behind a flag, off by default) | Mask legality fuzz (10 000 random legal steps never raise); each observation channel unit-tested incl. repair reach; score formula on constructed outcomes; reset/seed determinism; `prebought="random"` always legal; shaping flag off ⇒ returns identical to unshaped, flag on ⇒ shaped return sums to the true score |
| B8 | Baselines: Random, Greedy, BestSingle, Oracle (cached), army search, RAS-N, RPS-N; **baseline numbers on TUNE-val recorded** (inputs to G7) | Baseline outputs reproducible; Oracle ≥ every other baseline on each layout it is run on; best-of-N non-decreasing in N for nested samples; baseline file written before any pilot exists (timestamp/commit test) |
| B9 | D4 augmentation, `D4Wrapper`, TTA | Inverse identity, group closure, action bijection per symmetry; mask legality under transform; env + simulator equivariance on 100 layouts incl. Menders; TTA with identity-only policy equals the plain policy |
| B10 | PPO pipeline: features, train/eval scripts, curriculum, resume; **end-to-end throughput benchmark** | Tiny wiring run; phase A random-army legality; curriculum switch logged; resume restores timestep; model identity records sha/commit; throughput per job at 6 parallel recorded and compared to the §12 minimum |
| B11 | Masked Double DQN | Masked argmax never selects illegal actions; Double-Q target equals hand-computed value on a toy MDP; replay buffer round-trips; tiny run improves over random on a toy layout |
| B12 | Policy-guided planner (nested N = 8/32/128), RAS-N/RPS-N controls at equal N, cost projection and §7.2 fallback selector | N = 1 equals one sampled rollout; deterministic given seed; chosen plan's score = max of its N; N = 8 ⊂ 32 ⊂ 128 prefixes ⇒ non-decreasing; masks respected; RAS-N uses exactly N simulations; fallback selector picks the pre-registered option for synthetic timings |
| B13 | **Pilot (DEV only):** 3 PPO plain + 3 PPO+D4 seeds × 300k steps on TUNE-train → TUNE-val; **G7**; G8 re-check; **§7.3 armies × deployers diagnostic**; **step budget set from the curves** (§12); hyperparameters frozen | G7 evaluator reproduces its inputs from files (pass/fail on synthetic curves); decomposition control cells reproduce the eval exactly (max diff 0); test that FINAL, transfer and LAB-DESIGN ids are never loaded; frozen config file recorded with hash; step-budget rule function tested on synthetic slopes. **If G7 fails: §7.4 shaped-reward pilot, then redesign — the grid does not launch** |
| B14 | Launch grid (user-run, only after G7 pass): PPO plain ×10, PPO+D4 ×10, DDQN ×10 at the step budget from B13 | `status.py` and resume tests updated; dry-run command list asserted; launcher refuses to start unless `data/v2/gates.json` has G7 = pass; ledger still has zero `eval` entries |
| B15 | Full evaluation, FINAL once on a clean commit; bootstrap tables; familiar vs transfer; planner, RAS-N and TTA rows | Ledger shows exactly one `eval` session, on a clean commit, after the B14 grid finished and not before; evaluator asserts layout ids ⊂ declared groups; bootstrap reproducible by seed; D4 canonical view equals identity wrapper; model identity complete |
| B16 | Diagnostics (buy-vs-deploy again, curves, per-seed) on DEV/validation only | FINAL never loaded (test); control cells reproduce eval numbers exactly |
| B17 | "Why this move?" | Top-5 probs sum ≤ 1 and respect mask; heatmap sums to the deploy mass; counterfactual score equals a direct simulation |
| B18 | App (blueprint theme, command bar, map, Decision lens, scrubber) | Renders every layout family without error; frame count equals sim frames; scrubber index bounds; export JSON/HTML round-trips |
| B19 | Duel mode | Hidden choices stay hidden until submit; all three sides scored by the same simulator; ties and illegal human input handled |
| B20 | **Remove `siege/`** (and prototype-only scripts, tests and docs superseded by v2) from the `v2-redesign` branch, once every reused piece lives in `breachlens/` | A repo-wide test asserts no file under `breachlens/`, `rl/`, `app/`, `scripts/`, `tests/` imports `siege`; the full suite is green with `siege/` deleted; `git show prototype-v1:siege/combat.py` still works (tag intact); nothing under `results/`, `models/` is deleted |
| B21 | README, docs regenerated from code, final PROGRESS.md; README carries the line `Inspired by AUTHOR_NAME (AUTHOR_URL).` using the **placeholders `AUTHOR_NAME` and `AUTHOR_URL`** until the owner supplies real values | **The test fails while the strings `AUTHOR_NAME` or `AUTHOR_URL` appear anywhere in README.md**, and also fails if the `Inspired by` line is missing or the URL is not `http(s)://…`; doc constants regenerated from code and checked by test. B21 is therefore not closable until the owner replaces both placeholders |

### B5 report (required content, regenerated per revision → `data/v2/b5_report.md` + `.json`)

For **every unit** (Scout, Sniper, Bulwark, Blaster, Mender):

| Field | Definition |
|---|---|
| Effective hp per gold, **including armor** | For each defender type `d` (Gun, Lancer, Flak): `EHP_d = hp · dmg_d / max(dmg_d − armor, 1)` (armor is flat per hit; Flak uses its centre damage, neighbours' 7 reported too), divided by cost. Plus the unweighted mean over the three defenders |
| DPS per gold | `damage per tick ÷ cost`; splash units also report "structure-hits per tick at a 3-building cluster ÷ cost". Mender reports 0 DPS and `heal per tick ÷ cost` separately |
| Defender types that threaten it | `d` threatens unit `u` iff `hits_to_kill(u, d) ≤ 3` (armor applied) **and** `d.max_range ≥ u.range` (so `u` cannot attack `d`'s region from outside it); also lists "outranged by" and "inside blind spot at ≤ 2.0" |
| Single-unit-army score table | Rows = 5 unit types; columns = each familiar family (TUNE) and LAB-DESIGN: mean score, mean `D`, breach rate, mean units bought and gold spent, each with a 95 % bootstrap CI over layouts; the best row per family is marked |

**Extra (B5):** per family, walls attacked per battle per unit type (distinct walls that were a unit's primary target), and the share of battle time spent attacking walls (wall-attack unit-ticks ÷ alive attacker unit-ticks; and the share of battle ticks on which any unit attacked a wall), both from the single-unit-army battles and from the Oracle's armies.

Also included: the G1–G9 results (point estimates and bootstrap bounds for G2–G4), the paired-difference tables behind them, the defender mix per family, and the revision number. The report is a script output (`scripts_gates.py --report`), not hand-written.

### B2 simulator speed gate

**Minimum required:** (a) mean battle time **≤ 60 ms** and p95 **≤ 150 ms** over 200 random legal armies (≥ 10 units) spread across all 4 families, single core; (b) the random-masked-agent env loop (simulator included, no network) **≥ 300 steps/s** in one process. Derivation: the prototype's PPO overhead was ≈ 6 ms/step; at ~28 steps/episode that is ≈ 170 ms/episode plus ≤ 60 ms of simulation ⇒ ≈ 120 steps/s per job, so (b) ⇒ **≥ 100 steps/s per PPO job at 6 parallel (≥ 600 steps/s aggregate)**, which is what the §12 wall-clock budget needs.

If the gate fails: (1) time-boxed optimization inside B2 (BFS-field cache, tighter early exit, numba/vectorized tick loop) and re-measure; (2) if still failing, **reduce the horizon to 150 ticks** and re-measure; (3) if still failing, **reduce the grid to 12×12** (border 44, action space `5 + 5·44 + 2 = 227`, spec constants amended). Nothing after B2 is built until the gate passes. Steps (2)–(3) are design revisions and each uses one of the 3 B5 revision slots (needing both leaves 1 for the gates); if the gate still fails after both, stop for a user decision.

### Movement tests (required by B2/B3/B4)

- **No deadlock on exact-diagonal approaches (B2):** fixtures for every unit type — unit at `(a, a)`, target at `(a+k, a+k)`, k = 2…10 — with (i) open ground, (ii) a single wall on the exact diagonal, (iii) walls on the diagonal and both orthogonals, (iv) two symmetric gaps, (v) a Mender following an ally on an exact diagonal. Assertion: every unit reaches attack position or starts attacking the blocking wall within `d + ceil(wall_hp / dmg) + 2` ticks. A global invariant checked on 2 000 random battles: no unit with a live target makes zero progress (no move, no attack) for more than 2 consecutive ticks.
- **D4-exact in labyrinth (B4):** ≥ 2 000 (LAB-DESIGN layout × random army × random start tile) cases × 8 symmetries — identical scores, identical destroyed-structure sets (transformed) and identical per-tick unit positions up to the transform; adversarial id/order shuffles; the canonical-fallback counter is asserted > 0 for the suite so the tie path is genuinely exercised. Any mismatch blocks B4.

---

## 12. Training-time estimate and step-budget rule

### 12.1 Step budget — set from the G7 pilot curves, not fixed

The pilot (B13) trains 3 PPO plain and 3 PPO+D4 seeds for 300k steps with validation every 25k on TUNE-val. Rule, fixed now:

1. For each condition, average the validation score over its 3 seeds and take the least-squares slope over the last 100k steps (t = 200k … 300k, 5 points), in score per 100k steps. Let `s = max(slope_plain, slope_D4)` (D4 is slower; both conditions get the same budget so the comparison stays equal-budget).
2. `T_curve`: **s < 0.005 → 1.0M; 0.005 ≤ s < 0.015 → 1.5M; 0.015 ≤ s < 0.03 → 2.0M; s ≥ 0.03 → 3.0M.** Reasoning: the prototype's plain curve gained ≈ 0.05 per 100k early (0.28 → 0.37 over 175k) and ≈ 0.0014 per 100k near the end (0.014 per 1M); the thresholds sit between "clearly still learning" and "plateaued".
3. `T_cap = W · 3600 · r / ⌈R / P⌉` with wall-clock cap `W = 20 h` for the PPO grid, `r` = per-job steps/s at `P = 6` parallel measured in B10, `R = 20` runs (so `⌈R/P⌉ = 4` waves) ⇒ `T_cap = 18 000 · r` (e.g. r = 110 → 1.98M).
4. **`T = min(T_curve, T_cap)`, rounded down to a multiple of 250k.** If `T < 1.0M`, stop and ask the user (do not cut seeds below 10). DDQN uses the same number of environment steps. `T` is recorded in `data/v2/step_budget.json` with the slopes and `r`, and is never changed after the grid starts.

### 12.2 Wall-clock estimate (estimate; re-measured in B10/B12/B13)

Basis (prototype, PROGRESS.md): ~167 steps/s per job at 6 parallel; battle = 50 ticks ≈ 10 ms; episodes ≈ 17 steps. v2: battle up to 240 ticks / 20 units / BFS, gated at ≤ 60 ms mean (B2); episodes ≈ 28 steps ⇒ **≈ 100–130 steps/s per job**.

| Item | Wall time (estimate) |
|---|---:|
| Pilot (B13): 6 runs × 300k, one wave of 6 | ≈ 0.7–0.9 h |
| Shaped-reward pilot (only if G7 fails) | ≈ 1.5–3.5 h |
| Gates G1–G6, G8, G9 (B5, per iteration; Oracle searches on 240 + 40 layouts) | ≈ 1–1.5 h |
| PPO plain + D4, 20 runs, 4 waves, example `T = 1.5M` | 12.5–17 h (scales linearly with `T`, bounded by `T_cap`) |
| Masked DDQN, 10 runs, 2 waves, same env steps (1 update / 4 steps) | 3–5 h |
| Oracle search on the 300-layout pool | ≈ 0.5 h |
| Planner + RAS-N + RPS-N evaluation (§7.2) | ≈ 1.2–2.4 h (≤ 3 h by the fallback) |
| TTA + remaining evaluation | ≈ 0.5 h |
| **Total (example `T = 1.5M`, no shaped pilot)** | **≈ 20–29 h**, mostly unattended |

Sensitivity: every 10 ms of battle time adds ≈ 8 % to training and ≈ 10 % to planner cost. If B10 measures `r < 90` steps/s, `T_cap` (≈ 1.6M) binds before the curve rule does; below `r ≈ 56` the cap falls under 1.0M and the project stops for a user decision.

---

## 13. Decisions I made that you may want to overrule

1. **Package layout:** new `breachlens/` package beside `siege/`, not an in-place edit.
2. **Planner base policies:** both PPO plain and PPO+D4 (so 6 planner cells), not just the better one.
3. **Exact-tie rule:** split/abstain instead of id ordering (changes gameplay marginally in symmetric situations; makes D4 exactness provable).
4. **No reward shaping:** terminal score only.
5. **Hard curriculum switch** (ramp is an option, default off).
6. **Step budget** is derived from the pilot (§12.1), not fixed; 1.5M is only the worked example.
7. **G7 margin 0.05, G8 margins 0.10 / 0.20, G9 thresholds 0.60 / 0.10 / 0.45** (§3.4) — my proposals, frozen now; tell me if you want them different *before* B1.
8. **Revision budget:** 3 in B5 + 1 after G7 = 4 total, TUNE/LAB-DESIGN only (§3.5); the post-G7 revision is stats-or-reward only and triggers pool re-acceptance (with a ledger guard) if it touches stats.
9. **Movement ties** use a canonical-form fallback (always moves), not "abstain", because abstaining can deadlock a unit on a symmetric diagonal approach.
10. **Shaped-reward fallback** is potential-based with a preview-battle potential (§7.4); it only helps deployment, and is skipped if the §7.3 decomposition says buying is the bottleneck.
11. **"Inspired by" line (B21):** written as `Inspired by AUTHOR_NAME (AUTHOR_URL).`; B21's test fails until both placeholders are replaced.
12. **Defenders** (Gun / Lancer / Flak, §3.1b) are my proposed stats; walls and turrets carry **no** score value.
13. **G7 extension:** one-time continuation to 600k when the last-100k slope is ≥ +0.02 per 100k steps (I read your "+0.02" in score per 100k steps; say so if you meant per 1M).
