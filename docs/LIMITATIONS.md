# Known limitations (running list, feeds the README)

Numbers only. Source: revision-1 gate run (`data/v2/b5_report.md`, `data/v2/gates.json`), R3 sweep (`data/v2/b5r3_sweep/`), PROGRESS.md. Add to this list; do not delete entries.

## BreachLens v2 design (B5 closed at revision 1; docs/SPEC_V2.md 3.6)
- Revision budget: B5 closed with 2 of 3 revisions unspent (1 held for after G7, 1 spare). Balance gates are proxies; G7 decides learnability.
- G2 / G3, Blaster: monotone ban-loss <= 0.015 in every family; niche share 1.1 % (threshold 8 %). Blaster is rarely worth buying.
- G3, Scout: niche share 7.9 % (threshold 8 %). G2 passes for Scout only in spine_pods.
- G8, bunker_nest: army-search beats best single on 56 % of layouts (threshold 60 %); all mean gains >= +0.20.
- Wall-attack share: 0.50-0.62 of attacker time on bunker_nest and labyrinth.
- Army cap 20 never binds: Oracle armies use at most 12 units (G6 diagnostic).
- Archipelago layout rejection: 60.4 % (122 of 202 candidates); other families 7.0 % (spine_pods), 16.7 % (bunker_nest), 31.0 % (labyrinth).
- Blaster / Scout compete for the same role: in the R3 sweep, every Blaster buff arm made Blaster the best single unit (labyrinth best single 0.688-0.705, Oracle 0.976-0.993, both above the bands) and cut Scout's niche to 1-2.5 %. No arm applied.
- G4 was re-defined after the revision-1 result (changelog (g)); under the original definition it fails (Mender gain bunker_nest +0.017 < 0.02, LAB-DESIGN 0.000). Under (g) it passes (spine_pods +0.029, archipelago +0.031).
- One-time waiver, changelog (f): item 9 criterion (ii) for R2a (labyrinth best single 0.669 vs 0.65, n = 20).
- Sample sizes: sweeps used 20 layouts per family; the sampling error exceeded the margin in the item 9 case above.
- Design sets: TUNE-train 180, TUNE-val 60, LAB-DESIGN 40; stats were tuned on them, so gate numbers are optimistic for those sets.
- Greedy (B5 definition) buys 8 Bulwarks, identical to the single-Bulwark army.
- Oracle is a fixed-budget search oracle, not the optimum (no breach found on one DEV-validation layout that the acceptance search breached).

## Prototype lessons that still apply (PROGRESS.md)
- Resumed PPO runs are not bit-identical to uninterrupted runs (RNG state not in checkpoints; reseeded as seed + 100003 * k). Prototype seeds 13-15 were resumed twice.
- Two seeds of the same config differed by 0.08 on validation (SD about 0.05-0.06); a 3-seed mean has SE about 0.03. Differences under 0.05 are not distinguishable.
- `best.zip` is selected on 20 validation layouts and is optimistic; headline numbers use `final.zip`.
- Models stop at rollout granularity (about 1.0015M steps for a 1M budget).
- Config C (lr 1e-4, 5 epochs) collapsed to one troop type per seed (val 0.23 vs 0.33).
- The prototype transfer family had BestSingle = 0.000, so BestSingle is uninformative there; v2 reports transfer against Oracle / Greedy / Random.

## Pool (B6, `data/v2/pool_v1.json`, sha256 1346536dea209eb7953eb00eb0d69f24bed930a2a81d89daea7ec1b705f834e4)
- 300 layouts: DEV-train 150, DEV-validation 30, FINAL-familiar 60, FINAL-transfer (labyrinth) 60; 50 / 10 / 20 per familiar family.
- Candidates walked (accepted / candidates): DEV spine_pods 60/67, archipelago 60/223, bunker_nest 60/77; FINAL spine_pods 20/23, archipelago 20/79, bunker_nest 20/23, labyrinth 60/94.
- Rejections by rule (a candidate can fire several): DEV spine_pods count_buildings 2, single_type_trivial 4, too_hard 1; DEV archipelago count_defenders 34, count_buildings 15, count_value_total 12, single_type_trivial 113; DEV bunker_nest count_defenders 2, single_type_trivial 7, too_hard 8.
- FINAL rejections by rule: spine_pods count_buildings 3; archipelago count_defenders 13, count_buildings 10, count_value_total 7, single_type_trivial 35; bunker_nest count_defenders 1, single_type_trivial 2; labyrinth count_value_total 6, single_type_trivial 28. Zero too_hard in FINAL.
- Archipelago candidate rejection: DEV 163 of 223 (73.1 %), FINAL 59 of 79 (74.7 %); the 50 % flag (item 5) applies to the stat rules: stat rejection 113 of 173 stat-evaluated in DEV (65.3 %, above the 50 % flag), 35 of 55 in FINAL (63.6 %); the B5 TUNE walk rate was 60.4 %.
- D4 duplicate redraws against all other sets: 0 in every walk. Zero overlaps with TUNE-train, TUNE-val, LAB-DESIGN.
- DEV defender mix (DEV only): Gun share 0.59 / 0.54 / 0.59, Lancer 0.38 / 0.34 / 0.39 (spine_pods / archipelago / bunker_nest); sealed-core share 0.42 / 0.65 / 0.50.
- FINAL scores, baselines and Oracle results are not stored anywhere; the pool carries accept bits and rule ids only. Pool statistics of FINAL cannot be reported before the final eval.
- Freeze guarantee: the sha256 test (`test_pool_file_is_read_only_and_sha256_matches`, against `pool_v1.sha256` and the recorded hash), not file permissions, is what guarantees the pool is frozen. The 0444 mode is a convenience; anyone can chmod it. Likewise the ledger is an append-only convention, not an access-control mechanism; its integrity is checked by tests (build entries only, zero eval).

## G7 / learnability (B13d)
- G7 passed only with potential-based shaping (SPEC 7.4), on the second pre-registered attempt: unshaped pilot FAIL (mean 0.4284 vs bar 0.4369, 1 of 3 seeds above), shaped pilot PASS (mean 0.4650, seeds 0.4544 / 0.4875 / 0.4531, 3 of 3 above). Shaping is a training-only signal and is used for all grid learners; evaluation is always the true score. It consumed the post-G7 revision; 1 spare remains.
- The shaped-vs-unshaped pilot difference (+0.037, 0.4650 vs 0.4284) is 3 seeds vs 3 different seeds (9201-9203 vs 9001-9003), not a controlled comparison; per-seed spread is 0.40-0.49, so it is not evidence that shaping helps by that amount.
- On TUNE-val the shaped PPO policies breach 0.38-0.48 of layouts vs 0.65 for BestSingle: PPO trades breaches for destruction under the 0.6 / 0.4 score (D 0.48-0.50 vs 0.20). The learner is not better at breaching than the single-unit baseline.
- The D4 pilot was unshaped and flat (mean 0.4213, slope +0.0030 per 100k); there is no shaped D4 pilot, so the D4 slope entering the step budget is the unshaped one (it does not change T = 1.5M).
- The shaped plain curve peaks near 225k (0.476) and falls to 0.450 at 300k; its 200k-300k slope (+0.0083 per 100k) is noisy (per-seed -0.031 to +0.030).

## DDQN pilot (B14a)
- DDQN epsilon decays over 25% of steps, i.e. inside curriculum phase A (first T/3) in both the 300k pilot (75k < 100k) and the 1.5M grid (375k < 500k); the pilot is representative in this respect.
