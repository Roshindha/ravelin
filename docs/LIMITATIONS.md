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

## B15 FINAL evaluation (session 3ccef86e0e2349d4af22e210d4b46a07; results in docs/RESULTS.md)
- One FINAL session, one commit (`7e3aa42`) for all 116 cells; no crash, no restart, no cell recomputed, no cell discarded. Tagged `b15-final`.
- Ledger `dirty` flags: the session's first ledger eval entry records `dirty: false`; the other 237 eval entries (per-layout loads and cell records) record `dirty: true` (the 5 build entries from B6 are also all `dirty: true`; cause, from `git log`/`ls-tree`: the four pool-build entries (2026-11-14 14:36 UTC, 22:36 +0800) record commit `bab548e`, which contains neither `scripts_b6_build_pool.py` nor `breachlens/pool.py` nor the ledger; those files were first committed in `3cd773e` at 22:50 +0800, so the B6 code and ledger were uncommitted/untracked when the build ran. The fifth entry (`verify_final_acceptance`, 15:49 UTC = 23:49 +0800, commit `0ca8d71`, B7) falls between the B7 commit (23:40) and the B7.1 commit `e345ae9` (23:59), so it was most likely written with B7.1 work uncommitted; that last cause is inferred from timestamps, not recorded). The flag is evaluated before the append, which is why only eval entry 1 is `false`. The generator itself was not uncommitted: `breachlens/layouts.py` has the same sha256 (`49b8a18c...`) at `f1bf4ad`, `bab548e`, `3cd773e`, `f461116` (tag `pool-v1`), `0ca8d71`, `688bffa` and on disk now, and the pool build refuses unless the on-disk hash equals the recorded one. `pool.git_dirty()` is `git status --porcelain --untracked-files=normal` being non-empty, evaluated when each entry is written, so it is true for any modified tracked file or any untracked, non-ignored file. What was checked (B15d): `git --no-pager diff --stat 7e3aa42 688bffa` lists 130 files, all insertions (28310 +, 0 -): `data/v2/final_access.log` (+238 lines) and 129 files under `results/b15/`, and nothing else (no source, test, spec, config, model or pool file). This is CONSISTENT WITH the flag coming from the ledger append itself (the first append leaves the tracked ledger modified, and `results/b15/` is new and untracked), and the flip happens between eval entry 1 and entry 2, written in the same second (13:23:43). It does NOT by itself exclude transient uncommitted edits made during the session and reverted or committed-away before `688bffa`, because the commit-range diff only sees the end state. Stronger evidence from the recorded fields is limited: the ledger records only `commit`, `dirty`, `entry_point`, `time` (no dirty-path list, no source hash); `results/b15/session.json` records `git_dirty: false` at creation and `results/b15/preconditions.json` records `git_dirty False` for the "clean tree" check and again for "clean tree after the suite", where the suite is the full pytest run inside the preconditions (exit code 0, 679 passed in 954.95s), all evaluated before the first ledger append; every manifest line records `commit` 7e3aa42; and all 25,920 raw rows carry the same `git_commit` (7e3aa42), `sim_version`, `score_version` (b7-single-rounding) and `generator_hash` (49b8a18c...), with `git_dirty: True` on every row (computed after the ledger was already modified). These show the simulator, score and generator versions were constant across the session and that the tree was clean when it started; they provide no per-file proof that no tracked source file was edited mid-session. No result depends on it; no such edit is known.
- 10 seeds per learner and 60 layouts per group: intervals are wide (a primary interval is about +/-0.08 to +/-0.12 on a breach rate), and per-family rows rest on 20 layouts. "Not established" is not "no effect".
- 64 primary checks without multiplicity correction: about 3 would hold by chance at the 5 % level. 27 hold; they cluster in TTA (12), planner vs strongest equal-N control (12) and learners vs Greedy on familiar score (3). Everything outside P1-P5 is exploratory and labelled so.
- The primary interval is hierarchical (seeds, then layouts), wider than the section 7 learner-only interval, which is reported as secondary only. The two disagree on some cells (e.g. P1 familiar score, learner-only intervals above 0); only the primary decides "holds".
- On breach rate no learned policy beats BestSingle. BestSingle is also a poor reference on FINAL-transfer (breach 1.000, a ceiling) and on bunker_nest (0.30), so breach cannot discriminate on those slices; score is used there.
- Breach rate saturates for BestSingle, RAS-N and Oracle on labyrinth, so the planner-vs-RAS comparison on breach there is uninformative at the ceiling.
- The planner has privileged simulator access (SPEC 7, row r); its gains over the learner (E1) and over RAS-N / RPS-N measure a search procedure, not the policy. P4 is judged against the strongest control by *score*; on FINAL-familiar RAS-N has the higher breach rate than the control used.
- FINAL-transfer is a single family (labyrinth), easier in absolute terms (every baseline scores higher there); learner-vs-baseline gaps on transfer are not a controlled measure of generalisation.
- Generalisation gap uses DEV-train in-sample scores from B14c (seen data); it is descriptive. PPO plain and DDQN lose about 0.27-0.29 in score from DEV-train to FINAL-familiar; PPO+D4 loses 0.03 but starts lower.
- The 3 TTA rows reuse the same 30 models as the no-TTA rows; TTA adds 8 forward passes per decision and is not a separate training result. Planners were not run on DDQN or on TTA rows (pinned in (gg)).
- Oracle is a fixed-budget search (about 400 simulations), a lower bound on the optimum.
