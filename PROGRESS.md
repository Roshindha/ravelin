# PROGRESS

Project root: `~/base-siege` (NOT `~/RL-Project`, which is an empty 0-byte file that was left untouched).
Project spec: [docs/SPEC.md](docs/SPEC.md)

## Setup / commands
    cd ~/base-siege
    python3.11 -m venv .venv                      # already created
    .venv/bin/pip install -r requirements.txt     # full set already installed (torch/sb3-contrib/streamlit included)
    .venv/bin/pytest                              # run all tests
    .venv/bin/python scripts_smoke.py             # 100 random masked steps
    .venv/bin/python scripts_balance.py --n 200   # balance table: mean destruction by army x family
    .venv/bin/python scripts_contact_sheet.py     # results/layout_contact_sheet.png

## DONE
- STEP 1: scaffold (siege/ rl/ app/ results/ models/ tests/ docs/), requirements.txt, pytest.ini,
  .gitignore, README, git init.
- `siege/env.py`: `SiegeEnv`. Discrete(246) layout = 4 buy | 4x60 deploy | END_BUY(244) | WAIT(245).
  Obs Dict: grid (8,16,16) [ch0 deployed-troop occupancy, ch1 building hp, ch2 turret hp, ch3 wall hp,
  ch4-7 still zero/reserved], state (9) [gold, 4 owned counts, 3-way phase one-hot, tick].
  `action_masks()` implemented for buy/deploy/battle phases. Illegal action raises ValueError.
- STEP 2: `siege/troops.py` (4 troop types), `siege/bases.py` (layout generation), `siege/combat.py`
  (`simulate_battle`: deterministic tick-based combat, documented fixed tie-breaking rules). Wired the
  real simulator into `SiegeEnv` (battle resolves in one `simulate_battle` call the step the army is
  fully deployed; `env.battle_result`/`env.destruction` hold the outcome; grid channels 1-3 reflect
  building/turret/wall state). `siege/render.py`: matplotlib board renderer.
  Bug fixed: `combat._select_target` ignored each troop's own `target_preference` ordering (hardcoded
  building>turret>wall), so wall-preferring troops never prioritized walls. Fixed and regression-tested.
- A 200k-timestep MaskablePPO job was run as a pipeline smoke test only -- NOT a result, do not cite
  it. It confirmed env/model wiring works end to end but ran against the pre-balance single-family
  layout and un-tuned stats from step 2, so its "ranger-only -> 0.96 mean destruction" outcome reflects
  the balance bug fixed in step 3 below, not a real finding. `rl/train.py`/`rl/eval.py` (MaskablePPO +
  `MultiInputPolicy`, held-out-seed eval) and `tests/test_rl.py` (tiny wiring smoke test) were built at
  this point and are reused as-is in step 3/6 below.
- STEP 3: layout families, balance pass, acceptance rules, frozen DEV/FINAL pools. Full detail and the
  balance table/split-logic report are in README.md ("Layout families, balance, and the frozen pool");
  summary:
  - `siege/bases.py`: three generator families -- `ring_defence` (one walled core), `clustered_core`
    (two separately-walled clusters with gates), `scattered_outposts` (3-5 small independently-defended
    outposts). Each: 6-12 buildings, 2-5 turrets, walls, a core.
  - `scripts_balance.py` + `siege/armies.py`: fixed single-type/mixed test armies, clustered at one
    shared deploy point (see "why clustering" in docs/troops.md -- spreading evenly made it impossible
    for any mixed army to ever beat a single-type one), scored with the real simulator across 200
    layouts/family. Used to drive troop/turret stat tuning (`siege/troops.py`, `siege/bases.py`
    constants) until no single type dominated every family -- final table and full before/after
    rationale in docs/troops.md.
  - `siege/layouts.py`: acceptance rules (reject empty / unreachable / too_hard / ranger_trivial),
    `build_layout_pool()` generates exactly 241 accepted layouts, split into DEV (160) / FINAL (81)
    pools with `ring_defence` held out entirely as "transfer" (41, all FINAL) and the other two
    families "familiar" (80 DEV / 20 FINAL each). Saved as `results/layouts.json` (sha256 in
    `results/layouts.sha256`) -- records are just `{id, family, seed, pool, tag}`, layouts are
    regenerated deterministically from `(family, seed)`, no geometry stored.
  - `SiegeEnv(layout_pool=...)` samples a record per `reset()` via the env's own seeded `np_random` and
    regenerates that exact layout. `tests/test_layouts.py` (7 tests) covers the pool split, exact-layout
    reproduction, acceptance-rule unit cases, and (critically)
    `test_dev_familiar_sampler_never_leaks_final_or_transfer_ids`.
  - `results/layout_contact_sheet.png`: 12 samples per family (`scripts_contact_sheet.py`).
  - Tests: 24 passing total (`test_env.py` x10, `test_simulator.py` x6, `test_rl.py` x1,
    `test_layouts.py` x7).
- STEP 4: finalized `SiegeEnv` interface and masks for the frozen DEV pool, with action encode/decode
  helpers, fully documented observation/action/reward specs (`docs/observation.md`, `docs/actions.md`,
  `docs/reward.md`), and a legal-action mask contract that still passes Gymnasium `check_env`.
  `RandomLegalAgent`, `GreedyAgent`, the best single-troop army per family, and an `Oracle` (300-sample
  random search per layout) were evaluated on DEV/familiar plus held-out transfer layouts and saved to
  `results/baselines.csv`. The strongest family-wise signal is `Oracle` at 0.8967 overall destruction,
  followed by best single-troop at 0.4314; all baseline numbers are stored and ready for later PPO comparison.

### STEP 5: D4 augmentation (done)
- `rl/augment.py`: 8 symmetries (`s = 4*flip + k`), explicit 246-action permutation table per symmetry
  (buy/END_BUY/WAIT fixed; deploy keeps troop type, permutes the border-tile index), grid/mask/obs
  transforms, `transform_layout`/`transform_army`, and `D4Wrapper` (random symmetry per reset, maps the
  agent's action back to original coordinates before stepping; exposes `action_masks()` for MaskablePPO).
- The simulator was NOT D4-equivariant: 128/700 transformed battles differed. Culprits, all position-ordered
  tie-breaks in `siege/combat.py`: (1) equal-distance target ties broke by `(y, x)`; (2) movement ties
  (diagonal blocked, two equally-good orthogonal steps) broke by `(y, x)`; (3) turrets fired in `(y, x)` order
  and picked targets by `(y, x)`. Now ties go through D4-invariant properties first (target hp/value, remaining
  Euclidean distance, summed distance to the rest of the base), then object ids, never raw coordinates.
- Result: 0/7000 mismatches with ids carried along. Adversarially re-shuffling wall ids leaves 4/7000
  (one ring layout, a genuine tie in a symmetric configuration).
- Side effect: re-scoring the frozen 241 layouts with the fixed simulator, 23 now fail the acceptance rules
  (12 too_hard, 11 ranger_trivial). Resolved in STEP 5b (pool rebuilt).
- `tests/test_augment.py`: inverse identity, bijection, group closure, mask legality, and 100-layout
  env + simulator equivariance (fails on the old simulator, passes on the fixed one).

### STEP 5b: layout pool v2 under the D4-equivariant simulator (done)
- **Decision:** rebuild and re-freeze the pool (user's call). **Why:** the Step-5 simulator fix changed
  outcomes, and 23/241 v1 layouts (12 too_hard, 11 ranger_trivial) no longer passed the acceptance rules,
  so v1 was inconsistent with the simulator it would be evaluated on.
- v1 archived untouched as `results/layouts_v1.json` + `layouts_v1.sha256` (sha 07d1fc7d...25e2bf); superseded, see docs/troops.md.
- v2: same generators, acceptance rules and split logic; fresh seed stream `seed_base=5_000_000`
  (`scripts_rebuild_pool.py`; seeds, split seed and rejection counts in `results/layouts_build.json`).
  Acceptance is decided per layout and never uses pool membership or any FINAL-pool result.
- Counts: ring_defence 41 (all FINAL/transfer); clustered_core 80 DEV + 20 FINAL; scattered_outposts
  80 DEV + 20 FINAL. Total 241 = DEV 160 + FINAL 81.
- Rejections while building: clustered_core 12 ranger_trivial + 35 too_hard; ring_defence 22 too_hard;
  scattered_outposts 417 ranger_trivial (only ~19% of generated outposts layouts survive).
- **SHA256 (results/layouts.json): 7d9a394934aa36f3e68204aeac01fefd93502adf4cd933b7dbf9b4d56095a280**
- Fixed a latent reproducibility bug: the split and baseline RNG seeds used `hash(family)`, salted per
  process; now `zlib.crc32` (`siege.layouts.stable_family_hash`), with a cross-process regression test.
- **Residual symmetry issue (carried from Step 5):** 0/7000 mismatches with object ids carried along, but
  4/7000 if wall ids are adversarially shuffled (one ring_defence layout, a genuine tie in a symmetric
  configuration). Not fixed; the wrapper never relabels objects so it cannot trigger this.
- Balance (mean destruction, fixed armies, push-point deployment), frozen pool:
  ring_defence swarm .000 ranger .069 tank .316 splash .012 mixed_balanced .417 mixed_breach .232;
  clustered_core .126 / .155 / .127 / .000 / .407 / .096; scattered_outposts .639 / .314 / .478 / .016 / .476 / .162.
  No troop dominates everywhere (best single: tank / ranger / swarm). Splash is dead (<= .016) and
  scattered_outposts is too easy unfiltered (ranger .899 on fresh seeds, hence the 81% reject rate). No stats changed.
- Baselines re-run on new DEV (+ transfer) -> `results/baselines.csv`; contact sheet regenerated.
- Tests: leak guard unchanged and passing; added sha, archive, build-log, acceptance-under-current-sim,
  id/seed-uniqueness and cross-process split tests.

### STEP 5c: final balance pass + pool v3 (done)
- **Tuning discipline:** tuned only on a separate TUNING seed set (300 raw layouts/family, seeds 9M/19M/29M + 0..299,
  recorded in `results/layouts_build.json`, tests assert it is disjoint from v1/v2/v3 pools). 2 iterations used, then frozen.
  Iteration 1: splash area damage (r=1.5, falloff 0.6) + scattered_outposts turrets per outpost -> splash overshot
  (strictly best army on 77%/67% of ring/clustered layouts). Iteration 2: falloff 0.3, picked by a 4-point sweep
  (0.2-0.5) on the same tuning seeds -> accepted. Details + every stat change and reason: docs/troops.md.
- **Simulator changes:** splash area damage (new mechanic; radius-based so D4-invariant); duplicated attack code refactored into
  `_strike` (verified bit-identical on 400 battles before stats changed); turrets now fire **simultaneously**.
  Residual D4 id-shuffle mismatches: 4/7000 -> **0/7000** (new regression test).
- **Generator:** scattered_outposts turrets drawn per outpost (1-2) instead of 2-5 shared across 3-5 outposts.
  Raw rangers-only mean 0.881 -> 0.442; acceptance 17% -> 64%.
- **Greedy:** v1 ranked by damage/cost and bought 5x splash (weakest unit) -> scored below Random. Now ranks by
  Lanchester value hp*damage/cost^2 (6 tank + 2 swarm).
- **Pool v2 archived** (`layouts_v2.json/.sha256`, `layouts_build_v2.json`, `baselines_v2.csv`); v3 = seed_base 3,000,000, same split logic.
  Counts: ring_defence 41 FINAL(transfer); clustered_core 80 DEV + 20 FINAL; scattered_outposts 80 DEV + 20 FINAL (241 total).
  Acceptance rates: ring 83.7%, clustered 88.5%, scattered 63.7%.
- **SHA256 (layouts.json): e97cfbc7d43bf30842618511b9004764d5655b971ec6b208e7bd1e039edd0592**
- **Known issues found (not fixed, to avoid invalidating the tuning):**
  (1) RESOLVED (cleanup before step 6): `combat._normalize_walls` ignores hp when walls are passed as a `{(x, y): hp}` dict, so every
  wall fought at 80, not the generator's 70; all tuning/pools were measured at 80. `bases.WALL_HP`, docs/troops.md now say 80;
  simulator untouched; `test_walls_have_80_hp` pins it. Pool sha unchanged.
  (2) tank is the weakest unit now (never the best single type; oracle ban-loss ~0.01-0.04). (3) Ranger is the oracle's most
  irreplaceable unit (ban-loss 0.14-0.17); splash/swarm/tank are partly substitutable in oracle armies.
  (4) Wall targeting is a global rank (tank/splash chew through every wall before touching buildings).

### STEP 6: PPO training pipeline (done)
- Cleanup first: wall hp = 80 everywhere (generator constant, docs/troops.md, this file); `test_walls_have_80_hp`; pool sha unchanged.
- `rl/features.py` (`SiegeFeaturesExtractor`: 3-layer CNN on 8x16x16 -> 256, MLP on 9-d state -> 64, concat 320),
  `rl/policy_io.py` (`policy_outputs`: masked probs/logits, value, top-k decoded actions -- explainability hook),
  `rl/split.py` + `results/val_split.json` (fixed split of DEV-familiar: 140 train / 20 val, 10 per family, bound to layouts sha),
  `rl/evaluation.py`, `rl/train_ppo.py`, `rl/eval_ppo.py`. Tests: `tests/test_step6.py` (6). Total 120 passing.
- Env additions: `SiegeEnv(prebought=...)`, `set_prebought()`, `reset(options={"layout_record", "prebought"})`.
- Train: `.venv/bin/python -m rl.train_ppo --aug {none,d4} --seed N --timesteps N --curriculum {on,off} --run-name X`
  Eval: `.venv/bin/python -m rl.eval_ppo --run-name X [--ckpt final|best]` (validation layouts only; caches results/val_baselines.json).
- Training only ever sees the 140 train layouts; val feeds the callback (every 25k steps, full-task and deploy-only); FINAL/transfer never loaded.
- Curriculum: phase A = deploy-only, pre-bought **Greedy army (2 swarm + 6 tank = 200 gold)**, env starts in DEPLOY; phase B = full game.
  Switch at timestep `--phase-a-steps` (default timesteps//3), applied via `env.set_prebought(None)` (takes effect at each env's next reset), logged to `curriculum.log` and model_identity.json.
- Per run in `models/{run_name}/`: final.zip, best.zip (best val full-task score; selected on val, which is DEV), checkpoints/, model_identity.json
  (algorithm, aug, seed, timesteps, git commit + dirty flag, layouts sha, hyperparams, throughput), progress.csv, val_eval.csv, curriculum.log.
- Hyperparams: 8 envs (DummyVecEnv) x 256 steps, batch 256, 10 epochs, lr 3e-4, gamma 1.0, gae 0.95, ent 0.01, net [256,256]; torch 1 thread.
- **Throughput:** 30k smoke: none 271 steps/s, d4 268 steps/s (incl. val evals); 300k sanity 286 steps/s. Threads don't help, so run ~1 job/core.
- **Sanity run (seed 0, none, curriculum on, 300k, 1053 s)** on the 20 validation layouts (deterministic):

  | agent | clustered_core | scattered_outposts | overall |
  |---|---|---|---|
  | PPO final | 0.334 | 0.354 | 0.344 |
  | PPO best (t=300k) | 0.390 | 0.356 | 0.373 |
  | Random | 0.081 | 0.055 | 0.068 |
  | Greedy | 0.180 | 0.206 | 0.193 |
  | best single (splash / swarm; picked on train) | 0.111 | 0.572 | 0.341 |
  | Oracle | 0.962 | 0.699 | 0.831 |

  (The 0.096 / 0.304 / 0.878 figures from baselines.csv are over all 201 DEV+transfer layouts; the table re-scores baselines on the 20 val layouts, which are harder for Random/Greedy and easier for best-single swarm; Oracle lower on scattered.)
  Final vs best differ only by one extra PPO update after the 300k eval -- noisy policy, 20 layouts.
- **Army (final):** clustered_core mean [swarm 4.5, ranger 0.5, tank 1.5, splash 2.3]; scattered_outposts [17.9, 0.2, 0.1, 0.3]. 13 distinct armies
  over 20 layouts; 7 layouts buy 20 swarm, 2 buy 5 splash. Not collapsed to one troop, but it largely learned "swarm on scattered, mixed on
  clustered", ignores rangers (the oracle's most irreplaceable unit), and ~matches best-single swarm only on scattered (0.356 vs 0.572 -- still worse).
  Full-clear rate ~0. Verdict: learning something beyond Greedy/Random, far from Oracle at 300k; val curve still rising slowly (0.28 -> 0.37 over t=125k-300k).
  Phase-B switch caused a dip (0.16 -> 0.05) then recovery.
- **Budget recommendation for 10 seeds x 2 conditions (20 runs):** ~286 steps/s/process; run 7 in parallel (8 cores; verify contention first)
  = 3 waves. **1M timesteps/run** ~ 58 min/wave, ~3 h total (2M ~ 6 h if the curve is still rising). Reduce val-freq to 50k to save time.
  Caveat: 300k is one seed; consider a small lr/ent_coef check on VAL only before launching (step 7).
- Known: phase switch is per-episode lazy (<=1 in-flight episode/env still in phase A); D4 and none have similar speed; git dirty at run time (recorded).

### STEP 6b: one bounded diagnostic/tuning round (decision rule written BEFORE results)
- Judged ONLY on the 20 validation layouts (DEV). FINAL/transfer untouched. This is the ONLY tuning round; hyperparameters are frozen for the grid after it.
- Configs (300k steps, seeds 0 and 1, curriculum on): **A** fixed Greedy phase-A army, lr 3e-4 (current) | **B** random phase-A army, lr 3e-4 |
  **C** random phase-A army, lr 1e-4, n_epochs 5, target_kl 0.03.
- **Rule:** score = mean over both seeds of (average of the last 3 validation evaluations, full task, t=250k/275k/300k). Pick the best-scoring config;
  adopt it only if it beats A by >= 0.03, otherwise keep A. No other criterion, no re-runs.

- **Added:** `--phase-a-army {fixed,random}` (`SiegeEnv.set_prebought("random")` = fresh random legal army per episode, buys random affordable types, stops w.p. 0.1/buy), `--lr`, `--n-epochs`, `--target-kl`; test in tests/test_step6.py (121 tests total).
- **Diagnostic 1** (`scripts_diag_deploy.py`, sanity best.zip, deploy-only single-type armies, 20 val layouts, PPO vs scripted best-single deployment; logs_6b/diag1.txt):

  | troop (n) | PPO clustered | script clustered | PPO scattered | script scattered | PPO all | script all |
  |---|---|---|---|---|---|---|
  | swarm (20) | 0.096 | 0.023 | 0.407 | 0.572 | 0.251 | 0.298 |
  | ranger (8) | 0.084 | 0.120 | 0.000 | 0.227 | 0.042 | 0.173 |
  | tank (6) | 0.051 | 0.070 | 0.120 | 0.165 | 0.085 | 0.118 |
  | splash (5) | 0.126 | 0.111 | 0.000 | 0.000 | 0.063 | 0.056 |

  Supports the hypothesis for rangers (0.04 vs 0.17; zero on scattered) and, more mildly, tank/swarm; splash is fine (it is weak everywhere).
- **Results** (300k, seeds 0/1, score = mean over seeds of avg of last 3 val evals, t=250k/275k/300k; `scripts_6b_analysis.py`, logs_6b/analysis.txt):

  | config | seed 0 | seed 1 | MEAN | clip_fraction | approx_kl |
  |---|---|---|---|---|---|
  | A fixed army, lr 3e-4 | 0.364 | 0.287 | **0.3258** | 0.44 | 0.079 |
  | B random army, lr 3e-4 | 0.366 | 0.335 | **0.3505** | 0.46 | 0.077 |
  | C random army, lr 1e-4, 5 ep, target_kl 0.03 | 0.272 | 0.196 | 0.2340 | 0.08 | 0.010 |

  (clip_fraction / approx_kl are means over the few logged rollouts; log_interval=20 so only ~7 rows/run.)
- **Army composition per family (final.zip, mean counts [swarm, ranger, tank, splash]):**
  A s0 clustered [4.5,0.5,1.5,2.3] scattered [17.9,0.2,0.1,0.3]; A s1 [2.5,4.3,0.6,1.2] / [9.8,0.9,2.5,0.1].
  B s0 [4.2,2.4,0.6,1.9] / [15.7,0.8,0.7,0.0]; B s1 [4.6,2.6,0.0,2.2] / [6.7,0.3,2.5,1.1].
  C s0 [20,0,0,0] / [20,0,0,0]; C s1 [0,8,0,0] / [0,8,0,0] -> **C collapsed to one troop type per seed** (underfit/stuck, entropy still high).
  B buys noticeably more rangers on clustered_core (2.4-2.6 vs 0.5/4.3) and gets 0.42-0.51 there, but still swarm-heavy on scattered_outposts.
- **DECISION (by the pre-registered rule): KEEP A.** B is best but beat A by only 0.0247 < 0.03. Frozen grid hyperparameters = config A:
  fixed Greedy phase-A army, lr 3e-4, n_epochs 10, no target_kl, other hyperparameters as in step 6. C rejected (collapse).
  B was a near miss (+0.025, seed 1 gap larger than seed 0), so it is a reasonable candidate if anyone reopens the question, but no further tuning round will be run.
  Caveats: tuning used only 2 seeds, the 20-layout validation split, 300k steps; differences of ~0.03 are within seed noise (A seeds differ by 0.08).
  clip_fraction ~0.44 persists under A and B, so the "updates too large" hypothesis was addressed only through C, which learned too slowly/collapsed.
- **Throughput correction:** with 6 jobs in parallel each ran at ~167 steps/s (A/B) and ~253 (C, 5 epochs), i.e. ~1000 steps/s aggregate, not 286 each.
  Revised budget for 20 runs: 1M steps each = 20M steps ~ 5.5-6 h at 6 parallel; 500k ~ 3 h.
- Run dirs: models/tune_{A,B,C}_s{0,1}/; logs in logs_6b/.

### STEP 7: PPO experiment grid written (NOT launched)
- **Config A is frozen for the grid** (fixed Greedy phase-A army, lr 3e-4, 10 epochs). Config B (random phase-A army, `--phase-a-army random`) is a
  **candidate ablation** (+0.025 val over A in step 6b, below the 0.03 adoption bar, deploys rangers better). Config C (lr 1e-4, 5 epochs, target_kl 0.03)
  failed: collapsed to a single troop type per seed (all-swarm / all-ranger), val 0.23 vs A 0.33. The 6b deploy-diagnostic table (PPO ranger
  deployment 0.04 vs scripted 0.17) is in the STEP 6b notes above.
- **Primary checkpoint for all headline results = `final.zip`.** `best.zip` (selected on validation score) is kept but NOT used for headline numbers
  (selecting on the 20 validation layouts is optimistic; final.zip is the pre-declared, unselected artifact).
- `scripts/run_ppo_grid.sh` (`--dry-run` prints commands): aug {none,d4} x seeds 10-19, 1M timesteps, curriculum on, interleaved order
  (s10 none, s10 d4, s11 none, ...), 6 parallel via `xargs -P`, one log per run in `results/logs/<run>.log` (appended), run names `ppo_none_s10`, `ppo_d4_s10`, ...
  Env overrides: `SEEDS`, `TIMESTEPS`, `PARALLEL`, `PREFIX`, `EXTRA`. `scripts/run_one.sh` runs one run (short args: macOS xargs -I has a 255-byte limit).
- **Resumable:** `rl.train_ppo --resume`: skips if `final.zip` exists; else continues from the latest `checkpoints/ckpt_<steps>_steps.zip` (every 50k steps,
  so an interruption loses <= 50k steps); with no checkpoint it cleans stale artifacts and restarts. Restores weights, optimizer and `num_timesteps`; curriculum
  phase is recomputed from the timestep (before the boundary -> `A-resume`, after -> `B-resume`; boundary and requested timesteps come from the original
  model_identity.json). Earlier `progress.csv` is kept as `progress_part<k>.csv`; val_eval.csv rows after the checkpoint are dropped and the eval schedule/best score continue; identity
  json gets a `resumes` list. **Resumed runs are not bit-identical to uninterrupted ones** (RNG state is not in checkpoints; reseeded as seed+100003*k);
  the audit log `curriculum.log` keeps entries from the discarded segment. Tests: tests/test_step6.py (3 new; 123 tests total).
- `scripts/status.py [--prefix ppo] [--seeds 10-19]`: table of runs (finished / running / interrupted / not started), current timestep (max of latest
  checkpoint and latest validation eval, so granular to 25k), % progress, latest and best validation score, resume count. "running" = a matching `rl.train_ppo` process exists.
- Verified with a 4-run mini-grid (seeds 900, 901 x none/d4; the script always runs both augs per seed), 20k steps, 3 parallel: logs, status, kill -9 of one run
  at t>=10k (status -> interrupted), rerun -> resumed from ckpt_10000 in phase B, finished. Mini-grid models/logs deleted.
- Expected cost: ~167 steps/s per job at 6 parallel => 1M steps ~ 1.7 h/job, 20 jobs in ~3.4 waves => roughly 6-7 h wall.
- Commands to run (own terminal):
      cd ~/base-siege && caffeinate -i nohup scripts/run_ppo_grid.sh > results/logs/grid.out 2>&1 &
      .venv/bin/python scripts/status.py            # check progress anytime
      # after a crash/reboot/Ctrl-C: just run the same command again; finished runs are skipped, others resume

### STEP 8: full evaluation of the 20-model PPO grid (run ONCE)
- Run on clean commit `6fe8366` (all 20 runs were trained at `6c2a800`, clean; nothing to commit beforehand). layouts sha256 `e97cfbc7...0592` (matches all models).
  Command: `nohup .venv/bin/python -m rl.eval_all --workers 7 > results/logs/eval_all.log 2>&1 &` (~4 min wall). Code `rl/eval_all.py`, tests `tests/test_eval_all.py` (8 tests; 131 total).
- **D4 models are evaluated on the canonical view**: bare `SiegeEnv`, no `D4Wrapper` (identity symmetry, nothing random). Test checks it equals a `D4Wrapper(symmetry=IDENTITY)` rollout.
  Deterministic actions (masked argmax), `final.zip` only. Groups: DEV-train 140, validation 20 (both SEEN in training/checkpointing, never headline), FINAL-familiar 40 + transfer 41 (HEADLINE, 81 layouts, never seen).
  Evaluator asserts every layout id is in the declared groups; unknown group names raise.
- Outputs: `results/final_eval.csv` (20 x 241 rows with identity), `results/final_eval_summary.json`, `results/final_eval_tables/*.csv`
  (model_identity, per_model, condition_summary, condition_family_summary, paired_diff*, baselines_*, sampled_probe_*), charts `results/{destruction_by_condition,familiar_vs_transfer,army_composition}.png`,
  sampled probe per layout `results/final_eval_sampled_EXPLORATORY.csv`. Bootstrap: 10000 resamples over learners, seed 8080, percentile 95% CI.
- **Headline (mean destruction, 10 learners, 95% CI):** none 0.273 [0.257, 0.293] vs d4 0.202 [0.184, 0.220]; full clear none 0.094 vs d4 0.053.
  FINAL-familiar: none 0.278 / d4 0.197; transfer (ring_defence): none 0.268 / d4 0.207. Paired none-d4 headline +0.071 [0.054, 0.091], none wins 10/10 seeds
  (familiar +0.081, 10/10; transfer +0.061, 10/10). **D4 augmentation HURT at 1M steps** (also on seen DEV-train: 0.433 vs 0.276), i.e. consistent with D4 underfitting, not a held-out-only effect.
  By family: clustered_core none 0.352 vs d4 0.182; scattered_outposts 0.204 vs 0.212 (tie).
- Baselines (headline 81 layouts): Random 0.084, Greedy 0.229, BestSingle 0.189, Oracle 0.927 (FINAL-familiar 0.891, transfer 0.963). PPO beats Random and (narrowly, none only) Greedy; far below Oracle.
  Oracle budget: `oracle_search` defaults, 200 random armies + 200 hill-climb steps (~400 sims/layout), seed [id, 4242]; a lower bound on the optimum.
  BestSingle troop chosen on DEV only: clustered_core=splash, scattered_outposts=swarm; ring_defence has no DEV layouts, so the pooled-DEV best (swarm) is used, which scores 0.000 on ring_defence (weak baseline there; an artifact of the no-peeking rule).
- Army bought (mean counts swarm/ranger/tank/splash, FINAL): see `army_composition.png` and condition_family_summary.csv. None, clustered_core: 3.4/2.5/0.8/1.8; d4: 3.1/1.5/1.1/2.4. scattered_outposts is swarm-heavy (5.8-6.5).
- **Sampled-action probe (EXPLORATORY, 5 sampled rollouts/layout, torch seed per model x layout x rollout):** sampled minus deterministic mean destruction: none headline -0.058 [-0.078, -0.039], d4 -0.038 [-0.047, -0.028]; sampling hurts both. Deterministic stays the reported mode.
- Caveats: seeds s13-s15 (both conditions) were resumed twice (not bit-identical to uninterrupted runs); all models ~1.0015M steps (rollout granularity). No evaluator bug found; nothing was tuned or re-run after seeing FINAL results.

### STEP 9a: diagnostics on the finished grid (no training, no tuning, no re-evaluation)
- **FINAL was evaluated exactly once (step 8, commit 6fe8366). Everything after that is diagnostic and must not drive tuning; any further model choices use DEV/validation only.**
- `rl/diagnose_grid.py` + `tests/test_diagnose_grid.py` (3 tests); outputs in `results/diagnostics/` (1_*, 2_*, 3_*, 4_*, 5_provenance.json, 2_validation_curves.png, stdout.txt). Reads final_eval.csv, baselines_per_layout.csv, models/*/val_eval.csv.
  Only fresh computation: fixed single-troop scores (simulator) per group for the hindsight reference. Command: `.venv/bin/python -m rl.diagnose_grid`.
- **1. By group (Random / Greedy / BestSingle(DEV-chosen) / PPO plain / PPO D4 / NON-DEPLOYABLE hindsight best single / Oracle):**
  DEV-train .074/.326/.419/.433/.276/.419/-; validation .073/.193/.341/.314/.227/.346/-; FINAL-familiar .059/.175/.382/.278/.197/.382/.891; transfer .110/.281/.000/.268/.207/.301/.963.
  Reading: (a) plain PPO barely beats the fixed best-single baseline even on training layouts (.433 vs .419) and is BELOW it on FINAL-familiar (.278 vs .382). The gap is entirely scattered_outposts, where a fixed
  swarm army scores .545 (FINAL) / .575 (DEV-train) but plain PPO gets .204 / .364. On clustered_core plain PPO beats the best single (.352 vs .218 FINAL). (b) clustered_core is harder on FINAL (best single .263 -> .218, Greedy .304 -> .102)
  but plain PPO drops more (.502 -> .352), so about 0.1 of that gap is overfit, the rest layout difficulty. (c) transfer: DEV-chosen single troop (pooled swarm) = 0.000 vs hindsight splash .301, so that baseline is meaningless there; PPO plain .268 is about Greedy (.281).
- **2. Curves (validation, full task, every 25k):** mean change over last 250k steps (window around 750k -> last ~3 evals): plain +0.013 (6/10 seeds up, slope 0.014/M steps, essentially flat, peak of mean .332 at 975k),
  D4 +0.037 (9/10 seeds up, slope 0.18/M, still rising; val .194 -> .232). D4 is slower, not converged at 1M; plain has plateaued around 0.30-0.33.
- **3. Per seed (final.zip):** validation plain min/max .264/.352, D4 .150/.314; DEV-train plain .415/.460, D4 .224/.342; FINAL-headline plain .238/.345, D4 .151/.256.
  Worst plain seed beats best D4 seed on DEV-train (yes, .415 > .342) but NOT on validation (.264 < .314) nor FINAL-headline (.238 < .256); seed ranges overlap there although the paired mean difference is large.
  Seeds 13-15 (resumed twice) are flagged in 3_per_seed.csv; their numbers are within the other seeds' range.
- **4. Army (mean counts s/r/t/sp, FINAL clustered_core):** plain 3.4/2.5/0.8/1.8, D4 3.1/1.5/1.1/2.4, Greedy 2/0/6/0 (fixed, layout-independent). scattered_outposts: plain 5.8/1.0/2.3/1.1, D4 6.5/0.7/2.5/0.9. ring_defence: plain 2.5/1.2/3.1/1.1, D4 2.6/0.9/3.7/0.9.
  Oracle army counts were never stored; only gold shares from step 4 (DEV, 80 layouts/family, and transfer): ranger 28-40%, tank 25-31%, splash 23-34%, swarm 11-12%. PPO buys far more swarm (13-30% of gold, 5-6 units on scattered) and fewer rangers than the Oracle.
- **5. Provenance confirmed:** final_eval_summary.json top-level `commit` = eval commit 6fe8366; the TRAINING commit 6c2a800 (dirty=false) is recorded per model in `models[].train_git_commit` (all 20), layouts sha256 e97cfbc7... matches train and eval. Nothing missing; no final_eval files rewritten.

### STEP 9b: buy-vs-deploy decomposition (eval-only, DEV/validation layouts only)
- `rl/diagnose_buy_deploy.py` + `tests/test_diagnose_buy_deploy.py` (5 tests). 60 layouts = 20 validation + 40 DEV-train (20/family, seed 9090). FINAL-familiar/transfer never loaded (test asserts). 4 workers, 73 s.
  Outputs `results/diagnostics/9b/`: matrix.csv (ALL60 / validation / DEV-train-subset x family), episodes.csv, ranger_efficiency.csv, oracle_armies.json (60 armies + start tile), own_army_counts.csv, run.log.
  Armies are forced via real buy actions + END_BUY; scripted deployer = `place_on_border(army in Greedy ranking order, _default_start_idx)`; Oracle deployer = oracle's own order + start tile (same search/seed/budget as steps 4/8).
- Control check: own-buy x PPO reproduces final_eval.csv exactly (1200 episodes, max diff 0.0). Scripted x Greedy / best-single reproduce baselines_per_layout.csv.
- **Matrix, mean destruction, 60 layouts (PPO cells: mean over 10 learners [95% CI]):**

  | army \ deployer | PPO plain | PPO D4 | scripted | Oracle deploy |
  |---|---|---|---|---|
  | own buying | .399 [.384,.413] | .266 [.246,.289] | .382 / .367 (plain / D4 own armies) | - |
  | Greedy army | .274 [.261,.288] | .205 [.182,.228] | .299 | - |
  | DEV-chosen best single | .360 [.338,.380] | .320 [.309,.330] | .413 | - |
  | Oracle army | .331 [.312,.345] | .203 [.169,.238] | .397 | .856 |

  clustered_core: own .489/.249, Greedy .240/.116, best-single(splash) .233/.175, Oracle army .484/.256, scripted .447 (own) .307 .258 .466, oracle deploy .917.
  scattered_outposts: own .309/.284, Greedy .307/.294, best-single(swarm) .486/.465, Oracle army .178/.150, scripted .316 (own) .291 .568 .328, oracle deploy .794.
- **Ranger efficiency** (destruction per 100 gold, 8-ranger army): PPO plain .060, D4 .066, scripted .083 overall; clustered_core PPO .091/.113 vs scripted .042 (PPO better); scattered_outposts PPO .029/.020 vs scripted .124 (PPO ~4-6x worse).
  Marginal ranger contribution inside own armies (destruction lost when rangers are removed, per 100 ranger gold): plain .434, D4 .225, scripted on plain's armies .385. So plain deploys rangers fine inside mixed armies; D4 does not; and a ranger-only army on scattered_outposts is where PPO fails.
- **Training-distribution check:** phase A (first 1/3 = 333k steps) pre-bought ONLY the Greedy army (2 swarm + 6 tank). A test confirms the forced-buy state (grid + state vector, after END_BUY) is bit-identical to the phase-A observation, so case (b) is in-distribution for deployment.
  Cases (c)/(d) are NOT: those post-buy states appear in training only if the policy itself bought such an army in phase B (its own armies, e.g. ~5-6 swarm on scattered, 2-3 rangers on clustered, never a 20-swarm, 8-ranger or oracle mix on demand). PPO-deployer numbers for (c)/(d) therefore mix "cannot deploy" with "off-distribution state"; treat as a lower bound on deployment skill.
- **Interpretation:** clustered_core: buying is NOT the bottleneck (PPO's own army x PPO deploy .489 is its best cell; Oracle's army only matches it at .484; remaining 0.43 to the Oracle is deployment/ordering/start-tile that neither PPO nor the scripted deployer finds).
  scattered_outposts: BUYING is the main bottleneck (a swarm army deployed by the plain script gets .568 vs .309 for PPO's own army; swapping the deployer on PPO's own army changes nothing: .309 vs .316), with a secondary deployment loss (swarm .486 PPO vs .568 scripted; rangers/splash badly deployed).
  D4 is worse mainly at deployment (own army .266 vs scripted deployer on the same armies .367).
  Oracle's lead is mostly deployment: the same Oracle army scores .856 with oracle deployment vs .331-.397 with PPO/scripted. All of this is on DEV/validation only (nothing from FINAL), diagnostic, no tuning.

### STEP 10a: v2 redesign decision (BreachLens) -- DESIGN ONLY, no code
- **Decision:** prototype frozen at git tag `prototype-v1` (commit ef43f63); redesign continues on branch `v2-redesign`. `master` untouched.
  Design lives in [docs/SPEC_V2.md](docs/SPEC_V2.md): drone fleet vs automated citadel, 14x14, budget 240, 240 ticks, max 20 units, 52 border tiles,
  5 units (Scout/Sniper/Bulwark/Blaster/Mender), Discrete(267), 9 channels + 11 state, 4 families (labyrinth held out), pool of 300 (180 DEV / 120 FINAL),
  learners PPO / PPO+D4 / masked DDQN / policy-guided planner (N = 8, 32, 128), D4 TTA, "Why this move?", duel mode, blueprint-theme app.
- **Why:** prototype PPO stayed far below Oracle (0.27 vs 0.93), D4 underfit, one unit (splash/tank) was weak in every tuning pass, and phase A used a single fixed army.
  v2 targets those: explicit non-dominance/non-dead gates, random phase-A armies, equivariance-by-construction tie rules (split/abstain, no ids), larger step budget.
- Unit stats and the 6 balance gates in the spec are PROPOSED, not tested. Training-time estimate (22-29 h wall for the full grid) is an extrapolation to be re-measured in build steps B2/B10.
- 20 build steps (B1-B20) with a pytest requirement each are listed in the spec. Nothing is started until told.

### STEP 10b: SPEC_V2 amendments (design only, no code)
- Added gates **G7** (3 PPO seeds x 300k on TUNE must beat max(BestSingle, Greedy) by >= 0.05 score, 2/3 seeds, no collapse), **G8** (army search >= +0.10 and full Oracle >= +0.20 over best single unit, per familiar family), **G9** (LAB-DESIGN labyrinth: Oracle >= 0.60, Greedy 0.10-0.45). Thresholds are frozen in the spec; rationale from prototype seed spread (SD ~0.05-0.06) and the scattered_outposts finding that buying did not matter.
- Design sets TUNE (180/60) and LAB-DESIGN (40) on disjoint seed streams; labyrinth FINAL generated only at B6 after a recorded `generator_hash`; content-hash disjointness test. Revision budget: 2 total, <= 1 in B5, 1 reserved for after G7.
- Movement: exact tie-break chain for equal-length shortest paths (Euclid to target, turret cover, centroid, canonical-form fallback; always moves); pytest for no-deadlock on exact diagonals and D4-exactness in labyrinth.
- Planner: nested N=8/32/128 prefixes, RAS-N (random-army search + scripted deploy) and RPS-N controls at equal N, cost estimate 1.2-2.4 h, 3-step fallback if projected > 3 h.
- B2 speed gate: battle <= 60 ms mean, random loop >= 300 steps/s (=> >= 100 steps/s/job); else 150 ticks, then 12x12.
- Step budget: from pilot slopes (table) capped by T_cap = 18000*r; shaped-reward fallback (potential-based preview) defined, used only if G7 fails; armies x deployers pilot diagnostic on TUNE-val; B20 removes siege/, B21 README with "Inspired by" line (source name needed from the owner).

### STEP 10c: SPEC_V2 amendments (design only, no code)
- G2/G3/G4 now need the point threshold AND a 95% paired bootstrap lower bound > 0 (same layouts, same Oracle/template seeds, bootstrap over layouts, 10000 resamples, seed 8080; thresholds in `data/v2/gate_thresholds.json`, doc-checked by test).
- G7: BestSingle = per-family best single-unit army chosen on TUNE-train, scored on TUNE-val. One-time pre-registered extension to 600k if the mean last-100k slope >= +0.02 score/100k (my reading of the unit); otherwise fail.
- Revision budget now 3 in B5 + 1 after G7 (4 total), TUNE/LAB-DESIGN only. Post-G7 revision = stats or reward only, never generators; stat changes re-accept the pool from the same seed streams (new sha) before any agent evaluation. FINAL-access ledger (`data/v2/final_access.log`) + pytest that no agent evaluation of FINAL exists before the final eval.
- Defenders: Gun (r4), Lancer (r6, min 2, threatens range-5 units), Flak (area, r5, punishes clumped Scouts) with stats and generator-mix rules. Walls and turrets carry no score value.
- B5 report per unit: effective hp/gold incl. armor (per defender type), dps/gold, threatening defenders, single-unit-army score table per family.
- B21 README line uses placeholders AUTHOR_NAME / AUTHOR_URL; test fails until replaced.

### STEP B1 (v2): scaffold, constants, action codec, border tiles (done, branch v2-redesign)
- New package `breachlens/`: `constants.py` (14x14, budget 240, 240 ticks, cap 20, 52 border tiles, 5 unit names, Discrete(267) layout, obs shape constants 9ch/11 state), `actions.py` (clockwise border from top-left, `encode_buy`/`encode_deploy`/`decode_action`).
- `tests/test_b1_constants.py` (14 tests): 52 unique clockwise contiguous edge tiles, codec bijection over 267, range errors, constants parsed from SPEC_V2 section 2 and compared. Full suite 153 passing (prototype tests untouched; `siege/` still present until B20).
- No unit stats, simulator, env or layouts yet (B2+). Run: `.venv/bin/pytest tests/test_b1_constants.py`.

### STEP B2 (v2): units/defenders/walls/core combat + simulator speed gate (done, branch v2-redesign)
- `data/v2/unit_stats.json` is the only place unit/defender/structure stats live; `breachlens/config.py` loads it once (`CFG`), scales hp-like values by S=840, ranges/speeds are integer tenths, and exposes `sim_version()` = sha256(stats file bytes + NUL + `SIM_CODE_VERSION`). Bump `SIM_CODE_VERSION` on any behaviour change.
- `breachlens/sim.py`: Scout/Sniper/Bulwark/Blaster, Gun/Lancer/Flak, walls, buildings, core. Integer-only outcomes; tick order per spec 3.3; nearest-in-preference targeting (equal keys -> exact damage split); BFS movement with rule-7 keys 1-3; blocked units pick the wall on the cheapest ghost path; early exit (cleared / all_dead / stalemate / horizon). `Prepared` caches fields per layout (hold one per layout in the env).
- (B2 history) Mender was rejected and key 4 stubbed with a raw-coordinate stopgap; both replaced in B3 below.
- `breachlens/_bench_layouts.py`: PROVISIONAL crude layouts of the 4 families, used only by speed benchmark and tests; B4 replaces them (not balanced, not for gates).
- Speed gate (`scripts_speed.py` -> `data/v2/speed.json`): PASSED by a wide margin, see numbers in the B2 report; tests `tests/test_b2_sim.py`, `tests/test_b2_speed.py`.
- Spec-silent interpretations (see report): speed via integer accumulator (steps = acc//10 per tick); stalemate = no hp change anywhere and no movement for 12 ticks; in-range walls are attacked per rule 6 (so a passing Scout stops to chew an adjacent wall).

### STEP B3 (v2): Mender, rule-7 key 4 (canonical-form fallback), D4 equivariance (done, branch v2-redesign)
- `breachlens/sim.py`: Mender (heal key chain `(dist^2, hp fraction, -cost)`, exact split among ties, heals add, no overheal / revive,
  nothing for units killed this tick, heals computed from the start-of-tick snapshot and applied after damage; follows the centroid of living
  non-Mender allies, stops within 1.5, holds with no allies or when walled off, never chews walls). Rule-7 key 4 = `canon_pick`: tied candidates
  get the world serialized (all alive structures + units, candidate decision marked) and the lexicographically smallest string over the 8 D4
  transforms wins. **The raw-coordinate stopgap is deleted** (test greps the source); no id or raw coordinate decides an outcome.
  `fallback_calls` now counts only genuine canonical-form calls; new `canon_equal_picks` counts calls whose best candidates had EQUAL strings.
- Sites that use key 4: movement step, movement target, blocked-wall choice, Blaster splash centre, Flak burst centre. Attack/heal ties split
  (floor of fixed-point hp; exact for k <= 8 since S = 840). Flak: units in the burst cell with equal (hp fraction, cost) share centre+neighbour damage.
- Interpretations the spec leaves open (see B3 report): (1) every tied decision reads the start-of-movement-phase snapshot (positions, speed accumulators,
  pending step counts) plus the deciding unit's own steps, so no processing order exists to matter (stronger than the spec's sequential working copy);
  (2) when canonical strings are EQUAL the first candidate in iteration order is taken (worlds are D4 images, outcome-equivalent up to a symmetry of the
  world; this is the only place iteration order is consulted); (3) Menders heal AND move in the same tick; (4) a Mender is never counted idle when it has no
  route/no allies (it has no target); (5) tick-1 tied movement targets are recomputed every tick (only untied targets are cached).
- Bug found by the adversarial id-shuffle test and fixed before commit: speed accumulators were advanced inside the movement loop, so a canonical
  decision saw a half-updated world (order dependent). Now all accumulators advance before any decision. `SIM_CODE_VERSION` = `breachlens-sim-b3.0`.
- Tests: `tests/test_b3_mender.py` (55) + updated `tests/test_b2_sim.py` (stopgap assertions replaced; determinism / idle tests now include Menders). Helpers:
  `breachlens/_d4check.py` (independent D4 tables, transform/summary/stabilizer, parallel equivariance runner), `scripts_b3_smoke_d4.py`.
  D4 equivariance: 10,000 random battles x 8 symmetries (Menders in >= 50% of armies), 0 mismatches; 2,000 x 8 with independent shuffles of army order and
  building/wall/defender lists per frame, 0 mismatches. Full suite 251 passing.
- Smoke D4 (200 battles x 8, provisional layouts, seed 3030): 0 mismatches (mutation check: disabling the canonical key gives 24).
- Speed (`scripts_speed.py`, gate still passed by a wide margin): battle warm mean 0.441 -> 0.999 ms (5 unit types incl. Mender), p95 0.79 -> 2.03 ms;
  same 4-type workload as B2: 0.441 -> 0.634 ms (+44%); env 1 proc 26,971 -> 17,775 steps/s (-34%), 6 procs 107,986 -> 74,955 (-31%). Slowdown > 30% is
  flagged: canonical fallback (~2 calls/battle) and the Mender BFS (~25% of time with Menders). Not optimised further (gate is 60 ms / 300 steps/s).

### STEP B4 (v2): layout generators, structural acceptance, TUNE / LAB-DESIGN sets (done, branch v2-redesign)
- `breachlens/layouts.py`: 4 generators (`spine_pods`, `archipelago`, `bunker_nest`, `labyrinth`), each a pure function of `(family, seed)` (`candidate`), plus
  `check()` = the structural acceptance rules (empty, no value building, overlap, interior-only, per-family building/defender/wall/value-total ranges, value weights in 1..3,
  Gun majority >= 50 % in familiar families, labyrinth uses gun+lancer+flak, **core and every building reachable without breaking walls**). A rejected seed is skipped, the next seed of the
  stream is tried. Layout module never imports config/sim (test) => acceptance is stat-independent. Each candidate is built in a canonical frame then put in a random D4 frame.
  Content hash is D4-canonical (a mirrored duplicate counts as the same layout).
- Seed streams (disjoint, all above the prototype's max seed 25,000,516): `T` base 100,000,000 (TUNE), `L` 200,000,000 (LAB-DESIGN), `F` 300,000,000 (reserved for the pool, **never generated or loaded in B4**);
  per-family stride 1,000,000. `build_design_sets()` / `load_design_set()` refuse any seed outside T/L.
- `data/v2/design_sets.json` (written by `scripts_b4_design_sets.py`): TUNE-train 180 (60/family), TUNE-val 60 (20/family), LAB-DESIGN 40, with ids, seeds, content hashes, stream ranges, rejection counts.
  `generator_hash` NOT recorded (end of B5). FINAL-access ledger (`data/v2/final_access.log`) does not exist: zero entries.
- NOT done in B4 (needs the Oracle/baselines): the stat-dependent acceptance rules too_hard / single-type-trivial; they are applied when the pool is built (B6). TUNE/LAB-DESIGN are therefore structurally accepted only.
- `breachlens/armies.py`: `random_army`, `free_tiles` (moved out of `_bench_layouts.py`, which is now only the B2/B3 simulator-test fixture) and `scripted_army` (2 Bulwark, 2 Sniper, 2 Blaster, 1 Mender, 4 Scout, clustered from the border tile nearest the core).
- `breachlens/_d4check.py` gained the real-layout D4 runner (`equivariance_chunk_real`, `run_parallel_real`) incl. per-tick position equivalence (up to the transform). `scripts_b4_report.py` -> `data/v2/b4_report.json`
  (full-scale D4 run + battle-length table); `scripts_b4_contact_sheets.py` -> `results/design/contact_sheet_<family>.png`; `scripts_speed.py` now benchmarks the real layouts.
- Tests: `tests/test_b4_layouts.py` (32). Full-scale D4 on real layouts: 4,100 battles x 8 (2,000 labyrinth, Mender in 85 %) + 810 with adversarial id/list shuffles: 0 mismatches, 0 per-tick-position mismatches.

### STEP B5 (v2): gate script, acceptance rules, B5 report (done, branch v2-redesign) -- **7 of 8 evaluated gates FAIL; no stat/generator/rule changed; generator_hash NOT recorded**
- Simulator: new exit `no_attackers` (no live unit with damage > 0; ends the battle immediately, counted separately from stalemate, cannot change structures or score -- tested on 300 real battles with the exit on/off).
  `SIM_CODE_VERSION` = `breachlens-sim-b5.0`; extra read-only counters (attack / wall-attack ticks, distinct walls hit, ticks with any wall attack). B2/B3 fixtures that use damage-0 dummies run with the exit switched off (`_no_attackers_exit=False`).
  Re-run after the change: D4 smoke 200 battles x 8 = 0 mismatches; real-layout D4 4,100 x 8 plain + 810 x 8 adversarial shuffles = 0 mismatches, 0 trace mismatches (`data/v2/b4_report.json`); speed gate still passed (`speed.json`).
- `breachlens/oracle.py` (the only search implementation; B8 reuses it): full Oracle (200 random + 200 hill-climb, army + start tile, warm starts = 5 single-unit armies + Bulwark/Mender template + Greedy) and army search (128 + 128, scripted deployment).
  `armies.py` gained `place_on_border`, `default_start`, `single_unit_types`, `greedy_types`/`greedy_rank`, `random_types`. `acceptance.py`: `too_hard` (acceptance Oracle, rng tag 5151, finds no breach) and `single_type_trivial` (best single-unit army > 0.99) + builder
  that draws further seeds from streams T / L. `gates.py` + `scripts_gates.py`: raw per-layout evaluation, paired bootstrap (10k, seed 8080, one-sided 98.75 % bound for G2/G3), G1-G6/G8/G9, B5 report.
- SPEC_V2 amended (pre-registered, written before the gate run): rule 8 `no_attackers`; G2/G3 one-sided 98.75 % bound (Bonferroni over 4 families); Oracle / army-search / Greedy / template / G6 / G9-stalemate / acceptance definitions; extra wall metrics. `data/v2/gate_thresholds.json` created (the spec said it existed; it did not) and pinned by `tests/test_b5_gates.py`.
- **Stat-dependent acceptance rejection rates (design sets rebuilt, 1142 s):** spine_pods 91.2 % (832/912), archipelago 99.4 % (13,254/13,334), bunker_nest 60.0 % (120/200), labyrinth 74.2 % (115/155). **All four families flagged > 50 %.** Every rejection was `single_type_trivial`; `too_hard` fired 0 times.
  Streams used: T consumed 912 / 13,334 / 200 seeds of 1,000,000 per family; L 155. FINAL / stream F never generated or loaded; ledger `data/v2/final_access.log` does not exist (zero entries).
- **Gate results (revision 0; `data/v2/gates.json`, `b5_report.md/json`, raw per-layout `b5_raw.json`; run on the working tree on top of 67ae8b1):**
  G1 FAIL (Sniper is the best single-unit army on 75 % of layouts, limit 60 %) | G2 FAIL (no unit has paired Oracle ban-loss >= 0.02 in any family; max 0.009) | G3 FAIL (Sniper's template wins all 3 families; pigeonhole: 5 units, 3 tested families)
  | G4 FAIL (Mender gain -0.001 on bunker_nest, +0.008 on LAB-DESIGN; Mender-only = 0.000 passes) | G5 FAIL (best single 0.69 / 0.73 / 0.53; Oracle breach 1.00 in every family, limit 0.95) | G6 FAIL (Oracle uses > 12 units on 1 % of layouts, need 30 %; 20-Scout army never >= Oracle)
  | **G8 PASS** (army search +0.23 / +0.24 / +0.37, Oracle +0.27 / +0.30 / +0.47) | G9 FAIL (Oracle 1.000 ok, breach 1.00 ok, 0 stalemates ok, but Greedy mean 0.636 > 0.45).
  Dominant cause: Oracle saturation (mean 0.997-1.000, breach 100 %) -> ban-losses ~0 and nothing is "needed"; Sniper outranges the Gun majority. Proposals (at most 3) are in the B5 report message; the owner decides, none applied.
- Not recorded (gates failed): `data/v2/generator_hash.json`. Tests: `tests/test_b5_sim.py` (8), `test_b5_oracle.py` (8), `test_b5_gates.py` (17); full suite 316 passed.

### STEP B5-R1 (v2): design revision 1 of 3 -- **sweep found no arm meeting >= 4 of 5 criteria; NOTHING APPLIED, no revision spent, gate script NOT re-run**
- G3 amended (pre-registered, commit 1dbb34f, before any sweep result): per-layout niche (>= 8 % of the 280 pooled layouts, margin >= 0.03 over the runner-up of 6 templates, 95 % lower bound > 0); no one-sided bound. Thresholds + tests updated. Revision-0 data re-scored under it (reference only): Sniper 48 % and Mender 25 % pass; Scout 2.9 %, Bulwark 6.1 %, Blaster 1.1 % fail.
- Sweep (exploration, not a revision; TUNE-train first 20/family + LAB-DESIGN first 20; TUNE-val untouched): `scripts_b5r1_sweep.py`, outputs in `data/v2/b5r1_sweep/` (arm_*.json, sweep_summary.md/json). Override hooks: `BREACHLENS_STATS_PATH`, `BREACHLENS_GEN_PARAMS` (layouts.GEN_PARAMS: lancer_share, defender_count_mult; defaults = revision 0, bit-identical layouts); gate script asserts they are unset.
- Result by the fixed rule: criteria met A 0, B 0, C 1, D 0, E 1 (of 5). Oracle still saturated (breach 1.00 everywhere, score 0.87-1.00), Sniper is best single in every family in every arm, stat rejection > 40 % in archipelago/labyrinth for all arms, Greedy on labyrinth 0.58-0.64 (> 0.45). Only C fixes the Sniper share (0.53); only E puts best-single in range (0.32-0.64 in all families, hp x1.8).
- Budget: 3 revisions in B5 still unspent; owner decides the next design (see report message).

### STEP B5-R1b (v2): gate changelog, monotone ban-loss estimator, R1b sweep -- **no arm meets >= 4 of 5 (best R1b 3/5); NOTHING APPLIED, no revision spent, gate script NOT re-run**
- Gate changelog (SPEC 3.4, commit 5d901cb, before any sweep): G5 = Oracle score in [0.80, 0.97] + Oracle breach >= 0.6 (ceiling removed); G6 = reported diagnostic (not in all_pass); G9 Greedy band 0.10-0.65; G2/G4 use the monotone ban-loss estimator. `gate_thresholds.json` + tests updated. The stored revision-0 `gates.json`/report are historical (old definitions); the reproducibility test only re-checks G1/G3/G8 against them.
- `oracle.monotone_ban_pair` (single Oracle module): full search warm-started with the ablated best; unit-free armies seen by the full search also count for the ablated side; loss >= 0 by construction. On R0 reference data (80 layouts) old vs new differ little (no negative independent losses occurred; max diff 0.003, e.g. labyrinth sniper 0.012 -> 0.009), because R0 is saturated.
- Sweep parameters (override hooks only, asserted unset in gate runs/tests, defaults bit-identical): `class_damage_pct` per unit, `GEN_PARAMS.sealed_fraction` (separate rng, ring of 8 walls, displaced objects, waiver only for sealed layouts, +8 wall max), `rules.unit_cap` (used by armies + Oracle), core hp / Mender stats via stats file. Sweep: `scripts_b5r1b_sweep.py`, outputs `data/v2/b5r1b_sweep/`. Selection rule pre-registered as SPEC 3.4 item 8 (committed d42de0f before the sweep).
- Result: criteria met R0 0, R1a 1, R1b 3, R1c 3, R1d 2 (of 5). Sniper stays best single in every family and arm (pooled share 0.6-0.8); rejection > 40 % in archipelago and labyrinth in all arms; R1c == R1b (cap 14 never binds: Oracle uses <= 12 units); R1d pushes Oracle score above 0.97 (Mender buff). See `sweep_summary.md`.
- 3 revisions remain unspent; FINAL never generated or loaded, ledger empty; generator_hash not recorded.

### STEP B5-R2 (v2): Sniper-cost / island-gap / Mender sweep -- **no arm qualifies (best 4/5 but fails (ii)); NOTHING APPLIED, no revision spent, gate script NOT re-run**
- Gate changelog (e) + SPEC 3.4 item 9 committed (7f2c753) before the sweep: criterion (iii) = rejection <= 65 % per family, disclosed in README. Recorded as POST-HOC (made after seeing R1b; R1b would already meet 4/5 under it). Cap lever dropped (never binds); G6 stays a diagnostic.
- Sweep (`scripts_b5r2_sweep.py`, `data/v2/b5r2_sweep/`; new hook `GEN_PARAMS.island_gap`, default 4 = bit-identical; interpreted as island-center Chebyshev spacing): R2a 4/5, R2d 4/5, R2b 3/5, R2c 1/5. Best arm R2a fails (ii): labyrinth best single = Sniper 0.669 (> 0.65); bunker_nest 0.288. R2c: archipelago rejection 89.8 %, Oracle 1.00. Sniper cost 28 vs 30 gives identical best-single numbers.
- Full suite: 343 tests, all pass, exit 0 (pytest.ini addopts -q plus -q suppresses the summary line). FINAL never generated/loaded; ledger empty; generator_hash not recorded. 3 revisions unspent.

### STEP B5-R2b (v2): R2a applied as design revision 1 of 3 -- **gates G1 G5 G9 PASS; G2 G3 G4 G8 FAIL (G6 diagnostic); nothing fixed; generator_hash NOT recorded**
- Gate changelog (f), commit 85bb792, written BEFORE applying: one-time waiver of item 9 criterion (ii) for R2a (labyrinth best single 0.669 vs 0.65 on n = 20, sampling error > margin); a deliberate post-hoc exception by the rule's owner; no threshold changed.
- Revision 1 = R2a (commit 21f71ff): `data/v2/unit_stats.json` (building/core/defender hp x1.4 -> Gun 126 / Lancer 98 / Flak 140, building 84, core 840 = x1.4 x3, Sniper cost 28, Sniper 40 % vs building/wall/core, Mender unchanged heal 8 / hp 40, cap 20); generator defaults `lancer_share` 0.4, `sealed_fraction` 0.5, `island_gap` 4 (hook kept, NOT applied). Versions: `SIM_CODE_VERSION` = `breachlens-sim-b5.1-rev1`, `stats_version` = `v2-rev1`, `layouts.GENERATOR_VERSION` = `v2-rev1`. `SCRIPTED_COMPOSITION` Scouts 4 -> 3 (Sniper 28 made the 4-Scout army cost 248 > 240).
- Design sets regenerated from the same streams T / L: rejection spine_pods 7.0 % (6/86), archipelago 60.4 % (122/202, flagged > 50 %), bunker_nest 16.7 % (16/96), labyrinth 31.0 % (18/58); every rejection single_type_trivial except 3 too_hard in bunker_nest. Sealed cores in TUNE-train: 27/60, 40/60, 30/60; LAB-DESIGN 24/40. Contact sheets (3 sealed examples each, tagged) in `results/design/` (git-ignored). Streams F / FINAL never generated or loaded; ledger empty.
- Gate run ONCE from clean commit 21f71ff (`scripts_gates.py`, 409 s; now needs `--record-hash` to record generator_hash). Report `data/v2/b5_report.md` (adds monotone ban-loss for every unit x 4 families, Oracle gold shares). **G1 PASS** (max best-single share 0.54, Sniper) | **G2 FAIL** (Blaster passes in no family; Scout only spine_pods; Mender fails bunker_nest 0.017) | **G3 FAIL** (Scout 7.9 % < 8 %, Blaster 1.1 %; Sniper 33.6 %, Bulwark 9.3 %, Mender 15.0 % pass) | **G4 FAIL** (Mender gain bunker_nest 0.017 < 0.02, LAB-DESIGN 0.000; Mender-only 0.000 ok) | **G5 PASS** (best single 0.456 / 0.503 / 0.277, Oracle 0.885 / 0.928 / 0.835, breach 1.00 / 1.00 / 0.96) | G6 diagnostic (Oracle > 12 units 0 %) | **G8 FAIL** (only bunker_nest army-search better on 56 % < 60 %; all gains >= +0.20) | **G9 PASS** (Oracle 0.956, Greedy 0.603).
- Tests updated for revision 1 (stat pins, sealed-core waivers, Gun-hp-pinned turret scenario, float tolerance for sim-vs-integer-key scores, gates reproducibility now all gates at revision 1); `data/v2/b4_report.json` + `speed.json` regenerated (D4: 0 mismatches). Full plain `pytest`: see final report. 2 revisions of 3 remain.

### STEP B5-R3 (v2): Blaster / Scout sweep -- **no arm meets the three required items; NOTHING APPLIED, no revision spent (2 of 3 remain), gate script NOT re-run**
- Gate changelog committed BEFORE the sweep (7a289d4): (g) G4 part (a) = Mender monotone gain >= 0.02 with 95 % lower bound > 0 in at least one familiar TUNE family or LAB-DESIGN (was bunker_nest or LAB-DESIGN; post-hoc, made after seeing the revision-1 result); (h) time-box: after the next applied revision, any of G2/G3/G4/G8 still failing is a documented limitation and we go to B6 + the G7 pilot unless G7 fails; no more waivers; the last revision is held for after G7. SPEC 3.4 item 10 = the R3 selection rule. `gates.g4`, `scripts_gates.py` and the reproducibility test updated (stored revision-1 `gates.json` G4 is under the old definition; the test checks it on the old sets).
- Read-only recomputation of G4 on the stored revision-1 raw data under (g) (no new simulation, not a gate run): spine_pods +0.029 [0.013], archipelago +0.031 [0.016] pass; bunker_nest +0.017, LAB-DESIGN 0.000 fail; Mender-only ok -> G4 would PASS at revision 1. `gates.json` is not regenerated.
- Sweep (`scripts_b5r3_sweep.py`, `data/v2/b5r3_sweep/`; base = revision 1; same sample/seeds): Oracle score is above 0.97 in labyrinth in every arm (0.976 / 0.993 / 0.987 / 0.989) and in archipelago for R3b / R3d (0.975 / 0.987), so required item R2 fails everywhere; best-single is out of [0.25, 0.65] only in labyrinth, in every arm (Blaster 0.688-0.705), so required item R3 also fails everywhere (the pooled Sniper share <= 0.60 part holds: 0.16-0.35). Required R1 (Blaster ban-loss) passes in every arm (e.g. bunker_nest 0.048-0.078, lo95 > 0). Blaster niche 31-55 % (P1 yes), Scout niche 1-2.5 % (P2 no). Pooled Sniper best-single share 0.16-0.35. Blaster becomes the best single in most families (Sniper dominance gone, Blaster takes over, esp. R3c).
- Full plain `pytest` (no -q): `343 passed in 363.95s (0:06:03)`, exit 0. FINAL never generated/loaded; ledger empty; generator_hash not recorded.

### STEP B5-CLOSE (v2): **B5 closed at revision 1** (2 of 3 revisions remain: 1 held for after G7, 1 spare); generator_hash recorded; FINAL not generated or loaded, ledger empty
- Decision recorded in SPEC_V2 3.6; known limitations with numbers in `docs/LIMITATIONS.md` (new, running list for the README). Commit f1bf4ad (decision, LIMITATIONS, `--record-hash` now records the failing-gate list and revision even when gates fail; test `test_generator_hash_record_matches_gates` replaces the "only if all pass" test).
- Final gate run ONCE from clean commit f1bf4ad (`python scripts_gates.py --record-hash`, 481 s, dirty=false), G4 as aligned (changelog (g)): **G1 PASS | G2 FAIL | G3 FAIL | G4 PASS | G5 PASS | G6 diagnostic (Oracle > 12 units 0 %) | G8 FAIL | G9 PASS**. Documented-failing gates: G2, G3, G8. Numbers identical to the revision-1 run except G4 (definition).
- `data/v2/generator_hash.json`: generator_hash 49b8a18c53228c6ec9271f0082bbf6d6ec0d6fdbc3c5755721bfa0f36ec42095, commit f1bf4ad, revision 1, documented_failing_gates [G2, G3, G8].
- Full plain `pytest`: `343 passed in 395.37s (0:06:35)`, exit 0.

### STEP B6 (v2): 300-layout pool built from stream F and frozen; FINAL never scored, ledger has build entries only
- `breachlens/pool.py` (loaders, ledger, walk), `scripts_b6_build_pool.py`, `tests/test_b6_pool.py` (22 tests). `layouts.py` untouched, so `generator_hash` 49b8a18c... and stats version 3048422b... equal data/v2/generator_hash.json / design_sets.json (`check_frozen()` runs at build and in a test).
- Pool: DEV-train 50/50/50 + DEV-validation 10/10/10 + FINAL-familiar 20/20/20 (spine_pods / archipelago / bunker_nest) + FINAL-transfer labyrinth 60 = 300. Stream F of each family is split: DEV = first 500k seeds, FINAL = last 500k (labyrinth FINAL only); within a walk the first 50 accepted = train, next 10 = validation.
- Acceptance as in the gate runs (structural rules + too_hard + single_type_trivial). Workers return rule ids only; FINAL stores accept bit + rule ids (`walk.*.candidate_bits`). D4 duplicate check against TUNE-train, TUNE-val, LAB-DESIGN and the pool: 0 redraws in every walk, 0 overlaps.
- Candidates (accepted/candidates): DEV 60/67, 60/223, 60/77; FINAL 20/23, 20/79, 20/23, labyrinth 60/94. Rejection counts per rule are in docs/LIMITATIONS.md.
- Ledger `data/v2/final_access.log`: 4 entries, all purpose=build (one walk per FINAL family / labyrinth), zero eval. Loader role guards: train / validation / tuning raise PermissionError on FINAL / transfer ids; eval needs unlock + purpose eval.
- Freeze: `data/v2/pool_v1.json` read-only (0444), sha256 1346536dea209eb7953eb00eb0d69f24bed930a2a81d89daea7ec1b705f834e4 in `pool_v1.sha256`. Old ledger-empty assertions in test_b4_layouts / test_b5_gates relaxed to "no eval entries".
- No contact sheets or plots made. DEV stats (DEV only): sealed-core share 0.42 / 0.65 / 0.50, Gun 0.59 / 0.54 / 0.59, Lancer 0.38 / 0.34 / 0.39, Flak 0.04 / 0.13 / 0.01.
- Full plain `pytest`: `365 passed in 566.00s (0:09:26)`, exit 0.

### STEP B6.1 (v2): integrity fix, git tag pool-v1 (commit f461116)
- FINAL acceptance re-check moved out of the suite to the manual `scripts_b6_verify_final.py` (refuses if the ledger has an `eval` entry; ONE `build` entry; prints counts only). Not run. `BREACHLENS_FINAL_LEDGER` removed: the ledger path is the module constant `pool.LEDGER_PATH`; tests redirect it only by monkeypatching it for refusal / synthetic-record tests. AST + runtime tests: no non-test code loads FINAL layout content without a ledger write first. Ledger tests: tags subset of {build}, eval 0, build 4. Literal generator-hash assertion (49b8a18c...). Pool sha + stream-F split in Gate changelog (i); LIMITATIONS: the sha test, not file permissions, is the freeze guarantee. Suite: 370 passed in 486.64s.

### STEP B7.1 (v2): integrity fixes; ledger now 5 build, 0 eval
- LAB-DESIGN guard: roles train / validation / tuning load DEV-train / DEV-validation (pool) and TUNE-train / TUNE-val only (tuning = both); LAB-DESIGN raises PermissionError for them (by split, and by stream L so a relabelled record is refused too). New role `design` (`pool.ROLES`) loads TUNE-train, TUNE-val and LAB-DESIGN and nothing from the pool (DEV, FINAL, transfer all raise). `BreachEnv` roles unchanged (train / validation / tuning / eval). The gate and acceptance code reads design sets through `breachlens.layouts`, not `pool.py`, so no pool role applies to it (a test asserts they never import pool). Tests added in `tests/test_b6_pool.py`; env test updated (LAB-DESIGN now raises for tuning).
- Score refactor vs frozen pool: `sim_version` unchanged (3048422b...a338, `SIM_CODE_VERSION` not bumped although sim rounding changed by at most 1 ulp). Only acceptance threshold on a score: `single_type_trivial` best single > 0.99. In-suite DEV re-acceptance passes (180/180). `scripts_b6_verify_final.py` run once: 120/120 accepted (20/20/20/60), no rejections, ONE build ledger entry (commit 0ca8d71, dirty), 0 eval. Gate changelog (j) added.
- Full plain `pytest`: `403 passed in 490.53s (0:08:10)`, exit 0.

### STEP B7 (v2): BreachEnv, canonical score, bounded caches; no FINAL content touched, ledger unchanged (4 build, 0 eval)
- `breachlens/score.py`: the ONE score formula, an exact integer ratio `(6 d_num + 4 d_den C) / (10 d_den)` (d_den 0 read as 1), converted to float once (int / int). `sim.simulate` (`BattleResult.score`), `oracle.score_key` / `SearchResult.score`, the env reward and `info["score"]` all call it; a test greps non-test code (excluding the prototype `siege/`, removed in B20) for a second formula. The old simulator expression rounded twice (`6d/(10 den) + 0.4`), so a stored score can differ from a recomputed one by 1 ulp; see Deviations. There is no v2 evaluator or CSV writer yet (B15 / B8 must import `breachlens.score`).
- `breachlens/env.py` `BreachEnv`: Gymnasium API, Discrete(267), `action_masks()`, obs Dict(grid float32 (9,14,14) in [0,1], state float32 (11,)) as spec 4.2 (channel definitions in `_Static`); illegal action raises `IllegalActionError`; `action_space.sample()` draws from the current mask; passes `gymnasium.utils.env_checker.check_env` (default BUY-start game, pool and synthetic layouts). Terminal-only reward (the canonical score on the step that ends deployment); `shaping=False` default, `shaping=True` = spec 7.4 potential (Phi = 0.5 x preview score, 120 ticks, 0 in BUY and before the first deployment), shaped return sums to the true score (1e-12), `info["score"]` always true. `prebought="random"` (spec 6 phase A, via `armies.random_types`) or a count list. Max steps 41 (20 buys + END_BUY + 20 deploys), reached in a test, never exceeded in 1,000 fuzz episodes.
- Layout access only through `pool` loaders: `BreachEnv(records= | source= | layouts=, role=...)`; roles train / validation / tuning raise PermissionError on any FINAL / transfer id or split; role eval needs `unlock=True` (each FINAL load = one `eval` ledger entry, none made). `pool.design_records` + `LAYOUT_ROLE_SPLITS` let train / validation / tuning load TUNE-train / TUNE-val / LAB-DESIGN through `load_layout`. `layouts=` (layout dicts) is the synthetic path for tests and symmetry checks. D4 helpers: `d4_cell_perm`, `d4_border_perm`, `d4_action_perm`, `d4_transform_grid`, `d4_transform_mask` (conventions of `layouts.transform`).
- Bounded caches (`sim.LRU`): the five `Prepared` caches are LRUs: movement-field cache 2048 entries (about 2.1 KB each, <= 4.4 MB per layout), the four small caches 4096 entries; observed after 200 random battles on one layout: at most 1,395 / 376 / 720 / 145 / 50 entries (fields / region / cand / blocked / splash). The env keeps an LRU of `Prepared` objects (`layout_cache`, default 64). Size 0 disables a cache; results are identical for default, unbounded, tiny (2 / 3) and off.
- Tests `tests/test_b7_env.py` (27): score on constructed outcomes and monotonicity; no second formula; env reward == Oracle score == simulator score with `==` on 200 layouts (100 DEV-train + 100 TUNE-train) for a random masked-agent army and an Oracle army (6 + 6 search steps) replayed through the env; check_env; spaces; max steps; 1,000 masked-random episodes (mask equals an independent restatement of spec 4.1 at every step, 3 illegal actions per step all raise and change nothing, termination within 41 steps); budget / cap / END_BUY / WAIT semantics; `prebought="random"` legal in 500 resets with all five types seen; bit-identical replay of obs / reward / info and reset(seed) honoured; D4 on 100 layouts x 8 symmetries (obs grid transformed, mask permuted, state and every reward identical, scores equal) with Menders; every channel on a hand-built layout incl. repair reach and the terminal board; shaping on / off; LRU and cache on / off / tiny identity; role matrix; the env itself never calls `generate` / `candidate`.
- Preliminary env-only throughput (`scripts_b7_throughput.py`, `data/v2/env_throughput.json`; masked-random agent, one process, arm64, 30 s per config; full estimate is B10): DEV-train full game 6,313 steps/s (10.7 steps / episode, 592 episodes/s); prebought=random 3,476 steps/s; buy-until-broke (about 21 steps / episode) 8,679 steps/s; without a `Prepared` cache (rebuilt every episode) 4,686 steps/s; TUNE-train 6,113 steps/s; max RSS 48 MB. The B2 gate asks for >= 300 steps/s. Uniformly random choice among legal actions picks END_BUY early, so these armies are small; no trained-policy number exists yet.
- Deviations / decisions: (1) WAIT is legal in DEPLOY at any time (as in the prototype), including with nothing deployed (zero-score battle) and forfeits undeployed units. (2) Illegal actions raise (spec) and `check_env` passes because `sample()` is mask-aware; with `prebought="random"` the checker's pre-reset sample can be illegal after the reseeded reset, so `check_env` is claimed for the default game only. (3) Channels 1-6 show the CURRENT board (after the battle: final hp, destroyed = 0); channel 0 and 8 keep the deployment positions; state tick = battle ticks / 240 after the battle. (4) Score values may differ from stored B5 numbers by 1 ulp (single conversion), which `assert ==` tests would have rejected.
- Full plain `pytest`: `397 passed in 486.29s (0:08:06)`, exit 0.
- Not done (by design): no baselines (B8), no D4 wrapper / TTA (B9), no PPO (B10).

### STEP B8 (v2): baselines + evaluator (done, branch v2-redesign); FINAL never loaded, ledger 5 build / 0 eval
- Pre-registered first (commit 3713e5b): Gate changelog (k) = BestSingle chosen per family on the matching training split (DEV-validation <- DEV-train, TUNE-val <- TUNE-train, training sets in-sample; B15 FINAL-familiar <- DEV-train per family, FINAL-transfer <- ONE type pooled over DEV-train familiar), diagnostics BestSingle-hindsight and BestSingle-perlayout, Greedy = B5 definition (Mender pairing dropped), Random = uniform over legal BreachEnv actions with 5 layout-id-derived seeds, B8 scope. Changelog (l) records the boundary check.
- **Deviation from the SPEC section 11 B8 row (by owner decision):** B8 covers Random, Greedy, BestSingle (+2 diagnostics) and Oracle only. Army search stays in `oracle.py`; RAS-N and RPS-N move to the planner step (B12).
- `breachlens/evaluator.py` (policy x set x role -> per-layout rows + per-group summary; breach rate headline, mean score and D, bootstrap over layouts 10,000 resamples seed 8080; every row carries score_version, sim_version, generator_hash, pool_sha256, git_commit, seed, role, set; missing field -> ValueError; role eval needs unlock=True and writes one `eval` ledger entry per evaluation, tested only with a monkeypatched `LEDGER_PATH` and synthetic layouts), `breachlens/baselines.py`, `SCORE_VERSION = "b7-single-rounding"` in `score.py`, `scripts_b8_baselines.py`, `scripts_b8_checks.py`, `tests/test_b8_baselines.py` (21). Outputs in `data/v2/baselines/` (layout rows, summary csv/json, best-single choice, Oracle cache, timing, checks). Run on commit 1d0b924 with 7 processes; row `git_dirty` is true only because untracked `logs_b8/` existed.
- Findings: Greedy (B5 definition) buys 8 Bulwark, i.e. it IS the single-Bulwark army; BestSingle = Sniper everywhere except TUNE archipelago (Bulwark); the train-chosen type equals the hindsight type on all 4 sets. Oracle >= every other baseline on all 420 layouts. B5 consistency exact for Oracle; Greedy within 1.1e-16 (22 layouts). Boundary check: 0 / 367 DEV bit mismatches, 0 candidates at exactly 99/100.
- Wall time per baseline (4 sets, 7 procs): Oracle 49.7 s; every other baseline <= 1.4 s.
- Full plain `pytest`: `425 passed in 492.08s (0:08:12)`, exit 0 (the first run failed the B6 AST audit list until `evaluator.py` and `scripts_b8_checks.py` were added to it; the new B8 test pins that the checks script only calls `walk` with "DEV").

## Decisions
- Env raises on illegal actions (catches bugs early) -> Gymnasium `check_env` can't be used (samples
  unmasked); we test reset/seed determinism only.
- Border tiles: 60 cells, clockwise from top-left (`siege.env.BORDER`).
- Seeding: `reset(seed=...)` seeds `self.np_random`, which also drives layout sampling/generation when
  no frozen pool is given, so layouts are reproducible per seed; smoke/tests separately use
  `np.random.default_rng(seed)` for action sampling so layout and action-choice randomness don't fight
  over the same stream.
- Battle resolves in a single `simulate_battle` call (not ticked through separate `env.step()` calls);
  `env.tick` jumps straight to `MAX_BATTLE_TICKS` and `terminated=True` the same step the army is fully
  deployed (or WAIT is called during deploy with zero army). `simulate_battle` itself still runs its own
  internal tick loop (`max_ticks=MAX_BATTLE_TICKS`) for the frame-by-frame record used by `render.py`.
- Python 3.11 venv from /opt/homebrew/bin/python3.11 (system python is 3.14).
- `ring_defence` is the held-out "transfer" family (see STEP 3 above) -- chosen because it's the
  simplest, most "classic" single-core style, making it a meaningful generalization probe for a policy
  trained only on the two multi-cluster/multi-outpost families.
- Test armies for balance/acceptance are clustered at one shared border push point
  (`siege.armies.place_on_border(..., start_idx)`), not spread evenly across all 60 tiles -- spreading
  means each unit fights alone through its own gap, so no mixed-army synergy can ever show up regardless
  of stats. See docs/troops.md for the full reasoning.

### STEP B9 (D4 augmentation, D4Wrapper, TTA) -- done
- Item 0 (commit c66025a, before any B9 code): Gate changelog (m) pins the G7 bar = mean over the 60 TUNE-val layouts of max(BestSingle, Greedy) per layout + 0.05 = **0.4369** (unrounded 0.436862743301702; `data/v2/g7_bar.json`; test recomputes it from `b8_layout_rows.csv`). Per-family max (0.4308) and pooled max (0.3808) recorded as not used. Ledger eval entries now carry `session_id` (`pool.eval_session`, `pool.count_eval_sessions`; the evaluator wraps one evaluation in one session); `git_dirty()` ignores gitignored untracked files; `logs_*/` in .gitignore; two LIMITATIONS notes (Greedy = 8 Bulwarks; Oracle = fixed-budget search). Ledger still 5 build / 0 eval.
- `rl/augment.py` rewritten for BreachEnv (14x14, 52-tile border, 267 actions, 9 channels incl. repair reach, 11-d state invariant): own D4 tables (s = 4*flip + k, `layouts._tr` convention), `compose`, `inverse_symmetry`, `ACTION_PERM` / `TILE_PERM`, `transform_grid/obs/mask/action/army`, `untransform_probs`, `D4Wrapper` (random per reset, seeded, or pinned; identity = canonical view). New `rl/tta.py`: `tta_scores` / `tta_action` (8 views, map back through the inverse action permutation, average, masked argmax; model adapters come with `rl/policy_io.py` later). The prototype version is kept as `rl/augment_v1.py` (+ `tests/test_augment_v1.py`) for the prototype trainer `rl/train_ppo.py` and its tests until B10 / B20; `tests/test_augment.py` is the v2 test.
- `tests/test_augment.py` (12 tests): tables equal the independent `_d4check` helpers and `env.d4_*`; inverse identity; group closure / associativity / table composition; action bijection per symmetry; per-channel grid transform incl. repair; mask legality on real episodes; env + simulator equivariance on 100 layouts (50 DEV-train + 50 TUNE-train) x 8 symmetries through `D4Wrapper` (>= 50 % of armies hold a Mender), bare simulator with `transform_army`; wrapper identity / seeding; TTA with identity only equals the plain policy, TTA of a symmetrised policy is a fixed point and TTA is itself equivariant, Q values map back through the inverse permutation.
- No training, no FINAL access (ledger unchanged), no pool or gate change.

## Known issues / placeholders
- Splash is still the weakest archetype even after the v2 buff (docs/troops.md) -- hypothesis is its
  speed (0.6, slowest in the game) means more ticks exposed to turret fire before reaching a wall than
  its HP buff compensates for. Worth another look once `app/` can actually show a battle.
- Grid channels 4-7 are still zero/reserved (e.g. per-troop-type channels, turret range, distance-to-target).
- `app/` is still an empty package (no UI yet, and it's now last -- see NEXT).
- Movement in `combat.py` is exactly one grid cell per tick regardless of a troop's `speed` stat — speed
  is stored and passed through but not yet used to scale step count or action frequency.
- `rl/train.py` uses a single env (no vectorization/parallelism) and default MaskablePPO hyperparameters
  -- fine for the step-2 smoke test, will want parallel envs for the real step-6/7 runs.
- `models/maskable_ppo_siege.zip` from the throwaway smoke run exists locally but is not committed;
  treat it as stale, to be superseded once step 6 re-runs training against the frozen DEV pool.
- `SiegeEnv.action_masks()`/action layout are the same 246-action interface from step 1; step 4 is about
  formalizing/baselining this against the new frozen pools, not changing the interface itself unless
  the baselines surface a real problem with it.

## NEXT
- v2: B9 DONE (D4 augmentation, D4Wrapper, TTA; next is B10, only when told). B8 DONE (baselines + evaluator). B7.1 DONE (LAB-DESIGN guard, score re-verification); B7 DONE (BreachEnv + canonical score + bounded caches; B6.1 tag pool-v1); next is B8, only when told. Before that: B6 DONE (pool frozen, sha 1346536d...). Before that: B5 CLOSED at revision 1 (G2/G3/G8 documented limitations, hash recorded); next is B6 (build the 300-layout pool from stream F), only when told. History: B1-B5 in progress: revision 1 (R2a) applied, G2/G3/G8 FAIL (G4 passes under changelog (g)), B5-R3 sweep found no qualifying arm (nothing applied, 2 revisions remain), awaiting owner (B5 gates FAILED at revision 0; B5-R1 and B5-R1b sweeps found no acceptable arm, nothing applied, awaiting owner; generator_hash not recorded); B6 starts only when told AND the gates pass.
- (STEP 8 evaluation done above.) Next: DDQN, TRPO, then app, explainability, README, only when told.
- STEP 4: run the full Gym env interface (246 actions: 4 buy + 4 troops x 60 border deploy tiles +
  END_BUY + WAIT) and masks against the frozen DEV pool, plus simple scripted baselines (rangers-only,
  tanks-only, a mixed heuristic -- reuse `siege/armies.py`'s fixed compositions) evaluated on DEV for
  sanity before any learning is involved.
- (STEP 6/6b/7 done above; next is launching the grid yourself, then DDQN/TRPO/evaluation/app/explainability) STEP 6 was: re-run the PPO pipeline properly (`rl/train.py`/`rl/eval.py` already exist; adapt to sample
  only `siege.layouts.load_layout_pool(pool="DEV", tag="familiar")`) now that balance is fixed.
- STEP 7: launch the real PPO grid (hyperparameter sweep).
- STEP 8: `app/` -- streamlit UI to visualize a trained agent's buy/deploy choices and battles, now last.
