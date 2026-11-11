# PROGRESS

Project root: `~/base-siege` (NOT `~/RL-Project`, which is an empty 0-byte file that was left untouched).

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
- STEP 4: run the full Gym env interface (246 actions: 4 buy + 4 troops x 60 border deploy tiles +
  END_BUY + WAIT) and masks against the frozen DEV pool, plus simple scripted baselines (rangers-only,
  tanks-only, a mixed heuristic -- reuse `siege/armies.py`'s fixed compositions) evaluated on DEV for
  sanity before any learning is involved.
- STEP 5: D4 (dihedral group of order 4 -- 90-degree rotations + mirror) augmentation of layouts/obs.
- STEP 6: re-run the PPO pipeline properly (`rl/train.py`/`rl/eval.py` already exist; adapt to sample
  only `siege.layouts.load_layout_pool(pool="DEV", tag="familiar")`) now that balance is fixed.
- STEP 7: launch the real PPO grid (hyperparameter sweep).
- STEP 8: `app/` -- streamlit UI to visualize a trained agent's buy/deploy choices and battles, now last.
