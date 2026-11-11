# PROGRESS

Project root: `~/base-siege` (NOT `~/RL-Project`, which is an empty 0-byte file that was left untouched).

## Setup / commands
    cd ~/base-siege
    python3.11 -m venv .venv                      # already created
    .venv/bin/pip install -r requirements.txt     # full set; only gymnasium, numpy, pytest, matplotlib installed so far
    .venv/bin/pytest                              # run all tests
    .venv/bin/python scripts_smoke.py             # 100 random masked steps

## DONE
- STEP 1: scaffold (siege/ rl/ app/ results/ models/ tests/ docs/), requirements.txt, pytest.ini,
  .gitignore, README, git init.
- `siege/env.py`: `SiegeEnv`. Discrete(246) layout = 4 buy | 4x60 deploy | END_BUY(244) | WAIT(245).
  Obs Dict: grid (8,16,16) [ch0 deployed-troop occupancy, ch1 building hp, ch2 turret hp, ch3 wall hp,
  ch4-7 still zero/reserved], state (9) [gold, 4 owned counts, 3-way phase one-hot, tick].
  `action_masks()` implemented for buy/deploy/battle phases. Illegal action raises ValueError.
- STEP 2: `siege/troops.py` (4 troop types: swarm/ranger/tank/splash — cost, hp, damage, speed, range,
  target_preference), `siege/bases.py` (`Building`/`Turret` dataclasses + `generate_base_layout(rng)`:
  random walled core with 3-5 buildings and 1-2 turrets), `siege/combat.py`
  (`simulate_battle`: deterministic tick-based combat with documented, fixed tie-breaking rules).
- Wired the real simulator into `SiegeEnv`: `reset()` generates a base layout from `self.np_random`
  (reproducible per seed); deploying the last owned troop or calling WAIT in the deploy phase runs
  `simulate_battle` once and uses `BattleResult.reward` (fraction of building value destroyed) as the
  step reward; `env.battle_result`/`env.destruction` hold the outcome; grid channels 1-3 reflect
  live/post-battle building, turret and wall state.
- `siege/render.py`: matplotlib board renderer + `save_battle_frames` (used to produce the
  `results/example_battle/` demo frames).
- Tests: 16 passing (`tests/test_env.py` x9, `tests/test_simulator.py` x7): constants, shapes, masks,
  determinism, reset-seed contract, base-layout generation, full buy->deploy->battle episode with real
  destruction, troop-cannot-walk-through-walls, and a regression test for the target-preference bug below.
- Committed: "Step 2: simulator" (df2f730); simulator-to-env wiring + bugfix not yet committed (see NEXT).

## Bugs fixed
- `combat._select_target` ranked targets by a hardcoded building(0) > turret(1) > wall(2) priority
  regardless of a troop's own `target_preference` tuple, so tanks/splash units (meant to prefer
  walls) actually chased buildings/turrets first whenever any were in range. Fixed by ranking by the
  troop's own preference order instead. Caught by running a 5-tank rush through `SiegeEnv.step()` end
  to end and seeing 0 destruction after 50 ticks even though troops reached the wall.

## Decisions
- Env raises on illegal actions (catches bugs early) -> Gymnasium `check_env` can't be used (samples
  unmasked); we test reset/seed determinism only.
- Border tiles: 60 cells, clockwise from top-left (`siege.env.BORDER`).
- Seeding: `reset(seed=...)` seeds `self.np_random`, which also drives `generate_base_layout` so the
  base is reproducible per seed; smoke/tests separately use `np.random.default_rng(seed)` for action
  sampling so layout and action-choice randomness don't fight over the same stream.
- Battle resolves in a single `simulate_battle` call (not ticked through separate `env.step()` calls);
  `env.tick` jumps straight to `MAX_BATTLE_TICKS` and `terminated=True` the same step the army is fully
  deployed (or WAIT is called during deploy with zero army). `simulate_battle` itself still runs its own
  internal tick loop (`max_ticks=MAX_BATTLE_TICKS`) for the frame-by-frame record used by `render.py`.
- Core base layout is a random 6x6 interior (rows/cols 5-10) ringed by walls one tile out (4 and 11);
  3-5 buildings and 1-2 turrets placed on distinct core cells per `generate_base_layout`.
- Python 3.11 venv from /opt/homebrew/bin/python3.11 (system python is 3.14).

## Known issues / placeholders
- Troop costs/stats in `siege/troops.py` are a first pass, not yet balance-tested by actual training.
- Grid channels 4-7 are still zero/reserved (e.g. per-troop-type channels, turret range, distance-to-target).
- torch / sb3 / sb3-contrib / streamlit / pandas not yet installed in the venv (matplotlib is).
- `app/` and `rl/` are still empty packages (no training script or UI yet).
- Movement in `combat.py` is exactly one grid cell per tick regardless of a troop's `speed` stat — speed
  is stored and passed through but not yet used to scale step count or action frequency.

## NEXT
STEP 3: `rl/` training — pick an algorithm (sb3-contrib MaskablePPO fits the action-mask design),
install torch/sb3-contrib, write a training script against `SiegeEnv`, and a short eval script that
reports average destruction over N random/held-out base layouts.
