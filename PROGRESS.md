# PROGRESS

Project root: `~/base-siege` (NOT `~/RL-Project`, which is an empty 0-byte file that was left untouched).

## Setup / commands
    cd ~/base-siege
    python3.11 -m venv .venv                      # already created
    .venv/bin/pip install -r requirements.txt     # full set; only gymnasium, numpy, pytest installed so far
    .venv/bin/pytest                              # run all tests
    .venv/bin/python scripts_smoke.py             # 100 random masked steps

## DONE
- STEP 1: scaffold (siege/ rl/ app/ results/ models/ tests/ docs/), requirements.txt, pytest.ini,
  .gitignore, README, git init (no commits yet).
- `siege/env.py`: minimal `SiegeEnv`. Discrete(246) layout = 4 buy | 4x60 deploy | END_BUY(244) | WAIT(245).
  Obs Dict: grid (8,16,16) [only channel 0 = deployed occupancy, rest zeros], state (9) [gold, 4 owned counts, 3-way phase one-hot, tick].
  `action_masks()` implemented for buy/deploy/battle phases. Illegal action raises ValueError.
- Tests: 8 passing (`tests/test_env.py`): constants, shapes, 100-step masked rollout, masks, determinism, reset-seed contract.

## Decisions
- Env raises on illegal actions (catches bugs early) -> Gymnasium `check_env` can't be used (samples unmasked);
  we test reset/seed determinism only.
- Border tiles: 60 cells, clockwise from top-left (`siege.env.BORDER`).
- Seeding: `reset(seed=...)` seeds `self.np_random`; tests/smoke use `np.random.default_rng(seed)`.
- Python 3.11 venv from /opt/homebrew/bin/python3.11 (system python is 3.14).

## Known issues / placeholders
- Combat is a stub (`_run_battle_stub`: battle ends instantly, reward 0). Troop costs are placeholders [10,20,30,40].
- No bases/layouts yet, so all observation channels except occupancy are zeros; observation docs not yet written.
- torch / sb3 / sb3-contrib / streamlit / pandas / matplotlib not yet installed in the venv.
- No git commit yet.

## NEXT
STEP 2: `siege/troops.py` (4 troop types: cost, hp, dmg, range, speed) and `siege/bases.py`
(building types, base representation) + tests; then deterministic `combat.py`.
