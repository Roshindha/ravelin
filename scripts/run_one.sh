#!/usr/bin/env bash
# One resumable PPO run (called by run_ppo_grid.sh):  run_one.sh <aug> <seed> <run-name>   [env: TIMESTEPS, EXTRA]
set -uo pipefail
cd "$(dirname "$0")/.."
aug="$1"; seed="$2"; name="$3"
.venv/bin/python -m rl.train_ppo --aug "$aug" --seed "$seed" --timesteps "${TIMESTEPS:-1000000}" --curriculum on \
  --run-name "$name" --resume ${EXTRA:-} >> "results/logs/${name}.log" 2>&1 \
  || echo "FAILED: $name (see results/logs/${name}.log)" >&2
