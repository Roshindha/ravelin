#!/usr/bin/env bash
# PPO experiment grid: {--aug none, --aug d4} x seeds 10..19, 1M timesteps, config A (fixed Greedy phase-A army,
# lr 3e-4, 10 epochs), curriculum on. Order is interleaved: s10 none, s10 d4, s11 none, s11 d4, ...
# Resumable: finished runs (final.zip) are skipped; interrupted runs continue from their latest checkpoint
# (trainer's --resume). Logs: results/logs/<run>.log (appended). Models: models/<run>/.
#
#   scripts/run_ppo_grid.sh              # run the grid, 6 jobs in parallel
#   scripts/run_ppo_grid.sh --dry-run    # print commands only
# Overrides (env vars): SEEDS="10 11 ..." TIMESTEPS=1000000 PARALLEL=6 PREFIX=ppo EXTRA="--val-freq 25000"
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-10 11 12 13 14 15 16 17 18 19}"
TIMESTEPS="${TIMESTEPS:-1000000}"
PARALLEL="${PARALLEL:-6}"
PREFIX="${PREFIX:-ppo}"
EXTRA="${EXTRA:-}"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

mkdir -p results/logs models
CMDS="$(mktemp)"
trap 'rm -f "$CMDS"' EXIT
n_todo=0; n_done=0
for seed in $SEEDS; do
  for aug in none d4; do
    name="${PREFIX}_${aug}_s${seed}"
    if [ -f "models/${name}/final.zip" ]; then
      n_done=$((n_done+1)); [ $DRY = 1 ] && echo "# skip (finished): ${name}"
      continue
    fi
    echo "${aug} ${seed} ${name}" >> "$CMDS"
    n_todo=$((n_todo+1))
  done
done

echo "grid: ${n_todo} to run, ${n_done} already finished, ${PARALLEL} parallel, ${TIMESTEPS} timesteps"
if [ $DRY = 1 ]; then
  while read -r aug seed name; do
    echo ".venv/bin/python -m rl.train_ppo --aug ${aug} --seed ${seed} --timesteps ${TIMESTEPS} --curriculum on --run-name ${name} --resume ${EXTRA} >> results/logs/${name}.log 2>&1"
  done < "$CMDS"
  exit 0
fi
[ "$n_todo" -gt 0 ] || { echo "nothing to do"; exit 0; }
# One job per core; each command appends to its own log. A failed job does not stop the others.
export TIMESTEPS EXTRA
xargs -P "$PARALLEL" -L1 scripts/run_one.sh < "$CMDS"
echo "grid finished at $(date)"
