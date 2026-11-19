#!/bin/zsh
# B10 benchmark stage 2 (threads = 1, chosen in stage 1). One benchmark at a time; the parallel runs start P processes on purpose.
cd "$(dirname "$0")/.."
export PYTHONPATH=.
B=data/v2/b10_bench
PY=.venv/bin/python
S="scripts/b10_throughput.py"
for cfg in "plain B none" "d4 B d4" "plain A none" "d4 A d4"; do
  set -- ${=cfg}
  $PY $S train --aug $3 --phase $2 --threads 1 --warmup 60 --seconds 180 --out $B/single_$1_$2.json > /dev/null
done
for aug in none d4; do
  n=$([ $aug = none ] && echo plain || echo d4)
  $PY $S env --aug $aug --phase A --agent uniform --seconds 30 --out $B/env_A_$n.json > /dev/null
  $PY $S env --aug $aug --phase B --agent uniform --seconds 30 --out $B/env_B_uniform_$n.json > /dev/null
  $PY $S env --aug $aug --phase B --agent buy_all --seconds 30 --out $B/env_B_buy_all_$n.json > /dev/null
done
$PY $S infer --phase B --threads 1 --seconds 30 --out $B/infer_B.json > /dev/null
$PY $S parallel --n 3 --aug none --phase B --threads 1 --warmup 60 --seconds 180 --out $B/par3_plain_B.json > /dev/null
for cfg in "plain B none" "d4 B d4" "plain A none" "d4 A d4"; do
  set -- ${=cfg}
  $PY $S parallel --n 6 --aug $3 --phase $2 --threads 1 --warmup 60 --seconds 180 --out $B/par6_$1_$2.json > /dev/null
done
echo STAGE2_DONE
