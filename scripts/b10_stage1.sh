#!/bin/zsh
# B10 benchmark stage 1: thread-count probes (plain, phase B). One benchmark at a time; the P=6 probes start 6 processes on purpose.
cd "$(dirname "$0")/.."
export PYTHONPATH=.
B=data/v2/b10_bench
for th in 1 2 4; do
  .venv/bin/python scripts/b10_throughput.py train --aug none --phase B --threads $th --warmup 30 --seconds 60 --out $B/probe_p1_t$th.json > /dev/null
done
for th in 1 2; do
  .venv/bin/python scripts/b10_throughput.py parallel --n 6 --aug none --phase B --threads $th --warmup 30 --seconds 90 --out $B/probe_p6_t$th.json > /dev/null
done
echo STAGE1_DONE
