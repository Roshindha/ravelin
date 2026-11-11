# Base Siege: Reinforcement Learning for Army Selection and Deployment

An RL agent buys troops under a 200-resource budget and deploys them against
unfamiliar base layouts in a custom 16x16 Gymnasium combat environment.
See PROGRESS.md for current status and exact commands.

## Quick start
    python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/pytest
    .venv/bin/python scripts_smoke.py

## Layout families, balance, and the frozen pool

Base layouts come from three generator families (`siege/bases.py`):

- **ring_defence** -- one consolidated core behind a single continuous wall ring.
- **clustered_core** -- two separately-walled clusters, each with its own punched gate.
- **scattered_outposts** -- 3-5 small, independently-defended outposts spread across the board.

### Balance

Troop/turret stats were tuned against the real simulator (`scripts_balance.py`: fixed
single-type and mixed armies, clustered at a shared push point, scored over 200 generated
layouts per family) until no single troop type dominated every family. See `docs/troops.md`
for the full iteration history (the first pass let a rangers-only army clear ~90-97% of every
layout -- a real balance bug, not a result worth citing). Current mean destruction by army and
family:

| family | swarm | ranger | tank | splash | mixed_balanced | mixed_breach |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ring_defence | 0.000 | 0.082 | **0.278** | 0.023 | 0.255 | 0.238 |
| clustered_core | 0.086 | 0.165 | 0.176 | 0.003 | **0.291** | 0.075 |
| scattered_outposts | 0.763 | **0.901** | 0.810 | 0.321 | 0.831 | 0.641 |

Tank wins `ring_defence` (one heavy wall rewards high HP over raw DPS), a mixed army wins
`clustered_core` (splitting focus across two separately-defended clusters rewards combined
arms over any single type), and ranger wins `scattered_outposts` (open sight lines, thin
defenses reward range/DPS). No type wins everywhere.

### Acceptance rules

A generated layout is rejected (and logged, not counted) if any of:
- **empty**: no buildings, or total building value is 0.
- **unreachable**: a building/turret sits off the board or directly on a wall tile (a generator
  bug check -- walls themselves are destructible and ranged troops ignore them entirely when
  targeting, so a sealed wall ring is not "unreachable" in the gameplay sense).
- **too_hard**: every one of the 6 fixed test armies (4 single-type, 2 mixed) scores below 10%
  destruction.
- **ranger_trivial**: a rangers-only army scores >= 99% destruction (too easy, dominated by one
  type).

### Frozen pool (`results/layouts.json`, 241 layouts, sha256 in `results/layouts.sha256`)

Each record is `{id, family, seed, pool, tag}` -- small and fully reproducible, since
`generate_base_layout(np.random.default_rng(seed), family=family)` deterministically regenerates
the exact same layout; no geometry is stored.

One family, **ring_defence, is held out entirely as "transfer"**: all 41 of its accepted layouts
go to FINAL and are never trained or tuned on, only used to check whether a policy generalizes
to a base style (one consolidated core) it has never seen, having only trained on the two
multi-cluster/multi-outpost families. The other two families are "familiar": their accepted
layouts are split 80/20 into DEV (training/tuning) and FINAL (frozen, held out, same
distribution as DEV -- measures in-distribution generalization).

| family | pool | tag | count |
| --- | --- | --- | ---: |
| ring_defence | FINAL | transfer | 41 |
| clustered_core | DEV | familiar | 80 |
| clustered_core | FINAL | familiar | 20 |
| scattered_outposts | DEV | familiar | 80 |
| scattered_outposts | FINAL | familiar | 20 |
| **total** | | | **241** |

`SiegeEnv(layout_pool=...)` accepts a list of records and samples one per `reset()` via the env's
own seeded `np_random`, regenerating it deterministically from its `(family, seed)`. Training
must only ever pass `siege.layouts.load_layout_pool(pool="DEV", tag="familiar")`
(`tests/test_layouts.py::test_dev_familiar_sampler_never_leaks_final_or_transfer_ids` guards this).

A 12-samples-per-family contact sheet is at `results/layout_contact_sheet.png`
(`scripts_contact_sheet.py`).
