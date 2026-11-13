# BreachLens (v2) — design spec

Status: **design only** (step 10a). No code has been written for v2. The prototype is frozen at git tag `prototype-v1`; all v2 work lives on branch `v2-redesign`.
Supersedes: [SPEC.md](SPEC.md) (Base Siege). Where this file is silent, SPEC.md's rules still apply (PROGRESS.md after every step, pytest per module, fixed seeds, one step at a time, report in ≤ 5 lines).

Every number marked **(proposed)** is a starting value to be tested by the balance gates in §3.4, not a result. Every timing marked **(estimate)** is extrapolated from prototype measurements in §9 and must be re-measured in build step B10 before any grid is launched.

---

## 1. Concept

A drone fleet breaches an automated citadel. The agent buys a fleet under a budget, then deploys it on the border; the deterministic simulator runs the battle. The agent never steers units.

- **Score** = `0.6 · D + 0.4 · C`
  - `D` = value-weighted destruction = `Σ v_b·[b destroyed] / Σ v_b` over **non-core** buildings (`v_b ∈ {1,2,3}` assigned by the generator; turrets and walls have no value, they are obstacles).
  - `C` = 1 if the core is destroyed, else 0.
- **Success metric = breach rate** (mean `C`). Mean score and `D` are reported alongside.
- Reward is terminal: the score of the finished battle, nothing else (no step penalties, no shaping). The prototype's "full clear" is replaced by breach.

### What the prototype taught us (and what v2 changes)

| Prototype finding (PROGRESS.md) | v2 response |
|---|---|
| Splash was dead; tank was weakest after tuning | Every unit must pass explicit non-dominance *and* non-deadness gates (§3.4), tuned on a separate seed set, 2 iterations max |
| D4 hurt at 1M steps (underfit, deployment worst) | D4 kept as a condition, but also tested as test-time ensembling (§7.1); step budget raised; Mender tie-breaks designed to be equivariant from the start (§3.3) |
| Phase-A used a fixed Greedy army → PPO only ever saw one post-buy state before phase B | Phase A uses **random armies from step 0** (§6) |
| Wall targeting was a global rank; tanks/splash chewed every wall first | Target choice is nearest-within-preference, not global (§3.3) |
| `ring_defence` as transfer made "best single" meaningless (0.000) | BestSingle baseline is chosen per *family-agnostic DEV pool*, and transfer is reported against Oracle/Greedy/Random plus a hindsight reference only |
| Residual 4/7000 D4 mismatches came from id-ordered ties | **Ids never decide anything.** Exact ties are resolved by a symmetric action (split or abstain) — §3.3 |
| Battle had 50 ticks | 240 ticks, early exit when nothing can change |

---

## 2. World parameters (fixed)

| Parameter | Value |
|---|---|
| Grid | 14 × 14 |
| Budget | 240 gold |
| Battle horizon | 240 ticks (cap; battle exits early when no unit can act or the base is cleared) |
| Max units | 20 |
| Border tiles | 52 (`4·14 − 4`), indexed clockwise from top-left, as in the prototype |
| Unit types | 5 |
| Phases | BUY → DEPLOY → BATTLE (battle resolved inside one `simulate_battle` call, frames recorded for replay) |

---

## 3. Units, turrets, simulator rules

### 3.1 Units (proposed stats)

| Unit | Cost | HP | Damage/tick | Range | Speed | Armor | Target pref. | Special |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **Scout** | 8 | 16 | 6 | 1.2 | 1.5 | 0 | building > turret > wall | Cheap swarm body |
| **Sniper** | 24 | 28 | 12 | 5.0 | 1.0 | 0 | turret > building > wall | Outranges range-4 turrets |
| **Bulwark** | 30 | 170 | 16 | 1.4 | 0.7 | 4 | wall > turret > building | Flat −4 on each hit taken; draws turret fire (turrets shoot nearest unit) |
| **Blaster** | 36 | 60 | 26 | 2.5 | 0.7 | 0 | building > wall > turret | Splash r = 1.5, 50 % falloff, hits structures only |
| **Mender** | 28 | 40 | 0 | — | 1.0 | 0 | — (no attack) | Heals **nearest damaged ally within range 2 per tick**, 8 hp/tick |

Turrets (generator constants, proposed): HP 80–120, damage 10–16, range 4–6, fire at the nearest unit, simultaneous volleys. Walls: HP 80, destructible. Cores: HP 200.

**The 20-unit cap is a design lever**: 20 Scouts cost only 160, so a swarm leaves 80 gold unspendable — the agent is pushed to trade cheap bodies for expensive ones, and the cap makes "buy everything cheap" a deliberate choice, not a free lunch.

### 3.2 Why each unit should be worth buying (hypotheses to test, not claims)

| Unit | Wins when | Loses when |
|---|---|---|
| Scout | `archipelago`: many lightly guarded pods; turrets can only kill so many bodies per volley, so mass beats HP | Against dense overlapping turret fire (`bunker_nest`), 16 hp dies in one volley |
| Sniper | `spine_pods`: turrets of range ≤ 4 guard the spine; snipers kill them without return fire | Range-5/6 turrets (a minority, by design) kill 28 hp units in 2–3 volleys; useless vs walls |
| Bulwark | `bunker_nest`: armor makes 10–16 dmg turret hits cost 6–12; nearest-unit targeting spares squishier allies; breaks walls | Open layouts: speed 0.7 means it arrives late, and 30 gold buys 3.75 Scouts |
| Blaster | Clustered pods (`spine_pods` pods, `bunker_nest` interiors): one shot hits 3+ buildings, wall lines | `archipelago` islands (nothing to splash); 60 hp is fragile |
| Mender | Sustained fire on a tanky target (Bulwark + Mender in `bunker_nest`, `labyrinth` corridors); +8 hp/tick offsets one turret | Burst deaths (Scouts); costs a unit slot and does zero damage on its own |

### 3.3 Simulator rules that matter for exactness

1. **Fixed-point arithmetic.** All hp, damage, heal values are integers scaled by `S = 840` (= lcm(1…8)), so splitting by any k ≤ 8 is exact and no float comparison can differ across symmetries.
2. **Tick order.** (a) snapshot start-of-tick state; (b) every turret and unit selects a target from the snapshot; (c) all damage and heals computed from the snapshot; (d) apply damage (armor first, then hp), then apply heals to units that survived; clamp at max hp; (e) remove dead; (f) movement (from the snapshot of survivors).
3. **Mender heal rule.** Candidates: allies alive at start of tick, `hp < max`, not itself, Euclidean distance ≤ 2.0. Choose by the key chain `(distance, hp_fraction, −cost)` — all D4-invariant. If still tied, **split the heal equally across all tied candidates** (exact, because of `S`). Heals from several Menders add. No overheal, no revive, a unit killed this tick receives nothing.
4. **Mender movement.** Moves toward the centroid of living non-Mender allies (D4-covariant), 1 cell/tick, stops within 1.5 of it; with no allies left, holds. Candidate steps ordered by invariant keys (distance to centroid, then summed distance to remaining allies); **if still tied, the Mender does not move this tick.** (A symmetric stay-put is equivariant; "pick the first by (y, x)" is not.)
5. **General tie principle (applies to all targeting and movement):** tie-breaks go through D4-invariant properties (hp, value, Euclidean distance, summed distance to the rest of the base). **Object ids and raw coordinates never decide.** If an exact tie survives, the action is symmetric: attacks split damage, heals split, movement abstains.
6. **Target choice** = most-preferred class that has any target in range, then nearest within that class (not a global rank across the map). Fixes the prototype's wall-first behaviour.
7. **Pathing.** Greedy straight-line movement (prototype) cannot navigate `labyrinth`. v2 moves along a BFS distance field to the current target (walls are passable at a cost only by destroying them; units attack a wall that blocks their next step). Field ties follow rule 5.
8. **Early exit.** Battle ends when the core and all value buildings are destroyed, all units are dead, or no unit changed hp/position for 12 consecutive ticks (stalemate); cap 240.
9. Splash is radius-based (D4-invariant) and never damages units.

### 3.4 Balance gates (pre-registered; evaluated on a TUNING seed set only)

Tuning set: raw layouts from seed streams disjoint from the pools (a test asserts disjointness). Max **2 iterations**, then stats freeze. Gates, all measured with the real simulator on scripted armies and the Oracle search:

| Gate | Pass condition |
|---|---|
| G1 no dominant unit | No unit type appears in the best fixed single-type army on > 60 % of tuning layouts, and no unit's removal from the Oracle's allowed set costs > 0.25 mean score in *every* family |
| G2 no dead unit | For each unit, Oracle ban-loss ≥ 0.02 mean score in ≥ 1 family |
| G3 niche exists | Each unit is the strictly best member of some scripted template (§3.2 column "wins when") in ≥ 1 family by ≥ 0.03 |
| G4 Mender specifically | Removing Menders from the Oracle army costs ≥ 0.02 in `bunker_nest` or `labyrinth`-like tuning layouts; and a Mender-only army scores < 0.01 (it cannot win alone) |
| G5 not trivial / not hopeless | Best single-type army mean score ∈ [0.25, 0.65] per family; Oracle breach rate ∈ [0.6, 0.95] |
| G6 cap pressure | Oracle armies use > 12 units in ≥ 30 % of layouts, but 20 Scouts is not the best army anywhere |

If a gate fails after iteration 2, the gate is reported as failed in PROGRESS.md and the decision (ship / re-scope) goes to the user — no silent third iteration.

---

## 4. Interface

### 4.1 Action space — Discrete(267)

| Indices | Meaning |
|---|---|
| 0–4 | Buy Scout / Sniper / Bulwark / Blaster / Mender |
| 5 … 264 | Deploy: `5 + type·52 + border_tile` (5 types × 52 tiles = 260) |
| 265 | END_BUY |
| 266 | WAIT |

Total `5 + 260 + 1 + 1 = 267`. Legal-action masks (`action_masks()`): buys need affordability and unit count < 20; deploys need units of that type still in the barracks, a free border tile (one unit per tile per deploy step), and DEPLOY phase; END_BUY only in BUY phase (and ≥ 1 unit bought); WAIT only when legal as in the prototype (zero-army / battle). Illegal action raises, as in the prototype.

### 4.2 Observation — Dict

`grid` (9, 14, 14), float32, all in [0, 1]:

| Ch | Content |
|---|---|
| 0 | deployed-unit occupancy (count / 4, clipped) |
| 1 | building hp fraction (0 where none) |
| 2 | turret hp fraction |
| 3 | wall hp fraction |
| 4 | building value weight / 3 (core = 1) |
| 5 | core mask |
| 6 | turret threat: number of turrets whose range covers the cell / 4, clipped |
| 7 | free legal-deploy border tiles (1 = deployable now) |
| 8 | **repair reach** — for each cell, number of deployed Menders whose range-2 disc covers it / 3, clipped; zero in BUY phase |

`state` (11): gold/240, owned-not-yet-deployed counts ×5 (÷ 20), phase one-hot ×3, units bought/20 (cap headroom), tick/240.

All channels are built from rotation/reflection-covariant quantities (discs, counts, masks), so the D4 transform of the grid is the grid of the transformed world. A test pins this per channel.

---

## 5. Layouts and the frozen pool

### 5.1 Families (14 × 14)

| Family | Role | Shape |
|---|---|---|
| `spine_pods` | familiar | A central spine of turrets/walls; 3–4 pods of 3–4 buildings branch off; core at a spine end. Favors Snipers, Blasters |
| `archipelago` | familiar | 4–6 isolated islands of 1–3 buildings with 1–2 turrets each; core on one island. Favors Scouts |
| `bunker_nest` | familiar | Core inside two nested wall rings with dense turrets between them; satellite buildings outside. Favors Bulwark + Mender |
| `labyrinth` | **held-out transfer, never trained on** | Serpentine wall corridors with doors, turrets at junctions, core at the end of the path. Requires BFS pathing; not generated or loaded by any training/tuning code path |

Acceptance rules (carried from the prototype, re-tuned in the balance step): reject empty, unreachable core, too_hard (Oracle breach impossible), single-type-trivial (best single type scores > threshold).

### 5.2 Pool of 300

| Group | `spine_pods` | `archipelago` | `bunker_nest` | `labyrinth` | Total |
|---|---:|---:|---:|---:|---:|
| DEV-train | 50 | 50 | 50 | — | 150 |
| DEV-validation | 10 | 10 | 10 | — | 30 |
| FINAL-familiar | 20 | 20 | 20 | — | 60 |
| FINAL-transfer | — | — | — | 60 | 60 |
| **Total** | 80 | 80 | 80 | 60 | **300** |

DEV = 180, FINAL = 120.

- Records are `{id, family, seed, pool, tag}`; geometry regenerated from `(family, seed)` as in the prototype. New seed streams (disjoint from the prototype's v1–v3 pools and from the tuning set).
- Frozen as `data/v2/layouts_v2.json` with a **new sha256** in `data/v2/layouts_v2.sha256`, recorded in every model's identity file.
- **FINAL/transfer are never used for training or tuning**, enforced by: loader refuses FINAL/transfer unless called from the evaluation entry point with an explicit unlock flag; a leak-guard test (`dev sampler never yields FINAL or labyrinth ids`); FINAL evaluated exactly once on a clean commit, logged. Hyperparameters, planner N values (8/32/128 are declared, not selected), curriculum, and stat tuning are all fixed using DEV/validation/tuning only.

---

## 6. Curriculum

Deployment-first, **with random armies from the first step**.

- Phase A (first 1/3 of timesteps): env starts in DEPLOY with a *fresh random legal army per episode* (random affordable types, stop with p = 0.1 per buy, ≤ 20 units, ≤ 240 gold, all five types possible including Menders).
- Phase B: full BUY → DEPLOY game.
- Switch by timestep, per-episode lazy (as in the prototype), logged to `curriculum.log` and the model identity.
- Open option (user decision, default **off**): a linear ramp of the pre-bought probability from 1 → 0 over the next 1/6 of training to soften the phase-switch dip seen in the prototype (0.16 → 0.05).

---

## 7. Learners and evaluation

| Condition | Description | Trained learners |
|---|---|---:|
| PPO plain | MaskablePPO, CNN(9×14×14) + MLP(11) features | 10 |
| PPO + D4 | same, random D4 symmetry per episode via `D4Wrapper` | 10 |
| Masked Double DQN | own PyTorch implementation, action masks in both argmax and target, replay buffer, same feature extractor | 10 |
| Planner N = 8 / 32 / 128 | **no new training**: for each PPO learner (plain and D4) sample N complete plans (buy + deploy) from the *stochastic* policy, simulate each with the deterministic simulator, play the best | 10 learners × 2 PPO types |

Planner details: plans sampled with masks; the simulator is the exact one used in play, so the planner has privileged access to the simulator — this is stated in the report. N is reported as a **compute budget** (simulator calls and wall-clock per layout). Control baselines at the same N: *random-plan best-of-N* (uniformly random legal plans), to show how much the policy guidance adds beyond brute-force search.

Evaluation protocol (inherits Step 8): deterministic actions (masked argmax) for PPO/DDQN; `final.zip`/final checkpoint only; D4 models evaluated on the canonical view; groups DEV-train / validation (seen, never headline) / FINAL-familiar (60) / FINAL-transfer (60); bootstrap over **learners** (10 000 resamples, percentile 95 % CI); paired differences; metrics: breach rate (headline), mean score, mean `D`, army composition per family; familiar vs transfer for every condition; baselines Random, Greedy, BestSingle (chosen on DEV), Oracle (search budget declared), random-plan best-of-N. A sampled-vs-deterministic probe is kept as exploratory.

### 7.1 Extras

- **D4 test-time ensembling (TTA):** for each of the 8 symmetries, transform obs and mask, get masked policy probabilities (Q-values for DDQN), map back through the inverse action permutation, average, then argmax. Reported as extra eval rows (`+TTA`) for PPO plain, PPO+D4 and DDQN; costs 8 forward passes, no retraining.
- **"Why this move?"** For every replay step the **Decision lens** shows: top-5 actions with probabilities, the value estimate, a deployment-probability heatmap over the 52 border tiles per unit type, and a simulator-backed counterfactual (planner rollout score for the top-3 alternatives, labelled with its N). Built on `policy_outputs` (kept structured for this).
- **Duel mode:** human vs agent vs oracle on the same layout and budget. The agent's and oracle's choices are hidden until the human submits; all three armies/deployments run through the same simulator and are compared on score and breach. Oracle = cached search with its declared budget (so it is a lower bound on optimum and labelled that way).

---

## 8. App

Streamlit shell, **no sidebar** (`initial_sidebar_state="collapsed"`, sidebar CSS removed). Dark blueprint theme (navy background, thin cyan grid lines, amber/white accents; palette as CSS variables, high-contrast text).

```
┌────────────────────────────────────────────────────────────────────┐
│ COMMAND BAR: layout ▾ | agent ▾ | mode: Play·Replay·Duel | budget │
│              [Buy] [Deploy] [Run] [Why?] [Heatmap ◐]  [Export]     │
├───────────────────────────────────────────────┬────────────────────┤
│                                               │  DECISION LENS     │
│          CENTRAL MAP  (14×14 SVG)             │  top-5 actions     │
│   buildings · turrets · walls · units         │  value estimate    │
│   heatmap toggle: deploy-prob / threat /      │  counterfactuals   │
│   repair reach                                │  army + gold       │
├───────────────────────────────────────────────┴────────────────────┤
│ TIMELINE SCRUBBER  ◀ ▶  |────●──────────────| tick 87 / 240         │
└────────────────────────────────────────────────────────────────────┘
```

The map is inline SVG rendered from precomputed battle frames so scrubbing does not re-run the simulator; the scrubber is the only widget that triggers a re-render. CPU inference with bounded model caches; JSON and standalone-HTML export as in SPEC.md.

---

## 9. Prototype → v2 module map

New package `breachlens/` is created next to `siege/` (copy-then-modify), so the prototype stays importable and testable until v2 is complete; `siege/` is deleted at the end of the branch (it lives on the `prototype-v1` tag).

| Prototype module | Action | Notes |
|---|---|---|
| `siege/troops.py` | **Rewrite** | 5 types, armor, heal, fixed-point stats |
| `siege/combat.py` | **Rewrite** | Fixed-point, tick order, Mender, armor, BFS pathing, nearest-in-preference targeting, symmetric tie handling, early exit. Prototype `_strike`/frames record idea reused |
| `siege/bases.py` | **Rewrite** | Four new generators, value weights, new turret constants |
| `siege/layouts.py` | **Modify** | Keep record format, `(family, seed)` regeneration, acceptance framework, `stable_family_hash`, sha handling; new pool counts/groups, new rules, labyrinth guard |
| `siege/env.py` | **Modify (heavy)** | 14×14, 52-tile border, 267 actions, 9 ch + 11 state, unit cap, random pre-bought armies, new score |
| `siege/armies.py` | **Modify** | `place_on_border` for 52 tiles, 5 types |
| `siege/baselines.py` | **Modify** | Random: reuse. Greedy: rewrite (5 types, Mender pairing). BestSingle: modify. Oracle: modify (5 types, cap, new budget). New: random-plan best-of-N |
| `siege/render.py` | **Rewrite** | Blueprint theme, heatmaps, SVG frames |
| `rl/augment.py` | **Modify** | Re-derive 52-tile border permutation and 267-action table; add repair-channel transform; `D4Wrapper` logic reused |
| `rl/features.py` | **Modify** | Input 9×14×14 and 11-d state |
| `rl/policy_io.py` | **Modify** | 267 actions, per-type heatmaps, TTA |
| `rl/split.py` | **Modify** | 150/30 split, 10 per family validation, bound to new sha |
| `rl/evaluation.py` | **Reuse** | Rollout/helper functions, minor signature changes |
| `rl/train_ppo.py` | **Modify** | Random phase-A default, new ids, identity fields; resume logic reused as is |
| `rl/train.py`, `rl/eval.py` | **Drop** | Throwaway step-2 smoke code |
| `rl/eval_ppo.py` | **Modify** | Validation eval for new env |
| `rl/eval_all.py` | **Modify (heavy)** | New groups, DDQN/planner/TTA conditions, breach-rate headline, write-once FINAL guard |
| `rl/diagnose_grid.py`, `rl/diagnose_buy_deploy.py` | **Reuse later / modify** | Same buy-vs-deploy decomposition on new data, after the grid; not needed for build |
| new `rl/ddqn.py`, `rl/planner.py`, `rl/tta.py`, `rl/explain.py`, `app/*` | **New** | |
| `scripts/run_ppo_grid.sh`, `run_one.sh`, `status.py` | **Modify (small)** | Condition list, step budget |
| `scripts_balance.py`, `scripts_tuning.py` | **Modify** | Become the G1–G6 gate script on the tuning set |
| `scripts_rebuild_pool.py`, `scripts_contact_sheet.py` | **Modify** | v2 pool builder, contact sheet incl. labyrinth |
| `scripts_smoke.py` | **Modify** | Trivial |
| `scripts_6b_analysis.py`, `scripts_diag_deploy.py`, `scripts_sweep_splash.py` | **Drop** | Prototype-specific; remain on the tag |
| `tests/test_augment.py` | **Modify** | Equivariance tests re-derived; keep structure |
| `tests/test_simulator.py`, `test_env.py`, `test_layouts.py`, `test_step4_contract.py`, `test_step6.py`, `test_eval_all.py`, `test_diagnose_*.py` | **Modify** | Same intent, new constants; most assertions rewritten |
| `docs/*.md` (actions, observation, reward, troops) | **Rewrite** | Regenerated from v2 constants |
| Prototype results (`results/`, `models/`, `logs_6b/`) | **Keep untouched** | Read-only history; v2 writes to `data/v2/`, `results_v2/`, `models_v2/` |

---

## 10. Risks

| # | Risk | Why it matters | Mitigation |
|---|---|---|---|
| R1 | **D4 exactness with the Mender** | Heal-target selection, Mender movement, and simultaneous heal+damage are new tie sources; id- or coordinate-ordered ties silently break equivariance (the prototype needed a dedicated step to reach 0 mismatches) | §3.3 rules 1–5: fixed-point integers; invariant key chain; exact ties → split heal / abstain move; ids never used; tick-snapshot ordering so heal order cannot matter. Tests: ≥ 10 000 random battles × 8 symmetries with Menders in ≥ 50 % of armies; adversarial **id and wall-order shuffles**; hand-built symmetric fixtures (Mender between two equally damaged allies, symmetric centroid ties) that must come out bit-identical. B3 does not close until mismatches are 0 |
| R2 | Float/ordering drift | Equivariance can fail from 1e-16 differences | Integer fixed-point state; no float comparisons in decisions except distances compared via squared-integer coordinates |
| R3 | Balance of five units | Prototype took 2 tuning iterations and still left tank weak | Gates G1–G6 pre-registered; tuning seeds disjoint; hard cap on iterations; failed gates reported, not hidden |
| R4 | Simulator speed (240 ticks, 20 units, BFS) | Dominates both training throughput and Oracle/planner cost | Early exit; cache BFS fields per layout/target; benchmark gate in B2 (target < 40 ms mean battle); if missed, vectorize/numba before any grid |
| R5 | Labyrinth pathing | New movement model only needed for the transfer family, risk of stuck units or non-terminating stalemates | Stalemate exit rule; reachability acceptance rule; contact-sheet review; tests with constructed corridors |
| R6 | Planner leaks simulator knowledge | Looks like "the agent" is stronger than it is | Always labeled as search-augmented, reported with compute budget and compared to random-plan best-of-N at equal N |
| R7 | Larger action space (267) and 20-step buys → harder exploration | Prototype PPO plateaued far below Oracle | Random phase-A armies; larger step budget; one bounded pilot round (B13) on validation, then frozen |
| R8 | D4 may underfit again | Prototype D4 was still rising at 1M | Larger budget; TTA as a cheaper symmetry use; report either outcome — the question is the experiment |
| R9 | Evaluation leakage by convenience | Tempting to peek at FINAL for N, TTA, or stats | Write-once FINAL protocol + loader guard + tests (§5.2) |
| R10 | Streamlit scrubbing latency / no-sidebar styling | UX of the signature feature | Precomputed frames, SVG map, only scrubber triggers rerun; fall back to standalone HTML export |
| R11 | Scope | Five conditions, three extras, an app | Strict step order (§11); app/extras only after the grid is evaluated |

---

## 11. Build steps (each ends with pytest green; one step at a time, only when told)

| Step | Deliverable | Pytest requirement |
|---|---|---|
| B1 | Scaffold `breachlens/`, constants, action codec, border tiles | Border has 52 unique clockwise tiles; codec is a bijection over 267 actions; constants match this doc (budget 240, cap 20, ticks 240) |
| B2 | Units/turrets/walls + core combat (no Mender), fixed-point, nearest-in-preference targeting, BFS pathing, early exit; **speed benchmark** | Hand-built scenarios (range, armor, splash, wall block); determinism over 100 seeds; early-exit cases; benchmark recorded (not asserted) |
| B3 | Mender, tie rules, full tick order; **D4 equivariance of the simulator** | 0 mismatches across ≥ 10 000 random battles × 8 symmetries; adversarial id/order shuffles; symmetric-tie fixtures; heal rules (no overheal, no revive, split) |
| B4 | Four layout generators + acceptance rules | Determinism from `(family, seed)`; connectivity/reachability incl. labyrinth; building counts and value weights in range; each family contact sheet renders |
| B5 | Balance gate script + ≤ 2 tuning iterations on the TUNING set; stats frozen | Gate script deterministic; tuning seeds disjoint from all pool seeds; gate results written to `data/v2/balance.json`; every unit has stats pinned by test |
| B6 | Build and freeze the 300-layout pool; new sha256 | Counts per group/family exactly as §5.2; sha matches; ids unique; leak-guard test (DEV sampler never yields FINAL/labyrinth); acceptance holds under the current simulator |
| B7 | `BreachEnv`: 267 actions, 9 ch + 11 state, masks, random pre-bought armies, terminal score | Mask legality fuzz (10 000 random legal steps never raise); each observation channel unit-tested incl. repair reach; score formula on constructed outcomes; reset/seed determinism; `prebought="random"` always legal |
| B8 | Baselines: Random, Greedy, BestSingle, Oracle (cached), random-plan best-of-N | Baseline outputs reproducible; Oracle ≥ every other baseline on each layout it is run on; best-of-N non-decreasing in N for nested samples |
| B9 | D4 augmentation, `D4Wrapper`, TTA | Inverse identity, group closure, action bijection per symmetry; mask legality under transform; env + simulator equivariance on 100 layouts incl. Menders; TTA with identity-only policy equals the plain policy |
| B10 | PPO pipeline: features, train/eval scripts, curriculum, resume; **throughput benchmark** | Tiny wiring run; phase A random-army legality; curriculum switch logged; resume restores timestep; model identity records sha/commit; throughput recorded to re-estimate §12 |
| B11 | Masked Double DQN | Masked argmax never selects illegal actions; Double-Q target equals hand-computed value on a toy MDP; replay buffer round-trips; tiny run improves over random on a toy layout |
| B12 | Policy-guided planner (N = 8/32/128) | N = 1 equals one sampled rollout; deterministic given seed; chosen plan's score = max of its N; nested sample sets give non-decreasing score; masks respected |
| B13 | One bounded pilot round on validation (1–2 seeds, short), hyperparameters frozen, evaluation protocol pre-registered in PROGRESS.md | Pilot scripts have a test that FINAL/transfer ids are never loaded; frozen config file is read-only-checked |
| B14 | Launch grid (user-run): PPO plain ×10, PPO+D4 ×10, DDQN ×10 | `status.py` and resume tests updated; dry-run command list asserted |
| B15 | Full evaluation, FINAL once on a clean commit; bootstrap tables; familiar vs transfer; planner and TTA rows | Evaluator asserts layout ids ⊂ declared groups; bootstrap reproducible by seed; D4 canonical view equals identity wrapper; model identity complete |
| B16 | Diagnostics (buy-vs-deploy, curves, per-seed) on DEV/validation only | FINAL never loaded (test); control cells reproduce eval numbers exactly |
| B17 | "Why this move?" | Top-5 probs sum ≤ 1 and respect mask; heatmap sums to the deploy mass; counterfactual score equals a direct simulation |
| B18 | App (blueprint theme, command bar, map, Decision lens, scrubber) | Renders every layout family without error; frame count equals sim frames; scrubber index bounds; export JSON/HTML round-trips |
| B19 | Duel mode | Hidden choices stay hidden until submit; all three sides scored by the same simulator; ties and illegal human input handled |
| B20 | README, docs regeneration, final PROGRESS.md | Doc constants regenerated from code and checked by test |

---

## 12. Training-time estimate at 14×14 (estimate)

Basis (measured in the prototype, PROGRESS.md): ~167 steps/s per job with 6 jobs in parallel on 8 cores; 1M steps ≈ 1.7 h/job; battle = 50 ticks at ≈ 10 ms; episodes ≈ 17 steps.

Adjustments for v2:

- Battle: 240 ticks, up to 20 units, BFS, Menders → raw sim 5–10× slower (~50–100 ms); early exit and BFS caching should cut that to roughly **40–80 ms (estimate)**.
- Episodes: up to 20 buys + 20 deploys + END_BUY → ~28 steps on average, which amortizes sim cost over more steps.
- Net per-job throughput: **100–130 steps/s** at 6 parallel.
- Step budget: **1.5M** per PPO run (prototype plain plateaued near 0.75M; D4 had not converged at 1M; action space is larger).

| Item | Runs | Per run | Waves (6 parallel) | Wall time |
|---|---:|---:|---:|---:|
| PPO plain | 10 | 3.2–4.2 h | | |
| PPO + D4 | 10 | 3.2–4.2 h | 4 waves total for 20 runs | **13–17 h** |
| Masked DDQN (1.5M env steps, 1 update/4 steps) | 10 | ~1.5–2.5 h | 2 waves | **3–5 h** |
| Oracle search, 300 layouts (≈ 400 sims each) | — | — | 7 workers | ≈ 0.5 h |
| Planner eval: 20 PPO learners × 120 FINAL layouts × (8+32+128 = 168 plans) ≈ 400 k plans × ~0.1 s | — | — | 7 workers | ≈ 1.5–2 h |
| TTA + all other evaluation | — | — | | ≈ 0.5 h |
| Tuning, gates, pilot (B5, B13) | — | — | | ≈ 3–4 h |
| **Total compute** | | | | **≈ 22–29 h wall**, mostly unattended (grid) |

Sensitivity: every 10 ms of battle time adds ≈ 8 % to training and ≈ 10 % to planner cost. If B2 measures > 100 ms/battle, simulator optimization comes before any grid. If B10 throughput is < 90 steps/s, drop to 1.2M steps/run (≈ 10–13 h for PPO) rather than cutting seeds below 10.

---

## 13. Decisions I made that you may want to overrule

1. **Package layout:** new `breachlens/` package beside `siege/`, not an in-place edit.
2. **Planner base policies:** both PPO plain and PPO+D4 (so 6 planner cells), not just the better one.
3. **Exact-tie rule:** split/abstain instead of id ordering (changes gameplay marginally in symmetric situations; makes D4 exactness provable).
4. **No reward shaping:** terminal score only.
5. **Hard curriculum switch** (ramp is an option, default off).
6. **1.5M steps** per PPO/DDQN run.
