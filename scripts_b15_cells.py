"""B15 cell registry: the set of cells, their order, names, condition ids and the raw-file schema (docs/SPEC_V2.md row (gg)).

Standard library only (no numpy, torch, simulator or pool import), so the analysis script can import it without being able to call a model or the
simulator. The session script (`scripts_b15_final.py`) and the analysis script (`scripts_b15_analysis.py`) both take the cell list from here.

Cell granularity (UNPINNED by (gg), chosen here, see UNPINNED): one cell = one (condition, seed) pair for everything that has seeds, one cell per
deterministic baseline. A control or planner cell holds the three nested prefixes N = 8 / 32 / 128 of ONE 128-plan batch (rows per layout and N).
Total 116 cells: 6 baselines + 30 learners + 30 learners with TTA + 30 controls (RAS / RPS / RPS-full x 10 seeds) + 20 planners.
"""
from __future__ import annotations

SEEDS = tuple(range(10))
N_VALUES = (8, 32, 128)
GROUPS = ("FINAL-familiar", "FINAL-transfer")
UNIT_TYPES = ("scout", "sniper", "bulwark", "blaster", "mender")
EVAL_SEED = 8080

BASELINES = ("random", "greedy", "bestsingle", "bestsingle_perlayout", "bestsingle_hindsight", "oracle")
LEARNERS = ("ppo_plain", "ppo_d4", "ddqn")                     # grid condition keys; run names grid_<cond>_s<k>
CONTROLS = ("ras", "rps", "rpsfull")
PLANNERS = ("ppo_plain", "ppo_d4")                             # planner on PPO learners only (row (gg) (a))

LABEL = {"random": "Random", "greedy": "Greedy", "bestsingle": "BestSingle", "bestsingle_perlayout": "BestSingle-perlayout (diagnostic)",
         "bestsingle_hindsight": "BestSingle-hindsight (diagnostic)", "oracle": "Oracle (upper reference, ~400 sims)",
         "ppo_plain": "PPO plain", "ppo_d4": "PPO+D4", "ddqn": "DDQN",
         "ppo_plain_tta": "PPO plain + TTA", "ppo_d4_tta": "PPO+D4 + TTA", "ddqn_tta": "DDQN + TTA",
         "ras": "RAS", "rps": "RPS", "rpsfull": "RPS-full"}
DETERMINISTIC = set(BASELINES)                                 # one value per layout, no seed level


def run_name(cond: str, seed: int) -> str:
    return f"grid_{cond}_s{seed}"


def cond_id(base: str, n: int | None = None) -> str:
    """Condition id used in the analysis: the base id, plus _<N> for the best-of-N cells."""
    return base if n in (None, "") else f"{base}_{int(n)}"


def cond_label(cid: str) -> str:
    base, _, n = cid.rpartition("_")
    if base and n.isdigit() and int(n) in N_VALUES:
        if base.startswith("planner_"):
            return f"Planner on {LABEL[base[len('planner_'):]]}, N={n}"
        return f"{LABEL[base]}-{n}"
    return LABEL[cid]


def _cell(kind: str, cond: str, seed: int | None = None, **kw) -> dict:
    name = cond if seed is None else f"{cond}__s{seed}"
    return {"name": name, "kind": kind, "cond": cond, "seed": seed, **kw}


def all_cells() -> list[dict]:
    """The cells in the order of row (gg) (e)(3): (i) baselines, (ii) learners, (iii) the same models with TTA, (iv) controls, (v) planners.
    Within a group seeds ascend; controls are method-major, planners PPO plain then PPO+D4."""
    cells = [_cell("baseline", b) for b in BASELINES]
    for c in LEARNERS:
        cells += [_cell("learner", c, s, run=run_name(c, s), algo="ddqn" if c == "ddqn" else "ppo", tta=False) for s in SEEDS]
    for c in LEARNERS:
        cells += [_cell("learner", c + "_tta", s, run=run_name(c, s), algo="ddqn" if c == "ddqn" else "ppo", tta=True) for s in SEEDS]
    for m in CONTROLS:
        cells += [_cell("control", m, s, method=m, ns=list(N_VALUES)) for s in SEEDS]
    for c in PLANNERS:
        cells += [_cell("planner", "planner_" + c, s, run=run_name(c, s), algo="ppo", ns=list(N_VALUES)) for s in SEEDS]
    names = [c["name"] for c in cells]
    assert len(set(names)) == len(names)
    for i, c in enumerate(cells):
        c["order"] = i
    return cells


def cell_conditions(cell: dict) -> list[str]:
    """Condition ids a cell contributes rows to."""
    if cell["kind"] in ("control", "planner"):
        return [cond_id(cell["cond"], n) for n in cell["ns"]]
    return [cell["cond"]]


def all_conditions() -> list[str]:
    out = []
    for c in all_cells():
        for k in cell_conditions(c):
            if k not in out:
                out.append(k)
    return out


def n_seeds(cid: str) -> int:
    return 1 if cid in DETERMINISTIC else len(SEEDS)


def expected_rows(cell: dict, n_layouts: int) -> int:
    return n_layouts * (len(cell["ns"]) if cell["kind"] in ("control", "planner") else 1)


# ---------------------------------------------------------------------------------------------------------------- raw file schema
ARMY_COLS = tuple(f"army_{u}" for u in UNIT_TYPES)
RAW_COLUMNS = ("cell", "condition", "N", "learner_seed", "tta", "group", "set", "family", "id", "layout_seed", "n_episodes",
               "score", "d", "breach", "n_bought", "n_deployed", *ARMY_COLS, "chosen_type", "model_sha256", "policy_hash", "extra",
               "session_id", "score_version", "sim_version", "generator_hash", "pool_sha256", "git_commit", "git_dirty", "seed", "role")

# Rules the code needs that row (gg) does not pin. Each was settled with the smallest mechanical choice that keeps (gg) intact and is listed in the
# B15a report for the owner. None of them changes a pinned rule, the cell list, N, seeds, layouts, budgets or the primary comparison list.
UNPINNED = {
    "cell granularity": "one cell = one (condition, seed); control / planner cells hold the nested N = 8 / 32 / 128 rows of one 128-plan batch (116 cells)",
    "raw file schema": "columns in RAW_COLUMNS (one row per cell, layout and N); `seed` is the evaluator seed 8080 (REQUIRED_META), the learner / control seed is `learner_seed`",
    "partial raw file": "an unfinished cell file (no manifest line) is moved to results/b15/discarded/ and logged, not deleted; a manifest line whose file is missing or does not hash is NOT recomputed: the session stops (corruption, not a crash)",
    "restart interface": "`--restart --cause machine|code-fix`; machine requires HEAD == the commit of the last manifest line, code-fix requires HEAD to differ; both require a clean tree, session.json and an eval ledger that holds only this session",
    "full suite precondition": "the real run executes `.venv/bin/python -m pytest` itself and logs its exit code (the pinned wording 'a green full suite' has no other checkable form); it must exit 0",
    "BestSingle-hindsight": "per FINAL group and per family, the best of the 5 single-unit types by mean score on that group's FINAL layouts (B8 definition, ties -> lowest type index); the five single-unit rows are computed inside the cell and only the chosen rows are written",
    "strongest control tie": "highest mean TRUE SCORE at (group, N) among RAS, RPS, RPS-full; an exact tie goes to the earlier of that order",
    "secondary interval for differences": "learner-only: seeds resampled with the same independence / sharing rule as the primary interval, layouts fixed; a deterministic-vs-deterministic comparison has no seed level and its secondary interval is the point value",
    "gap bootstrap seeds": "the generalisation gap uses ONE seed draw shared by the DEV-train and FINAL-familiar sets (same 10 trained models), then DEV-train layouts, then FINAL-familiar layouts, independently",
    "per-family P comparisons": "P1-P5 are judged per group only; per-family values appear in the condition tables, not as comparisons",
    "army composition": "mean units bought per type from the episode `army`; Greedy / BestSingle(-perlayout, -hindsight) get it from their fixed type lists; Random and Oracle episodes carry no army, so only n_bought / n_deployed exist for them",
    "exploratory list": "planner vs its own learner, planner N vs N, DDQN vs PPO, transfer vs familiar per learner / baseline (two independent layout sets, shared seeds); labelled exploratory, no multiplicity correction",
    "plots": "no plots are produced; the analysis writes CSV and Markdown tables only",
    "restart reloads FINAL": "a restart must load FINAL again through the loaders (new ledger entries, same session_id); count_eval_sessions stays 1",
}
