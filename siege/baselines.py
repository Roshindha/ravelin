from __future__ import annotations

import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from siege.armies import MIXED_ARMIES, best_affordable_army, mixed_army, place_on_border
from siege.bases import generate_base_layout
from siege.combat import simulate_battle
from siege.env import BORDER, BUDGET, N_BORDER
from siege.layouts import TRANSFER_FAMILY, FAMILIAR_FAMILIES, load_layout_pool, stable_family_hash
from siege.troops import TROOP_TYPES

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "baselines.csv"
ORACLE_USAGE_PATH = RESULTS_PATH.with_name("oracle_usage.csv")


def _layout_for_record(record: dict) -> dict:
    return generate_base_layout(np.random.default_rng(record["seed"]), family=record["family"])


def _score_layout(army: list[dict], layout: dict, max_ticks: int = 50) -> float:
    result = simulate_battle(
        army,
        layout["buildings"],
        layout["walls"],
        layout["turrets"],
        max_ticks=max_ticks,
    )
    return result.destruction, result


def _random_army(rng: np.random.Generator, allowed: list[str] | None = None) -> list[str]:
    remaining = BUDGET
    army: list[str] = []
    while remaining > 0:
        candidates = [spec.name for spec in TROOP_TYPES if spec.cost <= remaining and (allowed is None or spec.name in allowed)]
        if not candidates:
            break
        name = rng.choice(candidates)
        army.append(name)
        remaining -= next(spec.cost for spec in TROOP_TYPES if spec.name == name)
        if rng.random() < 0.25:
            break
    return army


def lanchester_value(spec) -> float:
    """Static combat value per gold^2: (hp * damage) / cost^2.

    Combat power of a group scales with (total HP) x (total DPS), so the fair per-gold comparison of
    two unit types is hp*damage/cost^2. (v1 GreedyAgent ranked by damage/cost, which put the
    weakest-in-practice unit, splash, first and bought five of it: 0.01 mean destruction, below Random.)
    """
    return spec.hp * spec.damage / spec.cost**2


def _greedy_army(layout: dict) -> list[str]:
    """Spend the budget on the highest Lanchester-value unit first, then fill leftover gold with the
    next best that fits. Layout-independent composition (the layout only matters for deployment)."""
    ranked = sorted(TROOP_TYPES, key=lambda spec: (lanchester_value(spec), spec.name), reverse=True)
    army: list[str] = []
    remaining = BUDGET
    for spec in ranked:
        count = remaining // spec.cost
        if count > 0:
            army.extend([spec.name] * count)
            remaining -= count * spec.cost
    return army


def _score_family_best_single(layouts: list[dict]) -> dict[str, list[str]]:
    best_by_family: dict[str, list[str]] = {}
    for family in FAMILIAR_FAMILIES + (TRANSFER_FAMILY,):
        fam_layouts = [r for r in layouts if r["family"] == family]
        if not fam_layouts:
            continue
        best = {name: 0.0 for name in [spec.name for spec in TROOP_TYPES]}
        for troop_name in best:
            score = 0.0
            for record in fam_layouts:
                layout = _layout_for_record(record)
                army = place_on_border(best_affordable_army(troop_name), start_idx=0)
                score += _score_layout(army, layout)[0]
            score /= len(fam_layouts)
            best[troop_name] = score
        winner = max(best, key=lambda name: best[name])
        best_by_family[family] = [winner]
    return best_by_family


def _default_start_idx(layout: dict) -> int:
    buildings = layout["buildings"]
    if not buildings:
        return 0
    nearest = min(buildings, key=lambda b: abs(b["x"]) + abs(b["y"]))
    best = min(range(N_BORDER), key=lambda i: sum(abs(BORDER[i][d] - nearest["x" if d == 0 else "y"]) for d in (0, 1)))
    return best


def _place_random_deploy(army: list[str], rng: np.random.Generator) -> list[dict]:
    return place_on_border(army, start_idx=int(rng.integers(0, N_BORDER)))


def random_agent_score(layout: dict, rng: np.random.Generator) -> float:
    army = _random_army(rng)
    if not army:
        return 0.0
    return _score_layout(_place_random_deploy(army, rng), layout)[0]


def greedy_agent_score(layout: dict, rng: np.random.Generator) -> float:
    army = _greedy_army(layout)
    if not army:
        return 0.0
    start = _default_start_idx(layout)
    return _score_layout(place_on_border(army, start_idx=start), layout)[0]


def best_single_agent_score(layout: dict, troop_name: str) -> float:
    army = best_affordable_army(troop_name)
    start = _default_start_idx(layout)
    return _score_layout(place_on_border(army, start_idx=start), layout)[0]


_COSTS = {spec.name: spec.cost for spec in TROOP_TYPES}
_NAMES = [spec.name for spec in TROOP_TYPES]


def oracle_search(layout: dict, rng: np.random.Generator, n_random: int = 200, n_refine: int = 200, allowed: list[str] | None = None) -> tuple[float, list[str], int]:
    """Best (destruction, army names, push start tile) found by random search + local hill-climbing.

    Phase 1 samples `n_random` random armies/start tiles. Phase 2 hill-climbs from the best one with
    single-troop replace/add/remove, order swaps and start-tile shifts, accepting strict improvements
    only. It is a (strong) lower bound on the true optimum, i.e. a ceiling estimate, not exact.
    """
    best_score, best_army, best_start = -1.0, [], 0
    for _ in range(n_random):
        army = _random_army(rng, allowed)
        if not army:
            continue
        start = int(rng.integers(0, N_BORDER))
        score = _score_layout(place_on_border(army, start_idx=start), layout)[0]
        if score > best_score:
            best_score, best_army, best_start = score, army, start
    pool_names = _NAMES if allowed is None else list(allowed)
    for _ in range(n_refine):
        army, start = list(best_army), best_start
        spent = sum(_COSTS[n] for n in army)
        move = int(rng.integers(0, 5))
        if move == 0 and army:  # replace one troop
            i = int(rng.integers(len(army)))
            new = pool_names[int(rng.integers(len(pool_names)))]
            if spent - _COSTS[army[i]] + _COSTS[new] > BUDGET:
                continue
            army[i] = new
        elif move == 1:  # add one troop
            new = pool_names[int(rng.integers(len(pool_names)))]
            if spent + _COSTS[new] > BUDGET:
                continue
            army.insert(int(rng.integers(len(army) + 1)), new)
        elif move == 2 and len(army) > 1:  # remove one troop
            army.pop(int(rng.integers(len(army))))
        elif move == 3:  # shift the push point
            start = (start + int(rng.integers(-6, 7))) % N_BORDER
        elif move == 4 and len(army) > 1:  # reorder (order == deployment slot == troop id)
            i, j = (int(x) for x in rng.choice(len(army), size=2, replace=False))
            army[i], army[j] = army[j], army[i]
        else:
            continue
        score = _score_layout(place_on_border(army, start_idx=start), layout)[0]
        if score > best_score:
            best_score, best_army, best_start = score, army, start
    return max(best_score, 0.0), best_army, best_start


def oracle_agent_score(layout: dict, rng: np.random.Generator, n_search: int = 200) -> float:
    return oracle_search(layout, rng, n_random=n_search)[0]


def _oracle_worker(record: dict) -> tuple[int, float, list[str]]:
    layout = _layout_for_record(record)
    score, army, _ = oracle_search(layout, np.random.default_rng([int(record["id"]), 4242]))
    return int(record["id"]), score, army


def oracle_results(records: list[dict]) -> dict[int, tuple[float, list[str]]]:
    """Run the oracle once per record (in parallel). Returns {record id: (score, army)}."""
    with Pool() as pool:
        out = pool.map(_oracle_worker, records, chunksize=2)
    return {rid: (score, army) for rid, score, army in out}


def _evaluate_records(records: list[dict], strategy_name: str, *, troop_name: str | None = None, rng_seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(rng_seed)
    scores = []
    clears = 0
    for record in records:
        layout = _layout_for_record(record)
        if strategy_name == "RandomLegalAgent":
            score = random_agent_score(layout, rng)
        elif strategy_name == "GreedyAgent":
            score = greedy_agent_score(layout, rng)
        elif strategy_name == "BestSingleTroopAgent":
            score = best_single_agent_score(layout, troop_name or "ranger")
        elif strategy_name == "Oracle":
            score = oracle_agent_score(layout, rng)
        else:
            raise ValueError(strategy_name)
        scores.append(score)
        if score >= 0.999:
            clears += 1
    return float(np.mean(scores)), float(clears / len(scores)) if scores else 0.0


def run_baselines() -> list[dict[str, str | float | int]]:
    dev_records = load_layout_pool(pool="DEV", tag="familiar")
    transfer_records = load_layout_pool(pool="FINAL", tag="transfer")
    records_by_family = {family: [r for r in dev_records if r["family"] == family] for family in FAMILIAR_FAMILIES}
    records_by_family[TRANSFER_FAMILY] = transfer_records

    strategies = [
        ("RandomLegalAgent", None),
        ("GreedyAgent", None),
        ("Oracle", None),
    ]
    all_records = dev_records + transfer_records
    oracle = oracle_results(all_records)
    best_single_names = {family: max(
        [spec.name for spec in TROOP_TYPES],
        key=lambda name: sum(
            best_single_agent_score(_layout_for_record(r), name)
            for r in records_by_family[family]
        ) / len(records_by_family[family])
    ) for family in records_by_family}

    rows: list[dict[str, str | float | int]] = []
    def _oracle_stats(recs: list[dict]) -> tuple[float, float]:
        sc = [oracle[r["id"]][0] for r in recs]
        return float(np.mean(sc)), float(np.mean([x >= 0.999 for x in sc]))

    for agent_name, _ in strategies:
        for family, family_records in records_by_family.items():
            if agent_name == "Oracle":
                mean, clear_rate = _oracle_stats(family_records)
            else:
                mean, clear_rate = _evaluate_records(family_records, agent_name, rng_seed=stable_family_hash(family))
            rows.append({
                "agent": agent_name,
                "family": family,
                "mean_destruction": round(mean, 4),
                "full_clear_rate": round(clear_rate, 4),
                "layouts": len(family_records),
            })
        if agent_name == "Oracle":
            mean, clear_rate = _oracle_stats(all_records)
        else:
            mean, clear_rate = _evaluate_records(all_records, agent_name, rng_seed=1234)
        rows.append({
            "agent": agent_name,
            "family": "overall",
            "mean_destruction": round(mean, 4),
            "full_clear_rate": round(clear_rate, 4),
            "layouts": len(all_records),
        })

    for family, troop_name in best_single_names.items():
        fam_records = records_by_family[family]
        mean, clear_rate = _evaluate_records(fam_records, "BestSingleTroopAgent", troop_name=troop_name, rng_seed=4321)
        rows.append({
            "agent": "BestSingleTroopAgent",
            "family": family,
            "mean_destruction": round(mean, 4),
            "full_clear_rate": round(clear_rate, 4),
            "layouts": len(fam_records),
            "best_troop": troop_name,
        })

    overall_best_single = dev_records + transfer_records
    best_troop_overall = max([spec.name for spec in TROOP_TYPES], key=lambda name: sum(best_single_agent_score(_layout_for_record(r), name) for r in overall_best_single) / len(overall_best_single))
    mean, clear_rate = _evaluate_records(overall_best_single, "BestSingleTroopAgent", troop_name=best_troop_overall, rng_seed=999)
    rows.append({
        "agent": "BestSingleTroopAgent",
        "family": "overall",
        "mean_destruction": round(mean, 4),
        "full_clear_rate": round(clear_rate, 4),
        "layouts": len(overall_best_single),
        "best_troop": best_troop_overall,
    })

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ORACLE_USAGE_PATH.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "troop", "include_share", "gold_share", "layouts"])
        names = [spec.name for spec in TROOP_TYPES]
        for family, recs in records_by_family.items():
            armies = [oracle[r["id"]][1] for r in recs]
            for t in names:
                inc = float(np.mean([t in a for a in armies]))
                gold = float(np.mean([sum(_COSTS[x] for x in a if x == t) / max(1, sum(_COSTS[x] for x in a)) for a in armies]))
                w.writerow([family, t, round(inc, 4), round(gold, 4), len(recs)])
    with RESULTS_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["agent", "family", "mean_destruction", "full_clear_rate", "layouts", "best_troop"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    rows = run_baselines()
    print(f"Saved {len(rows)} baseline rows to {RESULTS_PATH}")
    for row in rows:
        print(row)
