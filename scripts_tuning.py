"""STEP 5c tuning harness. Runs ONLY on the TUNING seed set (disjoint from the v1/v2/v3 pools).

Per family it generates raw, unfiltered layouts from fresh seeds, then reports:
  * mean destruction of the fixed armies (same deployment rule as acceptance),
  * acceptance rate + rejection breakdown under the real acceptance rules,
  * how often each fixed army is the best (ties shared) and how often a splash army strictly wins,
  * the oracle's best armies: share that include each troop type, and mean gold share.
Never touches the FINAL pool or results/layouts.json.

Usage: .venv/bin/python scripts_tuning.py [--n 300] [--oracle-n 100] [--tag iter1]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from siege.armies import place_on_border
from siege.baselines import _score_layout, oracle_search
from siege.bases import FAMILIES, generate_base_layout
from siege.layouts import ARMY_NAMES, acceptance_reason, generate_and_score
from siege.troops import TROOP_TYPES

TUNING_SEED_BASE = 9_000_000  # disjoint from v1 (0), v2 (5_000_000) and the v3 pool (see scripts_rebuild_pool)
FAMILY_STRIDE = 10_000_000
SINGLES = [s.name for s in TROOP_TYPES]
NEEDED_DROP = 0.05  # a type is 'needed' in an oracle army if deleting it costs >= 5 points of destruction
BAN_DROP = 0.03  # banning a type 'matters' on a layout if the oracle's best falls by >= 3 points
COST = {s.name: s.cost for s in TROOP_TYPES}


def tuning_seeds(n: int) -> dict[str, list[int]]:
    return {f: [TUNING_SEED_BASE + i * FAMILY_STRIDE + k for k in range(n)] for i, f in enumerate(FAMILIES)}


def _work(args):
    family, seed, do_oracle = args
    layout, scores = generate_and_score(seed, family)
    out = {"family": family, "seed": seed, "scores": scores, "reason": acceptance_reason(layout, scores)}
    if do_oracle:
        score, army, start = oracle_search(layout, np.random.default_rng(seed + 17))
        placed = place_on_border(army, start_idx=start)
        # Ablation: delete every unit of one type (others keep their tiles) and re-fight.
        drop = {}
        for t in SINGLES:
            if t in army:
                rest = [u for u in placed if u["type"] != t]
                drop[t] = score - (_score_layout(rest, layout)[0] if rest else 0.0)
        # Ban test: best army the oracle can find when one type is forbidden (2 seeded searches, take
        # the better, so the comparison is not dominated by search noise). Full search = best of 2 too.
        full = max(score, oracle_search(layout, np.random.default_rng(seed + 18))[0])
        banned = {}
        for t in SINGLES:
            allowed = [x for x in SINGLES if x != t]
            banned[t] = max(oracle_search(layout, np.random.default_rng(seed + 19 + k), allowed=allowed)[0] for k in range(2))
        out["oracle"] = {"score": score, "army": army, "drop": drop, "full": full, "banned": banned}
    return out


def run(n: int, oracle_n: int) -> dict:
    jobs = [(f, s, k < oracle_n) for f, seeds in tuning_seeds(n).items() for k, s in enumerate(seeds)]
    with Pool() as pool:
        rows = pool.map(_work, jobs, chunksize=4)
    report: dict = {"n": n, "oracle_n": oracle_n, "families": {}}
    for family in FAMILIES:
        fr = [r for r in rows if r["family"] == family]
        means = {a: float(np.mean([r["scores"][a] for r in fr])) for a in ARMY_NAMES}
        rej = Counter(r["reason"] for r in fr)
        wins = {a: 0.0 for a in ARMY_NAMES}
        splash_strict = 0
        for r in fr:
            top = max(r["scores"].values())
            tied = [a for a in ARMY_NAMES if r["scores"][a] >= top - 1e-9]
            for a in tied:
                wins[a] += 1 / len(tied)
            sp = max(r["scores"]["splash"], r["scores"]["mixed_breach"])
            other = max(r["scores"][a] for a in ("swarm", "ranger", "tank", "mixed_balanced"))
            splash_strict += sp > other + 1e-9
        orc = [r["oracle"] for r in fr if "oracle" in r]
        include = {t: float(np.mean([t in o["army"] for o in orc])) for t in SINGLES} if orc else {}
        gold = {t: float(np.mean([sum(COST[x] for x in o["army"] if x == t) / max(1, sum(COST[x] for x in o["army"])) for o in orc])) for t in SINGLES} if orc else {}
        loss = {t: float(np.mean([max(0.0, o["full"] - o["banned"][t]) for o in orc])) for t in SINGLES} if orc else {}
        loss_share = {t: float(np.mean([o["full"] - o["banned"][t] >= BAN_DROP for o in orc])) for t in SINGLES} if orc else {}
        needed = {t: float(np.mean([o["drop"].get(t, 0.0) >= NEEDED_DROP for o in orc])) for t in SINGLES} if orc else {}
        report["families"][family] = {
            "oracle_needed_share": needed,
            "oracle_ban_mean_loss": loss,
            "oracle_ban_loss_share": loss_share,
            "mean_destruction": means,
            "accept_rate": float(rej[None] / len(fr)),
            "rejections": {str(k): v for k, v in rej.items() if k is not None},
            "best_army_share": {a: wins[a] / len(fr) for a in ARMY_NAMES},
            "splash_army_strictly_best_share": splash_strict / len(fr),
            "oracle_mean": float(np.mean([o["score"] for o in orc])) if orc else None,
            "oracle_include_share": include,
            "oracle_gold_share": gold,
        }
    return report


def show(rep: dict) -> None:
    print(f"{'family':<20}" + "".join(f"{a:>16}" for a in ARMY_NAMES))
    for f, d in rep["families"].items():
        print(f"{f:<20}" + "".join(f"{d['mean_destruction'][a]:>16.3f}" for a in ARMY_NAMES))
    print("\nacceptance / rejections")
    for f, d in rep["families"].items():
        print(f"  {f:<20} accept={d['accept_rate']:.2f}  {d['rejections']}")
    print("\nshare of layouts where army is best (ties shared); last col = splash army strictly best")
    for f, d in rep["families"].items():
        print(f"  {f:<20}" + "".join(f"{d['best_army_share'][a]:>8.2f}" for a in ARMY_NAMES) + f"   splash-strict={d['splash_army_strictly_best_share']:.2f}")
    print("\noracle: mean score | include-share per type | gold-share per type")
    for f, d in rep["families"].items():
        if d["oracle_mean"] is None:
            continue
        inc = " ".join(f"{t}={d['oracle_include_share'][t]:.2f}" for t in SINGLES)
        gld = " ".join(f"{t}={d['oracle_gold_share'][t]:.2f}" for t in SINGLES)
        print(f"  {f:<20} {d['oracle_mean']:.3f} | {inc} | {gld}")
        print(f"  {'':<20} ban-loss mean: " + " ".join(f"{t}={d['oracle_ban_mean_loss'][t]:.3f}" for t in SINGLES) + " | share of layouts where banning costs >=0.03: " + " ".join(f"{t}={d['oracle_ban_loss_share'][t]:.2f}" for t in SINGLES))
        print(f"  {'':<20} needed (drop>={NEEDED_DROP}): " + " ".join(f"{t}={d['oracle_needed_share'][t]:.2f}" for t in SINGLES))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--oracle-n", type=int, default=100)
    ap.add_argument("--tag", default="adhoc")
    args = ap.parse_args()
    rep = run(args.n, args.oracle_n)
    show(rep)
    out = Path("results") / f"tuning_{args.tag}.json"
    out.write_text(json.dumps(rep, indent=2, sort_keys=True))
    print("saved", out)


if __name__ == "__main__":
    main()
