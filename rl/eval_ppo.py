"""Evaluate a trained run on the VALIDATION layouts only (DEV-familiar, 10/family) and compare to baselines.

    .venv/bin/python -m rl.eval_ppo --run-name sanity_none_s0 [--ckpt final|best]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from rl.evaluation import TROOP_NAMES, evaluate_policy, summarize
from rl.split import load_split
from siege.baselines import (_layout_for_record, best_single_agent_score, greedy_agent_score, oracle_results, random_agent_score)
from siege.layouts import FAMILIAR_FAMILIES

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
VAL_BASELINES = Path(__file__).resolve().parent.parent / "results" / "val_baselines.json"


def val_baselines(train: list[dict], val: list[dict]) -> dict:
    """Random / Greedy / per-family best single troop / Oracle, scored on the validation layouts."""
    if VAL_BASELINES.exists():
        return json.loads(VAL_BASELINES.read_text())
    out: dict = {"best_single_troop": {}}
    rows: dict[str, dict[str, list[float]]] = {k: {f: [] for f in FAMILIAR_FAMILIES} for k in ("Random", "Greedy", "BestSingle", "Oracle")}
    orc = oracle_results(val)
    for fam in FAMILIAR_FAMILIES:
        tr = [r for r in train if r["family"] == fam]
        best = max(TROOP_NAMES, key=lambda n: np.mean([best_single_agent_score(_layout_for_record(r), n) for r in tr]))  # picked on TRAIN layouts
        out["best_single_troop"][fam] = best
        for r in [v for v in val if v["family"] == fam]:
            lay = _layout_for_record(r)
            rng = np.random.default_rng(r["id"])
            rows["Random"][fam].append(float(np.mean([random_agent_score(lay, rng) for _ in range(50)])))
            rows["Greedy"][fam].append(greedy_agent_score(lay, rng))
            rows["BestSingle"][fam].append(best_single_agent_score(lay, best))
            rows["Oracle"][fam].append(orc[r["id"]][0])
    for k, per in rows.items():
        out[k] = {**{f: float(np.mean(v)) for f, v in per.items()}, "overall": float(np.mean([x for v in per.values() for x in v]))}
    VAL_BASELINES.write_text(json.dumps(out, indent=2))
    return out


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--ckpt", default="final")
    args = p.parse_args(argv)
    train, val = load_split()
    model = MaskablePPO.load(MODELS_DIR / args.run_name / f"{args.ckpt}.zip", device="cpu")
    rows = evaluate_policy(model, val)
    s = summarize(rows)
    base = val_baselines(train, val)

    print(f"\n== {args.run_name} [{args.ckpt}] on {len(val)} validation layouts (deterministic) ==")
    print(f"{'agent':<22}" + "".join(f"{f:>20}" for f in (*FAMILIAR_FAMILIES, "overall")))
    print(f"{'PPO':<22}" + "".join(f"{s[f]:>20.3f}" for f in FAMILIAR_FAMILIES) + f"{s['overall']:>20.3f}")
    for k in ("Random", "Greedy", "BestSingle", "Oracle"):
        label = k if k != "BestSingle" else "BestSingle(" + "/".join(base["best_single_troop"][f] for f in FAMILIAR_FAMILIES) + ")"
        print(f"{label:<22}" + "".join(f"{base[k][f]:>20.3f}" for f in FAMILIAR_FAMILIES) + f"{base[k]['overall']:>20.3f}")
    print(f"PPO full-clear rate: " + ", ".join(f"{f} {s[f + '_clear']:.2f}" for f in FAMILIAR_FAMILIES))
    print("\nArmy bought (mean count per layout; troops =", TROOP_NAMES, ")")
    for fam in (*FAMILIAR_FAMILIES, None):
        sub = [r for r in rows if fam is None or r["family"] == fam]
        print(f"  {fam or 'all':<20}", np.round(np.mean([r['army'] for r in sub], axis=0), 2).tolist())
    gold = np.array([[sum(c * sp for c, sp in zip(r["army"], (10, 25, 30, 40)))] for r in rows]).mean()
    print(f"  mean gold spent: {gold:.0f}/200")
    from collections import Counter
    comps = Counter(tuple(r["army"]) for r in rows)
    print(f"  distinct armies bought: {len(comps)} / {len(rows)} layouts; most common {comps.most_common(3)}")
    res = {"run": args.run_name, "ckpt": args.ckpt, "ppo": s, "baselines": base, "n_distinct_armies": len(comps)}
    (MODELS_DIR / args.run_name / f"val_report_{args.ckpt}.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
