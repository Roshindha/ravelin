"""B8: run Random / Greedy / BestSingle (+ 2 diagnostics) / Oracle on DEV-train, DEV-validation, TUNE-train, TUNE-val ONLY.
No FINAL, transfer or LAB-DESIGN layout is loaded; the ledger is not touched. Writes data/v2/baselines/. One copy only (caffeinate -dimsu)."""
import json
import sys
import time
from pathlib import Path

from breachlens import baselines as B
from breachlens import evaluator as E
from breachlens import oracle as O
from breachlens.config import sim_version
from breachlens.constants import UNIT_TYPES
from breachlens.score import SCORE_VERSION

OUT = Path(__file__).resolve().parent / "data" / "v2" / "baselines"
SETS = (("DEV-train", "train"), ("DEV-validation", "validation"), ("TUNE-train", "train"), ("TUNE-val", "validation"))
TRAIN_OF = {"DEV-train": "DEV-train", "DEV-validation": "DEV-train", "TUNE-train": "TUNE-train", "TUNE-val": "TUNE-train"}
PROCS = 7


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / "b8_oracle_cache.json"
    cache = {}
    if cache_path.exists():
        c = json.loads(cache_path.read_text())
        if c.get("sim_version") == sim_version() and c.get("score_version") == SCORE_VERSION:
            cache = c["episodes"]
    items = {s: E.load_set(s, role) for s, role in SETS}
    role_of = dict(SETS)
    rows, summary, timing = [], [], {}
    single_rows = {s: [] for s, _ in SETS}

    def run(policy, s, label=None):
        t0 = time.perf_counter()
        r = E.evaluate(policy, s, role_of[s], items=items[s], procs=PROCS)
        for x in r["rows"] + r["summary"]:
            if label:
                x["policy"] = label
        timing.setdefault(label or policy.name, {})[s] = round(time.perf_counter() - t0, 2)
        rows.extend(r["rows"]); summary.extend(r["summary"])
        print(f"{label or policy.name:24s} {s:15s} {r['summary'][-1]['breach_mean']:.3f} {r['summary'][-1]['score_mean']:.3f}  {timing[label or policy.name][s]}s", flush=True)
        return r

    for t in range(5):
        for s, _ in SETS:
            single_rows[s] += run(B.SingleUnit(t), s)["rows"]
    choice = {s: B.select_best_single(single_rows[s]) for s, _ in SETS}
    train_choice = {s: choice[TRAIN_OF[s]] for s, _ in SETS}
    for s, _ in SETS:
        run(B.BestSingle(train_choice[s]["by_family"], name="BestSingle"), s)
        run(B.BestSingle(choice[s]["by_family"], name="BestSingle-hindsight"), s)
        run(B.BestSinglePerLayout(), s)
    for s, _ in SETS:
        run(B.Greedy(), s)
    for s, _ in SETS:
        run(B.RandomPolicy(), s)
    new_cache = dict(cache)
    for s, _ in SETS:
        r = run(B.Oracle(cache=cache), s)
        for k, eps in r["episodes"].items():
            new_cache[k] = {x: v for x, v in eps[0].items() if x != "cached"}
    meta = E.run_meta(E.EVAL_SEED, "mixed (train / validation per set)", "DEV-train, DEV-validation, TUNE-train, TUNE-val")
    # per-row meta already carries role / set; this file-level record is informational
    wall = {p: round(sum(v.values()), 2) for p, v in timing.items()}
    E.write_csv(rows, OUT / "b8_layout_rows.csv")
    E.write_csv(summary, OUT / "b8_summary.csv")
    E.write_json(summary, OUT / "b8_summary.json")
    (OUT / "b8_best_single.json").write_text(json.dumps({"rule": "highest mean score per family on the matching training split; ties -> lowest type index",
        "types": list(UNIT_TYPES), "train_chosen_for_set": {s: train_choice[s]["by_family"] for s, _ in SETS},
        "hindsight_for_set": {s: choice[s]["by_family"] for s, _ in SETS}, "means_on_training_split": {s: choice[s]["means"] for s in ("DEV-train", "TUNE-train")},
        "pooled_dev_train_type_for_labyrinth_transfer_B15": choice["DEV-train"]["pooled"], "pooled_means_dev_train": choice["DEV-train"]["pooled_means"]}, indent=1))
    (OUT / "b8_timing.json").write_text(json.dumps({"wall_seconds_per_baseline": wall, "wall_seconds_per_baseline_and_set": timing, "procs": PROCS,
                                                    "oracle_cache_used_entries": len(cache)}, indent=1))
    cache_path.write_text(json.dumps({"sim_version": sim_version(), "score_version": SCORE_VERSION, "episodes": new_cache}, sort_keys=True))
    (OUT / "b8_run_meta.json").write_text(json.dumps(meta, indent=1))
    print("wall per baseline", wall)


if __name__ == "__main__":
    sys.exit(main())
