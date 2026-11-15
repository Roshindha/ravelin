"""B13a: mean army size / units deployed of every equal-N control and of a random-init planner on TUNE-val (role tuning). No FINAL, nothing trained.
    caffeinate -dimsu .venv/bin/python scripts_b13a_controls.py   -> data/v2/b13a_control_sizes.json"""
import json
import sys

import numpy as np
import torch as th

from breachlens import evaluator as E
from breachlens import pool as P
from rl import planner as PL
from rl import train_ppo_v2 as T


def main():
    th.set_num_threads(1)
    before = P.LEDGER_PATH.read_bytes()
    items = E.load_set("TUNE-val", "tuning")
    th.manual_seed(0)
    model = T.build_model(T.build_venv("TUNE-train", "none", 0, 1, None), 0, T.PPO_CONFIG)
    out = {"set": "TUNE-val", "n_layouts": len(items), "seeds": [0, 1, 2], "policy": "random-init seeded PPO (not trained)", "rows": {}}
    for m in ("planner", "RAS", "RPS", "RPSfull"):
        agg = {"all_plans_bought": [], "all_plans_deployed": [], **{f"chosen_N{n}_{k}": [] for n in PL.N_VALUES for k in ("bought", "deployed", "score")}}
        for seed in out["seeds"]:
            cells = PL.make_cells(m, seed, probs_fn=PL.ModelProbs(model) if m == "planner" else None)
            for c in cells:
                res = E.evaluate(c, "TUNE-val", "tuning", items=items)
                s = res["summary"][-1]
                agg[f"chosen_N{c.N}_bought"].append(s["mean_n_bought"])
                agg[f"chosen_N{c.N}_deployed"].append(s["mean_n_deployed"])
                agg[f"chosen_N{c.N}_score"].append(s["score_mean"])
            for ps in cells[0].cache.plans.values():
                agg["all_plans_bought"] += [p["n_bought"] for p in ps]
                agg["all_plans_deployed"] += [p["n_deployed"] for p in ps]
        out["rows"][PL.METHOD_LABEL[m]] = {k: round(float(np.mean(v)), 3) for k, v in agg.items()}
    assert P.LEDGER_PATH.read_bytes() == before
    json.dump(out, open("data/v2/b13a_control_sizes.json", "w"), indent=1, sort_keys=True)
    for k, v in out["rows"].items():
        print(k, v)


if __name__ == "__main__":
    sys.exit(main())
