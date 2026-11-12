"""STEP 6b diagnostic 1: deploy-only, single-troop armies, trained policy vs scripted best-single deployment (validation layouts only)."""
import sys
import numpy as np
from sb3_contrib import MaskablePPO
from rl.evaluation import TROOP_NAMES, evaluate_policy
from rl.split import load_split
from siege.armies import best_affordable_army
from siege.baselines import _layout_for_record, best_single_agent_score
from siege.layouts import FAMILIAR_FAMILIES

run = sys.argv[1] if len(sys.argv) > 1 else "sanity_none_s0"
_, val = load_split()
model = MaskablePPO.load(f"models/{run}/best.zip", device="cpu")
print(f"{run}/best.zip, deploy-only single-type armies, 20 val layouts (mean destruction)")
print(f"{'troop':<8}{'n':>3}" + "".join(f"{f[:9]+' PPO':>16}{f[:9]+' script':>16}" for f in FAMILIAR_FAMILIES) + f"{'ALL PPO':>10}{'ALL script':>12}")
for k, name in enumerate(TROOP_NAMES):
    n = len(best_affordable_army(name))
    counts = [0] * 4; counts[k] = n
    rows = evaluate_policy(model, val, prebought=counts)
    line, allp, alls = f"{name:<8}{n:>3}", [], []
    for fam in FAMILIAR_FAMILIES:
        p = [r["destruction"] for r in rows if r["family"] == fam]
        sc = [best_single_agent_score(_layout_for_record(r), name) for r in val if r["family"] == fam]
        line += f"{np.mean(p):>16.3f}{np.mean(sc):>16.3f}"; allp += p; alls += sc
    print(line + f"{np.mean(allp):>10.3f}{np.mean(alls):>12.3f}")
