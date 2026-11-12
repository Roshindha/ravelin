import numpy as np, pandas as pd
from sb3_contrib import MaskablePPO
from rl.evaluation import TROOP_NAMES, evaluate_policy
from rl.split import load_split
_, val = load_split()
res = {}
for cfg in "ABC":
    sc, rows = [], []
    for sd in (0, 1):
        d = f"models/tune_{cfg}_s{sd}"
        v = pd.read_csv(f"{d}/val_eval.csv"); v = v[v["mode"] == "full"]
        last3 = v.sort_values("timestep").tail(3)["overall"].mean(); sc.append(last3)
        p = pd.read_csv(f"{d}/progress.csv")
        rows.append((sd, last3, p["train/clip_fraction"].mean(), p["train/clip_fraction"].tail(10).mean(), p["train/approx_kl"].mean(), p["train/approx_kl"].tail(10).mean(), p["train/entropy_loss"].tail(5).mean()))
        ev = evaluate_policy(MaskablePPO.load(f"{d}/final.zip", device="cpu"), val)
        for fam in ("clustered_core", "scattered_outposts"):
            a = np.mean([r["army"] for r in ev if r["family"] == fam], axis=0)
            print(f"  {cfg} s{sd} {fam:<19} army {np.round(a,1).tolist()}  val={np.mean([r['destruction'] for r in ev if r['family']==fam]):.3f}")
    res[cfg] = np.mean(sc)
    print(f"{cfg}: per-seed last3 = {[round(x,3) for x in sc]}  MEAN={res[cfg]:.4f}")
    for r in rows: print(f"   seed {r[0]}: clip_frac mean {r[2]:.3f} (last10 {r[3]:.3f}) | approx_kl mean {r[4]:.4f} (last10 {r[5]:.4f}) | entropy_loss(last5) {r[6]:.2f}")
best = max(res, key=res.get)
print("best:", best, "margin over A:", round(res[best] - res["A"], 4), "-> ADOPT" if best != "A" and res[best] - res["A"] >= 0.03 else "-> KEEP A")
