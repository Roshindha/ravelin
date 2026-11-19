"""B14c item 2: SEEN-data summary of the finished grid (DEV-train, DEV-validation only). Monitoring: nothing is selected, no later choice may depend on it.

Evaluates each of the 30 final models (final.zip / final.pt; deterministic masked argmax, canonical view, no TTA) on DEV-train (150) and DEV-validation (30)
through the B8 evaluator (roles train / validation; FINAL is refused by those roles), writes per-layout rows, a condition table with hierarchical
bootstrap CIs (resample seeds, then layouts within each resampled seed; 10 000 resamples; rng seed 8080), the B8 baselines next to them, and DEV
validation learning curves (every 25k) to results/b14/.   .venv/bin/python scripts/b14_seen.py
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # the repo root is importable when run as `python scripts/<name>.py`

import csv
import sys
from collections import defaultdict

import numpy as np

from scripts import b14_grid as G
from breachlens import evaluator as EV
from breachlens.env import BreachEnv

ROOT = G.ROOT
MODELS, OUT = ROOT / "models_v2", ROOT / "results" / "b14"
SETS = (("DEV-train", "train"), ("DEV-validation", "validation"))
LABEL = {"ppo_plain": "PPO plain", "ppo_d4": "PPO+D4", "ddqn": "DDQN"}
COLOR = {"ppo_plain": "#0072B2", "ppo_d4": "#D55E00", "ddqn": "#009E73"}     # fixed order, colour-blind-safe (Okabe-Ito)
FAMS = ("spine_pods", "archipelago", "bunker_nest")
NRES, BSEED = 10_000, 8080


def hboot(mat: np.ndarray) -> tuple:
    """mat = seeds x layouts. Resample seeds (with replacement), then layouts within each drawn seed; returns (mean, lo95, hi95)."""
    s, n = mat.shape
    rng = np.random.default_rng(BSEED)
    si = rng.integers(0, s, size=(NRES, s))
    li = rng.integers(0, n, size=(NRES, s, n))
    means = mat[si[:, :, None], li].mean(axis=(1, 2))
    return float(mat.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_policy(job, items):
    d = MODELS / job.name
    if job.algo == "ppo":
        from sb3_contrib import MaskablePPO
        from rl.eval_ppo_v2 import ModelPolicy
        return ModelPolicy(MaskablePPO.load(d / "final.zip", device="cpu"), name=job.name)
    import torch as th
    from rl import ddqn as D
    ck = th.load(d / "final.pt", map_location="cpu", weights_only=False)
    space = BreachEnv(layouts=[items[0][1]], role="train").observation_space
    net = D.QNet(space, ck["n_actions"], ck["hyperparameters"]["net_arch"])
    net.load_state_dict(ck["online"])
    return D.DDQNPolicy(net, name=job.name)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfgs = {c: G.load_config(c) for c in G.CONFIG_FILES}
    jobs = G.job_list(cfgs)
    rows = []
    for setname, role in SETS:
        items = EV.load_set(setname, role)
        for j in jobs:
            res = EV.evaluate(load_policy(j, items), setname, role, items=items)
            for r in res["rows"]:
                rows.append({"run": j.name, "cond": j.cond, "seed": j.seed, "set": setname, "id": r["id"], "family": r["family"], "score": r["score"],
                             "breach": r["breach"], "d": r["d"], "n_bought": r["n_bought"]})
        print("evaluated", setname, flush=True)
    with open(OUT / "seen_layout_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # control: DEV-validation per-family means at T equal the trainer's logged t = 1.5M validation eval (max abs diff)
    diffs = []
    for j in jobs:
        for lr in csv.DictReader(open(MODELS / j.name / "val_eval.csv")):
            if int(lr["timestep"]) == 1_500_000 and lr["group"] in FAMS:
                mine = [r["score"] for r in rows if r["run"] == j.name and r["set"] == "DEV-validation" and r["family"] == lr["group"]]
                diffs.append(abs(float(np.mean(mine)) - float(lr["score_mean"])))
    ctrl = max(diffs)
    # tables
    def mat(cond, setname, fam, key):
        ids = sorted({r["id"] for r in rows if r["set"] == setname and (fam == "ALL" or r["family"] == fam)})
        d = {(r["run"], r["id"]): r[key] for r in rows if r["set"] == setname}
        runs = [j.name for j in jobs if j.cond == cond]
        return np.array([[d[(rn, i)] for i in ids] for rn in runs])
    base = {(r["policy"], r["set"], r["group"]): r for r in csv.DictReader(open(ROOT / "data/v2/baselines/b8_summary.csv"))}
    out_rows, md = [], []
    for setname, _ in SETS:
        tag = "DEV-train (150 layouts; in-sample, trained on)" if setname == "DEV-train" else "DEV-validation (30 layouts; monitored every 25k, never used to select)"
        md += [f"### {tag}", "", "| row | breach rate [95% CI] | mean score [95% CI] | mean D |", "|---|---|---|---|"]
        for fam in ("ALL", *FAMS):
            if fam != "ALL":
                md.append(f"| *{fam}* | | | |")
            for c in LABEL:
                b, s, d = hboot(mat(c, setname, fam, "breach")), hboot(mat(c, setname, fam, "score")), float(mat(c, setname, fam, "d").mean())
                md.append(f"| {LABEL[c]} ({'ALL' if fam == 'ALL' else fam}) | {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}] | {s[0]:.4f} [{s[1]:.4f}, {s[2]:.4f}] | {d:.3f} |")
                out_rows.append({"set": setname, "group": fam, "condition": LABEL[c], "breach": b[0], "breach_lo95": b[1], "breach_hi95": b[2],
                                 "score": s[0], "score_lo95": s[1], "score_hi95": s[2], "d": d, "n_seeds": 10})
            for pol in ("Random", "Greedy", "BestSingle", "BestSingle-perlayout", "Oracle"):
                r = base[(pol, setname, fam)]
                md.append(f"| {pol} ({fam}) | {float(r['breach_mean']):.3f} [{float(r['breach_lo95']):.3f}, {float(r['breach_hi95']):.3f}] | "
                          f"{float(r['score_mean']):.4f} [{float(r['score_lo95']):.4f}, {float(r['score_hi95']):.4f}] | {float(r['d_mean']):.3f} |")
                out_rows.append({"set": setname, "group": fam, "condition": pol + " (B8, CI over layouts)", "breach": float(r["breach_mean"]), "breach_lo95": float(r["breach_lo95"]),
                                 "breach_hi95": float(r["breach_hi95"]), "score": float(r["score_mean"]), "score_lo95": float(r["score_lo95"]),
                                 "score_hi95": float(r["score_hi95"]), "d": float(r["d_mean"]), "n_seeds": 0})
        md.append("")
    with open(OUT / "seen_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    (OUT / "seen_summary.md").write_text(
        "# B14c item 2: SEEN-data summary (DEV only; seen, never headline)\n\nMonitoring only: nothing is selected, and no number here changes any later choice. Final models at T = 1.5M, "
        "deterministic masked argmax, canonical view, no TTA. Learner CIs: hierarchical bootstrap (resample the 10 seeds, then layouts within each drawn seed), 10 000 resamples, rng seed 8080. "
        "Baseline CIs are the B8 ones (bootstrap over layouts). BestSingle = Sniper in every family (chosen on DEV-train; DEV-train row is in-sample).\n\n"
        f"Control: per-family DEV-validation means recomputed here vs the trainers' logged t = 1.5M validation eval: max abs diff {ctrl:.2e}.\n\n" + "\n".join(md))
    curves(jobs)
    print(f"control max diff {ctrl:.3e}")
    return 0


def curves(jobs) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = {(r["policy"], r["set"], r["group"]): float(r["score_mean"]) for r in csv.DictReader(open(ROOT / "data/v2/baselines/b8_summary.csv"))}
    base_b = {(r["policy"], r["set"], r["group"]): float(r["breach_mean"]) for r in csv.DictReader(open(ROOT / "data/v2/baselines/b8_summary.csv"))}
    cur = defaultdict(lambda: defaultdict(dict))       # cond -> seed -> t -> (score, breach), equal-weight mean of the 3 families (10 layouts each)
    for j in jobs:
        acc = defaultdict(list)
        for lr in csv.DictReader(open(MODELS / j.name / "val_eval.csv")):
            if lr["group"] in FAMS:
                acc[int(lr["timestep"])].append((float(lr["score_mean"]), float(lr["breach_mean"])))
        for t, v in acc.items():
            cur[j.cond][j.seed][t] = tuple(np.mean(v, axis=0))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for k, (ax, ttl) in enumerate(zip(axs, ("Mean true score", "Breach rate"))):
        for c in LABEL:
            ts = sorted(cur[c][0])
            m = np.array([[cur[c][s][t][k] for t in ts] for s in range(10)])
            ax.fill_between(np.array(ts) / 1e6, m.min(0), m.max(0), color=COLOR[c], alpha=0.13, lw=0)
            ax.plot(np.array(ts) / 1e6, m.mean(0), color=COLOR[c], lw=2, label=f"{LABEL[c]} (mean of 10 seeds; band = min-max)")
        for pol, ls in (("BestSingle", "--"), ("Greedy", ":")):
            v = (base if k == 0 else base_b)[(pol, "DEV-validation", "ALL")]
            ax.axhline(v, color="#555555", lw=1.2, ls=ls, label=f"{pol} (B8, DEV-validation)")
        ax.axvline(0.5, color="#999999", lw=0.8)
        ax.set_xlabel("environment steps (millions); vertical line = end of phase A")
        ax.set_title(f"DEV-validation (30 layouts, seen): {ttl}")
        ax.grid(alpha=0.2)
    axs[0].legend(fontsize=7.5, loc="lower right")
    fig.savefig(OUT / "dev_validation_curves.png", dpi=140)
    plt.close(fig)
    # per-seed lines
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True, constrained_layout=True)
    for ax, c in zip(axs, LABEL):
        for s in range(10):
            ts = sorted(cur[c][s])
            ax.plot(np.array(ts) / 1e6, [cur[c][s][t][0] for t in ts], color=COLOR[c], lw=1, alpha=0.7)
        ax.axhline(base[("BestSingle", "DEV-validation", "ALL")], color="#555555", lw=1.2, ls="--")
        ax.set_title(f"{LABEL[c]}: 10 seeds, DEV-validation score")
        ax.set_xlabel("steps (millions)")
        ax.grid(alpha=0.2)
    fig.savefig(OUT / "dev_validation_curves_per_seed.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
