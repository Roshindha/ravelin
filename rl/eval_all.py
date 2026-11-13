"""STEP 8: full evaluation of the 20 PPO grid models (final.zip) on all 241 layouts.

    nohup .venv/bin/python -m rl.eval_all > results/logs/eval_all.log 2>&1 &

Views: every model (including D4-trained ones) is evaluated on the CANONICAL view: a bare ``SiegeEnv``
with no ``D4Wrapper`` (identity symmetry, nothing random). Actions are deterministic (argmax of the masked policy)
for the main results. The sampled-action run is a separate EXPLORATORY probe.

Layout groups (the only ones the evaluator will ever load):
    DEV-train (140) and validation (20): SEEN in training / checkpoint selection -> shown, never headline.
    FINAL-familiar (40) and transfer (41): never seen in training or tuning -> HEADLINE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS = ROOT / "results"
TABLES = RESULTS / "final_eval_tables"

GROUPS = ("DEV-train", "validation", "FINAL-familiar", "transfer")
HEADLINE_GROUPS = ("FINAL-familiar", "transfer")
SEEN_GROUPS = ("DEV-train", "validation")
GROUP_SIZES = {"DEV-train": 140, "validation": 20, "FINAL-familiar": 40, "transfer": 41}
SEEDS = tuple(range(10, 20))
AUGS = ("none", "d4")
N_SAMPLED_ROLLOUTS = 5
N_BOOT = 10000
BOOT_SEED = 8080
RANDOM_BASELINE_ROLLOUTS = 50
EVAL_SEED = 20268  # base for sampled-action torch seeds
TROOPS = ["swarm", "ranger", "tank", "splash"]


# ----------------------------------------------------------------------------------------------
# layout groups
def assign_groups() -> dict[int, str]:
    """{layout id: group}. Uses only the frozen layouts.json + val_split.json."""
    from rl.split import load_split
    from siege.layouts import load_layout_pool

    train, val = load_split()
    out = {r["id"]: "DEV-train" for r in train}
    out.update({r["id"]: "validation" for r in val})
    for r in load_layout_pool(pool="FINAL"):
        assert r["id"] not in out, "FINAL layout also in DEV"
        out[r["id"]] = "transfer" if r["tag"] == "transfer" else "FINAL-familiar"
    return out


def load_group_records(groups: tuple[str, ...] | list[str]) -> list[dict]:
    """Records (with a ``group`` key) for exactly the declared groups, sorted by id. Unknown group -> error."""
    for g in groups:
        if g not in GROUPS:
            raise ValueError(f"unknown layout group {g!r}; declared groups are {GROUPS}")
    from siege.layouts import load_layout_pool

    gmap = assign_groups()
    allowed = set(groups)
    recs = [dict(r, group=gmap[r["id"]]) for r in load_layout_pool() if gmap[r["id"]] in allowed]
    return sorted(recs, key=lambda r: r["id"])


def layouts_sha256() -> str:
    from siege.layouts import LAYOUTS_JSON

    return hashlib.sha256(Path(LAYOUTS_JSON).read_bytes()).hexdigest()


def git_head() -> tuple[str, bool]:
    import subprocess

    h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    return h, dirty


def run_names() -> list[str]:
    return [f"ppo_{aug}_s{s}" for aug in AUGS for s in SEEDS]


def parse_run(name: str) -> tuple[str, int]:
    _, aug, s = name.split("_")
    return aug, int(s[1:])


# ----------------------------------------------------------------------------------------------
# episodes
def rollout(model, env, rec: dict, deterministic: bool) -> dict:
    """One full-task episode on the canonical view (bare SiegeEnv). Returns destruction/army/steps."""
    obs, _ = env.reset(seed=int(rec["id"]), options={"layout_record": rec, "prebought": False})
    done, n = False, 0
    while not done:
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=deterministic)
        obs, _, term, trunc, _ = env.step(int(action))
        done, n = term or trunc, n + 1
    return {"destruction": float(env.destruction), "army": (env.owned + env.deployed).tolist(), "n_steps": n}


def evaluate_records(model, records: list[dict], allowed_ids: set[int], deterministic: bool = True,
                     n_rollouts: int = 1, seed_key: int = 0) -> list[dict]:
    """Evaluate ``model`` on ``records``. Refuses any record whose id is not in ``allowed_ids``.

    Sampled rollouts re-seed torch per (model seed_key, layout id, rollout) so results do not depend on
    chunking / process scheduling.
    """
    import torch

    from siege.env import SiegeEnv

    env = SiegeEnv()  # canonical view: identity symmetry, no D4Wrapper
    rows = []
    for rec in records:
        if int(rec["id"]) not in allowed_ids:
            raise RuntimeError(f"layout {rec['id']} is outside the declared groups")
        for k in range(n_rollouts):
            if not deterministic:
                torch.manual_seed(EVAL_SEED + 1_000_003 * seed_key + 1_009 * int(rec["id"]) + k)
            r = rollout(model, env, rec, deterministic)
            rows.append({"layout_id": int(rec["id"]), "rollout": k, **r})
    return rows


def _task(args: tuple) -> tuple[str, str, list[dict]]:
    run, mode, records, allowed_ids, seed_key = args
    import torch

    torch.set_num_threads(1)
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(MODELS_DIR / run / "final.zip", device="cpu")
    det = mode == "deterministic"
    rows = evaluate_records(model, records, allowed_ids, deterministic=det,
                            n_rollouts=1 if det else N_SAMPLED_ROLLOUTS, seed_key=seed_key)
    return run, mode, rows


def run_policies(runs: list[str], records: list[dict], mode: str, workers: int, chunk: int = 20) -> dict[str, list[dict]]:
    allowed = {int(r["id"]) for r in records}
    tasks = []
    for run in runs:
        _, seed = parse_run(run)
        key = seed * 2 + (run.startswith("ppo_d4"))
        for i in range(0, len(records), chunk):
            tasks.append((run, mode, records[i:i + chunk], allowed, key))
    out: dict[str, list[dict]] = {r: [] for r in runs}
    t0 = time.time()
    with Pool(workers) as pool:
        for n, (run, _, rows) in enumerate(pool.imap_unordered(_task, tasks), 1):
            out[run] += rows
            if n % 20 == 0 or n == len(tasks):
                print(f"[{mode}] {n}/{len(tasks)} chunks done, {time.time() - t0:.0f}s", flush=True)
    for run in runs:
        out[run].sort(key=lambda r: (r["layout_id"], r["rollout"]))
    return out


# ----------------------------------------------------------------------------------------------
# baselines
def best_single_troops(dev_records: list[dict]) -> dict[str, str]:
    """Best fixed single troop per family, chosen on DEV layouts ONLY (train+validation).
    The transfer family has no DEV layouts: it uses the troop that is best over all DEV layouts pooled."""
    from siege.baselines import _layout_for_record, best_single_agent_score

    assert all(r["group"] in SEEN_GROUPS for r in dev_records)
    lay = {r["id"]: _layout_for_record(r) for r in dev_records}
    fam_ids: dict[str, list[int]] = {}
    for r in dev_records:
        fam_ids.setdefault(r["family"], []).append(r["id"])
    out = {}
    for fam, ids in fam_ids.items():
        out[fam] = max(TROOPS, key=lambda t: np.mean([best_single_agent_score(lay[i], t) for i in ids]))
    allids = [r["id"] for r in dev_records]
    out["__pooled__"] = max(TROOPS, key=lambda t: np.mean([best_single_agent_score(lay[i], t) for i in allids]))
    return out


def run_baselines(all_records: list[dict], workers: int) -> tuple[pd.DataFrame, dict]:
    """Per-layout baseline scores on all records; Oracle only on the headline (FINAL) groups."""
    from siege.baselines import (_layout_for_record, best_single_agent_score, greedy_agent_score, oracle_results,
                                 random_agent_score)
    from siege.layouts import TRANSFER_FAMILY

    dev = [r for r in all_records if r["group"] in SEEN_GROUPS]
    best = best_single_troops(dev)
    rows = []
    for r in all_records:
        lay = _layout_for_record(r)
        rng = np.random.default_rng([int(r["id"]), 777])
        rnd = float(np.mean([random_agent_score(lay, rng) for _ in range(RANDOM_BASELINE_ROLLOUTS)]))
        gr = greedy_agent_score(lay, rng)
        troop = best.get(r["family"], best["__pooled__"]) if r["family"] != TRANSFER_FAMILY else best["__pooled__"]
        rows.append({"layout_id": r["id"], "family": r["family"], "group": r["group"], "Random": rnd, "Greedy": gr,
                     "BestSingle": best_single_agent_score(lay, troop), "BestSingle_troop": troop})
    df = pd.DataFrame(rows)
    head = [r for r in all_records if r["group"] in HEADLINE_GROUPS]
    orc = oracle_results(head)  # reused code: 200 random + 200 hill-climb sims per layout
    df["Oracle"] = df["layout_id"].map(lambda i: orc[i][0] if i in orc else np.nan)
    meta = {"best_single_troop": best,
            "oracle_budget": "siege.baselines.oracle_search defaults: 200 random armies + 200 hill-climb refinements "
                             "(~400 simulations / layout), rng seed [layout_id, 4242]; lower bound on the true optimum",
            "random_rollouts_per_layout": RANDOM_BASELINE_ROLLOUTS,
            "oracle_groups": list(HEADLINE_GROUPS)}
    return df, meta


# ----------------------------------------------------------------------------------------------
# statistics
def boot_ci(x, rng: np.random.Generator, n_boot: int = N_BOOT) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(per_model: pd.DataFrame, value_cols: list[str], by: list[str]) -> pd.DataFrame:
    """per_model: one row per (run, aug, seed, *by). Aggregates over learners within each aug x by."""
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for keys, g in per_model.groupby(["aug"] + by, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for c in value_cols:
            x = g[c].to_numpy()
            lo, hi = boot_ci(x, rng)
            rows.append(dict(zip(["aug"] + by, keys), metric=c, n_learners=len(x), mean=x.mean(),
                             std=x.std(ddof=1) if len(x) > 1 else np.nan, ci95_lo=lo, ci95_hi=hi))
    return pd.DataFrame(rows)


def paired_diff(per_model: pd.DataFrame, value_cols: list[str], by: list[str]) -> pd.DataFrame:
    """none - d4 for the same seed, bootstrap over seeds."""
    rng = np.random.default_rng(BOOT_SEED + 1)
    rows = []
    for keys, g in per_model.groupby(by, sort=True) if by else [((), per_model)]:
        keys = keys if isinstance(keys, tuple) else (keys,)
        for c in value_cols:
            p = g.pivot(index="seed", columns="aug", values=c).dropna()
            d = (p["none"] - p["d4"]).to_numpy()
            lo, hi = boot_ci(d, rng)
            rows.append(dict(zip(by, keys), metric=c, n_seeds=len(d), mean_diff_none_minus_d4=d.mean(),
                             ci95_lo=lo, ci95_hi=hi, seeds_none_wins=int((d > 0).sum()), seeds_d4_wins=int((d < 0).sum()),
                             seeds_tied=int((d == 0).sum())))
    return pd.DataFrame(rows)


def per_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """df: one row per model x layout. -> one row per (run, group) and (run, group, family) metrics (family 'ALL' = pooled)."""
    d = df.copy()
    d["full_clear"] = (d["destruction"] >= 0.999).astype(float)
    out = []
    for fam_mode in ("ALL", "family"):
        keys = ["run", "aug", "seed", "group"] + (["family"] if fam_mode == "family" else [])
        g = d.groupby(keys, sort=True)
        t = g.agg(destruction=("destruction", "mean"), full_clear=("full_clear", "mean"), n_layouts=("layout_id", "count"),
                  **{f"army_{t}": (f"army_{t}", "mean") for t in TROOPS}).reset_index()
        if fam_mode == "ALL":
            t["family"] = "ALL"
        out.append(t)
    return pd.concat(out, ignore_index=True)


def add_headline(df: pd.DataFrame) -> pd.DataFrame:
    """Append pseudo-group HEADLINE (= FINAL-familiar + transfer pooled) rows."""
    h = df[df["group"].isin(HEADLINE_GROUPS)].copy()
    h["group"] = "HEADLINE"
    return pd.concat([df, h], ignore_index=True)


# ----------------------------------------------------------------------------------------------
def model_identity_table(runs: list[str], commit_now: str, sha: str) -> pd.DataFrame:
    rows = []
    for run in runs:
        d = json.loads((MODELS_DIR / run / "model_identity.json").read_text())
        ck = MODELS_DIR / run / "final.zip"
        rows.append({"run": run, "algorithm": d["algorithm"], "aug": d["aug"], "seed": d["seed"],
                     "timesteps_trained": d["timesteps_trained"], "n_resumes": len(d.get("resumes", [])),
                     "train_git_commit": d["git_commit"], "train_git_dirty": d["git_dirty"],
                     "eval_git_commit": commit_now, "layouts_sha256_train": d["layouts_json_sha256"],
                     "layouts_sha256_eval": sha, "final_zip_sha256": hashlib.sha256(ck.read_bytes()).hexdigest()})
    return pd.DataFrame(rows)


def to_long(res: dict[str, list[dict]], recs_by_id: dict[int, dict], ident: pd.DataFrame, sha: str, commit: str) -> pd.DataFrame:
    idm = ident.set_index("run")
    rows = []
    for run, rs in res.items():
        aug, seed = parse_run(run)
        for r in rs:
            rec = recs_by_id[r["layout_id"]]
            rows.append({"run": run, "algorithm": "MaskablePPO", "aug": aug, "seed": seed,
                         "timesteps": int(idm.loc[run, "timesteps_trained"]), "commit": commit, "layouts_sha256": sha,
                         "layout_id": r["layout_id"], "family": rec["family"], "group": rec["group"],
                         "rollout": r["rollout"], "destruction": r["destruction"],
                         **{f"army_{t}": a for t, a in zip(TROOPS, r["army"])}, "n_steps": r["n_steps"]})
    return pd.DataFrame(rows)


def _row(df: pd.DataFrame, aug: str, g: str):
    return df[(df.aug == aug) & (df.group == g) & (df.metric == "destruction")].iloc[0]


def make_charts(summary_cond: pd.DataFrame, fam_cond: pd.DataFrame, base: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"none": "#1f77b4", "d4": "#e07b00"}
    lab = {"none": "PPO (no aug)", "d4": "PPO + D4"}
    # 1. destruction by condition with CI (all groups; headline emphasised)
    groups = ["HEADLINE", "FINAL-familiar", "transfer", "validation", "DEV-train"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.35
    for j, aug in enumerate(AUGS):
        m = [_row(summary_cond, aug, g) for g in groups]
        mean = np.array([r["mean"] for r in m]); lo = np.array([r["ci95_lo"] for r in m]); hi = np.array([r["ci95_hi"] for r in m])
        ax.bar(np.arange(len(groups)) + (j - .5) * w, mean, w, yerr=[mean - lo, hi - mean], color=col[aug], label=lab[aug], capsize=3)
    for gi, g in enumerate(groups):
        for name, ls in (("Greedy", "--"), ("Oracle", ":")):
            v = base.get(g, {}).get(name)
            if v is not None and not np.isnan(v):
                ax.hlines(v, gi - .45, gi + .45, color="k", linestyles=ls, lw=1, label=name if gi == 0 or (name == "Oracle" and g == "HEADLINE") else None)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g if g in HEADLINE_GROUPS or g == "HEADLINE" else g + "\n(seen in training)" for g in groups], fontsize=8)
    ax.set_ylabel("mean destruction (95% bootstrap CI over 10 learners)")
    ax.set_title("Destruction by condition (HEADLINE = FINAL-familiar + transfer)")
    h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h)); ax.legend(u.values(), u.keys(), fontsize=8)
    fig.tight_layout(); fig.savefig(out / "destruction_by_condition.png", dpi=150); plt.close(fig)
    # 2. familiar vs transfer
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for j, aug in enumerate(AUGS):
        m = [_row(summary_cond, aug, g) for g in HEADLINE_GROUPS]
        mean = np.array([r["mean"] for r in m]); lo = np.array([r["ci95_lo"] for r in m]); hi = np.array([r["ci95_hi"] for r in m])
        ax.bar(np.arange(2) + (j - .5) * w, mean, w, yerr=[mean - lo, hi - mean], color=col[aug], label=lab[aug], capsize=3)
    ax.set_xticks(range(2)); ax.set_xticklabels(["FINAL-familiar (40)", "transfer: ring_defence (41)"])
    for gi, g in enumerate(HEADLINE_GROUPS):
        for name, ls in (("Greedy", "--"), ("Oracle", ":")):
            ax.hlines(base[g][name], gi - .45, gi + .45, color="k", linestyles=ls, lw=1, label=name)
    h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h)); ax.legend(u.values(), u.keys(), fontsize=8)
    ax.set_ylabel("mean destruction"); ax.set_title("Familiar vs transfer (never-seen layouts)")
    fig.tight_layout(); fig.savefig(out / "familiar_vs_transfer.png", dpi=150); plt.close(fig)
    # 3. army composition (mean counts per troop type, headline families)
    fams = ["clustered_core", "scattered_outposts", "ring_defence"]
    tc = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs, labels = [], []
    for fi, fam in enumerate(fams):
        for j, aug in enumerate(AUGS):
            x = fi * 3 + j
            sub = fam_cond[(fam_cond.aug == aug) & (fam_cond.family == fam)]
            bottom = 0.0
            for t, c in zip(TROOPS, tc):
                v = float(sub[(sub.metric == f"army_{t}")]["mean"].iloc[0])
                ax.bar(x, v, bottom=bottom, color=c, label=t if (fi == 0 and j == 0) else None)
                bottom += v
            xs.append(x); labels.append(f"{aug}")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    for fi, fam in enumerate(fams):
        ax.text(fi * 3 + .5, -4.2, fam, ha="center", fontsize=8)
    ax.set_ylabel("mean troops bought per layout"); ax.set_title("Army bought on FINAL layouts (mean over 10 learners)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.subplots_adjust(bottom=.15)
    fig.savefig(out / "army_composition.png", dpi=150); plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--only-runs", default=None, help="comma list (debug only; do not use for the real run)")
    args = p.parse_args(argv)
    RESULTS.mkdir(exist_ok=True); TABLES.mkdir(exist_ok=True)
    runs = run_names() if not args.only_runs else args.only_runs.split(",")
    commit, dirty = git_head()
    sha = layouts_sha256()
    print(f"commit {commit} dirty={dirty} layouts_sha256={sha}", flush=True)
    assert sha == (RESULTS / "layouts.sha256").read_text().strip()
    assert not dirty, "refusing to run the final evaluation on a dirty tree"

    records = load_group_records(GROUPS)
    sizes = {g: sum(r["group"] == g for r in records) for g in GROUPS}
    assert sizes == GROUP_SIZES and len(records) == 241, sizes
    print("groups:", sizes, flush=True)
    recs_by_id = {r["id"]: r for r in records}
    ident = model_identity_table(runs, commit, sha)
    assert (ident.layouts_sha256_train == sha).all()
    ident.to_csv(TABLES / "model_identity.csv", index=False)

    t0 = time.time()
    det = run_policies(runs, records, "deterministic", args.workers)
    long = to_long(det, recs_by_id, ident, sha, commit)
    long.drop(columns=["rollout"]).to_csv(RESULTS / "final_eval.csv", index=False)
    print(f"deterministic eval done in {time.time() - t0:.0f}s, rows={len(long)}", flush=True)

    # --- baselines
    bl, bmeta = run_baselines(records, args.workers)
    bl.to_csv(TABLES / "baselines_per_layout.csv", index=False)
    print("baselines done", flush=True)

    # --- aggregation
    full = add_headline(long)
    pm = per_model_table(full)
    pm.to_csv(TABLES / "per_model.csv", index=False)
    vc = ["destruction", "full_clear"] + [f"army_{t}" for t in TROOPS]
    cond = aggregate(pm[pm.family == "ALL"], vc, ["group"])
    fam = aggregate(pm[pm.family != "ALL"], vc, ["group", "family"])
    cond.to_csv(TABLES / "condition_summary.csv", index=False)
    fam.to_csv(TABLES / "condition_family_summary.csv", index=False)
    pdiff = paired_diff(pm[pm.family == "ALL"], ["destruction", "full_clear"], ["group"])
    pdiff_f = paired_diff(pm[pm.family != "ALL"], ["destruction", "full_clear"], ["group", "family"])
    pdiff.to_csv(TABLES / "paired_diff_none_minus_d4.csv", index=False)
    pdiff_f.to_csv(TABLES / "paired_diff_none_minus_d4_by_family.csv", index=False)

    bfull = add_headline(bl.drop(columns=["BestSingle_troop"]).rename(columns={"layout_id": "layout_id"}))
    bsum = {}
    brows = []
    for g in ["DEV-train", "validation", "FINAL-familiar", "transfer", "HEADLINE"]:
        sub = bfull[bfull.group == g]
        bsum[g] = {n: float(sub[n].mean()) if sub[n].notna().any() else None for n in ("Random", "Greedy", "BestSingle", "Oracle")}
        for fam_name in ["ALL"] + sorted(sub.family.unique()):
            s2 = sub if fam_name == "ALL" else sub[sub.family == fam_name]
            for n in ("Random", "Greedy", "BestSingle", "Oracle"):
                if s2[n].notna().any():
                    brows.append({"group": g, "family": fam_name, "baseline": n, "mean_destruction": s2[n].mean(),
                                  "full_clear_rate": (s2[n] >= 0.999).mean(), "n_layouts": int(s2[n].notna().sum())})
    pd.DataFrame(brows).to_csv(TABLES / "baselines_summary.csv", index=False)
    base_for_chart = {g: {n: (v if v is not None else np.nan) for n, v in d.items()} for g, d in bsum.items()}

    # --- sampled probe
    t1 = time.time()
    smp = run_policies(runs, records, "sampled", args.workers)
    slong = to_long(smp, recs_by_id, ident, sha, commit)
    # per layout: mean over the 5 rollouts
    keys = ["run", "algorithm", "aug", "seed", "timesteps", "commit", "layouts_sha256", "layout_id", "family", "group"]
    sl = slong.groupby(keys, sort=True).agg(destruction=("destruction", "mean"), n_rollouts=("rollout", "count"),
                                           **{f"army_{t}": (f"army_{t}", "mean") for t in TROOPS}).reset_index()
    sl.to_csv(RESULTS / "final_eval_sampled_EXPLORATORY.csv", index=False)
    spm = per_model_table(add_headline(sl))
    spm = spm[spm.family == "ALL"]
    dm = per_model_table(full)
    dm = dm[dm.family == "ALL"]
    mrg = dm.merge(spm, on=["run", "aug", "seed", "group", "family"], suffixes=("_det", "_smp"))
    mrg["delta_destruction_smp_minus_det"] = mrg["destruction_smp"] - mrg["destruction_det"]
    mrg.to_csv(TABLES / "sampled_probe_per_model_EXPLORATORY.csv", index=False)
    probe = aggregate(mrg.assign(destruction=mrg["delta_destruction_smp_minus_det"]), ["destruction"], ["group"])
    probe["note"] = "EXPLORATORY: sampled minus deterministic mean destruction; 5 sampled rollouts/layout"
    probe.to_csv(TABLES / "sampled_probe_summary_EXPLORATORY.csv", index=False)
    print(f"sampled probe done in {time.time() - t1:.0f}s", flush=True)

    make_charts(cond, fam, base_for_chart, RESULTS)

    def tab(df, **filt):
        return json.loads(df.to_json(orient="records"))

    summary = {
        "commit": commit, "layouts_sha256": sha, "checkpoint": "final.zip", "action_selection": "deterministic",
        "view": "canonical (bare SiegeEnv, identity symmetry; no D4Wrapper) for ALL models incl. D4-trained",
        "groups": GROUP_SIZES, "headline_groups": list(HEADLINE_GROUPS),
        "seen_in_training_groups": list(SEEN_GROUPS),
        "n_bootstrap": N_BOOT, "bootstrap_seed": BOOT_SEED, "ci": "95% percentile bootstrap over learners (seeds)",
        "models": tab(ident), "condition_summary": tab(cond), "condition_family_summary": tab(fam),
        "paired_diff_none_minus_d4": tab(pdiff), "paired_diff_by_family": tab(pdiff_f),
        "baselines": {"group_means": bsum, **bmeta},
        "sampled_probe_EXPLORATORY": {"rollouts_per_layout": N_SAMPLED_ROLLOUTS, "summary": tab(probe)},
        "wall_seconds_total": time.time() - t0,
    }
    (RESULTS / "final_eval_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
