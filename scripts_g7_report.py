"""G7 decision report (B13a). Computes the decision from the run logs exactly as pinned in Gate changelog (s), (t), (u) and prints PASS / FAIL per criterion.

    .venv/bin/python scripts_g7_report.py                      # models_v2, H = 300k
    .venv/bin/python scripts_g7_report.py --horizon 600000     # after the extension
Writes logs_b13/g7_decision.json. It never writes data/v2/gates.json: recording the gate result is a separate, explicit step.
D4 runs are printed as non-gating diagnostics; the step-budget slopes of 12.1 (200k-300k) are printed for both conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from breachlens import g7 as G
from breachlens import manifest as MF

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "data" / "v2" / "g7_pilot_config.json"


def frozen() -> dict:
    return json.loads(FROZEN.read_text())["config"]


def config_sha(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def provenance_problems(run_dirs: list, models_dir: Path, smoke: bool) -> list:
    """Manifest checker on every run; identity finished; provenance columns equal across all runs and rows; not dirty."""
    import csv
    problems, commits, shas = [], set(), set()
    for d in run_dirs:
        try:
            MF.check_manifest_file(d / "manifest.json")
        except Exception as e:                                                   # noqa: BLE001 - report every kind of failure
            problems.append(f"{d.name}: manifest: {e}")
        ident = json.loads((d / "model_identity.json").read_text())
        if ident.get("status") != "finished":
            problems.append(f"{d.name}: run not finished (status {ident.get('status')})")
        if ident["val_set"] != "TUNE-val" or ident["train_set"] != "TUNE-train":
            problems.append(f"{d.name}: wrong sets {ident['train_set']} -> {ident['val_set']}")
        if ident["environment"]["torch_threads"] != 1:
            problems.append(f"{d.name}: torch threads {ident['environment']['torch_threads']} != 1")
        if not smoke:
            c = dict(frozen())
            hp = ident["hyperparameters"]
            if {k: hp[k] for k in c["ppo"] if k != "net_arch"} != {k: v for k, v in c["ppo"].items() if k != "net_arch"} or hp["net_arch"] != c["ppo"]["net_arch"]:
                problems.append(f"{d.name}: PPO hyperparameters differ from the frozen config")
            if ident["config"]["phase_a_steps"] != c["curriculum"]["phase_a_steps"]:
                problems.append(f"{d.name}: phase_a_steps differs from the frozen config")
        with (d / "val_eval.csv").open(newline="") as fh:
            for r in csv.DictReader(fh):
                commits.add(r["git_commit"])
                shas.add((r["pool_sha256"], r["generator_hash"], r["score_version"], r["sim_version"]))
                if r["git_dirty"] not in ("False", "false", "0") and not smoke:
                    problems.append(f"{d.name}: validation row written with a dirty tree (t={r['timestep']})")
                    break
                if r["role"] != "validation" or r["set"] != "TUNE-val":
                    problems.append(f"{d.name}: validation row has role {r['role']!r} / set {r['set']!r}")
    if len(commits) > 1:
        problems.append(f"validation rows come from {len(commits)} different git commits: {sorted(c[:8] for c in commits)}")
    if len(shas) > 1:
        problems.append("validation rows carry different pool / generator / score / sim versions")
    return problems


def main(argv=None) -> int:
    cfg = frozen()
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default=str(ROOT / "models_v2"))
    p.add_argument("--horizon", type=int, default=cfg["timesteps"], help="300000, or 600000 after the extension")
    p.add_argument("--freq", type=int, default=cfg["val_freq"])
    p.add_argument("--plain-seeds", default=",".join(map(str, cfg["plain_seeds"])))
    p.add_argument("--d4-seeds", default=",".join(map(str, cfg["d4_seeds"])))
    p.add_argument("--out", default=str(ROOT / "logs_b13" / "g7_decision.json"))
    p.add_argument("--smoke", action="store_true", help="dry-run output: any seed count and step horizon; the verdict is NOT a G7 result")
    a = p.parse_args(argv)
    md = Path(a.models_dir)
    bar = json.loads(G.BAR_PATH.read_text())["g7_bar"]
    ps = [int(x) for x in a.plain_seeds.split(",") if x]
    ds = [int(x) for x in a.d4_seeds.split(",") if x]
    plain_dirs = [md / cfg["run_names"]["plain"].format(seed=s) for s in ps]
    d4_dirs = [md / cfg["run_names"]["d4"].format(seed=s) for s in ds]
    problems = provenance_problems(plain_dirs + d4_dirs, md, a.smoke)
    print(f"G7 report  models={md}  horizon={a.horizon}  bar={bar:.4f}  plain seeds={ps}  d4 seeds={ds}")
    for pr in problems:
        print(f"  PROBLEM  {pr}")
    plain = {s: G.read_run(d) for s, d in zip(ps, plain_dirs)}
    d4 = {s: G.read_run(d) for s, d in zip(ds, d4_dirs)}
    dec = G.decide(plain, bar, a.horizon, a.freq, expect_seeds=None if a.smoke else G.N_PLAIN)
    judged = dec.get("second_look", dec["first_look"])
    print(f"\nG7 look at t = {judged['window']}: mean score {judged['mean_score']:.4f} vs bar {bar:.4f}")
    for s, m in judged["per_seed"].items():
        c = judged["collapse"][s]
        print(f"  seed {s}: score {m['score']:.4f}  D {m['d']:.3f}  breach {m['breach']:.3f}  bought {m['n_bought']:.1f}  distinct armies {c['distinct_armies']}  "
              f"max type share/family {', '.join(f'{k} {v:.2f}' for k, v in c['max_type_share_per_family'].items())}")
    for k, v in judged["criteria"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"  {'PASS' if judged['pass'] else 'FAIL'}  G7 at H={judged['horizon']}")
    if "second_look" in dec:
        f = dec["first_look"]
        print(f"  (300k look: {'PASS' if f['pass'] else 'FAIL'}, mean slope {dec['mean_slope_per_100k']:+.4f} per 100k)")
    else:
        print(f"  slope (200k..300k, per 100k steps, mean over plain seeds) = {dec['mean_slope_per_100k']:+.4f}  threshold +{G.SLOPE_THRESHOLD}  "
              f"{'>=' if dec['slope_ge_threshold'] else '<'}")
    print(f"\nVERDICT: {dec['verdict']}" + ("   [NOT A VALID G7: smoke / seed count]" if (a.smoke or not judged["valid_g7"]) else "") + ("   [PROVENANCE PROBLEMS]" if problems else ""))
    # non-gating diagnostics
    if d4:
        m4 = {s: G.seed_metric(r, a.horizon, a.freq) for s, r in d4.items()}
        print("\nD4 diagnostics (non-gating): " + ", ".join(f"seed {s} {m['score']:.4f}" for s, m in m4.items()) + f"  mean {np.mean([m['score'] for m in m4.values()]):.4f}")
    try:
        t0 = a.horizon if a.horizon <= 300_000 else 300_000
        cond = lambda runs: G.slope_per_100k({"score": {t: float(np.mean([r['score'][t] for r in runs.values()])) for t in G.slope_ts(t0, a.freq)}}, t0, a.freq)
        print(f"12.1 step-budget slopes (condition-mean curve, 200k..300k): plain {cond(plain):+.4f}" + (f"  D4 {cond(d4):+.4f}" if d4 else "") + "  per 100k steps")
    except ValueError as e:
        print(f"12.1 slopes unavailable: {e}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"decision": dec, "problems": problems, "smoke": a.smoke}, indent=1, default=str))
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
