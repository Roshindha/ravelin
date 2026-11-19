"""B14c item 1: integrity audit of the finished 30-run grid -> results/b14/grid_audit.md (+ .json). Read-only: no training, no FINAL access.

Checks per run (all 30): identity finished at exactly T = 1 500 000; zero resumes (identity `resumes` list AND exactly one scheduler start in the run log);
the manifest checker re-run (`scripts_b14_grid.check_job_outputs`, i.e. `breachlens.manifest.check_manifest_file` + val ids inside the manifest);
one git commit in every identity file and every validation row, git_dirty False everywhere; the frozen config files hash to the shas of row (ee) and every
identity's hyperparameters equal the frozen file; shaping on; final model present with its sha256. Also: ledger 5 build / 0 eval, 61 validation evals per run.
Any deviation is listed and the exit code is 1.

    .venv/bin/python scripts/b14_audit.py
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # the repo root is importable when run as `python scripts/<name>.py`

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

from scripts import b14_grid as G
from breachlens import pool as P

ROOT = G.ROOT
MODELS = ROOT / "models_v2"
LOGS = ROOT / "logs_b14"
OUT = ROOT / "results" / "b14"
SPEC = ROOT / "docs" / "SPEC_V2.md"
T = 1_500_000
VAL_T = list(range(0, T + 1, 25_000))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def row_ee_shas() -> dict:
    row = [ln for ln in SPEC.read_text().splitlines() if ln.startswith("| 2026-11-15 | (ee)")][0]
    return {"ppo_plain": re.search(r"PPO plain `data/v2/grid_config_ppo_plain.json` = \*\*([0-9a-f]{64})", row).group(1),
            "ppo_d4": re.search(r"PPO\+D4 `data/v2/grid_config_ppo_d4.json` = \*\*([0-9a-f]{64})", row).group(1),
            "ddqn": re.search(r"DDQN `data/v2/grid_config_ddqn.json` = \*\*([0-9a-f]{64})", row).group(1),
            "queue": re.search(r"`data/v2/grid_queue.json` = \*\*([0-9a-f]{64})", row).group(1)}


def run(write: bool = True) -> tuple:
    """The audit itself. write=False (B15 precondition): nothing under results/ is touched, so the tracked audit file keeps its pinned sha256."""
    cfgs = {c: G.load_config(c) for c in G.CONFIG_FILES}
    shas, want = G.config_shas(), row_ee_shas()
    dev = []                                           # deviations
    for k in want:
        if shas[k] != want[k]:
            dev.append(f"config file {k}: sha {shas[k]} != row (ee) {want[k]}")
    problems = G.configs_vs_code(cfgs, G.load_queue())
    dev += [f"configs vs code: {p}" for p in problems]
    jobs = G.job_list(cfgs)
    runs, commits = [], set()
    for j in jobs:
        d = MODELS / j.name
        ident = json.loads((d / "model_identity.json").read_text())
        chk = G.check_job_outputs(MODELS, j, T)                                   # the checker, re-run now
        cfg = cfgs[j.cond]
        final = d / G.FINAL_FILE[j.algo]
        rows = list(csv.DictReader(open(d / "val_eval.csv")))
        ts = sorted({int(r["timestep"]) for r in rows})
        row_commits = {r["git_commit"] for r in rows}
        row_dirty = {r["git_dirty"] for r in rows}
        hp_file = cfg["ppo"] if j.algo == "ppo" else cfg["ddqn"]
        hp_id = {k: v for k, v in ident["hyperparameters"].items() if k != "net_arch"}
        hp_file_cmp = {k: v for k, v in hp_file.items() if k != "net_arch"}
        starts = sum(1 for ln in (LOGS / f"{j.name}.log").read_text().splitlines() if "start (attempt" in ln)
        res = json.loads((LOGS / f"{j.name}.result.json").read_text()) if (LOGS / f"{j.name}.result.json").exists() else {}
        r = {"run": j.name, "cond": j.cond, "seed": j.seed, "status": ident.get("status"), "timesteps_trained": ident.get("timesteps_trained"),
             "resumes_identity": len(ident.get("resumes", [])), "scheduler_starts": starts, "checker_ok": chk["ok"], "manifest_ok": chk["manifest_ok"],
             "identity_commit": ident["git_commit"], "row_commits": sorted(row_commits), "identity_dirty": ident["git_dirty"], "row_dirty": sorted(row_dirty),
             "shaping": ident.get("shaping"), "aug": ident["aug"], "seed_identity": ident["seed"], "n_val_evals": len(ts), "val_t_ok": ts == VAL_T,
             "hp_equal_frozen_file": all(hp_id.get(k) == hp_file_cmp.get(k) for k in hp_file_cmp if k in hp_id), "timesteps_requested": ident.get("timesteps_requested"),
             "phase_b_start": ident.get("curriculum_phase_b_start_timestep"), "final_file": final.name, "final_present": final.exists(),
             "final_sha256": sha256_file(final) if final.exists() else None, "params_sha256": ident.get("params_sha256"),
             "result_exit_code": res.get("exit_code")}
        commits |= {ident["git_commit"], *row_commits}
        bad = []
        if r["status"] != "finished" or r["timesteps_trained"] != T or r["timesteps_requested"] != T:
            bad.append("not finished at exactly T")
        if r["resumes_identity"] != 0 or starts != 1:
            bad.append(f"resumes: identity {r['resumes_identity']}, scheduler starts {starts}")
        if not (r["checker_ok"] and r["manifest_ok"]):
            bad.append(f"checker: {chk}")
        if r["row_commits"] != [r["identity_commit"]]:
            bad.append("validation-row commit differs from identity commit")
        if r["identity_dirty"] is not False or r["row_dirty"] != ["False"]:
            bad.append("git_dirty not False")
        if r["shaping"] != "on":
            bad.append("shaping not on")
        if not r["final_present"]:
            bad.append("final model missing")
        if not r["val_t_ok"]:
            bad.append(f"validation evals are not t = 0, 25k, ..., {T}: {len(ts)} distinct")
        if not r["hp_equal_frozen_file"]:
            bad.append("identity hyperparameters differ from the frozen config file")
        if (r["aug"], r["seed_identity"], r["phase_b_start"]) != (cfg["aug"], j.seed, T // 3):
            bad.append("aug / seed / phase boundary differ from the frozen config")
        if r["result_exit_code"] != 0:
            bad.append("scheduler result exit code not 0")
        r["deviations"] = bad
        dev += [f"{j.name}: {b}" for b in bad]
        runs.append(r)
    if len(commits) != 1:
        dev.append(f"more than one git commit across identity files and validation rows: {sorted(commits)}")
    led = P.read_ledger()
    n_eval = sum(1 for e in led if e.get("purpose") == "eval")
    n_build = sum(1 for e in led if e.get("purpose") == "build")
    if (n_build, n_eval) != (5, 0) or P.count_eval_sessions(led) != 0:
        dev.append(f"ledger is {n_build} build / {n_eval} eval (expected 5 / 0)")
    head = P._git("rev-parse", "HEAD")
    out = {"n_runs": len(runs), "T": T, "commit_set": sorted(commits), "head_at_audit": head, "ledger": {"build": n_build, "eval": n_eval},
           "config_shas": shas, "row_ee_shas": want, "deviations": dev, "runs": runs}
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "grid_audit.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    L = ["# B14c item 1: grid integrity audit", "",
         f"Verdict: **{'PASS: no deviations' if not dev else 'FAIL: ' + str(len(dev)) + ' deviation(s)'}**", "",
         f"- Runs audited: {len(runs)} (PPO plain x10, PPO+D4 x10, DDQN x10), T = {T:,}; produced by `scripts/b14_audit.py` (read-only; no training, no FINAL access).",
         f"- Git commit in every identity file and every validation row: {', '.join(f'`{c}`' for c in sorted(commits))} ({len(commits)} distinct); HEAD at audit time `{head}`.",
         f"- Ledger `data/v2/final_access.log`: {n_build} build / {n_eval} eval entries.",
         "- Config files vs Gate changelog row (ee): " + "; ".join(f"{k} {'ok' if shas[k] == want[k] else 'MISMATCH'}" for k in want) + ".",
         "- Manifest checker re-run on every run (`check_job_outputs`: `check_manifest_file`, val ids inside the manifest, final model, retention, T): "
         f"{sum(r['checker_ok'] for r in runs)}/{len(runs)} ok.",
         f"- Resumes: identity `resumes` list empty and exactly one scheduler start in the run log for {sum(r['resumes_identity'] == 0 and r['scheduler_starts'] == 1 for r in runs)}/{len(runs)} runs.",
         f"- git_dirty False in identity and in all validation rows: {sum(r['identity_dirty'] is False and r['row_dirty'] == ['False'] for r in runs)}/{len(runs)}; shaping on: {sum(r['shaping'] == 'on' for r in runs)}/{len(runs)}.",
         f"- Validation evals at t = 0, 25k, ..., 1.5M ({len(VAL_T)} each): {sum(r['val_t_ok'] for r in runs)}/{len(runs)}.", "",
         "## Deviations", ""] + ([f"- {d}" for d in dev] if dev else ["None."]) + ["", "## Per run", "",
         "| run | status | t trained | resumes | starts | checker | commit | dirty | shaping | final model | sha256 of final model |", "|---|---|---:|---:|---:|---|---|---|---|---|---|"]
    for r in runs:
        L.append(f"| {r['run']} | {r['status']} | {r['timesteps_trained']:,} | {r['resumes_identity']} | {r['scheduler_starts']} | {'ok' if r['checker_ok'] else 'FAIL'} | "
                 f"`{r['identity_commit'][:7]}` | {r['identity_dirty']} | {r['shaping']} | {r['final_file']} | `{r['final_sha256']}` |")
    if write:
        (OUT / "grid_audit.md").write_text("\n".join(L) + "\n")
    return out, L, dev


def main() -> int:
    out, L, dev = run(write=True)
    print("\n".join(L[:20]))
    return 1 if dev else 0


if __name__ == "__main__":
    sys.exit(main())
