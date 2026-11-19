"""B11: determinism and resume check on the REAL DDQN config (TUNE-train -> TUNE-val; not a training result).

Two fresh runs with the same seed must be bit-identical (online / target / Adam / replay-buffer hash, validation rows, curriculum log); resuming from a
checkpoint (phase A before learning starts; phase B with learning running; after the first target-net sync) must equal the uninterrupted run; a real
SIGKILL then resume too; negative control: resume without restoring the RNG streams differs.
Usage: python scripts/b11_determinism.py [--threads 1] [--timesteps 12000]. Writes data/v2/b11_determinism.json. Runs go to a scratch dir.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # the repo root is importable when run as `python scripts/<name>.py`

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cli(out, name, extra, threads, timesteps, ckpt):
    cmd = [sys.executable, "-m", "rl.train_ddqn_v2", "--run-name", name, "--timesteps", str(timesteps), "--val-freq", str(ckpt), "--checkpoint-freq", str(ckpt),
           "--threads", str(threads), *extra]
    return cmd, dict(os.environ, PYTHONPATH=str(ROOT), BREACHLENS_MODELS_DIR=str(out))


def run(out, name, extra, threads, timesteps, ckpt, expect_fail=False):
    cmd, env = cli(out, name, extra, threads, timesteps, ckpt)
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if (r.returncode != 0) != expect_fail:
        raise RuntimeError(f"{name}: exit {r.returncode}\n{r.stderr[-2000:]}")
    return r


def summary(d):
    ident = json.loads((d / "model_identity.json").read_text())
    with (d / "val_eval.csv").open(newline="") as fh:
        rows = [{k: v for k, v in r.items() if k != "eval_seconds"} for r in csv.DictReader(fh)]
    return {"params_sha256": ident["params_sha256"], "params_optimizer_sha256": ident["params_optimizer_sha256"], "updates": ident["updates"],
            "timesteps": ident["timesteps_trained"], "val_rows": rows, "val_layouts": (d / "val_layouts.jsonl").read_text(),
            "curriculum": (d / "curriculum.log").read_text(), "resumes": ident.get("resumes", [])}


def same(a, b):
    return {k: a[k] == b[k] for k in ("params_sha256", "params_optimizer_sha256", "updates", "timesteps", "val_rows", "val_layouts", "curriculum")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--timesteps", type=int, default=12_000)
    p.add_argument("--ckpt", type=int, default=2_000)
    a = p.parse_args()
    out = Path(tempfile.mkdtemp(prefix="b11_det_"))
    T, ck, th = a.timesteps, a.ckpt, a.threads
    rec = {"threads": th, "timesteps": T, "checkpoint_freq": ck, "curriculum_boundary": T // 3, "config": "real DDQN config (rl.ddqn.DDQN_CONFIG)",
           "learning_starts": 5000, "first_target_sync_at_timestep": 9000, "cases": {}}
    t0 = time.time()
    run(out, "A1", [], th, T, ck)
    run(out, "A2", [], th, T, ck)
    base = summary(out / "A1")
    case = {"params_sha256": base["params_sha256"], "params_optimizer_sha256": base["params_optimizer_sha256"], "updates": base["updates"],
            "fresh_twice": same(base, summary(out / "A2"))}
    for at, what in ((2000, "phase A, before learning starts"), (6000, "phase B, learning running, buffer part-filled"), (10000, "after the first target-net sync")):
        name = f"R{at}"
        run(out, name, ["--debug-interrupt-at", str(at)], th, T, ck, expect_fail=True)
        assert not (out / name / "final.pt").exists()
        run(out, name, ["--resume"], th, T, ck)
        s = summary(out / name)
        case[f"resume_after_checkpoint_{at}"] = {**same(base, s), "resumed_from": s["resumes"][0]["at_timestep"], "situation": what}
    cmd, env = cli(out, "K", [], th, T, ck)
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ckdir = out / "K" / "checkpoints" / "ckpt_6000"
    while proc.poll() is None and not ckdir.exists():
        time.sleep(0.05)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    killed = sorted(d.name for d in (out / "K" / "checkpoints").iterdir())
    run(out, "K", ["--resume"], th, T, ck)
    s = summary(out / "K")
    case["sigkill_then_resume"] = {**same(base, s), "checkpoints_present_at_kill": killed, "resumed_from": s["resumes"][0]["at_timestep"]}
    run(out, "N", ["--debug-interrupt-at", "6000"], th, T, ck, expect_fail=True)
    run(out, "N", ["--resume", "--debug-skip-rng-restore"], th, T, ck)
    case["negative_control_no_rng_restore"] = {"differs_from_uninterrupted": summary(out / "N")["params_sha256"] != base["params_sha256"]}
    rec["cases"]["ddqn"] = case
    rec["all_identical"] = all(v for k, d in case.items() if isinstance(d, dict) and k != "negative_control_no_rng_restore" for kk, v in d.items() if isinstance(v, bool))
    rec["negative_control_differs"] = case["negative_control_no_rng_restore"]["differs_from_uninterrupted"]
    rec["wall_seconds"] = round(time.time() - t0)
    (ROOT / "data" / "v2" / "b11_determinism.json").write_text(json.dumps(rec, indent=1))
    shutil.rmtree(out, ignore_errors=True)
    print(json.dumps({k: v for k, v in rec.items() if k != "cases"}, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if isinstance(vv, bool)}) for k, v in case.items()}, indent=1))


if __name__ == "__main__":
    main()
