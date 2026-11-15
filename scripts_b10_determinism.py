"""B10: determinism and resume check on the REAL PPO config (TUNE-train -> TUNE-val; not a training result).

For aug in (none, d4): two fresh runs with the same seed must be bit-identical (policy and Adam-state hashes, validation rows, curriculum log);
resuming from a checkpoint (interrupt before / after the curriculum switch) must equal the uninterrupted run; for plain also a real SIGKILL.
Usage: python scripts_b10_determinism.py [--threads N] [--timesteps 10000]. Writes data/v2/b10_determinism.json. Runs go to a scratch dir.
"""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parent


def cli(out: Path, name: str, extra: list[str], threads: int, timesteps: int, ckpt: int):
    cmd = [sys.executable, "-m", "rl.train_ppo_v2", "--run-name", name, "--timesteps", str(timesteps), "--val-freq", str(ckpt), "--checkpoint-freq", str(ckpt),
           "--threads", str(threads), *extra]
    return cmd, dict(os.environ, PYTHONPATH=str(ROOT), BREACHLENS_MODELS_DIR=str(out))


def run(out, name, extra, threads, timesteps, ckpt, expect_fail=False):
    cmd, env = cli(out, name, extra, threads, timesteps, ckpt)
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if (r.returncode != 0) != expect_fail:
        raise RuntimeError(f"{name}: exit {r.returncode}\n{r.stderr[-2000:]}")
    return r


def summary(d: Path) -> dict:
    ident = json.loads((d / "model_identity.json").read_text())
    with (d / "val_eval.csv").open(newline="") as fh:
        rows = [{k: v for k, v in r.items() if k != "eval_seconds"} for r in csv.DictReader(fh)]
    return {"params_sha256": ident["params_sha256"], "params_optimizer_sha256": ident["params_optimizer_sha256"], "timesteps": ident["timesteps_trained"],
            "val_rows": rows, "val_layouts": (d / "val_layouts.jsonl").read_text(), "curriculum": (d / "curriculum.log").read_text(), "resumes": ident.get("resumes", [])}


def same(a: dict, b: dict) -> dict:
    return {k: a[k] == b[k] for k in ("params_sha256", "params_optimizer_sha256", "timesteps", "val_rows", "val_layouts", "curriculum")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--timesteps", type=int, default=10_000)
    p.add_argument("--ckpt", type=int, default=2_500)
    a = p.parse_args()
    out = Path(tempfile.mkdtemp(prefix="b10_det_"))
    T, ck, th = a.timesteps, a.ckpt, a.threads
    rec = {"threads": th, "timesteps": T, "checkpoint_freq": ck, "curriculum_boundary": T // 3, "rollout_size": 2500, "config": "real PPO config (rl.train_ppo_v2.PPO_CONFIG)", "cases": {}}
    t0 = time.time()
    for aug in ("none", "d4"):
        x = ["--aug", aug]
        run(out, f"{aug}_A1", x, th, T, ck)
        run(out, f"{aug}_A2", x, th, T, ck)
        base = summary(out / f"{aug}_A1")
        case = {"params_sha256": base["params_sha256"], "params_optimizer_sha256": base["params_optimizer_sha256"],
                "fresh_twice": same(base, summary(out / f"{aug}_A2"))}
        for at in (2500, 5000):                                     # before / after the curriculum switch at 3333
            name = f"{aug}_R{at}"
            run(out, name, x + ["--debug-interrupt-at", str(at)], th, T, ck, expect_fail=True)
            assert not (out / name / "final.zip").exists()
            run(out, name, ["--resume"], th, T, ck)
            s = summary(out / name)
            case[f"resume_after_checkpoint_{at}"] = {**same(base, s), "resumed_from": s["resumes"][0]["at_timestep"]}
        rec["cases"][aug] = case
    # real SIGKILL (plain): kill -9 as soon as the checkpoint at 5000 exists, then resume from whatever checkpoint is newest
    x = ["--aug", "none"]
    cmd, env = cli(out, "none_K", x, th, T, ck)
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ckdir = out / "none_K" / "checkpoints" / "ckpt_5000"
    while proc.poll() is None and not ckdir.exists():
        time.sleep(0.05)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    killed_at = sorted(d.name for d in (out / "none_K" / "checkpoints").iterdir())
    run(out, "none_K", ["--resume"], th, T, ck)
    s = summary(out / "none_K")
    rec["cases"]["none"]["sigkill_then_resume"] = {**same(summary(out / "none_A1"), s), "checkpoints_present_at_kill": killed_at, "resumed_from": s["resumes"][0]["at_timestep"]}
    rec["all_identical"] = all(v for c in rec["cases"].values() for k, d in c.items() if isinstance(d, dict) for kk, v in d.items() if isinstance(v, bool))
    rec["wall_seconds"] = round(time.time() - t0)
    (ROOT / "data" / "v2" / "b10_determinism.json").write_text(json.dumps(rec, indent=1))
    shutil.rmtree(out, ignore_errors=True)
    print(json.dumps({k: v for k, v in rec.items() if k != "cases"}, indent=1))
    for aug, c in rec["cases"].items():
        print(aug, json.dumps({k: (v if not isinstance(v, dict) else all(x for x in v.values() if isinstance(x, bool))) for k, v in c.items()}))


if __name__ == "__main__":
    main()
