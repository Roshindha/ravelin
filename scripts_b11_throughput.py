"""B11: end-to-end DDQN throughput benchmark (CPU only; TUNE-train; real config; not a training result). Same windows as B10: 60 s warm-up + 180 s window,
1 torch thread, full game (phase B), TUNE-train, single process and N parallel processes. Run under `caffeinate -dimsu`.
  train     one DDQN job: steps/s over the window, time split (act forward / env.step / masks / gradient updates / rest), peak RSS, replay size; with
            --ckpt-size also writes one checkpoint (replay buffer included) to a temp dir and reports its size.
  parallel  N `train` jobs started together.   assemble  collect data/v2/b11_bench/*.json into data/v2/b11_throughput.json.
The epsilon schedule is the real one for a 300k-step run (eps decays over the first 75k steps), so the early window is a snapshot, as in B10.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "data" / "v2" / "b11_bench"


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def dir_mb(p: Path):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def bench_train(a):
    from rl import ddqn as D
    from rl import train_ddqn_v2 as T
    th.set_num_threads(a.threads)
    hp = D.DDQN_CONFIG | {"net_arch": D.DDQN_CONFIG["net_arch"]}
    venv = T.build_venv("TUNE-train", "none", 0, hp["n_envs"], "random" if a.phase == "A" else None)
    tr = T.Trainer(hp, venv, 0, 300_000, 100_000 if a.phase == "A" else 0)
    tr.timers = {"act": 0.0, "env_step": 0.0, "masks": 0.0, "update": 0.0}
    t0 = time.perf_counter()
    mark0 = None
    while True:
        now = time.perf_counter() - t0
        if mark0 is None and now >= a.warmup:
            mark0 = (time.perf_counter(), tr.t, dict(tr.timers), tr.updates)
        if now >= a.warmup + a.seconds:
            break
        tr.vector_step()
    t1, s1, tm1, u1 = time.perf_counter(), tr.t, dict(tr.timers), tr.updates
    dt, steps = t1 - mark0[0], s1 - mark0[1]
    parts = {k: round(tm1[k] - mark0[2][k], 2) for k in tm1}
    parts["rest"] = round(dt - sum(parts.values()), 2)
    out = {"mode": "train", "phase": a.phase, "threads": th.get_num_threads(), "warmup_s": a.warmup, "window_s": round(dt, 1), "window_steps": steps,
           "steps_per_s": round(steps / dt, 1), "updates_in_window": u1 - mark0[3], "seconds_by_part_in_window": parts,
           "share_by_part": {k: round(v / dt, 3) for k, v in parts.items()}, "peak_rss_mb": round(peak_mb(), 1), "replay_size_at_end": len(tr.buffer),
           "replay_full": tr.buffer.n == tr.buffer.cap, "mean_episode_score_last100": round(float(np.mean(tr.eps_log)), 4) if tr.eps_log else None,
           "total_timesteps_run": tr.t}
    if a.ckpt_size:
        d = Path(tempfile.mkdtemp(prefix="b11_ck_"))
        t = time.perf_counter()
        tr.save_checkpoint(d / "ckpt", {"val_last_eval_t": -1})
        out["checkpoint"] = {"seconds_to_write": round(time.perf_counter() - t, 2), "size_mb": round(dir_mb(d / "ckpt"), 1),
                             "buffer_mb": round(dir_mb(d / "ckpt" / "buffer"), 1), "nets_mb": round((d / "ckpt" / "nets.pt").stat().st_size / 1e6, 1),
                             "state_pkl_mb": round((d / "ckpt" / "state.pkl").stat().st_size / 1e6, 1), "replay_len": len(tr.buffer),
                             "peak_rss_mb_after_save": round(peak_mb(), 1)}
        shutil.rmtree(d, ignore_errors=True)
    return out


def bench_parallel(a):
    procs, outs = [], []
    for i in range(a.n):
        f = BENCH / f"_tmp_par_{os.getpid()}_{i}.json"
        outs.append(f)
        procs.append(subprocess.Popen([sys.executable, __file__, "train", "--phase", a.phase, "--threads", str(a.threads), "--warmup", str(a.warmup),
                                       "--seconds", str(a.seconds), "--out", str(f)], cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT))))
    for p in procs:
        p.wait()
    res = [json.loads(f.read_text()) for f in outs]
    for f in outs:
        f.unlink()
    rates = [r["steps_per_s"] for r in res]
    return {"mode": "parallel", "n": a.n, "phase": a.phase, "threads": a.threads, "per_job_steps_per_s": rates, "per_job_mean": round(float(np.mean(rates)), 1),
            "per_job_min": min(rates), "aggregate_steps_per_s": round(sum(rates), 1), "peak_rss_mb_per_job": [r["peak_rss_mb"] for r in res],
            "share_by_part_job0": res[0]["share_by_part"], "replay_full": [r["replay_full"] for r in res]}


def assemble(a):
    from rl import ddqn as D
    from rl import train_ppo_v2 as P
    get = lambda n: json.loads((BENCH / f"{n}.json").read_text()) if (BENCH / f"{n}.json").exists() else None   # noqa: E731
    rec = {"hardware": P.environment_info(), "ddqn_config": D.DDQN_CONFIG | {"train_set": "TUNE-train"},
           "single": get("single_B"), "single_phase_A": get("single_A"), "parallel": {k: get(k) for k in ("par3_B", "par6_B")}}
    (ROOT / "data" / "v2" / "b11_throughput.json").write_text(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "parallel", "assemble"])
    p.add_argument("--phase", default="B")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--warmup", type=float, default=60)
    p.add_argument("--seconds", type=float, default=180)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--ckpt-size", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    BENCH.mkdir(parents=True, exist_ok=True)
    res = {"train": bench_train, "parallel": bench_parallel, "assemble": assemble}[a.mode](a)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
