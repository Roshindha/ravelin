"""B10: end-to-end PPO throughput benchmark (CPU only; TUNE-train layouts; no FINAL, no pool role beyond "train"). Not a training result.

Modes (one benchmark at a time unless `parallel`; run under `caffeinate -dimsu`):
  train     one PPO job with the real config for `--warmup` + `--seconds` wall seconds; steps/s over the window after warm-up, time split
            (env.step / action masks / policy forward / PPO update / rest), peak RSS, mean episode length.   --aug none|d4  --phase A|B
  parallel  N `train` jobs started together (separate processes), per-job and aggregate steps/s.
  env       env-only: random masked agent through the same VecEnv (Monitor [+ D4Wrapper]); --agent uniform|buy_all.
  infer     inference-only: batched policy forward on recorded real observations (n_envs rows per call).
  assemble  collect the raw files in data/v2/b10_bench/ into data/v2/b10_throughput.json.
Phase A = every episode starts in DEPLOY with a random legal army; phase B = full BUY -> DEPLOY game (the policy changes while it trains, so a
phase-B window is a snapshot of the early policy; see PROGRESS for the caveat).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "data" / "v2" / "b10_bench"
END_BUY = 265


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # macOS: bytes


def make(aug, phase, threads, shaping=False):
    from rl import train_ppo_v2 as T
    th.set_num_threads(threads)
    hp = T.PPO_CONFIG | {"net_arch": T.PPO_CONFIG["net_arch"]}
    venv = T.build_venv("TUNE-train", aug, 0, hp["n_envs"], "random" if phase == "A" else None, shaping=shaping)
    return T, hp, venv


def bench_train(a) -> dict:
    from stable_baselines3.common.callbacks import BaseCallback
    import sb3_contrib.ppo_mask.ppo_mask as pm
    T, hp, venv = make(a.aug, a.phase, a.threads, shaping=a.shaping == "on")
    model = T.build_model(venv, 0, hp)
    acc = {"env_step": 0.0, "masks": 0.0, "policy": 0.0, "update": 0.0}

    def timed(key, fn):
        def w(*x, **k):
            t = time.perf_counter()
            try:
                return fn(*x, **k)
            finally:
                acc[key] += time.perf_counter() - t
        return w
    venv.step_wait = timed("env_step", venv.step_wait)
    pm.get_action_masks = timed("masks", pm.get_action_masks)
    model.policy.forward = timed("policy", model.policy.forward)
    model.train = timed("update", model.train)

    marks, t0 = [], time.perf_counter()

    class Mark(BaseCallback):
        stop = False

        def _on_rollout_start(self):
            now = time.perf_counter()
            marks.append((now, int(self.num_timesteps), dict(acc)))
            if now - t0 >= a.warmup + a.seconds:
                self.stop = True

        def _on_step(self):
            return not self.stop
    model.learn(total_timesteps=10 ** 9, callback=Mark(), progress_bar=False)
    w0 = next(m for m in marks if m[0] - t0 >= a.warmup)
    w1 = marks[-1]
    dt, steps = w1[0] - w0[0], w1[1] - w0[1]
    parts = {k: round(w1[2][k] - w0[2][k], 2) for k in acc}
    parts["rest"] = round(dt - sum(parts.values()), 2)
    eps = [e["l"] for e in model.ep_info_buffer]
    out = {"mode": "train", "aug": a.aug, "phase": a.phase, "threads": th.get_num_threads(), "warmup_s": a.warmup, "window_s": round(dt, 1),
           "window_steps": steps, "steps_per_s": round(steps / dt, 1), "seconds_by_part_in_window": parts,
           "share_by_part": {k: round(v / dt, 3) for k, v in parts.items()}, "peak_rss_mb": round(peak_mb(), 1), "rollouts_in_window": len(marks) - marks.index(w0) - 1,
           "mean_episode_steps_last100": round(float(np.mean(eps)), 1) if eps else None, "mean_train_score_last100": round(float(np.mean([e["r"] for e in model.ep_info_buffer])), 4) if eps else None,
           "total_timesteps_run": int(model.num_timesteps)}
    return out


def bench_env(a) -> dict:
    T, hp, venv = make(a.aug, a.phase, a.threads, shaping=a.shaping == "on")
    venv.seed(0)
    venv.reset()
    rng = np.random.default_rng(0)
    steps, t_mask, t_step = 0, 0.0, 0.0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < a.seconds:
        t = time.perf_counter()
        masks = np.stack(venv.env_method("action_masks"))
        t_mask += time.perf_counter() - t
        acts = []
        for m in masks:
            legal = np.flatnonzero(m)
            if a.agent == "buy_all" and len(legal) > 1:
                legal = legal[legal != END_BUY]
            acts.append(int(rng.choice(legal)))
        t = time.perf_counter()
        venv.step(np.array(acts))
        t_step += time.perf_counter() - t
        steps += len(acts)
    dt = time.perf_counter() - t0
    return {"mode": "env", "aug": a.aug, "phase": a.phase, "agent": a.agent, "steps_per_s": round(steps / dt, 1), "seconds": round(dt, 1),
            "share_env_step": round(t_step / dt, 3), "share_masks": round(t_mask / dt, 3), "peak_rss_mb": round(peak_mb(), 1)}


def bench_infer(a) -> dict:
    from stable_baselines3.common.utils import obs_as_tensor
    T, hp, venv = make("none", a.phase, a.threads)
    model = T.build_model(venv, 0, hp)
    venv.seed(0)
    obs = venv.reset()
    rng = np.random.default_rng(0)
    batches = []
    for _ in range(120):                                              # real observations + masks from random masked play
        masks = np.stack(venv.env_method("action_masks"))
        batches.append(({k: v.copy() for k, v in obs.items()}, masks))
        obs, *_ = venv.step(np.array([int(rng.choice(np.flatnonzero(m))) for m in masks]))
    n, t0 = 0, time.perf_counter()
    model.policy.set_training_mode(False)
    while time.perf_counter() - t0 < a.seconds:
        o, m = batches[n % len(batches)]
        with th.no_grad():
            model.policy(obs_as_tensor(o, model.device), action_masks=m)
        n += 1
    dt = time.perf_counter() - t0
    return {"mode": "infer", "threads": th.get_num_threads(), "batch": hp["n_envs"], "calls_per_s": round(n / dt, 1), "steps_per_s": round(n * hp["n_envs"] / dt, 1),
            "ms_per_call": round(1000 * dt / n, 3), "peak_rss_mb": round(peak_mb(), 1)}


def bench_parallel(a) -> dict:
    procs, outs = [], []
    for i in range(a.n):
        f = BENCH / f"_tmp_par_{os.getpid()}_{i}.json"
        outs.append(f)
        procs.append(subprocess.Popen([sys.executable, __file__, "train", "--aug", a.aug, "--phase", a.phase, "--threads", str(a.threads), "--shaping", a.shaping, "--warmup", str(a.warmup),
                                       "--seconds", str(a.seconds), "--out", str(f)], cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT))))
    for p in procs:
        p.wait()
    res = [json.loads(f.read_text()) for f in outs]
    for f in outs:
        f.unlink()
    rates = [r["steps_per_s"] for r in res]
    return {"mode": "parallel", "n": a.n, "aug": a.aug, "phase": a.phase, "threads": a.threads, "per_job_steps_per_s": rates, "per_job_mean": round(float(np.mean(rates)), 1),
            "per_job_min": min(rates), "aggregate_steps_per_s": round(sum(rates), 1), "peak_rss_mb_per_job": [r["peak_rss_mb"] for r in res],
            "share_by_part_job0": res[0]["share_by_part"], "mean_episode_steps": [r["mean_episode_steps_last100"] for r in res]}


def assemble(a) -> dict:
    def get(name):
        f = BENCH / f"{name}.json"
        return json.loads(f.read_text()) if f.exists() else None
    from rl import budget, train_ppo_v2 as T
    chosen = json.loads((BENCH / "thread_choice.json").read_text())
    p6 = {k: get(k) for k in ("par6_plain_B", "par6_d4_B", "par6_plain_A", "par6_d4_A")}
    full = [p6[k]["per_job_mean"] for k in ("par6_plain_B", "par6_d4_B") if p6[k]]
    r_full = min(full)
    blend = {}
    for aug in ("plain", "d4"):
        a_, b_ = p6.get(f"par6_{aug}_A"), p6.get(f"par6_{aug}_B")
        if a_ and b_:
            blend[aug] = round(1.0 / ((1 / 3) / a_["per_job_mean"] + (2 / 3) / b_["per_job_mean"]), 1)
    rec = {"hardware": {**T.environment_info(), "torch_threads_used": chosen["chosen"]}, "ppo_config": T.PPO_CONFIG | {"rollout_size": 2500, "train_set": "TUNE-train"},
           "thread_choice": chosen, "single_process": {k: get(k) for k in ("single_plain_B", "single_d4_B", "single_plain_A", "single_d4_A")},
           "env_only": {k: get(k) for k in ("env_A_plain", "env_B_uniform_plain", "env_B_buy_all_plain", "env_A_d4", "env_B_uniform_d4", "env_B_buy_all_d4")},
           "inference_only": get("infer_B"), "parallel": {"par3_plain_B": get("par3_plain_B"), **p6},
           "six_parallel": {"per_job_steps_per_s_full_game": r_full, "per_job_blended_one_third_A": blend, "basis": "min over plain / D4 of the mean per-job rate with 6 simultaneous jobs, phase B (full game)"},
           "peak_memory_mb": {"single": {k: v["peak_rss_mb"] for k, v in ((k, get(k)) for k in ("single_plain_B", "single_d4_B")) if v},
                              "per_job_at_6": {k: max(v["peak_rss_mb_per_job"]) for k, v in p6.items() if v}},
           "verdict": budget.throughput_verdict(r_full)}
    (ROOT / "data" / "v2" / "b10_throughput.json").write_text(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "parallel", "env", "infer", "assemble"])
    p.add_argument("--aug", default="none")
    p.add_argument("--phase", default="B")
    p.add_argument("--shaping", choices=["off", "on"], default="off", help="B13d: SPEC 7.4 shaping in the training env (throughput only)")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--warmup", type=float, default=60)
    p.add_argument("--seconds", type=float, default=180)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--agent", default="uniform")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    BENCH.mkdir(parents=True, exist_ok=True)
    res = {"train": bench_train, "parallel": bench_parallel, "env": bench_env, "infer": bench_infer, "assemble": assemble}[a.mode](a)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
