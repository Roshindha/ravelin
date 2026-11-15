"""B10: v2 MaskablePPO pipeline on BreachEnv (docs/SPEC_V2.md 6, 7, 11 B10, 12).

    .venv/bin/python -m rl.train_ppo_v2 --train-set TUNE-train --aug none --seed 0 --timesteps 300000 --run-name pilot_plain_s0 --resume

Data discipline: trains ONLY on the training set (TUNE-train for the G7 pilot, DEV-train for the grid) with role "train"; validates on the matching
validation set (TUNE-val / DEV-validation) with role "validation". Held-out sets are refused by the role guards; this module never
asks for them. There is no best.zip: nothing is selected on validation (evaluation uses final.zip only).

Curriculum (spec 6): phase A (first 1/3 of timesteps) = env starts in DEPLOY with a fresh random legal army per episode (`prebought="random"`);
phase B = full BUY -> DEPLOY game. Switch by timestep, per-episode lazy, logged to curriculum.log and the model identity. (The optional
linear ramp is NOT implemented: spec default is off.)

Rollout size = n_envs * n_steps = 2500, which divides 25k / 250k, so validation evals (every 25k) and the final model fall exactly on rollout
boundaries and the final model has exactly `--timesteps` steps (G7 reads t = 250k / 275k / 300k).

Resume (bit-identical, rollout boundaries only): a checkpoint is a directory `checkpoints/ckpt_<t>/` written atomically at the START of a rollout
(model zip + `state.pkl` = pickled training VecEnv incl. every env's RNG / episode state / D4 stream, last obs, episode starts, torch / numpy /
python RNG states, callback state). Resuming restores all of it, so the continued run equals an uninterrupted run bit for bit (tests/test_b10_ppo.py,
data/v2/b10_determinism.json). Bit-identity needs the same torch thread count; it is recorded in the identity and a mismatch is logged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import stable_baselines3
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from breachlens import evaluator as EV
from breachlens import manifest as MF
from breachlens.config import sim_version
from breachlens.env import BreachEnv
from breachlens.score import SCORE_VERSION

from .augment import D4Wrapper
from .eval_ppo_v2 import ModelPolicy
from .features import SiegeFeaturesExtractor

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("BREACHLENS_MODELS_DIR", ROOT / "models_v2"))   # env override: benchmark / determinism scripts
TRAIN_TO_VAL = {"TUNE-train": "TUNE-val", "DEV-train": "DEV-validation"}

# B10 starting configuration for the G7 pilot (frozen in B13 together with the step budget; nothing here is tuned yet).
# Prototype values except the rollout shape, chosen so that n_envs * n_steps = 2500 divides 25k (see module docstring).
PPO_CONFIG = dict(n_envs=10, n_steps=250, batch_size=250, n_epochs=10, learning_rate=3e-4, gamma=1.0, gae_lambda=0.95, clip_range=0.2,
                  ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, net_arch=[256, 256])
DEFAULT_THREADS = 1

RUNTIME_ONLY = {"resume", "threads", "debug_interrupt_at", "extend_to", "speed_freq"}   # CLI options that may differ on resume; everything else comes from the identity


class Interrupted(RuntimeError):
    """Raised by the `--debug-interrupt-at` test hook right after a checkpoint (simulates a kill at a known point)."""


# ---------------------------------------------------------------------------------------------------- hashes / identity
def _hash_tensors(h, obj) -> None:
    if isinstance(obj, th.Tensor):
        h.update(str(obj.dtype).encode() + str(tuple(obj.shape)).encode() + obj.detach().cpu().contiguous().numpy().tobytes())
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            h.update(str(k).encode())
            _hash_tensors(h, obj[k])
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _hash_tensors(h, v)
    else:
        h.update(repr(obj).encode())


def params_sha256(model: MaskablePPO, optimizer: bool = False) -> str:
    """sha256 over the policy parameters (sorted names, dtype, shape, raw bytes); `optimizer=True` also hashes the Adam state."""
    h = hashlib.sha256()
    _hash_tensors(h, model.policy.state_dict())
    if optimizer:
        _hash_tensors(h, model.policy.optimizer.state_dict())
    return h.hexdigest()


def cpu_model() -> str:
    import subprocess
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return platform.processor() or "unknown"


def environment_info() -> dict:
    return {"python": sys.version.split()[0], "torch": th.__version__, "stable_baselines3": stable_baselines3.__version__,
            "numpy": np.__version__, "cpu": cpu_model(), "cpu_count": os.cpu_count(), "torch_threads": th.get_num_threads(),
            "platform": platform.platform()}


def repo_identity() -> dict:
    m = EV.run_meta(0, "train", "identity")          # git commit / dirty, pool sha, generator hash, sim and score versions (one source of truth)
    return {k: m[k] for k in ("git_commit", "git_dirty", "pool_sha256", "generator_hash", "sim_version", "score_version")}


# ---------------------------------------------------------------------------------------------------- envs
def make_env(train_set: str, aug: str, seed: int, rank: int, prebought):
    def _init():
        env = BreachEnv(source=train_set, role="train", prebought=prebought)
        if aug == "d4":
            env = D4Wrapper(env, seed=seed * 1000 + rank)
        return Monitor(env)
    return _init


def build_venv(train_set: str, aug: str, seed: int, n_envs: int, prebought) -> DummyVecEnv:
    return DummyVecEnv([make_env(train_set, aug, seed, i, prebought) for i in range(n_envs)])


def build_model(venv: DummyVecEnv, seed: int, hp: dict) -> MaskablePPO:
    return MaskablePPO(
        "MultiInputPolicy", venv, seed=seed, verbose=0, device="cpu", n_steps=hp["n_steps"], batch_size=hp["batch_size"], n_epochs=hp["n_epochs"],
        learning_rate=hp["learning_rate"], gamma=hp["gamma"], gae_lambda=hp["gae_lambda"], clip_range=hp["clip_range"], ent_coef=hp["ent_coef"],
        vf_coef=hp["vf_coef"], max_grad_norm=hp["max_grad_norm"],
        policy_kwargs=dict(features_extractor_class=SiegeFeaturesExtractor, net_arch=dict(pi=list(hp["net_arch"]), vf=list(hp["net_arch"]))))


# ---------------------------------------------------------------------------------------------------- callbacks
def _truncate_csv(path: Path, t: int) -> None:
    if not path.exists():
        return
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = rows and list(rows[0].keys())
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(r for r in rows if int(r["timestep"]) <= t)


def _truncate_jsonl(path: Path, t: int) -> None:
    if path.exists():
        keep = [ln for ln in path.read_text().splitlines() if ln and json.loads(ln)["timestep"] <= t]
        path.write_text("".join(ln + "\n" for ln in keep))


class CurriculumCallback(BaseCallback):
    """Phase A -> B by timestep. Envs are built in phase A (prebought='random'); switching only changes what the NEXT reset does (lazy)."""

    def __init__(self, boundary: int, log_path: Path) -> None:
        super().__init__()
        self.boundary, self.log_path, self.switched = boundary, log_path, False

    def _log(self, phase: str, prebought) -> None:
        with self.log_path.open("a") as fh:
            fh.write(json.dumps({"timestep": int(self.num_timesteps), "phase": phase, "prebought": prebought}) + "\n")
        print(f"[curriculum] t={self.num_timesteps}: phase {phase}", flush=True)

    def _on_training_start(self) -> None:
        self.switched = self.num_timesteps >= self.boundary
        want = None if self.switched else "random"
        got = {env.unwrapped.prebought for env in self.training_env.envs}
        if got != {want}:
            raise RuntimeError(f"curriculum state mismatch at t={self.num_timesteps}: envs have {got}, expected {want!r}")
        if self.num_timesteps == 0:
            self._log("A", "random")

    def _on_step(self) -> bool:
        if not self.switched and self.num_timesteps >= self.boundary:
            self.switched = True
            for env in self.training_env.envs:
                env.unwrapped.set_prebought(None)
            self._log("B", None)
        return True


class ValidationCallback(BaseCallback):
    """Deterministic evaluation on the validation set at t = 0 and every `freq` timesteps (and at the end), through the B8 evaluator.

    Writes val_eval.csv (one row per group: ALL + each family) and val_layouts.jsonl (one row per layout incl. the bought army) = the G7 inputs."""

    FIELDS = ["timestep", "group", "n_layouts", "breach_mean", "breach_lo95", "breach_hi95", "score_mean", "score_lo95", "score_hi95", "d_mean",
              "mean_n_bought", "mean_n_deployed", "eval_seconds",
              "score_version", "sim_version", "generator_hash", "pool_sha256", "git_commit", "git_dirty", "seed", "role", "set"]      # provenance (B13a)

    def __init__(self, set_name: str, freq: int, run_dir: Path) -> None:
        super().__init__()
        self.set_name, self.freq, self.run_dir = set_name, freq, run_dir
        self.csv_path, self.jsonl_path = run_dir / "val_eval.csv", run_dir / "val_layouts.jsonl"
        self.items = None
        self.last_eval_t = -1
        self.seconds = 0.0

    def evaluate(self) -> None:
        t = int(self.num_timesteps)
        if self.last_eval_t == t:
            return
        self.last_eval_t = t
        if self.items is None:
            self.items = EV.load_set(self.set_name, "validation", entry_point="rl.train_ppo_v2")
        t0 = time.perf_counter()
        res = EV.evaluate(ModelPolicy(self.model, name=f"PPO@{t}"), self.set_name, "validation", items=self.items)
        dt = time.perf_counter() - t0
        self.seconds += dt
        new = not self.csv_path.exists()
        with self.csv_path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.FIELDS)
            if new:
                w.writeheader()
            for r in res["summary"]:
                w.writerow({"timestep": t, **{k: r[k] for k in self.FIELDS if k in r}, "eval_seconds": round(dt, 2)})
        with self.jsonl_path.open("a") as fh:
            for r in res["rows"]:
                ep = res["episodes"][r["id"]][0]
                fh.write(json.dumps({"timestep": t, "id": r["id"], "family": r["family"], "score": r["score"], "d": r["d"], "breach": r["breach"],
                                     "army": ep["army"], "n_deployed": ep["n_deployed"]}) + "\n")
        a = next(r for r in res["summary"] if r["group"] == "ALL")
        self.logger.record("val/score", a["score_mean"])
        self.logger.record("val/breach", a["breach_mean"])
        print(f"[val] t={t} score={a['score_mean']:.4f} breach={a['breach_mean']:.3f} bought={a['mean_n_bought']:.1f} ({dt:.1f}s)", flush=True)

    def _on_training_start(self) -> None:
        if self.num_timesteps == 0:
            self.evaluate()

    def _on_rollout_start(self) -> None:
        t = int(self.num_timesteps)
        if t > 0 and t % self.freq == 0:
            self.evaluate()

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        t = int(self.num_timesteps)
        if t % self.freq == 0:
            self.evaluate()


class SpeedCallback(BaseCallback):
    """Thermal check (B13a): every `freq` timesteps append (timestep, wall seconds, steps/s over the last window, steps/s since start) to speed.csv."""

    def __init__(self, run_dir: Path, freq: int) -> None:
        super().__init__()
        self.path, self.freq = run_dir / "speed.csv", freq
        self.t0 = self.last_wall = None
        self.last_t = 0

    def _on_training_start(self) -> None:
        self.t0 = self.last_wall = time.perf_counter()
        self.start_t = self.last_t = int(self.num_timesteps)

    def _on_rollout_start(self) -> None:
        t = int(self.num_timesteps)
        if self.freq <= 0 or t <= self.last_t or t % self.freq:
            return
        now = time.perf_counter()
        new = not self.path.exists()
        with self.path.open("a") as fh:
            if new:
                fh.write("timestep,wall_seconds,steps_per_s_window,steps_per_s_since_start\n")
            fh.write(f"{t},{now - self.t0:.1f},{(t - self.last_t) / (now - self.last_wall):.1f},{(t - self.start_t) / (now - self.t0):.1f}\n")
        print(f"[speed] t={t} window={(t - self.last_t) / (now - self.last_wall):.0f} steps/s", flush=True)
        self.last_t, self.last_wall = t, now

    def _on_step(self) -> bool:
        return True


class CheckpointCallback(BaseCallback):
    """Atomic, resumable checkpoint directories written at the START of a rollout (the only point where the full state is reproducible)."""

    def __init__(self, run_dir: Path, freq: int, callbacks_state, keep: int = 2, interrupt_at: int | None = None) -> None:
        super().__init__()
        self.dir, self.freq, self.keep, self.interrupt_at = run_dir / "checkpoints", freq, keep, interrupt_at
        self.get_cb_state = callbacks_state
        self.bucket = 0

    def _on_training_start(self) -> None:
        self.bucket = self.num_timesteps // self.freq

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        t = int(self.num_timesteps)
        if t == 0 or t // self.freq <= self.bucket:
            return
        self.bucket = t // self.freq
        self.save(t)
        if self.interrupt_at is not None and t >= self.interrupt_at:
            raise Interrupted(f"debug interrupt after checkpoint at t={t}")

    def save(self, t: int) -> None:
        tmp, final = self.dir / f"ckpt_{t}.tmp", self.dir / f"ckpt_{t}"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(final, ignore_errors=True)
        tmp.mkdir(parents=True)
        m = self.model
        self.model.save(tmp / "model.zip")
        state = {"num_timesteps": t, "venv": m.env, "last_obs": m._last_obs, "last_episode_starts": m._last_episode_starts,
                 "episode_num": m._episode_num, "th_rng": th.get_rng_state(), "np_rng": np.random.get_state(), "py_rng": random.getstate(),
                 "callbacks": self.get_cb_state(), "torch_threads": th.get_num_threads()}
        with (tmp / "state.pkl").open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename(tmp, final)                                  # atomic: a directory without the final name is never used
        done = sorted((int(d.name.split("_")[1]), d) for d in self.dir.glob("ckpt_*") if d.is_dir() and not d.name.endswith(".tmp"))
        for _, d in done[:-self.keep]:
            shutil.rmtree(d, ignore_errors=True)


def latest_checkpoint(run_dir: Path) -> Path | None:
    found = []
    for d in (run_dir / "checkpoints").glob("ckpt_*"):
        if d.is_dir() and not d.name.endswith(".tmp") and (d / "state.pkl").exists() and (d / "model.zip").exists():
            try:
                found.append((int(d.name.split("_")[1]), d))
            except ValueError:
                continue
    return max(found)[1] if found else None


def clean_run_dir(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    for name in ("final.zip", "val_eval.csv", "val_layouts.jsonl", "curriculum.log", "model_identity.json"):
        (run_dir / name).unlink(missing_ok=True)
    for f in run_dir.glob("progress*.csv"):
        f.unlink()
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)


def restore(ckpt: Path):
    """-> (model, state). RNG states are restored LAST (loading the model builds networks and consumes torch RNG)."""
    with (ckpt / "state.pkl").open("rb") as fh:
        state = pickle.load(fh)
    model = MaskablePPO.load(ckpt / "model.zip", env=state["venv"], device="cpu")
    assert model.num_timesteps == state["num_timesteps"], (model.num_timesteps, state["num_timesteps"])
    model._last_obs, model._last_episode_starts, model._episode_num = state["last_obs"], state["last_episode_starts"], state["episode_num"]
    th.set_rng_state(state["th_rng"])
    np.random.set_state(state["np_rng"])
    random.setstate(state["py_rng"])
    return model, state


# ---------------------------------------------------------------------------------------------------- main
def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--train-set", choices=sorted(TRAIN_TO_VAL), default="TUNE-train")
    p.add_argument("--aug", choices=["none", "d4"], default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--curriculum", choices=["on", "off"], default="on")
    p.add_argument("--phase-a-steps", type=int, default=None, help="default: timesteps // 3")
    p.add_argument("--run-name", required=True)
    p.add_argument("--val-freq", type=int, default=25_000)
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    for k, v in PPO_CONFIG.items():
        if k != "net_arch":
            p.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.add_argument("--speed-freq", type=int, default=10_000, help="steps/s log interval (thermal check), speed.csv")
    p.add_argument("--extend-to", type=int, default=None, help="continue a FINISHED run from its latest checkpoint to this many steps (G7 extension, changelog (u)); the curriculum boundary must be explicit")
    p.add_argument("--resume", action="store_true", help="skip if final.zip exists; continue from the latest checkpoint if any; else start fresh")
    p.add_argument("--debug-interrupt-at", type=int, default=None, help="test hook: raise after the first checkpoint at t >= this")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    th.set_num_threads(args.threads)
    run_dir = MODELS_DIR / args.run_name
    ident_path = run_dir / "model_identity.json"
    if args.extend_to is not None:
        if not (run_dir / "final.zip").exists() or latest_checkpoint(run_dir) is None:
            raise SystemExit(f"--extend-to needs a finished run with a checkpoint in {run_dir}")
        args.resume = True
    elif args.resume and (run_dir / "final.zip").exists():
        print(f"[skip] {args.run_name}: final.zip exists", flush=True)
        return run_dir
    ckpt = latest_checkpoint(run_dir) if args.resume else None
    if ckpt is None:
        clean_run_dir(run_dir)                              # fresh start: never mix with stale artifacts of an interrupted run
    run_dir.mkdir(parents=True, exist_ok=True)

    prior = json.loads(ident_path.read_text()) if ckpt is not None and ident_path.exists() else None
    if prior is not None:                                   # the original run's settings win on resume
        for k, v in prior["config"].items():
            if k not in RUNTIME_ONLY:
                setattr(args, k, v)
    if args.extend_to is not None:
        if args.extend_to <= prior["config"]["timesteps"]:
            raise SystemExit("--extend-to must exceed the original timesteps")
        if prior["config"]["curriculum"] == "on" and prior["config"]["phase_a_steps"] is None:
            raise SystemExit("--extend-to needs an explicit --phase-a-steps in the original run (the curriculum boundary must not move)")
        shutil.copy2(run_dir / "final.zip", run_dir / f"final_{prior['config']['timesteps']}.zip")
        args.timesteps = args.extend_to
    cfg = {k: v for k, v in vars(args).items()}
    hp = {k: getattr(args, k) for k in PPO_CONFIG if k != "net_arch"} | {"net_arch": PPO_CONFIG["net_arch"]}
    train_set, val_set = args.train_set, TRAIN_TO_VAL[args.train_set]
    rollout = hp["n_envs"] * hp["n_steps"]
    boundary = 0 if args.curriculum == "off" else (args.phase_a_steps if args.phase_a_steps is not None else args.timesteps // 3)

    if ckpt is None:
        venv = build_venv(train_set, args.aug, args.seed, hp["n_envs"], "random" if boundary > 0 else None)
        model = build_model(venv, args.seed, hp)
        n_resumes, cb_state = 0, {}
    else:
        model, state = restore(ckpt)
        venv, cb_state = model.env, state["callbacks"]
        n_resumes = len(prior.get("resumes", [])) + 1 if prior else 1
        if (run_dir / "progress.csv").exists():
            (run_dir / "progress.csv").rename(run_dir / f"progress_part{n_resumes}.csv")     # the CSV logger truncates; keep the earlier segment
        t = int(model.num_timesteps)
        _truncate_csv(run_dir / "val_eval.csv", t)
        _truncate_jsonl(run_dir / "val_layouts.jsonl", t)
        print(f"[resume] {args.run_name} from {ckpt.name} (t={t}), resume #{n_resumes}", flush=True)
    model.set_logger(configure(str(run_dir), ["csv"]))
    start_t = int(model.num_timesteps)

    env_info = environment_info() | {"torch_threads": th.get_num_threads()}
    if prior is None:
        identity = {"algorithm": "MaskablePPO", "run_name": args.run_name, "config": cfg, "train_set": train_set, "val_set": val_set, "aug": args.aug,
                    "seed": args.seed, "timesteps_requested": args.timesteps, "rollout_size": rollout, "curriculum": args.curriculum,
                    "curriculum_phase_a_army": "random" if args.curriculum == "on" else None,
                    "curriculum_phase_b_start_timestep": boundary if args.curriculum == "on" else None, **repo_identity(), "hyperparameters": hp,
                    "features_extractor": "rl.features.SiegeFeaturesExtractor", "observation": "Dict(grid (9,14,14), state (11,))", "n_actions": 267,
                    "environment": env_info, "status": "running"}
    else:
        identity = prior
        identity["status"] = "running"
        if args.extend_to is not None:
            identity.setdefault("extensions", []).append({"from_timesteps": prior["timesteps_requested"], "to_timesteps": args.extend_to, "checkpoint": ckpt.name})
            identity["timesteps_requested"] = args.extend_to
            identity["config"]["timesteps"] = args.extend_to
        if prior["environment"]["torch_threads"] != th.get_num_threads():
            identity.setdefault("warnings", []).append(f"resume #{n_resumes}: torch threads {prior['environment']['torch_threads']} -> {th.get_num_threads()} (bit-identity not guaranteed)")
        identity.setdefault("resumes", []).append({"checkpoint": ckpt.name, "at_timestep": start_t, **repo_identity()})
    ident_path.write_text(json.dumps(identity, indent=2))

    val = ValidationCallback(val_set, args.val_freq, run_dir)
    val.items = EV.load_set(val_set, "validation", entry_point="rl.train_ppo_v2")          # loaded now (not at the first eval) so the manifest exists before any step
    train_ids = sorted({r["id"] for e in venv.envs for r in e.unwrapped.records})
    manifest = MF.build_manifest(args.run_name, train_set, val_set, train_ids, [r["id"] for r, _ in val.items], git_commit=identity.get("git_commit"))
    MF.check_manifest(manifest)                                                          # refuses FINAL / transfer / LAB-DESIGN ids or streams before training starts
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    cur = CurriculumCallback(boundary, run_dir / "curriculum.log") if args.curriculum == "on" else None
    val.last_eval_t = cb_state.get("val_last_eval_t", -1)
    if cur is not None:
        cur.switched = cb_state.get("switched", False)
    callbacks = ([cur] if cur else []) + [val, SpeedCallback(run_dir, args.speed_freq), CheckpointCallback(
        run_dir, args.checkpoint_freq, lambda: {"val_last_eval_t": val.last_eval_t, "switched": cur.switched if cur else None},
        interrupt_at=args.debug_interrupt_at)]
    t0 = time.perf_counter()
    model.learn(total_timesteps=max(1, args.timesteps - start_t), callback=CallbackList(callbacks), log_interval=10 ** 9,
                reset_num_timesteps=(ckpt is None), progress_bar=False)
    wall = time.perf_counter() - t0
    model.save(run_dir / "final.zip")
    identity.update(status="finished", timesteps_trained=int(model.num_timesteps), params_sha256=params_sha256(model),
                    params_optimizer_sha256=params_sha256(model, optimizer=True), wall_seconds_last_segment=round(wall, 1),
                    validation_seconds_last_segment=round(val.seconds, 1), steps_per_second_incl_validation=round((model.num_timesteps - start_t) / wall, 1))
    ident_path.write_text(json.dumps(identity, indent=2))
    print(f"[done] {model.num_timesteps} steps in {wall:.0f}s = {(model.num_timesteps - start_t) / wall:.0f} steps/s -> {run_dir}", flush=True)
    return run_dir


if __name__ == "__main__":
    main()
