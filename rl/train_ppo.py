"""Train MaskablePPO on the DEV-familiar training layouts (step 6 pipeline).

    .venv/bin/python -m rl.train_ppo --aug none --seed 0 --timesteps 300000 --curriculum on --run-name ppo_none_s0

Data discipline: trains ONLY on the 140 DEV-familiar training layouts; the 20 validation layouts
(results/val_split.json, 10 per family) feed the eval callback; FINAL and transfer layouts are never loaded.

Curriculum (``--curriculum on``): phase A = deploy-only with a fixed pre-bought army (the Greedy army,
see GREEDY_ARMY), phase B = full buy+deploy. The switch happens at ``--phase-a-steps`` env timesteps
(default: one third of ``--timesteps``) and is logged to curriculum.log / identity json.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.augment_v1 import D4Wrapper
from rl.evaluation import TROOP_NAMES, evaluate_policy, summarize
from rl.features import SiegeFeaturesExtractor
from rl.split import layouts_sha, load_split
from siege.baselines import _greedy_army
from siege.env import SiegeEnv

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ROOT = Path(__file__).resolve().parent.parent


def greedy_army_counts() -> list[int]:
    """Phase-A pre-bought army = the GreedyAgent's army (6 tank + 2 swarm = 200 gold)."""
    names = _greedy_army({})
    return [names.count(n) for n in TROOP_NAMES]


GREEDY_ARMY = greedy_army_counts()

HYPERPARAMS = dict(
    n_envs=8, n_steps=256, batch_size=256, n_epochs=10, learning_rate=3e-4, gamma=1.0, gae_lambda=0.95,
    clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, net_arch=[256, 256],
)


def git_info() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


class CurriculumCallback(BaseCallback):
    """Flips every training env from deploy-only (pre-bought army) to the full buy+deploy game at a timestep boundary."""

    def __init__(self, boundary: int, army: list[int], log_path: Path) -> None:
        super().__init__()
        self.boundary, self.army, self.log_path, self.switched = boundary, army, log_path, False

    def _apply(self, army, label: str) -> None:
        for env in self.training_env.envs:
            env.unwrapped.set_prebought(army)
        with self.log_path.open("a") as fh:
            fh.write(json.dumps({"timestep": int(self.num_timesteps), "phase": label, "prebought": army}) + "\n")
        print(f"[curriculum] t={self.num_timesteps}: phase {label} (prebought={army})", flush=True)

    def _on_training_start(self) -> None:
        # Envs were built already in phase A (prebought army) so even the first episodes are deploy-only.
        # On resume (num_timesteps > 0) the phase is restored from the timestep: before the boundary -> A, after -> B.
        resumed = self.num_timesteps > 0
        self.switched = self.boundary <= 0
        if self.boundary > 0 and resumed and self.num_timesteps >= self.boundary:
            self.switched = True
            self._apply(None, "B-resume")
        elif not self.switched:
            self._apply(self.army, "A-resume" if resumed else "A")

    def _on_step(self) -> bool:
        # Envs are only reset when an episode ends, so the new setting takes effect from the next episode.
        if not self.switched and self.num_timesteps >= self.boundary:
            self.switched = True
            self._apply(None, "B")
        return True


class ValidationCallback(BaseCallback):
    """Every ``freq`` timesteps: deterministic eval on the 20 validation layouts -> val_eval.csv; saves best.zip on full-task score."""

    def __init__(self, records: list[dict], freq: int, run_dir: Path) -> None:
        super().__init__()
        self.records, self.freq, self.run_dir, self.best = records, freq, run_dir, -1.0
        self.last = 0
        self.csv_path = run_dir / "val_eval.csv"
        fams = sorted({r["family"] for r in records})
        self.fields = ["timestep", "mode", "overall", *fams, *[f"army_{n}" for n in TROOP_NAMES]]

    def _eval(self) -> None:
        for mode, pre in (("full", None), ("deploy_only", GREEDY_ARMY)):
            s = summarize(evaluate_policy(self.model, self.records, prebought=pre))
            row = {"timestep": self.num_timesteps, "mode": mode, "overall": s["overall"],
                   **{f: s[f] for f in self.fields if f in s}, **{f"army_{n}": c for n, c in zip(TROOP_NAMES, s["mean_army"])}}
            new = not self.csv_path.exists()
            with self.csv_path.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=self.fields)
                if new:
                    w.writeheader()
                w.writerow(row)
            self.logger.record(f"val/{mode}_overall", s["overall"])
            if mode == "full":
                print(f"[val] t={self.num_timesteps} full={s['overall']:.3f} army={np.round(s['mean_army'], 1).tolist()}", flush=True)
                if s["overall"] > self.best:
                    self.best = s["overall"]
                    self.model.save(self.run_dir / "best.zip")

    def _on_training_start(self) -> None:
        if self.num_timesteps > 0:  # resumed: continue the eval schedule and best-score tracking, no t=0 eval
            self.last = (self.num_timesteps // self.freq) * self.freq
            if self.csv_path.exists():
                import pandas as pd
                allv = pd.read_csv(self.csv_path)
                allv = allv[allv["timestep"] <= self.num_timesteps]  # drop evals from the discarded post-checkpoint segment
                allv.to_csv(self.csv_path, index=False)
                v = allv[allv["mode"] == "full"]
                self.best = float(v["overall"].max()) if len(v) else -1.0
            return
        self._eval()  # t=0 baseline

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last >= self.freq:
            self.last = self.num_timesteps
            self._eval()
        return True


def make_env(records: list[dict], aug: str, seed: int, rank: int, prebought: list[int] | None = None):
    def _init():
        env = SiegeEnv(layout_pool=records, prebought=prebought)
        if aug == "d4":
            env = D4Wrapper(env, seed=seed * 1000 + rank)
        return Monitor(env)
    return _init


def latest_checkpoint(run_dir: Path) -> Path | None:
    """Newest ``checkpoints/ckpt_<steps>_steps.zip`` (by step count), or None."""
    found = []
    for f in (run_dir / "checkpoints").glob("ckpt_*_steps.zip"):
        try:
            found.append((int(f.stem.split("_")[1]), f))
        except (IndexError, ValueError):
            continue
    return max(found)[1] if found else None


def clean_run_dir(run_dir: Path) -> None:
    import shutil
    if not run_dir.exists():
        return
    for name in ("final.zip", "best.zip", "val_eval.csv", "curriculum.log", "model_identity.json"):
        (run_dir / name).unlink(missing_ok=True)
    for f in run_dir.glob("progress*.csv"):
        f.unlink()
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser()
    p.add_argument("--aug", choices=["none", "d4"], default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--curriculum", choices=["on", "off"], default="on")
    p.add_argument("--run-name", required=True)
    p.add_argument("--phase-a-steps", type=int, default=None, help="default: timesteps // 3")
    p.add_argument("--phase-a-army", choices=["fixed", "random"], default="fixed", help="fixed = Greedy army; random = fresh random legal army per episode")
    p.add_argument("--lr", type=float, default=HYPERPARAMS["learning_rate"])
    p.add_argument("--n-epochs", type=int, default=HYPERPARAMS["n_epochs"])
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--val-freq", type=int, default=25_000)
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--n-envs", type=int, default=HYPERPARAMS["n_envs"])
    p.add_argument("--resume", action="store_true", help="skip if final.zip exists; continue from the latest checkpoint if any; else start fresh")
    args = p.parse_args(argv)

    th.set_num_threads(1)  # measured: no speedup from more threads; 1 thread lets us run ~1 job per core in parallel
    hp = dict(HYPERPARAMS, n_envs=args.n_envs, learning_rate=args.lr, n_epochs=args.n_epochs, target_kl=args.target_kl)
    phase_a_army = GREEDY_ARMY if args.phase_a_army == "fixed" else "random"
    run_dir = MODELS_DIR / args.run_name
    ident_path = run_dir / "model_identity.json"
    if args.resume and (run_dir / "final.zip").exists():
        print(f"[skip] {args.run_name}: final.zip exists", flush=True)
        return run_dir
    ckpt_path = latest_checkpoint(run_dir) if args.resume else None
    if ckpt_path is None:
        clean_run_dir(run_dir)  # fresh start: never mix with stale artifacts of an interrupted run
    run_dir.mkdir(parents=True, exist_ok=True)
    train_records, val_records = load_split()
    assert all(r["pool"] == "DEV" and r["tag"] == "familiar" for r in train_records + val_records)
    assert not {r["id"] for r in train_records} & {r["id"] for r in val_records}

    boundary = 0 if args.curriculum == "off" else (args.phase_a_steps if args.phase_a_steps is not None else args.timesteps // 3)
    prior = json.loads(ident_path.read_text()) if ckpt_path is not None and ident_path.exists() else None
    if prior is not None:  # resuming: the original run's settings win (curriculum boundary, seed, ...)
        boundary = prior["curriculum_phase_b_start_timestep"] or 0
        args.timesteps = prior["timesteps_requested"]
    venv = DummyVecEnv([make_env(train_records, args.aug, args.seed, i, phase_a_army if boundary > 0 else None) for i in range(hp["n_envs"])])
    if ckpt_path is None:
        th.manual_seed(args.seed)
        model = MaskablePPO(
            "MultiInputPolicy", venv, seed=args.seed, verbose=0, device="cpu",
            n_steps=hp["n_steps"], batch_size=hp["batch_size"], n_epochs=hp["n_epochs"], learning_rate=hp["learning_rate"],
            gamma=hp["gamma"], gae_lambda=hp["gae_lambda"], clip_range=hp["clip_range"], ent_coef=hp["ent_coef"],
            vf_coef=hp["vf_coef"], max_grad_norm=hp["max_grad_norm"], target_kl=hp["target_kl"],
            policy_kwargs=dict(features_extractor_class=SiegeFeaturesExtractor, net_arch=dict(pi=hp["net_arch"], vf=hp["net_arch"])),
        )
        n_resumes = 0
    else:
        n_resumes = len(prior.get("resumes", [])) + 1 if prior else 1
        model = MaskablePPO.load(ckpt_path, env=venv, device="cpu")  # restores weights, optimizer, num_timesteps
        # RNG streams are not restorable from a checkpoint: reseed deterministically per resume (resumed runs are
        # reproducible given the same kill point, but not bit-identical to an uninterrupted run).
        model.set_random_seed(args.seed + 100_003 * n_resumes)
        venv.seed(args.seed + 100_003 * n_resumes)
        pc = run_dir / "progress.csv"
        if pc.exists():
            pc.rename(run_dir / f"progress_part{n_resumes}.csv")  # the CSV logger truncates; keep the earlier segment
        print(f"[resume] {args.run_name} from {ckpt_path.name} (t={model.num_timesteps}), resume #{n_resumes}", flush=True)
    model.set_logger(configure(str(run_dir), ["csv", "stdout"]))
    start_t = int(model.num_timesteps)

    if prior is None:
        identity = {
            "algorithm": "MaskablePPO", "run_name": args.run_name, "aug": args.aug, "seed": args.seed,
            "timesteps_requested": args.timesteps, "curriculum": args.curriculum,
            "curriculum_phase_a_army": phase_a_army if args.curriculum == "on" else None,
            "curriculum_phase_b_start_timestep": boundary if args.curriculum == "on" else None,
            **git_info(), "layouts_json_sha256": layouts_sha(), "n_train_layouts": len(train_records),
            "val_layout_ids": [r["id"] for r in val_records], "hyperparameters": hp,
            "features_extractor": "rl.features.SiegeFeaturesExtractor", "python": sys.version.split()[0],
            "torch": th.__version__, "status": "running",
        }
    else:
        identity = prior
        identity["status"] = "running"
        identity.setdefault("resumes", []).append({"checkpoint": ckpt_path.name, "at_timestep": start_t, **git_info()})
    ident_path.write_text(json.dumps(identity, indent=2))

    callbacks = [
        ValidationCallback(val_records, args.val_freq, run_dir),
        CheckpointCallback(save_freq=max(1, args.checkpoint_freq // hp["n_envs"]), save_path=str(run_dir / "checkpoints"), name_prefix="ckpt"),
    ]
    if args.curriculum == "on":
        callbacks.insert(0, CurriculumCallback(boundary, phase_a_army, run_dir / "curriculum.log"))
    t0 = time.time()
    model.learn(total_timesteps=max(1, args.timesteps - start_t), callback=CallbackList(callbacks), log_interval=20, reset_num_timesteps=(ckpt_path is None))
    wall = time.time() - t0
    model.save(run_dir / "final.zip")
    identity.update(status="finished", timesteps_trained=int(model.num_timesteps), wall_seconds_last_segment=round(wall, 1),
                    steps_per_second=round((model.num_timesteps - start_t) / wall, 1))
    ident_path.write_text(json.dumps(identity, indent=2))
    print(f"[done] {model.num_timesteps} steps in {wall:.0f}s = {model.num_timesteps / wall:.0f} steps/s -> {run_dir}", flush=True)
    return run_dir


if __name__ == "__main__":
    main()
