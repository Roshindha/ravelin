"""B11: v2 masked Double DQN pipeline on BreachEnv (docs/SPEC_V2.md 6, 7, 11 B11, 12).

    .venv/bin/python -m rl.train_ddqn_v2 --train-set TUNE-train --seed 0 --timesteps 300000 --run-name ddqn_s0 --resume

Same data discipline as rl/train_ppo_v2.py: trains ONLY on the training set (role "train"), validates on the matching validation set (role
"validation"); held-out sets are refused. Nothing is selected on validation: the evaluation model is `final.pt` (no best checkpoint).
Same curriculum as PPO (spec 6): phase A (first 1/3 of timesteps) = random pre-bought armies, then the full BUY -> DEPLOY game.
Same environment steps as PPO; one gradient update per `train_every` = 4 environment steps (spec 12.2).

Loop (n_envs lock-step envs, one vector step = n_envs transitions):  [validate at t % val_freq == 0] -> [checkpoint at t % checkpoint_freq == 0]
-> [lazy curriculum switch] -> act (epsilon-greedy over legal actions) -> step -> store (incl. next-state legal mask) -> gradient updates.
Resume (bit-identical): the checkpoint holds online + target net, Adam state, the replay buffer (exact), the pickled VecEnv (env RNG, episode
state), last obs / masks, the action/sampling RNG stream, torch / numpy / python RNG states, update counters and callback state. Needs the same
torch thread count (recorded).
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import pickle
import random
import shutil
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F

from breachlens import evaluator as EV

from . import ddqn as D
from .train_ppo_v2 import (TRAIN_TO_VAL, _hash_tensors, _truncate_csv, _truncate_jsonl, build_venv, environment_info, repo_identity)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("BREACHLENS_MODELS_DIR", ROOT / "models_v2"))
DEFAULT_THREADS = 1
RUNTIME_ONLY = {"resume", "threads", "debug_interrupt_at"}
LOG_EVERY = 5_000


class Interrupted(RuntimeError):
    """`--debug-interrupt-at` test hook: raised right after a checkpoint."""


def epsilon(t: int, total: int, hp: dict) -> float:
    """Linear decay eps_start -> eps_end over the first eps_decay_fraction of the total timesteps, then constant."""
    span = max(1.0, hp["eps_decay_fraction"] * total)
    return float(max(hp["eps_end"], hp["eps_start"] + (hp["eps_end"] - hp["eps_start"]) * min(1.0, t / span)))


def nets_sha256(online, target, opt=None, buffer=None) -> str:
    h = hashlib.sha256()
    _hash_tensors(h, online.state_dict())
    if target is not None:
        _hash_tensors(h, target.state_dict())
    if opt is not None:
        _hash_tensors(h, opt.state_dict())
    if buffer is not None:
        h.update(buffer.sha256().encode())
    return h.hexdigest()


class Trainer:
    """The DDQN training state machine. `venv` is injectable (tests use toy envs); everything random goes through `self.rng` (action selection and
    replay sampling) or torch's global generator (network init only)."""

    def __init__(self, hp: dict, venv, seed: int, total_timesteps: int, boundary: int, obs_space=None) -> None:
        self.hp, self.seed, self.total, self.boundary = hp, seed, total_timesteps, boundary
        self.venv, n = venv, hp["n_envs"]
        space = obs_space or venv.observation_space
        self.space, self.n_actions = space, venv.action_space.n
        th.manual_seed(seed)
        self.online = D.QNet(space, self.n_actions, hp["net_arch"])
        self.target = copy.deepcopy(self.online)
        self.target.requires_grad_(False)
        self.opt = th.optim.Adam(self.online.parameters(), lr=hp["learning_rate"])
        self.buffer = D.ReplayBuffer(hp["buffer_size"], n, space["grid"].shape, space["state"].shape[0], self.n_actions)
        self.rng = np.random.default_rng(seed)
        self.t = self.updates = self.credit = 0
        self.switched = boundary <= 0
        self.eps_log = deque(maxlen=100)          # train scores of the last finished episodes
        self.loss_acc, self.q_acc, self.n_loss = 0.0, 0.0, 0
        self.timers = None                         # benchmark: dict of accumulated seconds
        venv.seed(seed)
        self.obs = venv.reset()
        self.masks = np.stack(venv.env_method("action_masks"))

    # ------------------------------------------------------------------ one vector step
    def _timed(self, key, fn, *a):
        if self.timers is None:
            return fn(*a)
        t0 = time.perf_counter()
        try:
            return fn(*a)
        finally:
            self.timers[key] += time.perf_counter() - t0

    def act(self) -> np.ndarray:
        n = self.hp["n_envs"]
        u, r = self.rng.random(n), self.rng.random(n)            # always drawn: fixed RNG consumption
        self.online.eval()
        with th.no_grad():
            q = self.online(D.to_tensors(self.obs)).numpy()
        return D.select_actions(q, self.masks, epsilon(self.t, self.total, self.hp), u, r)

    def vector_step(self) -> None:
        a = self._timed("act", self.act)
        obs2, rew, done, infos = self._timed("env_step", self.venv.step, a)
        for i, d in enumerate(done):
            if d and "episode" in infos[i]:
                self.eps_log.append(float(infos[i]["episode"]["r"]))
        masks2 = self._timed("masks", lambda: np.stack(self.venv.env_method("action_masks")))
        # obs2 of a finished env is already the reset observation; the transition is terminal so it is never bootstrapped from
        self.buffer.add(self.obs, a, rew, done, obs2, masks2)
        self.obs, self.masks = obs2, masks2
        self.t += self.hp["n_envs"]
        if self.t >= self.hp["learning_starts"]:
            self.credit += self.hp["n_envs"]
            while self.credit >= self.hp["train_every"]:
                self.credit -= self.hp["train_every"]
                self._timed("update", self.update)

    def update(self) -> None:
        hp = self.hp
        b = self.buffer.to_torch(self.buffer.sample(hp["batch_size"], self.rng))
        y = D.compute_targets(self.online, self.target, {"next_obs": b["next_obs"], "next_mask": b["next_mask"], "reward": b["reward"], "done": b["done"]},
                              hp["gamma"])
        self.online.train()
        q = self.online(b["obs"]).gather(1, b["action"][:, None]).squeeze(1)
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.online.parameters(), hp["max_grad_norm"])
        self.opt.step()
        self.updates += 1
        if self.updates % hp["target_update_interval"] == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.loss_acc += loss.item()
        self.q_acc += q.mean().item()
        self.n_loss += 1

    # ------------------------------------------------------------------ persistence
    def save_checkpoint(self, d: Path, cb_state: dict) -> None:
        tmp = d.parent / (d.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(d, ignore_errors=True)
        tmp.mkdir(parents=True)
        th.save({"online": self.online.state_dict(), "target": self.target.state_dict(), "opt": self.opt.state_dict()}, tmp / "nets.pt")
        self.buffer.save(tmp / "buffer")
        state = {"t": self.t, "updates": self.updates, "credit": self.credit, "switched": self.switched, "venv": self.venv, "obs": self.obs,
                 "masks": self.masks, "rng": self.rng.bit_generator.state, "eps_log": list(self.eps_log), "th_rng": th.get_rng_state(),
                 "np_rng": np.random.get_state(), "py_rng": random.getstate(), "callbacks": cb_state, "torch_threads": th.get_num_threads(),
                 "loss": (self.loss_acc, self.q_acc, self.n_loss)}
        with (tmp / "state.pkl").open("wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename(tmp, d)

    @classmethod
    def from_checkpoint(cls, d: Path, hp: dict, seed: int, total: int, boundary: int, restore_rng: bool = True) -> tuple["Trainer", dict]:
        with (d / "state.pkl").open("rb") as fh:
            st = pickle.load(fh)
        tr = cls.__new__(cls)
        tr.hp, tr.seed, tr.total, tr.boundary, tr.venv = hp, seed, total, boundary, st["venv"]
        tr.space, tr.n_actions = tr.venv.observation_space, tr.venv.action_space.n
        tr.online = D.QNet(tr.space, tr.n_actions, hp["net_arch"])
        tr.target = copy.deepcopy(tr.online)
        tr.target.requires_grad_(False)
        tr.opt = th.optim.Adam(tr.online.parameters(), lr=hp["learning_rate"])
        nets = th.load(d / "nets.pt", weights_only=False)
        tr.online.load_state_dict(nets["online"])
        tr.target.load_state_dict(nets["target"])
        tr.opt.load_state_dict(nets["opt"])
        tr.buffer = D.ReplayBuffer.load(d / "buffer", hp["buffer_size"], hp["n_envs"], tr.space["grid"].shape, tr.space["state"].shape[0], tr.n_actions)
        tr.rng = np.random.default_rng(seed)
        tr.t, tr.updates, tr.credit, tr.switched = st["t"], st["updates"], st["credit"], st["switched"]
        tr.eps_log = deque(st["eps_log"], maxlen=100)
        tr.loss_acc, tr.q_acc, tr.n_loss = st["loss"]
        tr.timers = None
        tr.obs, tr.masks = st["obs"], st["masks"]
        if restore_rng:                                                       # last: building the nets consumed torch RNG
            tr.rng.bit_generator.state = st["rng"]
            th.set_rng_state(st["th_rng"])
            np.random.set_state(st["np_rng"])
            random.setstate(st["py_rng"])
        return tr, st

    def sha(self, full: bool = False) -> str:
        return nets_sha256(self.online, self.target if full else None, self.opt if full else None, self.buffer if full else None)


# ---------------------------------------------------------------------------------------------------- validation
class Validator:
    """Deterministic greedy evaluation on the validation set at t = 0 and every `freq` timesteps (and at the end) through the B8 evaluator; writes
    val_eval.csv and val_layouts.jsonl in the same format as the PPO pipeline (the G7-style inputs)."""

    FIELDS = ["timestep", "group", "n_layouts", "breach_mean", "breach_lo95", "breach_hi95", "score_mean", "score_lo95", "score_hi95", "d_mean",
              "mean_n_bought", "mean_n_deployed", "eval_seconds"]

    def __init__(self, set_name: str, run_dir: Path) -> None:
        self.set_name, self.csv_path, self.jsonl_path = set_name, run_dir / "val_eval.csv", run_dir / "val_layouts.jsonl"
        self.items, self.last_eval_t, self.seconds = None, -1, 0.0

    def evaluate(self, tr: Trainer) -> None:
        t = int(tr.t)
        if self.last_eval_t == t:
            return
        self.last_eval_t = t
        if self.items is None:
            self.items = EV.load_set(self.set_name, "validation", entry_point="rl.train_ddqn_v2")
        t0 = time.perf_counter()
        res = EV.evaluate(D.DDQNPolicy(tr.online, name=f"DDQN@{t}"), self.set_name, "validation", items=self.items)
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
        print(f"[val] t={t} score={a['score_mean']:.4f} breach={a['breach_mean']:.3f} bought={a['mean_n_bought']:.1f} ({dt:.1f}s)", flush=True)


def latest_checkpoint(run_dir: Path):
    found = []
    for d in (run_dir / "checkpoints").glob("ckpt_*"):
        if d.is_dir() and not d.name.endswith(".tmp") and (d / "state.pkl").exists() and (d / "buffer" / "meta.json").exists():
            try:
                found.append((int(d.name.split("_")[1]), d))
            except ValueError:
                continue
    return max(found)[1] if found else None


def clean_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        for name in ("final.pt", "val_eval.csv", "val_layouts.jsonl", "curriculum.log", "model_identity.json"):
            (run_dir / name).unlink(missing_ok=True)
        for f in run_dir.glob("progress*.csv"):
            f.unlink()
        shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)


def log_curriculum(path: Path, t: int, phase: str, prebought) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps({"timestep": int(t), "phase": phase, "prebought": prebought}) + "\n")
    print(f"[curriculum] t={t}: phase {phase}", flush=True)


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--train-set", choices=sorted(TRAIN_TO_VAL), default="TUNE-train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--curriculum", choices=["on", "off"], default="on")
    p.add_argument("--phase-a-steps", type=int, default=None, help="default: timesteps // 3")
    p.add_argument("--run-name", required=True)
    p.add_argument("--val-freq", type=int, default=25_000)
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    for k, v in D.DDQN_CONFIG.items():
        if k != "net_arch":
            p.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    p.add_argument("--shaping", choices=["off", "on"], default="off",
                   help="SPEC 7.4 potential-based shaping (the SAME BreachEnv implementation as the PPO shaped pilot, row z) in the TRAINING envs only; validation never shapes")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--debug-interrupt-at", type=int, default=None)
    p.add_argument("--debug-skip-rng-restore", action="store_true", help="negative control for the resume test: do NOT restore the RNG streams")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    th.set_num_threads(args.threads)
    run_dir = MODELS_DIR / args.run_name
    ident_path = run_dir / "model_identity.json"
    if args.resume and (run_dir / "final.pt").exists():
        print(f"[skip] {args.run_name}: final.pt exists", flush=True)
        return run_dir
    ckpt = latest_checkpoint(run_dir) if args.resume else None
    if ckpt is None:
        clean_run_dir(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    prior = json.loads(ident_path.read_text()) if ckpt is not None and ident_path.exists() else None
    if prior is not None:
        for k, v in prior["config"].items():
            if k not in RUNTIME_ONLY and k != "debug_skip_rng_restore":
                setattr(args, k, v)
    cfg = dict(vars(args))
    hp = {k: getattr(args, k) for k in D.DDQN_CONFIG if k != "net_arch"} | {"net_arch": D.DDQN_CONFIG["net_arch"]}
    if args.val_freq % hp["n_envs"] or args.checkpoint_freq % hp["n_envs"] or args.timesteps % hp["n_envs"]:
        raise SystemExit("timesteps, val-freq and checkpoint-freq must be multiples of n_envs (they are checked at vector-step boundaries)")
    train_set, val_set = args.train_set, TRAIN_TO_VAL[args.train_set]
    boundary = 0 if args.curriculum == "off" else (args.phase_a_steps if args.phase_a_steps is not None else args.timesteps // 3)

    if ckpt is None:
        venv = build_venv(train_set, "none", args.seed, hp["n_envs"], "random" if boundary > 0 else None, shaping=(args.shaping == "on"))
        tr = Trainer(hp, venv, args.seed, args.timesteps, boundary)
        n_resumes, cb_state = 0, {}
    else:
        tr, st = Trainer.from_checkpoint(ckpt, hp, args.seed, args.timesteps, boundary, restore_rng=not args.debug_skip_rng_restore)
        cb_state = st["callbacks"]
        n_resumes = len(prior.get("resumes", [])) + 1 if prior else 1
        if (run_dir / "progress.csv").exists():
            (run_dir / "progress.csv").rename(run_dir / f"progress_part{n_resumes}.csv")
        _truncate_csv(run_dir / "val_eval.csv", tr.t)
        _truncate_jsonl(run_dir / "val_layouts.jsonl", tr.t)
        print(f"[resume] {args.run_name} from {ckpt.name} (t={tr.t}), resume #{n_resumes}", flush=True)
    start_t = tr.t

    env_info = environment_info() | {"torch_threads": th.get_num_threads()}
    if prior is None:
        identity = {"algorithm": "MaskedDoubleDQN", "run_name": args.run_name, "config": cfg, "train_set": train_set, "val_set": val_set, "aug": "none",
                    "seed": args.seed, "timesteps_requested": args.timesteps, "curriculum": args.curriculum, "shaping": args.shaping,
                    "curriculum_phase_a_army": "random" if args.curriculum == "on" else None,
                    "curriculum_phase_b_start_timestep": boundary if args.curriculum == "on" else None, **repo_identity(), "hyperparameters": hp,
                    "features_extractor": "rl.features.SiegeFeaturesExtractor + MLP head", "observation": "Dict(grid (9,14,14), state (11,))", "n_actions": 267,
                    "environment": env_info, "status": "running"}
    else:
        identity = prior
        identity["status"] = "running"
        if prior["environment"]["torch_threads"] != th.get_num_threads():
            identity.setdefault("warnings", []).append(f"resume #{n_resumes}: torch threads changed (bit-identity not guaranteed)")
        identity.setdefault("resumes", []).append({"checkpoint": ckpt.name, "at_timestep": start_t, **repo_identity()})
    ident_path.write_text(json.dumps(identity, indent=2))

    val = Validator(val_set, run_dir)
    val.last_eval_t = cb_state.get("val_last_eval_t", -1)
    cur_log = run_dir / "curriculum.log"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_bucket = tr.t // args.checkpoint_freq
    if args.curriculum == "on":
        want = None if tr.switched else "random"
        got = {e.unwrapped.prebought for e in tr.venv.envs}
        if got != {want}:
            raise RuntimeError(f"curriculum state mismatch at t={tr.t}: envs have {got}, expected {want!r}")
        if tr.t == 0:
            log_curriculum(cur_log, 0, "A", "random")
    prog_path = run_dir / "progress.csv"
    last_log = tr.t // LOG_EVERY
    t0 = time.perf_counter()
    while tr.t < args.timesteps:
        if tr.t % args.val_freq == 0:
            val.evaluate(tr)
        if tr.t > 0 and tr.t // args.checkpoint_freq > ckpt_bucket:
            ckpt_bucket = tr.t // args.checkpoint_freq
            tr.save_checkpoint(ckpt_dir / f"ckpt_{tr.t}", {"val_last_eval_t": val.last_eval_t})
            for d in sorted(ckpt_dir.glob("ckpt_*"), key=lambda p: int(p.name.split("_")[1]))[:-2]:
                shutil.rmtree(d, ignore_errors=True)
            if args.debug_interrupt_at is not None and tr.t >= args.debug_interrupt_at:
                raise Interrupted(f"debug interrupt after checkpoint at t={tr.t}")
        if args.curriculum == "on" and not tr.switched and tr.t >= boundary:
            tr.switched = True
            for e in tr.venv.envs:
                e.unwrapped.set_prebought(None)
            log_curriculum(cur_log, tr.t, "B", None)
        tr.vector_step()
        if tr.t // LOG_EVERY > last_log:
            last_log = tr.t // LOG_EVERY
            new = not prog_path.exists()
            with prog_path.open("a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["timestep", "epsilon", "updates", "buffer", "loss", "q_mean", "train_score_last100"])
                w.writerow([tr.t, round(epsilon(tr.t, tr.total, hp), 4), tr.updates, len(tr.buffer), round(tr.loss_acc / max(1, tr.n_loss), 6),
                            round(tr.q_acc / max(1, tr.n_loss), 6), round(float(np.mean(tr.eps_log)), 4) if tr.eps_log else ""])
            tr.loss_acc, tr.q_acc, tr.n_loss = 0.0, 0.0, 0
    val.evaluate(tr) if tr.t % args.val_freq == 0 else None
    wall = time.perf_counter() - t0
    th.save({"online": tr.online.state_dict(), "hyperparameters": hp, "timesteps": tr.t, "n_actions": tr.n_actions}, run_dir / "final.pt")
    identity.update(status="finished", timesteps_trained=int(tr.t), params_sha256=tr.sha(), params_optimizer_sha256=tr.sha(full=True),
                    updates=tr.updates, wall_seconds_last_segment=round(wall, 1), validation_seconds_last_segment=round(val.seconds, 1),
                    steps_per_second_incl_validation=round((tr.t - start_t) / max(wall, 1e-9), 1))
    ident_path.write_text(json.dumps(identity, indent=2))
    print(f"[done] {tr.t} steps in {wall:.0f}s = {(tr.t - start_t) / max(wall, 1e-9):.0f} steps/s -> {run_dir}", flush=True)
    return run_dir


if __name__ == "__main__":
    main()
