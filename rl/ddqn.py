"""B11: masked Double DQN building blocks (docs/SPEC_V2.md 7, 11 B11). Own PyTorch implementation; no SB3 algorithm code.

Masks are used in BOTH places where they matter:
  * acting: epsilon-greedy draws a uniformly random LEGAL action; the greedy action is the masked argmax of the online Q-values;
  * the Double-Q target: a' = argmax over the LEGAL next actions of the ONLINE net, evaluated by the TARGET net;
    y = r + gamma * Q_target(s', a')  and  y = r for terminal transitions (no bootstrap).
The replay buffer therefore stores the legal-action mask of the NEXT state with every transition.

Replay memory layout (memory, not semantics): per-env ring buffers. The next observation of slot s is slot s+1 of the same env (the envs
are stepped in lock-step, so each env's stream is contiguous); only the newest slot's next observation is kept separately (`last_next_*`).
For a terminal transition the stored next-state mask is all False and the next observation is never read (no bootstrap).
Checkpoints are written losslessly (compressed, exact re-encodings; see ReplayBuffer.save).
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from breachlens.baselines import Policy, _ep
from breachlens.constants import N_UNIT_TYPES
from breachlens.env import GRID, N_GRID_CHANNELS, N_STATE, BreachEnv

from .features import SiegeFeaturesExtractor

DDQN_CONFIG = dict(n_envs=8, buffer_size=50_000, batch_size=64, learning_rate=1e-4, gamma=1.0, train_every=4, learning_starts=5_000,
                   target_update_interval=1_000, eps_start=1.0, eps_end=0.05, eps_decay_fraction=0.25, max_grad_norm=10.0, net_arch=[256, 256])


def default_obs_space() -> spaces.Dict:
    return spaces.Dict({"grid": spaces.Box(0.0, 1.0, (N_GRID_CHANNELS, GRID, GRID), np.float32), "state": spaces.Box(0.0, 1.0, (N_STATE,), np.float32)})


# ------------------------------------------------------------------------------------------------------------ network
class QNet(nn.Module):
    """Same feature extractor as PPO (CNN + MLP -> 320) followed by an MLP head -> one Q-value per action."""

    def __init__(self, obs_space: spaces.Dict, n_actions: int, net_arch=(256, 256)) -> None:
        super().__init__()
        self.features = SiegeFeaturesExtractor(obs_space)
        layers, d = [], self.features.features_dim
        for h in net_arch:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_actions))
        self.head = nn.Sequential(*layers)
        self.n_actions = n_actions

    def forward(self, obs: dict) -> th.Tensor:
        return self.head(self.features(obs))


def to_tensors(obs: dict) -> dict:
    return {k: th.as_tensor(np.asarray(v, dtype=np.float32)) for k, v in obs.items()}


def masked_q(q: th.Tensor, mask: th.Tensor) -> th.Tensor:
    return q.masked_fill(~mask, float("-inf"))


# ------------------------------------------------------------------------------------------------------------ action selection
def greedy_actions(q: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Masked argmax per row (lowest index wins an exact tie). Illegal actions can never be returned, whatever their Q-value."""
    q = np.asarray(q, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    if not masks.any(axis=1).all():
        raise ValueError("a row has no legal action")
    return np.argmax(np.where(masks, q, -np.inf), axis=1).astype(np.int64)


def select_actions(q: np.ndarray, masks: np.ndarray, eps: float, u: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Epsilon-greedy over LEGAL actions. u < eps -> explore: the legal action at rank floor(r * n_legal) (uniform over legal);
    otherwise masked argmax. `u`, `r` are uniform [0,1) draws, one pair per row, always drawn by the caller (fixed RNG consumption)."""
    masks = np.asarray(masks, dtype=bool)
    out = greedy_actions(q, masks)
    for i in np.flatnonzero(np.asarray(u) < eps):
        legal = np.flatnonzero(masks[i])
        out[i] = legal[min(int(r[i] * len(legal)), len(legal) - 1)]
    return out


# ------------------------------------------------------------------------------------------------------------ the Double-Q target
def double_q_target(q_online_next: th.Tensor, q_target_next: th.Tensor, next_mask: th.Tensor, reward: th.Tensor, done: th.Tensor, gamma: float) -> th.Tensor:
    """y = r + gamma * Q_target(s', argmax_{a' legal in s'} Q_online(s', a')); y = r where `done` (no bootstrap)."""
    a_star = masked_q(q_online_next, next_mask).argmax(dim=1, keepdim=True)
    boot = q_target_next.gather(1, a_star).squeeze(1)
    return th.where(done, reward, reward + gamma * boot)


def compute_targets(online, target, batch: dict, gamma: float) -> th.Tensor:
    with th.no_grad():
        return double_q_target(online(batch["next_obs"]), target(batch["next_obs"]), batch["next_mask"], batch["reward"], batch["done"], gamma)


# ------------------------------------------------------------------------------------------------------------ replay buffer
class ReplayBuffer:
    def __init__(self, capacity: int, n_envs: int, grid_shape, state_dim: int, n_actions: int) -> None:
        if capacity % n_envs:
            raise ValueError("capacity must be a multiple of n_envs")
        self.n_envs, self.cap, self.n_actions = n_envs, capacity // n_envs, n_actions
        self.grid_shape, self.state_dim = tuple(grid_shape), state_dim
        z = lambda shape, dt: np.zeros((n_envs, self.cap, *shape), dtype=dt)       # noqa: E731  (lazily paged by the OS)
        self.grid, self.state = z(self.grid_shape, np.float32), z((state_dim,), np.float32)
        self.action, self.reward, self.done = z((), np.int64), z((), np.float32), z((), bool)
        self.next_mask = z((n_actions,), bool)
        self.last_next_grid = np.zeros((n_envs, *self.grid_shape), np.float32)
        self.last_next_state = np.zeros((n_envs, state_dim), np.float32)
        self.pos, self.n = 0, 0                                                    # next write slot, filled slots per env

    def __len__(self) -> int:
        return self.n * self.n_envs

    def add(self, obs: dict, action, reward, done, next_obs: dict, next_mask) -> None:
        """One transition per env (arrays with leading dim n_envs). `next_mask` = legal actions of the NEXT state; for a terminal transition it is
        stored as all False (the next state does not exist / is never bootstrapped from)."""
        done = np.asarray(done, dtype=bool)
        next_mask = np.array(next_mask, dtype=bool)
        if not next_mask[~done].any(axis=1).all():
            raise ValueError("a non-terminal transition has an empty next-state mask")
        next_mask[done] = False
        p = self.pos
        self.grid[:, p], self.state[:, p] = obs["grid"], obs["state"]
        self.action[:, p], self.reward[:, p], self.done[:, p], self.next_mask[:, p] = action, reward, done, next_mask
        self.last_next_grid[:], self.last_next_state[:] = next_obs["grid"], next_obs["state"]
        self.pos = (p + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def _gather(self, e: np.ndarray, off: np.ndarray) -> dict:
        start = self.pos if self.n == self.cap else 0
        slot = (start + off) % self.cap
        newest = off == self.n - 1
        nxt = (slot + 1) % self.cap
        ng = np.where(newest[:, None, None, None], self.last_next_grid[e], self.grid[e, nxt])
        ns = np.where(newest[:, None], self.last_next_state[e], self.state[e, nxt])
        return {"obs": {"grid": self.grid[e, slot], "state": self.state[e, slot]}, "action": self.action[e, slot], "reward": self.reward[e, slot],
                "done": self.done[e, slot], "next_obs": {"grid": ng, "state": ns}, "next_mask": self.next_mask[e, slot]}

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        if self.n < 2:
            raise RuntimeError("buffer too small to sample")
        e = rng.integers(self.n_envs, size=batch_size)
        off = rng.integers(self.n, size=batch_size)
        return self._gather(e, off)

    def to_torch(self, b: dict) -> dict:
        return {"obs": to_tensors(b["obs"]), "next_obs": to_tensors(b["next_obs"]), "action": th.as_tensor(b["action"]), "reward": th.as_tensor(b["reward"]),
                "done": th.as_tensor(b["done"]), "next_mask": th.as_tensor(b["next_mask"])}

    # -- persistence (exact). Format 2 (B14b): one compressed buffer.npz. The in-memory arrays are unchanged (float32 grid / state, bool masks), so training
    # math and sha256() are untouched; only the files shrink, by exact re-encodings: grid -> uint8 codes + a per-channel codebook of the distinct float32 values
    # (each grid channel takes <= 7 distinct values; if a channel ever had > 256 the raw float32 grid is stored instead), next_mask -> np.packbits (bool is 1 bit of
    # information), everything else as is. Format 1 (no "format" key in meta.json: one .npy per array) is still read.
    _ARRAYS = ("grid", "state", "action", "reward", "done", "next_mask", "last_next_grid", "last_next_state")
    _PER_SLOT = ("grid", "state", "action", "reward", "done", "next_mask")
    FORMAT = 2

    @staticmethod
    def _encode_grid(g: np.ndarray):
        """-> (codes uint8 same shape, codebook float32 (C, 256) padded with NaN) or None if some channel has > 256 distinct values."""
        n_ch = g.shape[-3]
        book = np.full((n_ch, 256), np.nan, np.float32)
        codes = np.empty(g.shape, np.uint8)
        for c in range(n_ch):
            ch = g[..., c, :, :]
            u = np.unique(ch)                                              # sorted; NaN-free (asserted by the round trip below)
            if len(u) > 256 or np.isnan(u).any():
                return None
            book[c, :len(u)] = u
            codes[..., c, :, :] = np.searchsorted(u, ch).astype(np.uint8)
        return codes, book

    @staticmethod
    def _decode_grid(codes: np.ndarray, book: np.ndarray) -> np.ndarray:
        out = np.empty(codes.shape, np.float32)
        for c in range(codes.shape[-3]):
            out[..., c, :, :] = book[c][codes[..., c, :, :]]
        return out

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        pay = {}
        for name in self._ARRAYS:
            a = getattr(self, name)
            a = np.ascontiguousarray(a[:, :self.n] if name in self._PER_SLOT else a)
            if name in ("next_mask",):
                pay[name + "_bits"] = np.packbits(a, axis=-1)
            elif name in ("grid", "last_next_grid"):
                enc = self._encode_grid(a)
                if enc is None:
                    pay[name] = a
                else:
                    pay[name + "_codes"], pay[name + "_book"] = enc
            else:
                pay[name] = a
        np.savez_compressed(d / "buffer.npz", **pay)
        (d / "meta.json").write_text(f'{{"format": {self.FORMAT}, "pos": {self.pos}, "n": {self.n}, "cap": {self.cap}, "n_envs": {self.n_envs}}}')

    @classmethod
    def load(cls, d: Path, capacity: int, n_envs: int, grid_shape, state_dim: int, n_actions: int) -> "ReplayBuffer":
        import json
        meta = json.loads((d / "meta.json").read_text())
        fmt = meta.get("format", 1)
        if fmt not in (1, cls.FORMAT):
            raise ValueError(f"replay buffer checkpoint {d} has unknown format {fmt!r} (this code reads formats 1 and {cls.FORMAT}); refusing to load")
        b = cls(capacity, n_envs, grid_shape, state_dim, n_actions)
        if not (meta["cap"] == b.cap and meta["n_envs"] == n_envs):
            raise ValueError(f"replay buffer checkpoint {d} does not match the configured buffer (cap/n_envs {meta['cap']}/{meta['n_envs']} vs {b.cap}/{n_envs})")
        b.pos, b.n = meta["pos"], meta["n"]
        if fmt == 1:
            for name in cls._ARRAYS:
                a = np.load(d / f"{name}.npy")
                (getattr(b, name)[:, :b.n] if name in cls._PER_SLOT else getattr(b, name)[:])[...] = a
            return b
        with np.load(d / "buffer.npz") as z:
            for name in cls._ARRAYS:
                if name == "next_mask":
                    a = np.unpackbits(z["next_mask_bits"], axis=-1, count=n_actions).astype(bool)
                elif name in ("grid", "last_next_grid"):
                    a = z[name] if name in z.files else cls._decode_grid(z[name + "_codes"], z[name + "_book"])
                else:
                    a = z[name]
                (getattr(b, name)[:, :b.n] if name in cls._PER_SLOT else getattr(b, name)[:])[...] = a
        return b

    def sha256(self) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(f"{self.pos},{self.n},{self.cap},{self.n_envs}".encode())
        for name in self._ARRAYS:
            a = getattr(self, name)
            h.update(np.ascontiguousarray(a[:, :self.n] if name in ("grid", "state", "action", "reward", "done", "next_mask") else a).tobytes())
        return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------ evaluation policy
class DDQNPolicy(Policy):
    """Deterministic masked argmax of the online Q-values on the canonical view, through the B8 evaluator (like ModelPolicy for PPO)."""

    def __init__(self, net: QNet, name: str = "DDQN") -> None:
        self.net, self.name = net, name

    def act(self, obs: dict, mask: np.ndarray) -> int:
        self.net.eval()
        with th.no_grad():
            q = self.net(to_tensors({k: np.asarray(v)[None] for k, v in obs.items()}))[0].numpy()
        return int(greedy_actions(q[None], np.asarray(mask, dtype=bool)[None])[0])

    def episodes(self, rec: dict, layout: dict, P) -> list[dict]:
        env = BreachEnv(layouts=[layout], role="train")
        obs, _ = env.reset(seed=zlib.crc32(rec["id"].encode()), options={"layout_index": 0})
        done = False
        while not done:
            obs, _, done, _, _ = env.step(self.act(obs, env.action_masks()))
        army = (np.bincount([t for t, _ in env.army], minlength=N_UNIT_TYPES) + np.asarray(env.owned)).tolist()
        return [_ep(env.result, env.bought, len(env.army), army=army)]


def q_values_fn(net: QNet):
    """probs_fn(obs, mask) -> (N_ACTIONS,) raw Q-values of ONE observation: adapter for rl.tta with aggregate="mean" (Gate changelog (n), (o))."""
    def fn(obs: dict, mask: np.ndarray) -> np.ndarray:
        net.eval()
        with th.no_grad():
            return net(to_tensors({k: np.asarray(v)[None] for k, v in obs.items()}))[0].numpy().astype(np.float64)
    return fn
