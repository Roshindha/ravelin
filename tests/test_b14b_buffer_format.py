"""B14b: the compressed replay-buffer checkpoint format is an exact re-encoding (every field), the legacy format still loads, unknown formats are refused."""
import json

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from breachlens.env import BreachEnv
from rl import ddqn as D

SHAPE, SD, NA = (9, 14, 14), 11, 267


def filled(n_steps=17, cap=30, n_envs=3):
    env = DummyVecEnv([lambda i=i: BreachEnv(source="TUNE-train", role="train", prebought="random") for i in range(n_envs)])
    env.seed(0)
    obs = env.reset()
    buf = D.ReplayBuffer(cap, n_envs, SHAPE, SD, NA)
    rng = np.random.default_rng(0)
    for _ in range(n_steps):
        masks = np.stack(env.env_method("action_masks"))
        a = np.array([int(rng.choice(np.flatnonzero(m))) for m in masks])
        obs2, r, d, _ = env.step(a)
        buf.add(obs, a, r, d, obs2, np.stack(env.env_method("action_masks")))
        obs = obs2
    return buf


def same(a, b):
    assert (a.pos, a.n, a.cap, a.n_envs) == (b.pos, b.n, b.cap, b.n_envs)
    for name in D.ReplayBuffer._ARRAYS:
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and x.shape == y.shape, name
        if name in D.ReplayBuffer._PER_SLOT:
            x, y = x[:, :a.n], y[:, :a.n]
        assert np.array_equal(x, y), name                      # exact, bitwise for float32 (no NaNs in the data)
        assert x.tobytes() == y.tobytes(), name
    assert a.sha256() == b.sha256()


@pytest.mark.parametrize("steps", [0, 1, 7, 17])               # empty, partial, full + wrapped
def test_round_trip_every_field_is_exact(tmp_path, steps):
    buf = filled(steps) if steps else D.ReplayBuffer(30, 3, SHAPE, SD, NA)
    buf.save(tmp_path / "b")
    assert json.loads((tmp_path / "b" / "meta.json").read_text())["format"] == 2
    same(buf, D.ReplayBuffer.load(tmp_path / "b", 30, 3, SHAPE, SD, NA))


def test_legacy_format_1_still_loads(tmp_path):
    buf = filled()
    d = tmp_path / "old"
    d.mkdir()
    for name in D.ReplayBuffer._ARRAYS:
        a = getattr(buf, name)
        np.save(d / f"{name}.npy", a[:, :buf.n] if name in D.ReplayBuffer._PER_SLOT else a)
    (d / "meta.json").write_text(json.dumps({"pos": buf.pos, "n": buf.n, "cap": buf.cap, "n_envs": buf.n_envs}))
    same(buf, D.ReplayBuffer.load(d, 30, 3, SHAPE, SD, NA))


def test_unknown_format_and_mismatched_buffer_are_refused_with_a_clear_message(tmp_path):
    buf = filled()
    buf.save(tmp_path / "b")
    meta = json.loads((tmp_path / "b" / "meta.json").read_text())
    (tmp_path / "b" / "meta.json").write_text(json.dumps({**meta, "format": 99}))
    with pytest.raises(ValueError, match="unknown format 99"):
        D.ReplayBuffer.load(tmp_path / "b", 30, 3, SHAPE, SD, NA)
    (tmp_path / "b" / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="does not match the configured buffer"):
        D.ReplayBuffer.load(tmp_path / "b", 60, 3, SHAPE, SD, NA)


def test_a_channel_with_more_than_256_distinct_values_falls_back_to_raw_float32_exactly(tmp_path):
    buf = filled()
    rng = np.random.default_rng(1)
    buf.grid[:, :buf.n, 3] = rng.random(buf.grid[:, :buf.n, 3].shape, dtype=np.float32)     # channel 3 now has thousands of distinct values
    buf.save(tmp_path / "b")
    with np.load(tmp_path / "b" / "buffer.npz") as z:
        assert "grid" in z.files and "grid_codes" not in z.files
    same(buf, D.ReplayBuffer.load(tmp_path / "b", 30, 3, SHAPE, SD, NA))
