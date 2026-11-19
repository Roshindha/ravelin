"""B11: masked Double DQN (rl/ddqn.py, rl/train_ddqn_v2.py). TUNE-train / TUNE-val only; FINAL never loaded (ledger asserted unchanged)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from breachlens import pool as P
from breachlens.env import BreachEnv
from rl import ddqn as D
from rl import train_ddqn_v2 as T
from rl.augment import N_SYMMETRIES
from rl.tta import tta_action, tta_scores

ROOT = Path(__file__).resolve().parent.parent
TINY = ["--timesteps", "1600", "--n-envs", "4", "--buffer-size", "2000", "--batch-size", "16", "--learning-starts", "200",
        "--target-update-interval", "50", "--val-freq", "800", "--checkpoint-freq", "400"]


def _ledger() -> bytes:
    return P.LEDGER_PATH.read_bytes() if P.LEDGER_PATH.exists() else b""


@pytest.fixture(autouse=True)
def _models_dir(tmp_path, monkeypatch):
    before = _ledger()
    monkeypatch.setattr(T, "MODELS_DIR", tmp_path)
    yield
    assert _ledger() == before, "the FINAL-access ledger changed during a B11 test"


def run(name, extra=(), base=TINY):
    return T.main(["--run-name", name, *base, *extra])


def ident(name):
    return json.loads((T.MODELS_DIR / name / "model_identity.json").read_text())


# ------------------------------------------------------------------------------------------------ 1. masks never leak
def test_masked_argmax_never_selects_an_illegal_action_even_with_huge_illegal_q():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n = int(rng.integers(1, 6))
        masks = rng.random((n, 267)) < rng.choice([0.01, 0.1, 0.5, 0.9])
        masks[np.arange(n), rng.integers(267, size=n)] = True          # at least one legal action per row
        q = rng.normal(size=(n, 267))
        q[~masks] += 1e6                                                # illegal actions look irresistible
        a = D.greedy_actions(q, masks)
        assert masks[np.arange(n), a].all()
        assert (a == np.argmax(np.where(masks, q, -np.inf), axis=1)).all()
    with pytest.raises(ValueError):
        D.greedy_actions(np.zeros((1, 4)), np.zeros((1, 4), bool))


def test_epsilon_greedy_samples_only_legal_actions_and_covers_them():
    rng = np.random.default_rng(1)
    seen = np.zeros(267, bool)
    for _ in range(3000):
        n = 8
        masks = rng.random((n, 267)) < 0.05
        masks[np.arange(n), rng.integers(267, size=n)] = True
        q = rng.normal(size=(n, 267))
        q[~masks] += 1e6
        for eps in (0.0, 0.3, 1.0):
            a = D.select_actions(q, masks, eps, rng.random(n), rng.random(n))
            assert masks[np.arange(n), a].all()
        a = D.select_actions(q, masks, 1.0, np.zeros(n), rng.random(n))          # pure exploration: uniform over the legal set
        seen[a] = True
        assert (D.select_actions(q, masks, 0.0, rng.random(n), rng.random(n)) == D.greedy_actions(q, masks)).all()
    assert seen.sum() > 200                                              # exploration reaches many different actions
    # uniform over a small legal set
    m = np.zeros((1, 267), bool)
    m[0, [3, 10, 200]] = True
    counts = np.bincount([int(D.select_actions(np.zeros((1, 267)), m, 1.0, np.zeros(1), rng.random(1))[0]) for _ in range(3000)], minlength=267)
    assert set(np.flatnonzero(counts)) == {3, 10, 200} and counts[[3, 10, 200]].min() > 800


def test_real_env_episodes_under_pure_exploration_and_greedy_never_raise():
    env = BreachEnv(source="TUNE-train", role="train")
    rng = np.random.default_rng(2)
    net = D.QNet(D.default_obs_space(), 267)
    for eps in (1.0, 0.0):
        obs, _ = env.reset(seed=5)
        done, steps = False, 0
        while not done:
            with th.no_grad():
                q = net(D.to_tensors({k: v[None] for k, v in obs.items()})).numpy()
            a = D.select_actions(q, env.action_masks()[None], eps, rng.random(1), rng.random(1))
            obs, _, done, _, _ = env.step(int(a[0]))                    # BreachEnv raises IllegalActionError on an illegal action
            steps += 1
        assert steps >= 2


# ------------------------------------------------------------------------------------------------ 2. next mask + Double-Q target
class _Chain:
    """Toy: 3 states (one-hot in `state`), 4 actions. s0 -a-> s1 (r = 0.25), s1 -a-> s2 terminal (r = 1.0). Legal: s0 {0,1,2,3}, s1 {0,2,3}, s2 none."""
    LEGAL = {0: [0, 1, 2, 3], 1: [0, 2, 3], 2: []}

    @staticmethod
    def obs(s):
        st = np.zeros(3, np.float32)
        st[s] = 1.0
        return {"grid": np.zeros((1, 1, 1), np.float32), "state": st}

    @classmethod
    def mask(cls, s):
        m = np.zeros(4, bool)
        m[cls.LEGAL[s]] = True
        return m


class _Table(th.nn.Module):
    """Tabular Q: obs['state'] is one-hot, Q = state @ table."""

    def __init__(self, table):
        super().__init__()
        self.table = th.as_tensor(np.asarray(table, np.float32))

    def forward(self, obs):
        return obs["state"] @ self.table


def _toy_buffer():
    buf = D.ReplayBuffer(8, 2, (1, 1, 1), 3, 4)
    # env 0: s0 -> s1 (non-terminal, reward 0.25, action 0); env 1: s1 -> s2 (terminal, reward 1.0, action 2)
    obs = {"grid": np.zeros((2, 1, 1, 1), np.float32), "state": np.stack([_Chain.obs(0)["state"], _Chain.obs(1)["state"]])}
    nxt = {"grid": np.zeros((2, 1, 1, 1), np.float32), "state": np.stack([_Chain.obs(1)["state"], _Chain.obs(2)["state"]])}
    buf.add(obs, np.array([0, 2]), np.array([0.25, 1.0]), np.array([False, True]), nxt, np.stack([_Chain.mask(1), _Chain.mask(2)]))
    # one more step so that slot 0 has an in-buffer successor: env 0: s1 -> s2 terminal (r = 0.5); env 1: s0 -> s1 (r = 0.0)
    obs2 = nxt | {"state": np.stack([_Chain.obs(1)["state"], _Chain.obs(0)["state"]])}
    nxt2 = {"grid": np.zeros((2, 1, 1, 1), np.float32), "state": np.stack([_Chain.obs(2)["state"], _Chain.obs(1)["state"]])}
    buf.add(obs2, np.array([3, 1]), np.array([0.5, 0.0]), np.array([True, False]), nxt2, np.stack([_Chain.mask(2), _Chain.mask(1)]))
    return buf


def test_buffer_stores_the_next_state_legal_mask_and_zero_mask_for_terminals():
    buf = _toy_buffer()
    assert (buf.next_mask[0, 0] == _Chain.mask(1)).all() and (buf.next_mask[1, 1] == _Chain.mask(1)).all()
    assert not buf.next_mask[1, 0].any() and not buf.next_mask[0, 1].any()                   # terminal transitions: no next state
    with pytest.raises(ValueError):                                                             # non-terminal with an empty next mask is a bug
        buf.add(buf_obs := {"grid": np.zeros((2, 1, 1, 1), np.float32), "state": np.zeros((2, 3), np.float32)}, np.zeros(2, int), np.zeros(2),
                np.zeros(2, bool), buf_obs, np.zeros((2, 4), bool))


def test_buffer_next_obs_is_the_successor_slot_or_the_kept_last_next():
    buf = _toy_buffer()
    b = buf._gather(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
    nxt_state = b["next_obs"]["state"].argmax(axis=1)
    cur_state = b["obs"]["state"].argmax(axis=1)
    assert cur_state.tolist() == [0, 1, 1, 0] and nxt_state[[0, 3]].tolist() == [1, 1]          # slot 0 -> slot 1's obs; newest slot -> last_next
    assert nxt_state[1] == 2                                                                      # newest of env 0 (terminal) -> last_next = s2


def test_double_q_target_matches_the_hand_computed_value_exactly():
    # transitions (all exactly representable): gamma = 0.5
    #  A: s0 -a0-> s1, r = 0.25, not terminal; next legal {0,2,3}
    #     online Q(s1) = [5, 9, 1, 7]: the UNMASKED argmax is action 1 (illegal) -> the masked argmax is action 3 (7 > 5 > 1)
    #     target Q(s1) = [2, 3, 8, 4]:  Double-Q = Q_target(s1, 3) = 4    -> y = 0.25 + 0.5 * 4 = 2.25
    #     (plain max over legal of the target would be 8 -> 4.25; unmasked double would use Q_target(s1,1) = 3 -> 1.75)
    #  B: s1 -a2-> s2 terminal, r = 1.0 -> y = 1.0 (no bootstrap even though the tables have values at s2)
    #  C: s1 -a3-> s2 terminal, r = 0.5 -> y = 0.5
    #  D: s0 -a1-> s1, r = 0.0, not terminal -> y = 0.5 * 4 = 2.0
    online = _Table([[0, 0, 0, 0], [5, 9, 1, 7], [100, 100, 100, 100]])
    target = _Table([[0, 0, 0, 0], [2, 3, 8, 4], [50, 50, 50, 50]])
    buf = _toy_buffer()
    b = buf.to_torch(buf._gather(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])))
    y = D.compute_targets(online, target, b, gamma=0.5)
    assert y.tolist() == [2.25, 1.0, 0.5, 2.0]
    # the three wrong variants really are different on transition A
    legal = th.as_tensor(_Chain.mask(1))
    q_on, q_tg = online.table[1][None], target.table[1][None]
    assert D.masked_q(q_on, legal[None]).argmax().item() == 3 and q_on.argmax().item() == 1
    assert D.masked_q(q_tg, legal[None]).max().item() == 8.0 and q_tg[0, 1].item() == 3.0


def test_update_uses_the_double_q_target_end_to_end():
    """One real gradient step on the toy buffer moves Q(s,a) of the sampled transition towards y (and only the online net changes)."""
    space = spaces.Dict({"grid": spaces.Box(0, 1, (1, 3, 3), np.float32), "state": spaces.Box(0, 1, (3,), np.float32)})
    th.manual_seed(0)
    net = D.QNet(space, 4, (8,))
    tgt = type(net)(space, 4, (8,))
    tgt.load_state_dict(net.state_dict())
    before = {k: v.clone() for k, v in tgt.state_dict().items()}
    o = {"grid": np.zeros((2, 1, 3, 3), np.float32), "state": np.eye(3, dtype=np.float32)[[0, 1]]}
    b = {"obs": D.to_tensors(o), "next_obs": D.to_tensors(o), "action": th.tensor([0, 2]), "reward": th.tensor([0.25, 1.0]), "done": th.tensor([False, True]),
         "next_mask": th.as_tensor(np.stack([_Chain.mask(1), np.zeros(4, bool)]))}
    y = D.compute_targets(net, tgt, b, 0.5)
    assert y[1].item() == 1.0 and y[0].item() == pytest.approx(0.25 + 0.5 * tgt(b["next_obs"])[0, net(b["next_obs"])[0].masked_fill(~b["next_mask"][0], -1e9).argmax()].item())
    assert all(th.equal(before[k], v) for k, v in tgt.state_dict().items())


def test_buffer_round_trips_exactly_and_samples_identically(tmp_path):
    env = DummyVecEnv([lambda i=i: BreachEnv(source="TUNE-train", role="train", prebought="random") for i in range(3)])
    env.seed(0)
    obs = env.reset()
    buf = D.ReplayBuffer(30, 3, (9, 14, 14), 11, 267)          # 10 slots per env: wraps around
    rng = np.random.default_rng(0)
    for _ in range(17):
        masks = np.stack(env.env_method("action_masks"))
        a = np.array([int(rng.choice(np.flatnonzero(m))) for m in masks])
        obs2, r, d, _ = env.step(a)
        masks2 = np.stack(env.env_method("action_masks"))
        # the stored next mask is exactly the env's legal mask after the step (non-terminal)
        buf.add(obs, a, r, d, obs2, masks2)
        assert (buf.next_mask[:, (buf.pos - 1) % buf.cap][~d] == masks2[~d]).all() and not buf.next_mask[:, (buf.pos - 1) % buf.cap][d].any()
        obs = obs2
    assert buf.n == buf.cap and len(buf) == 30
    buf.save(tmp_path / "b")
    buf2 = D.ReplayBuffer.load(tmp_path / "b", 30, 3, (9, 14, 14), 11, 267)
    assert buf2.sha256() == buf.sha256() and (buf2.pos, buf2.n) == (buf.pos, buf.n)
    s1, s2 = buf.sample(16, np.random.default_rng(9)), buf2.sample(16, np.random.default_rng(9))
    for k in ("action", "reward", "done", "next_mask"):
        assert np.array_equal(s1[k], s2[k])
    for k in ("grid", "state"):
        assert np.array_equal(s1["obs"][k], s2["obs"][k]) and np.array_equal(s1["next_obs"][k], s2["next_obs"][k])
    # a partially filled buffer round-trips too
    small = D.ReplayBuffer(30, 3, (9, 14, 14), 11, 267)
    small.add(obs, a, r, d, obs2, masks2)
    small.save(tmp_path / "s")
    assert D.ReplayBuffer.load(tmp_path / "s", 30, 3, (9, 14, 14), 11, 267).sha256() == small.sha256()


def test_next_obs_of_a_non_terminal_transition_is_the_real_next_observation():
    env = DummyVecEnv([lambda i=i: BreachEnv(source="TUNE-train", role="train", prebought="random") for i in range(2)])
    env.seed(3)
    obs = env.reset()
    buf = D.ReplayBuffer(40, 2, (9, 14, 14), 11, 267)
    rng = np.random.default_rng(1)
    log = []
    for _ in range(12):
        masks = np.stack(env.env_method("action_masks"))
        a = np.array([int(rng.choice(np.flatnonzero(m))) for m in masks])
        obs2, r, d, _ = env.step(a)
        buf.add(obs, a, r, d, obs2, np.stack(env.env_method("action_masks")))
        log.append((obs, obs2, d))
        obs = obs2
    for e in range(2):
        for off in range(12):
            b = buf._gather(np.array([e]), np.array([off]))
            o, o2, d = log[off]
            if not d[e]:
                assert np.array_equal(b["next_obs"]["grid"][0], o2["grid"][e]) and np.array_equal(b["next_obs"]["state"][0], o2["state"][e])
            assert np.array_equal(b["obs"]["grid"][0], o["grid"][e])


# ------------------------------------------------------------------------------------------------ 3. training: wiring, determinism, resume
def test_tiny_wiring_run_end_to_end():
    d = run("w")
    assert {p.name for p in d.iterdir()} >= {"final.pt", "model_identity.json", "val_eval.csv", "val_layouts.jsonl", "curriculum.log", "checkpoints"}
    i = ident("w")
    assert i["status"] == "finished" and i["timesteps_trained"] == 1600 and i["train_set"] == "TUNE-train" and i["val_set"] == "TUNE-val"
    assert i["algorithm"] == "MaskedDoubleDQN" and i["pool_sha256"] and i["git_commit"]
    phases = [json.loads(ln) for ln in (d / "curriculum.log").read_text().splitlines()]
    assert [p["phase"] for p in phases] == ["A", "B"] and phases[1]["timestep"] >= 1600 // 3
    import csv
    with (d / "val_eval.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert sorted({int(r["timestep"]) for r in rows}) == [0, 800, 1600] and {r["group"] for r in rows} == {"ALL", "spine_pods", "archipelago", "bunker_nest"}
    assert not (d / "best.pt").exists()
    ck = th.load(d / "final.pt", weights_only=False)
    assert ck["timesteps"] == 1600 and ck["n_actions"] == 267
    net = D.QNet(D.default_obs_space(), 267, ck["hyperparameters"]["net_arch"])
    net.load_state_dict(ck["online"])


def test_trainer_only_offers_dev_side_sets():
    assert set(T.TRAIN_TO_VAL) == {"TUNE-train", "DEV-train"}
    with pytest.raises(SystemExit):
        T.parse_args(["--run-name", "x", "--train-set", "FINAL-familiar"])
    src = (ROOT / "rl" / "train_ddqn_v2.py").read_text() + (ROOT / "rl" / "ddqn.py").read_text()
    assert 'role="eval"' not in src and "purpose=" not in src


def test_two_fresh_runs_with_the_same_seed_are_bit_identical_and_a_different_seed_differs():
    run("a1"), run("a2"), run("b", ["--seed", "1"])
    a1, a2, b = ident("a1"), ident("a2"), ident("b")
    for k in ("params_sha256", "params_optimizer_sha256", "updates"):       # optimizer hash includes target net, Adam state and the replay buffer
        assert a1[k] == a2[k]
    assert a1["params_sha256"] != b["params_sha256"]
    for f in ("val_layouts.jsonl", "curriculum.log"):
        assert (T.MODELS_DIR / "a1" / f).read_text() == (T.MODELS_DIR / "a2" / f).read_text()


@pytest.mark.parametrize("at", [400, 1200])          # before / after the phase switch at 533 (and after learning starts at 200)
def test_resume_is_bit_identical_to_an_uninterrupted_run(at):
    run("full")
    with pytest.raises(T.Interrupted):
        run("part", ["--debug-interrupt-at", str(at)])
    assert not (T.MODELS_DIR / "part" / "final.pt").exists()
    run("part", ["--resume"])
    f, p = ident("full"), ident("part")
    assert p["resumes"][0]["at_timestep"] >= at
    for k in ("params_sha256", "params_optimizer_sha256", "updates", "timesteps_trained"):
        assert f[k] == p[k]
    for name in ("val_layouts.jsonl", "curriculum.log"):
        assert (T.MODELS_DIR / "full" / name).read_text() == (T.MODELS_DIR / "part" / name).read_text()
    import csv

    def rows(n):
        with (T.MODELS_DIR / n / "val_eval.csv").open() as fh:
            return [{k: v for k, v in r.items() if k != "eval_seconds"} for r in csv.DictReader(fh)]
    assert rows("full") == rows("part")


def test_negative_control_resume_without_restoring_the_rng_streams_is_not_bit_identical():
    run("nc_full")
    with pytest.raises(T.Interrupted):
        run("nc_part", ["--debug-interrupt-at", "800"])
    run("nc_part", ["--resume", "--debug-skip-rng-restore"])
    assert ident("nc_part")["params_sha256"] != ident("nc_full")["params_sha256"]


def test_epsilon_schedule_and_resume_restores_target_net_and_counters():
    hp = {"eps_start": 1.0, "eps_end": 0.05, "eps_decay_fraction": 0.25}
    assert T.epsilon(0, 1000, hp) == 1.0 and T.epsilon(250, 1000, hp) == pytest.approx(0.05) and T.epsilon(900, 1000, hp) == pytest.approx(0.05)
    assert T.epsilon(125, 1000, hp) == pytest.approx(0.525)
    with pytest.raises(T.Interrupted):
        run("c", ["--debug-interrupt-at", "800"])
    ck = T.latest_checkpoint(T.MODELS_DIR / "c")
    tr, st = T.Trainer.from_checkpoint(ck, {**D.DDQN_CONFIG, "n_envs": 4, "buffer_size": 2000, "batch_size": 16, "learning_starts": 200,
                                            "target_update_interval": 50}, 0, 1600, 533)
    assert tr.t == 800 and tr.updates == 151 and tr.credit == 0 and tr.switched and len(tr.buffer) == 800   # updates at t = 200, 204, ..., 800
    nets = th.load(ck / "nets.pt", weights_only=False)
    # the target was last synced at update 150 (interval 50); update 151 moved the online net, so they differ by exactly one step
    assert not all(th.equal(nets["online"][k], nets["target"][k]) for k in nets["online"])
    assert all(th.equal(tr.target.state_dict()[k], nets["target"][k]) for k in nets["target"])


# ------------------------------------------------------------------------------------------------ 4. learning on a toy layout
def test_tiny_run_improves_over_random_on_a_toy_layout():
    """One TUNE-train layout, a fixed pre-bought army (deployment-only game): DDQN's greedy deployment beats uniformly random legal deployment."""
    recs = BreachEnv(source="TUNE-train", role="train").records[:1]
    army = [0, 3, 3, 0, 0]
    hp = {**D.DDQN_CONFIG, "n_envs": 8, "buffer_size": 20_000, "batch_size": 64, "learning_starts": 1000, "target_update_interval": 200, "eps_decay_fraction": 0.3}
    venv = DummyVecEnv([lambda: BreachEnv(records=recs, role="train", prebought=army) for _ in range(8)])
    tr = T.Trainer(hp, venv, 0, 24_000, 0)

    def play(act_fn, n=20):
        env = BreachEnv(records=recs, role="train", prebought=army)
        out = []
        for s in range(n):
            obs, _ = env.reset(seed=1000 + s)
            done = False
            while not done:
                obs, r, done, _, _ = env.step(act_fn(obs, env.action_masks()))
            out.append(r)
        return float(np.mean(out))
    rng = np.random.default_rng(0)
    random_score = play(lambda o, m: int(rng.choice(np.flatnonzero(m))))
    while tr.t < 24_000:
        tr.vector_step()
    pol = D.DDQNPolicy(tr.online)
    learned = play(lambda o, m: pol.act(o, m))
    assert learned > random_score + 0.03, (learned, random_score)


# ------------------------------------------------------------------------------------------------ 5. TTA with Q-values (Gate changelog (n), (o))
def test_ddqn_tta_is_the_plain_mean_of_mapped_back_q_values_then_masked_argmax():
    env = BreachEnv(source="TUNE-train", role="train")
    obs, _ = env.reset(seed=4)
    mask = env.action_masks()
    th.manual_seed(0)
    net = D.QNet(D.default_obs_space(), 267)
    fn = D.q_values_fn(net)
    sc = tta_scores(fn, obs, mask, aggregate="mean")
    from rl.augment import transform_mask, transform_obs, untransform_probs
    manual = np.mean([untransform_probs(fn(transform_obs(obs, s), transform_mask(mask, s)), s) for s in range(N_SYMMETRIES)], axis=0)
    assert np.allclose(sc, manual)
    a = tta_action(fn, obs, mask, aggregate="mean")
    assert mask[a] and a == int(np.argmax(np.where(mask, manual, -np.inf)))
    # identity-only TTA equals the plain greedy action
    assert tta_action(fn, obs, mask, syms=[0], aggregate="mean") == int(D.greedy_actions(fn(obs, mask)[None], mask[None])[0])
