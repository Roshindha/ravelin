"""B10: v2 PPO pipeline (rl/train_ppo_v2.py, rl/eval_ppo_v2.py, rl/budget.py). TUNE-train / TUNE-val only; FINAL never loaded (ledger asserted unchanged)."""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch as th

from breachlens import pool as P
from breachlens.constants import N_UNIT_TYPES, PHASE_DEPLOY
from breachlens.env import UNIT_COST, BreachEnv
from rl import budget
from rl import eval_ppo_v2 as E
from rl import train_ppo_v2 as T
from rl.augment import N_SYMMETRIES, transform_mask, transform_obs
from rl.features import SiegeFeaturesExtractor
from rl.tta import tta_action, tta_scores

ROOT = Path(__file__).resolve().parent.parent
LEDGER = P.LEDGER_PATH
TINY = ["--timesteps", "640", "--n-envs", "2", "--n-steps", "16", "--batch-size", "16", "--n-epochs", "2", "--val-freq", "320", "--checkpoint-freq", "160"]


def _ledger_bytes() -> bytes:
    return LEDGER.read_bytes() if LEDGER.exists() else b""


@pytest.fixture(autouse=True)
def _models_dir(tmp_path, monkeypatch):
    before = _ledger_bytes()
    monkeypatch.setattr(T, "MODELS_DIR", tmp_path)
    yield
    assert _ledger_bytes() == before, "the FINAL-access ledger changed during a B10 test"


def run(name, extra=(), base=TINY):
    return T.main(["--run-name", name, *base, *extra])


def _rows(path):
    with Path(path).open(newline="") as fh:
        return [{k: v for k, v in r.items() if k != "eval_seconds"} for r in csv.DictReader(fh)]


# ------------------------------------------------------------------------------------------------ wiring
def test_tiny_wiring_run_end_to_end(tmp_path):
    d = run("w")
    assert {p.name for p in d.iterdir()} >= {"final.zip", "model_identity.json", "val_eval.csv", "val_layouts.jsonl", "curriculum.log", "progress.csv", "checkpoints"}
    assert not (d / "best.zip").exists()                     # nothing is selected on validation
    ident = json.loads((d / "model_identity.json").read_text())
    assert ident["status"] == "finished" and ident["timesteps_trained"] == 640 and ident["train_set"] == "TUNE-train" and ident["val_set"] == "TUNE-val"
    rows = _rows(d / "val_eval.csv")
    assert sorted({int(r["timestep"]) for r in rows}) == [0, 320, 640]                      # t = 0, every val-freq, and the end
    assert {r["group"] for r in rows} == {"ALL", "spine_pods", "archipelago", "bunker_nest"}
    lay = [json.loads(ln) for ln in (d / "val_layouts.jsonl").read_text().splitlines()]
    assert len(lay) == 3 * 60 and all(len(r["army"]) == N_UNIT_TYPES and 0 <= r["score"] <= 1 for r in lay)
    assert {r["id"] for r in lay if r["timestep"] == 0} == {r["id"] for r in lay if r["timestep"] == 640}
    from sb3_contrib import MaskablePPO
    m = MaskablePPO.load(d / "final.zip", device="cpu")
    assert m.num_timesteps == 640 and m.action_space.n == 267
    assert m.observation_space["grid"].shape == (9, 14, 14) and m.observation_space["state"].shape == (11,)


def test_trainer_only_offers_dev_side_sets():
    assert set(T.TRAIN_TO_VAL) == {"TUNE-train", "DEV-train"} and set(T.TRAIN_TO_VAL.values()) == {"TUNE-val", "DEV-validation"}
    with pytest.raises(SystemExit):
        T.parse_args(["--run-name", "x", "--train-set", "FINAL-familiar"])
    with pytest.raises(SystemExit):
        E.main(["--run-dir", "x", "--set", "FINAL-transfer"])
    with pytest.raises(SystemExit):
        E.main(["--run-dir", "x", "--set", "LAB-DESIGN"])
    with pytest.raises(PermissionError):
        BreachEnv(source="FINAL-familiar", role="train")
    src = (ROOT / "rl" / "train_ppo_v2.py").read_text() + (ROOT / "rl" / "eval_ppo_v2.py").read_text()
    assert 'role="eval"' not in src and "purpose=" not in src


def test_features_extractor_reads_v2_spaces():
    env = BreachEnv(source="TUNE-train", role="train")
    fe = SiegeFeaturesExtractor(env.observation_space)
    assert fe.features_dim == 320
    obs, _ = env.reset(seed=0)
    out = fe({k: th.as_tensor(v)[None] for k, v in obs.items()})
    assert out.shape == (1, 320)


# ------------------------------------------------------------------------------------------------ curriculum
@pytest.mark.parametrize("aug", ["none", "d4"])
def test_phase_a_random_armies_are_always_legal_and_varied(aug):
    venv = T.build_venv("TUNE-train", aug, 3, 3, "random")
    venv.seed(3)
    seen, n = np.zeros(N_UNIT_TYPES, bool), 0
    obs = venv.reset()
    for _ in range(120):
        for env in venv.envs:
            u = env.unwrapped
            counts = np.asarray(u.owned)
            assert u.phase == PHASE_DEPLOY                                           # DEPLOY phase from step 0
            assert 1 <= counts.sum() <= 20 and sum(c * k for c, k in zip(counts, UNIT_COST)) <= 240
            assert u.action_masks().any()
            seen |= counts > 0
            n += 1
        masks = np.stack(venv.env_method("action_masks"))
        acts = np.array([np.flatnonzero(m)[-1] for m in masks])                    # last legal action: a deploy (or WAIT)
        venv.step(acts)
    assert seen.all()                                                                # all five types appear, Menders included
    assert len({tuple(e.unwrapped.owned) for e in venv.envs}) >= 1


def test_curriculum_switch_is_logged_and_applied_lazily():
    d = run("c")
    log = [json.loads(ln) for ln in (d / "curriculum.log").read_text().splitlines()]
    assert [(e["phase"], e["prebought"]) for e in log] == [("A", "random"), ("B", None)]
    assert log[0]["timestep"] == 0 and log[1]["timestep"] == 214            # boundary = 640 // 3 = 213 -> first vec step with t >= 213 (n_envs 2)
    ident = json.loads((d / "model_identity.json").read_text())
    assert ident["curriculum_phase_b_start_timestep"] == 213 and ident["curriculum_phase_a_army"] == "random"
    # envs saved after the switch have prebought None
    ck = T.latest_checkpoint(d)
    st = pickle.loads((ck / "state.pkl").read_bytes())
    assert {e.unwrapped.prebought for e in st["venv"].envs} == {None}
    # lazy: changing prebought mid-episode does not touch the running episode
    env = BreachEnv(source="TUNE-train", role="train", prebought="random")
    env.reset(seed=0)
    before = list(env.owned)
    env.set_prebought(None)
    assert env.owned == before and env.phase == 1
    env.reset(seed=1)
    assert env.phase == 0 and env.owned == [0] * N_UNIT_TYPES


def test_curriculum_off_runs_the_full_game_from_step_zero():
    d = run("off", ["--curriculum", "off"])
    assert not (d / "curriculum.log").exists()
    ident = json.loads((d / "model_identity.json").read_text())
    assert ident["curriculum_phase_b_start_timestep"] is None


# ------------------------------------------------------------------------------------------------ identity
def test_model_identity_records_sha_commit_and_environment():
    d = run("i")
    ident = json.loads((d / "model_identity.json").read_text())
    gh = json.loads(P.GENERATOR_HASH_PATH.read_text())["generator_hash"]
    assert ident["pool_sha256"] == P.pool_sha256() == (ROOT / "data/v2/pool_v1.sha256").read_text().split()[0]
    assert ident["generator_hash"] == gh and len(ident["git_commit"]) == 40 and isinstance(ident["git_dirty"], bool)
    for k in ("sim_version", "score_version", "hyperparameters", "seed", "aug", "timesteps_requested", "rollout_size", "params_sha256", "config"):
        assert ident[k] not in (None, "")
    env = ident["environment"]
    assert env["torch_threads"] == th.get_num_threads() and {"python", "torch", "stable_baselines3", "cpu"} <= set(env)
    assert len(ident["params_sha256"]) == 64 and ident["n_actions"] == 267


# ------------------------------------------------------------------------------------------------ determinism and resume
def _final_state(d):
    from sb3_contrib import MaskablePPO
    m = MaskablePPO.load(d / "final.zip", device="cpu")
    ident = json.loads((d / "model_identity.json").read_text())
    return m, ident


def test_same_seed_twice_is_bit_identical_and_different_seed_differs():
    a, b, c = run("a"), run("b"), run("c", ["--seed", "1"])
    ia, ib, ic = (json.loads((x / "model_identity.json").read_text()) for x in (a, b, c))
    assert ia["params_sha256"] == ib["params_sha256"] and ia["params_optimizer_sha256"] == ib["params_optimizer_sha256"]
    assert ia["params_sha256"] != ic["params_sha256"]
    assert _rows(a / "val_eval.csv") == _rows(b / "val_eval.csv")
    assert (a / "val_layouts.jsonl").read_text() == (b / "val_layouts.jsonl").read_text()


@pytest.mark.parametrize("aug,interrupt_at", [("none", 160), ("none", 320), ("d4", 320), ("d4", 160)])
def test_resume_is_bit_identical_to_an_uninterrupted_run(aug, interrupt_at):
    full = run(f"full_{aug}_{interrupt_at}", ["--aug", aug])
    part = f"part_{aug}_{interrupt_at}"
    with pytest.raises(T.Interrupted):
        run(part, ["--aug", aug, "--debug-interrupt-at", str(interrupt_at)])
    pd = T.MODELS_DIR / part
    ck = T.latest_checkpoint(pd)
    assert ck is not None and int(ck.name.split("_")[1]) == interrupt_at
    m0, st = T.restore(ck)
    assert m0.num_timesteps == interrupt_at == st["num_timesteps"]                     # resume restores the timestep
    assert not (pd / "final.zip").exists()
    T.main(["--run-name", part, "--resume"])                                           # everything else comes from the identity
    fi, pi = (json.loads((x / "model_identity.json").read_text()) for x in (full, pd))
    assert pi["status"] == "finished" and pi["timesteps_trained"] == 640 and pi["resumes"][0]["at_timestep"] == interrupt_at
    assert pi["params_sha256"] == fi["params_sha256"] and pi["params_optimizer_sha256"] == fi["params_optimizer_sha256"]
    assert _rows(pd / "val_eval.csv") == _rows(full / "val_eval.csv")
    assert (pd / "val_layouts.jsonl").read_text() == (full / "val_layouts.jsonl").read_text()
    assert (pd / "curriculum.log").read_text() == (full / "curriculum.log").read_text()
    a, _ = _final_state(full)
    b, _ = _final_state(pd)
    for (k, va), (_, vb) in zip(a.policy.state_dict().items(), b.policy.state_dict().items()):
        assert th.equal(va, vb), k


def test_resume_skips_finished_runs_and_ignores_half_written_checkpoints():
    d = run("done")
    assert T.main(["--run-name", "done", "--resume"]) == d
    (d / "checkpoints" / "ckpt_99999.tmp").mkdir(parents=True)
    (d / "checkpoints" / "ckpt_88888").mkdir()                                          # no state.pkl / model.zip -> not a checkpoint
    ck = T.latest_checkpoint(d)
    assert ck is None or int(ck.name.split("_")[1]) < 640
    assert T.latest_checkpoint(T.MODELS_DIR / "nonexistent") is None


def test_checkpoints_are_pruned_to_the_newest_two():
    d = run("prune")
    names = sorted(int(p.name.split("_")[1]) for p in (d / "checkpoints").iterdir())
    assert names == [320, 480]                                                          # saved at 160 / 320 / 480 (rollout starts), keep=2


def test_env_pickle_roundtrip_continues_identically():
    env = BreachEnv(source="TUNE-train", role="train", prebought="random")
    seed = 4
    env.reset(seed=seed)
    while sum(env.owned) < 6:                                                            # an army that lasts a few deploy steps
        seed += 1
        env.reset(seed=seed)
    for _ in range(2):
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert not env.done
    blob = pickle.dumps(env)
    assert len(blob) < 2_000_000                                                         # layout caches are not part of the pickle
    twin = pickle.loads(blob)
    for _ in range(2):
        m = env.action_masks()
        assert (m == twin.action_masks()).all()
        a = int(np.flatnonzero(m)[0])
        o1, r1, d1, _, _ = env.step(a)
        o2, r2, d2, _, _ = twin.step(a)
        assert (o1["grid"] == o2["grid"]).all() and r1 == r2 and d1 == d2
    o1, _ = env.reset(seed=9)
    o2, _ = twin.reset(seed=9)
    assert (o1["grid"] == o2["grid"]).all()


# ------------------------------------------------------------------------------------------------ eval script / TTA adapter
def test_model_policy_matches_manual_rollout_and_is_deterministic():
    d = run("e")
    m, _ = _final_state(d)
    items = E.EV.load_set("TUNE-val", "validation")[:6]
    r1 = E.EV.evaluate(E.ModelPolicy(m), "TUNE-val", "validation", items=items)
    r2 = E.EV.evaluate(E.ModelPolicy(m), "TUNE-val", "validation", items=items)
    assert [x["score"] for x in r1["rows"]] == [x["score"] for x in r2["rows"]]
    rec, layout = items[0]
    env = BreachEnv(layouts=[layout], role="train")
    import zlib
    obs, _ = env.reset(seed=zlib.crc32(rec["id"].encode()), options={"layout_index": 0})
    done = False
    while not done:
        a, _ = m.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, _, done, _, _ = env.step(int(a))
    assert env.result.score == r1["rows"][0]["score"]
    ep = r1["episodes"][rec["id"]][0]
    assert sum(ep["army"]) == env.bought and len(ep["army"]) == N_UNIT_TYPES
    with pytest.raises(PermissionError):
        E.EV.load_set("LAB-DESIGN", "validation")


def test_tta_with_a_real_model_identity_only_equals_predict_and_logprob_tta_is_legal():
    m, _ = _final_state(run("t"))
    fn = E.masked_probs(m)
    env = BreachEnv(source="TUNE-val", role="validation")
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(12):
        mask = env.action_masks()
        p = fn(obs, mask)
        assert abs(p.sum() - 1) < 1e-5 and (p[~mask] == 0).all()
        a, _ = m.predict(obs, action_masks=mask, deterministic=True)
        assert tta_action(fn, obs, mask, syms=[0]) == int(a)
        full = tta_action(fn, obs, mask)
        assert mask[full]
        sc = tta_scores(fn, obs, mask)
        assert np.isneginf(sc[~mask]).all() and np.isfinite(sc[mask]).any()
        _, _, done, _, _ = env.step(int(rng.choice(np.flatnonzero(mask))))
        if done:
            obs, _ = env.reset(seed=1)
        else:
            obs = env._obs()


# ------------------------------------------------------------------------------------------------ throughput record (written by scripts_b10_throughput.py)
def test_throughput_verdict_function_on_synthetic_values():
    assert budget.t_cap(110) == pytest.approx(1_980_000)
    v = budget.throughput_verdict(110)
    assert v["meets_spec_minimum"] and v["t_cap_rounded_250k"] == 1_750_000 and not v["stop_for_user_decision"] and not v["below_90_cap_binds_early"]
    v = budget.throughput_verdict(89)
    assert not v["meets_spec_minimum"] and v["below_90_cap_binds_early"] and not v["stop_for_user_decision"]
    assert budget.throughput_verdict(55)["stop_for_user_decision"] and not budget.throughput_verdict(56)["stop_for_user_decision"]
    assert budget.throughput_verdict(100)["meets_spec_minimum"]


def test_throughput_record_exists_with_required_fields_and_matches_the_verdict():
    f = ROOT / "data" / "v2" / "b10_throughput.json"
    if not f.exists():
        pytest.skip("b10_throughput.json is written by scripts_b10_throughput.py (B10 benchmark run)")
    rec = json.loads(f.read_text())
    for k in ("hardware", "ppo_config", "thread_choice", "six_parallel", "verdict", "single_process", "env_only", "inference_only", "peak_memory_mb"):
        assert k in rec, k
    assert rec["hardware"]["torch_threads_used"] == rec["thread_choice"]["chosen"]
    r = rec["six_parallel"]["per_job_steps_per_s_full_game"]
    assert rec["verdict"] == budget.throughput_verdict(r)


def test_negative_control_resume_without_restoring_the_rng_streams_is_not_bit_identical(monkeypatch):
    """The bit-identity test has teeth: skipping the torch / numpy RNG restore (the prototype's behaviour) changes the result."""
    full = run("nc_full")
    with pytest.raises(T.Interrupted):
        run("nc_part", ["--debug-interrupt-at", "320"])
    monkeypatch.setattr(T.th, "set_rng_state", lambda s: None)
    monkeypatch.setattr(T.np.random, "set_state", lambda s: None)
    th.manual_seed(12345)
    np.random.seed(12345)
    T.main(["--run-name", "nc_part", "--resume"])
    fi, pi = (json.loads((T.MODELS_DIR / n / "model_identity.json").read_text()) for n in ("nc_full", "nc_part"))
    assert pi["params_sha256"] != fi["params_sha256"]


def test_real_config_determinism_record():
    f = ROOT / "data" / "v2" / "b10_determinism.json"
    if not f.exists():
        pytest.skip("written by scripts_b10_determinism.py")
    rec = json.loads(f.read_text())
    assert rec["all_identical"] is True and rec["rollout_size"] == 2500
    for aug in ("none", "d4"):
        c = rec["cases"][aug]
        assert c["fresh_twice"] and all(c["fresh_twice"].values())
        for k in ("resume_after_checkpoint_2500", "resume_after_checkpoint_5000"):
            assert all(c[k][x] for x in ("params_sha256", "params_optimizer_sha256", "val_rows", "val_layouts", "curriculum"))
    assert all(v for k, v in rec["cases"]["none"]["sigkill_then_resume"].items() if isinstance(v, bool))
