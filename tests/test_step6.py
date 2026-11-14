import json

import numpy as np
import pytest
import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv

from rl import train_ppo
from rl.augment_v1 import D4Wrapper
from rl.features import SiegeFeaturesExtractor
from rl.policy_io import policy_outputs
from rl.split import SPLIT_PATH, load_split, make_split
from siege.env import BUDGET, END_BUY, N_ACTIONS, PHASE_DEPLOY, SiegeEnv
from siege.layouts import load_layout_pool
from siege.troops import TROOP_COST


def test_val_split_is_fixed_dev_familiar_10_per_family_and_disjoint():
    train, val = load_split()
    assert len(val) == 20 and len(train) == 140
    assert all(r["pool"] == "DEV" and r["tag"] == "familiar" for r in train + val)
    assert {r["family"] for r in val} == {"clustered_core", "scattered_outposts"}
    for fam in ("clustered_core", "scattered_outposts"):
        assert sum(r["family"] == fam for r in val) == 10
    assert not {r["id"] for r in train} & {r["id"] for r in val}
    # the recorded file equals a fresh deterministic derivation
    assert json.loads(SPLIT_PATH.read_text()) == make_split()
    dev_ids = {r["id"] for r in load_layout_pool(pool="DEV", tag="familiar")}
    assert {r["id"] for r in train + val} == dev_ids


def test_prebought_curriculum_starts_in_deploy_with_greedy_army():
    army = train_ppo.GREEDY_ARMY
    assert army == [2, 0, 6, 0] and int(np.dot(army, TROOP_COST)) <= BUDGET
    train, _ = load_split()
    env = SiegeEnv(layout_pool=train, prebought=army)
    env.reset(seed=0)
    assert env.phase == PHASE_DEPLOY and env.owned.tolist() == army
    assert not env.action_masks()[END_BUY]
    env.set_prebought(None)
    env.reset(seed=0)
    assert env.phase != PHASE_DEPLOY and env.action_masks()[END_BUY]


def test_features_extractor_and_policy_outputs():
    train, _ = load_split()
    env = SiegeEnv(layout_pool=train)
    model = MaskablePPO("MultiInputPolicy", env, n_steps=64, seed=0, device="cpu",
                        policy_kwargs=dict(features_extractor_class=SiegeFeaturesExtractor))
    assert isinstance(model.policy.features_extractor, SiegeFeaturesExtractor)
    obs, _ = env.reset(seed=1)
    mask = env.action_masks()
    out = policy_outputs(model, obs, mask)
    assert out["probs"].shape == (N_ACTIONS,) and abs(out["probs"].sum() - 1) < 1e-4
    assert np.all(out["probs"][~mask] == 0)
    assert len(out["top"]) == 5 and isinstance(out["value"], float)


def test_d4_wrapper_provides_masks_to_vecenv():
    train, _ = load_split()
    venv = DummyVecEnv([lambda: D4Wrapper(SiegeEnv(layout_pool=train), seed=0)])
    venv.reset()
    assert get_action_masks(venv).shape == (1, N_ACTIONS)


@pytest.mark.parametrize("aug", ["none", "d4"])
def test_train_cli_tiny_run_writes_artifacts(aug, tmp_path, monkeypatch):
    monkeypatch.setattr(train_ppo, "MODELS_DIR", tmp_path)
    run = train_ppo.main(["--aug", aug, "--timesteps", "512", "--n-envs", "2", "--run-name", "t", "--val-freq", "256", "--phase-a-steps", "256", "--checkpoint-freq", "256"])
    for f in ("final.zip", "model_identity.json", "progress.csv", "val_eval.csv", "curriculum.log"):
        assert (run / f).exists(), f
    ident = json.loads((run / "model_identity.json").read_text())
    assert ident["algorithm"] == "MaskablePPO" and ident["aug"] == aug and ident["status"] == "finished"
    assert ident["layouts_json_sha256"] == "e97cfbc7d43bf30842618511b9004764d5655b971ec6b208e7bd1e039edd0592"
    phases = [json.loads(l) for l in (run / "curriculum.log").read_text().splitlines()]
    assert [p["phase"] for p in phases] == ["A", "B"] and phases[1]["timestep"] >= 256
    assert any((run / "checkpoints").glob("*.zip"))


def test_random_phase_a_army_is_legal_varied_and_phase_a_cli_option():
    train, _ = load_split()
    env = SiegeEnv(layout_pool=train, prebought="random")
    armies = set()
    for s in range(60):
        env.reset(seed=s)
        assert env.phase == PHASE_DEPLOY
        assert 0 < int(np.dot(env.owned, TROOP_COST)) <= BUDGET
        armies.add(tuple(env.owned.tolist()))
    assert len(armies) > 20
    assert any(a[1] > 0 for a in armies) and any(a[3] > 0 for a in armies)  # rangers and splash get deployed too
    env.reset(seed=0, options={"prebought": False})  # override -> normal shopping
    assert env.phase != PHASE_DEPLOY


def _tiny_args(name, extra=()):
    return ["--aug", "d4", "--timesteps", "1536", "--n-envs", "2", "--run-name", name, "--val-freq", "512",
            "--phase-a-steps", "700", "--checkpoint-freq", "256", *extra]


def _simulate_crash(run, keep_step):
    """Make a finished run look like a job killed right after checkpoint `keep_step`."""
    (run / "final.zip").unlink()
    for f in (run / "checkpoints").glob("ckpt_*_steps.zip"):
        if int(f.stem.split("_")[1]) > keep_step:
            f.unlink()
    ident = json.loads((run / "model_identity.json").read_text())
    ident["status"] = "running"
    (run / "model_identity.json").write_text(json.dumps(ident))


def test_resume_skips_finished_and_restarts_fresh_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(train_ppo, "MODELS_DIR", tmp_path)
    run = train_ppo.main(_tiny_args("r") + ["--resume"])
    mtime = (run / "final.zip").stat().st_mtime_ns
    train_ppo.main(_tiny_args("r") + ["--resume"])  # finished -> no-op
    assert (run / "final.zip").stat().st_mtime_ns == mtime
    for f in (run / "checkpoints").glob("*.zip"):
        f.unlink()
    (run / "final.zip").unlink()
    train_ppo.main(_tiny_args("r") + ["--resume"])  # interrupted before any checkpoint -> clean fresh start
    log = [json.loads(l) for l in (run / "curriculum.log").read_text().splitlines()]
    assert [e["phase"] for e in log] == ["A", "B"]  # no stale entries from the earlier attempt


def test_resume_before_and_after_curriculum_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(train_ppo, "MODELS_DIR", tmp_path)
    for name, keep, expect in (("pre", 512, "A-resume"), ("post", 1024, "B-resume")):
        run = train_ppo.main(_tiny_args(name))
        _simulate_crash(run, keep)
        ck = train_ppo.latest_checkpoint(run)
        t_ck = int(ck.stem.split("_")[1])
        assert (t_ck < 700) == (expect == "A-resume")
        train_ppo.main(_tiny_args(name, ["--resume"]))
        ident = json.loads((run / "model_identity.json").read_text())
        assert ident["status"] == "finished" and ident["timesteps_trained"] >= 1536  # continued to the requested total
        assert ident["resumes"][0]["at_timestep"] == t_ck  # restored the checkpoint's timestep, not 0
        log = [json.loads(l) for l in (run / "curriculum.log").read_text().splitlines()]
        resumed_log = log[next(i for i, e in enumerate(log) if e["phase"].endswith("-resume")):]  # audit log keeps the discarded segment
        if expect == "A-resume":  # resumed before the boundary: phase A restored, then the switch still happens at >= 700
            assert [e["phase"] for e in resumed_log] == ["A-resume", "B"] and resumed_log[-1]["timestep"] >= 700
        else:  # resumed after the boundary: straight into phase B, no second switch
            assert [e["phase"] for e in resumed_log] == ["B-resume"] and resumed_log[0]["prebought"] is None
        # validation schedule continues with no duplicates and no second t=0 eval
        v = __import__("pandas").read_csv(run / "val_eval.csv")
        full = v[v["mode"] == "full"]["timestep"].tolist()
        assert full == sorted(set(full)) and full.count(0) == 1
        assert (run / "progress_part1.csv").exists()
        assert (run / "final.zip").exists()
