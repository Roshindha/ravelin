import json

import numpy as np
import pytest
import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv

from rl import train_ppo
from rl.augment import D4Wrapper
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
