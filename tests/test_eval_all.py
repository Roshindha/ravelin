import numpy as np
import pytest

from rl import eval_all as E
from rl.eval_all import GROUPS, HEADLINE_GROUPS, SEEN_GROUPS

MODEL = E.MODELS_DIR / "ppo_d4_s10" / "final.zip"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="grid model not present")


def test_group_sizes_and_disjoint():
    recs = E.load_group_records(GROUPS)
    assert len(recs) == 241 and len({r["id"] for r in recs}) == 241
    for g, n in E.GROUP_SIZES.items():
        assert sum(r["group"] == g for r in recs) == n


def test_headline_groups_exclude_seen_ids():
    head = E.load_group_records(HEADLINE_GROUPS)
    seen = E.load_group_records(SEEN_GROUPS)
    assert len(head) == 81
    assert not ({r["id"] for r in head} & {r["id"] for r in seen})
    assert all(r["pool"] == "FINAL" for r in head)
    assert all(r["pool"] == "DEV" for r in seen)


def test_loader_only_returns_declared_groups_and_rejects_unknown():
    recs = E.load_group_records(("transfer",))
    assert {r["group"] for r in recs} == {"transfer"} and len(recs) == 41
    with pytest.raises(ValueError):
        E.load_group_records(("FINAL",))


@needs_model
def test_evaluator_refuses_layout_outside_declared_groups():
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(MODEL, device="cpu")
    val = E.load_group_records(("validation",))
    final = E.load_group_records(("FINAL-familiar",))
    allowed = {r["id"] for r in val}
    with pytest.raises(RuntimeError):
        E.evaluate_records(model, final[:1], allowed)


@needs_model
def test_reproducible_and_canonical_view():
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(MODEL, device="cpu")  # a D4-trained model
    recs = E.load_group_records(("validation",))[:3]
    ids = {r["id"] for r in recs}
    a = E.evaluate_records(model, recs, ids)
    b = E.evaluate_records(model, recs, ids)
    assert a == b
    s1 = E.evaluate_records(model, recs, ids, deterministic=False, n_rollouts=2, seed_key=5)
    s2 = E.evaluate_records(model, recs, ids, deterministic=False, n_rollouts=2, seed_key=5)
    assert s1 == s2
    # canonical view == D4Wrapper pinned to identity: same actions as the bare env
    from rl.augment import D4Wrapper, IDENTITY
    from siege.env import SiegeEnv
    env = D4Wrapper(SiegeEnv(), symmetry=IDENTITY)
    obs, _ = env.reset(seed=int(recs[0]["id"]), options={"layout_record": recs[0], "prebought": False})
    done = False
    while not done:
        act, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, _, t, tr, _ = env.step(int(act)); done = t or tr
    assert env.unwrapped.destruction == pytest.approx(a[0]["destruction"])


def test_evaluator_env_is_not_wrapped():
    import inspect
    src = inspect.getsource(E.evaluate_records)
    assert "D4Wrapper" not in src.replace("no D4Wrapper", "") and "SiegeEnv()" in src


def test_bootstrap_deterministic_and_paired():
    import pandas as pd
    x = np.arange(10.0)
    assert E.boot_ci(x, np.random.default_rng(1)) == E.boot_ci(x, np.random.default_rng(1))
    rows = [{"aug": a, "seed": s, "group": "g", "v": float(s) + (1 if a == "none" else 0)} for a in E.AUGS for s in range(10, 20)]
    out = E.paired_diff(pd.DataFrame(rows), ["v"], ["group"])
    assert out.iloc[0].mean_diff_none_minus_d4 == pytest.approx(1.0) and out.iloc[0].seeds_none_wins == 10


def test_best_single_uses_dev_only():
    with pytest.raises(AssertionError):
        E.best_single_troops(E.load_group_records(("FINAL-familiar",)))
