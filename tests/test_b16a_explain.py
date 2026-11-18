"""B16a: explainability core (rl/explain.py, Gate changelog row (ii)). Display models: seed 0 of each arm; DEV layouts only; one checkpoint at a time."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from breachlens import pool as P
from breachlens.constants import DEPLOY_BASE, END_BUY, N_ACTIONS, N_BORDER, WAIT
from rl import explain as X

LEDGER_SHA = "20a8b2877c91c9cf45c3692617ad3bd08d860ef0f7783ca3610a9a3fc44c9dbd"
ARMS = ("ppo_plain", "ppo_d4", "ddqn")
# (split, index): fixed in advance, not picked by how the explanations look
STATES = (("DEV-validation", 0), ("DEV-validation", 11), ("DEV-validation", 25), ("DEV-train", 3), ("DEV-train", 77))
STEPS = (0, 4, 9, 13, 17)          # a BUY state, late BUY / END_BUY, and DEPLOY states when the episode is long enough


def ledger_sha() -> str:
    return hashlib.sha256(P.LEDGER_PATH.read_bytes()).hexdigest()


def _states(model, algo, tta):
    """Yield (explanation, mask) for the listed (layout, step) states that exist."""
    for split, idx in STATES:
        rec, layout = X.load_dev_layout(split, idx)
        for step in STEPS:
            try:
                env, obs, taken = X.play_to_step(model, algo, rec, layout, step, tta)
            except ValueError:
                break
            yield rec, step, env, obs, env.action_masks()


def eval_action(model, algo, obs, mask, tta):
    """The action the B15 evaluation policy takes, written out independently of rl.explain."""
    from rl.tta import tta_action
    if tta:
        if algo == "ppo":
            from rl.eval_ppo_v2 import masked_probs
            return tta_action(masked_probs(model), obs, mask, aggregate="logprob")
        from rl.ddqn import q_values_fn
        return tta_action(q_values_fn(model), obs, mask, aggregate="mean")
    if algo == "ppo":
        return int(model.predict(obs, action_masks=mask, deterministic=True)[0])
    from rl.ddqn import DDQNPolicy
    return DDQNPolicy(model).act(obs, mask)


@pytest.fixture(scope="module", params=ARMS)
def loaded(request):
    m, algo, run = X.load_display_model(request.param)       # one checkpoint at a time (module scope per arm)
    yield request.param, m, algo, run
    del m


def test_display_models_are_seed_zero():
    assert X.DISPLAY_MODELS == {"ppo_plain": ("grid_ppo_plain_s0", "ppo"), "ppo_d4": ("grid_ppo_d4_s0", "ppo"), "ddqn": ("grid_ddqn_s0", "ddqn")}


@pytest.mark.parametrize("tta", [False, True])
def test_fidelity_argmax_equals_eval_policy(loaded, tta):
    arm, m, algo, _ = loaded
    n = ok = 0
    for rec, step, env, obs, mask in _states(m, algo, tta):
        ex = X.explain(m, algo, obs, mask, tta=tta)
        a = eval_action(m, algo, obs, mask, tta)
        n += 1
        ok += ex["greedy_action"] == a
        assert ex["greedy_action"] == a, (arm, tta, rec["id"], step)
        assert ex["top"][0]["action"] == a
        probs = np.asarray(ex["distribution"]["probs"])
        assert probs[a] == probs[mask].max()                                                 # displayed argmax is the chosen action (ties aside)
    print(f"FIDELITY {arm} tta={tta}: {ok}/{n}")
    assert n >= 15 and ok == n


@pytest.mark.parametrize("tta", [False, True])
def test_probabilities_and_heatmap(loaded, tta):
    arm, m, algo, _ = loaded
    n_deploy = 0
    for rec, step, env, obs, mask in _states(m, algo, tta):
        ex = X.explain(m, algo, obs, mask, tta=tta)
        p = np.asarray(ex["distribution"]["probs"])
        assert p.shape == (N_ACTIONS,)
        assert abs(p.sum() - 1.0) <= 1e-6, (arm, tta, rec["id"], step, p.sum())
        assert np.all(p[~mask] == 0.0) and np.all(p[mask] >= 0.0)
        h = ex["heatmap"]
        heat = np.asarray(h["heat"])
        assert heat.shape == (5, N_BORDER)
        deploy_sum = float(p[DEPLOY_BASE:END_BUY].sum())
        assert h["heat_total"] == pytest.approx(deploy_sum, abs=1e-12)
        assert float(np.sum(h["heat_tile"])) == pytest.approx(deploy_sum, abs=1e-12)
        assert float(np.sum(h["heat_unit"])) == pytest.approx(deploy_sum, abs=1e-12)
        assert deploy_sum == pytest.approx(1.0 - p[:DEPLOY_BASE].sum() - p[END_BUY] - p[WAIT], abs=1e-6)
        assert np.all(heat[:, ~env._st.border_free] == 0.0) if env.phase == 1 else True
        n_deploy += env.phase == 1
        assert len(ex["top"]) == min(5, int(mask.sum()))
        assert all(mask[t["action"]] for t in ex["top"]) and all(t["label"] for t in ex["top"])
        assert "SHAPED" in ex["value"]["label"] and np.isfinite(ex["value"]["estimate"])
        if algo == "ddqn":
            assert ex["distribution"]["label"].startswith("not a policy") and ex["distribution"]["temperature"] == 0.1
            assert ex["value"]["estimate"] == pytest.approx(max(q for q in ex["distribution"]["q"] if q is not None))
        json.dumps(ex)                                                                       # plain data
    assert n_deploy >= 1


@pytest.mark.parametrize("tta", [False, True])
def test_determinism(loaded, tta):
    arm, m, algo, run = loaded
    a = X.explain_layout_step(arm, "DEV-validation", 0, 5, tta, loaded=(m, algo, run))
    b = X.explain_layout_step(arm, "DEV-validation", 0, 5, tta, loaded=(m, algo, run))
    assert a == b and json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_split_labels(loaded):
    arm, m, algo, run = loaded
    v = X.explain_layout_step(arm, "DEV-validation", 0, 0, False, loaded=(m, algo, run))
    t = X.explain_layout_step(arm, "DEV-train", 0, 0, False, loaded=(m, algo, run))
    assert v["split"] == "DEV-validation" and v["in_sample"] is False
    assert t["split"] == "DEV-train" and t["in_sample"] is True and "IN-SAMPLE" in t["note"]
    assert "IN-SAMPLE" in X.format_text(t) and "IN-SAMPLE" not in X.format_text(v)


def test_action_labels_cover_the_map():
    assert X.action_label(0) == "Buy scout" and X.action_label(4) == "Buy mender"
    assert X.action_label(DEPLOY_BASE).startswith("Deploy scout at border tile 0")
    assert X.action_label(DEPLOY_BASE + 52 * 4 + 51).startswith("Deploy mender at border tile 51")
    assert X.action_label(END_BUY).startswith("End buying") and X.action_label(WAIT).startswith("Wait")


def test_guards_refuse_held_out_layouts():
    before = ledger_sha()
    assert before == LEDGER_SHA
    recs = json.loads(P.POOL_PATH.read_text())["records"]
    final_id = next(r["id"] for r in recs if r["split"] == "FINAL-familiar")
    transfer_id = next(r["id"] for r in recs if r["split"] == "FINAL-transfer")
    for split in ("FINAL-familiar", "FINAL-transfer", "LAB-DESIGN", "TUNE-val", "TUNE-train"):
        with pytest.raises(PermissionError):
            X.load_dev_layout(split, 0)
    for lid in (final_id, transfer_id):
        for split in ("DEV-train", "DEV-validation"):
            with pytest.raises(PermissionError):
                X.load_dev_layout(split, layout_id=lid)
    with pytest.raises(PermissionError):
        X.explain_layout_step("ppo_plain", "FINAL-familiar", 0, 0)
    assert ledger_sha() == before == LEDGER_SHA


def test_ledger_unchanged_after_explain_tests():
    assert ledger_sha() == LEDGER_SHA
    assert P.count_eval_sessions() == 1


def test_cli_prints_text(capsys):
    ex = X.main(["--arm", "ddqn", "--split", "DEV-validation", "--index", "0", "--step", "2"])
    out = capsys.readouterr().out
    assert "Why this move?" in out and "not a policy" in out and "SHAPED" in out and ex["step"] == 2
