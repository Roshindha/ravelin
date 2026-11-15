"""B12: policy-guided planner, RAS-N / RPS-N controls, fallback selector (rl/planner.py).

TUNE-val layouts (role tuning) and synthetic layouts only; no policy is trained (random-init seeded PPO and stubs). The real FINAL-access ledger is never
written: FINAL-path tests monkeypatch the ledger path to a temp file and the autouse fixture asserts the real file is unchanged.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch as th

from breachlens import evaluator as E
from breachlens import pool as P
from breachlens.constants import N_UNIT_TYPES
from breachlens.env import BreachEnv, END_BUY, N_ACTIONS, WAIT
from breachlens.score import score_ratio, score_value
from rl import planner as PL
from rl import train_ppo_v2 as T

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _ledger_untouched():
    before = P.LEDGER_PATH.read_bytes()
    yield
    assert P.LEDGER_PATH.read_bytes() == before, "the FINAL-access ledger changed during a B12 test"


@pytest.fixture(scope="module")
def items():
    return E.load_set("TUNE-val", "tuning")[:3]


def _random_init_model(seed: int):
    th.manual_seed(seed)
    venv = T.build_venv("TUNE-train", "none", seed, 1, None)
    return T.build_model(venv, seed, T.PPO_CONFIG)


@pytest.fixture(scope="module")
def model():
    return _random_init_model(0)


class Stub:
    """Deterministic stub policy: fixed preference per action (buy anything, never END_BUY / WAIT unless forced) and optional mass on ILLEGAL actions."""

    def __init__(self, illegal_mass: float = 0.0, nan: bool = False):
        self.illegal_mass, self.nan = illegal_mass, nan

    def __call__(self, obs, mask):
        p = np.ones(N_ACTIONS)
        p[END_BUY] = p[WAIT] = 1e-9                                             # only ever drawn when nothing else is legal
        p[~np.asarray(mask)] = self.illegal_mass
        if self.nan:
            p[~np.asarray(mask)] = np.nan
        return p


def _plans(method, rec, layout, seed, n, probs_fn=None, ph="stub"):
    cache = PL.PlanCache()
    from breachlens.sim import Prepared
    return cache.get(method, rec, layout, Prepared(layout), seed, probs_fn, n, ph)


def _sig(p):
    return (tuple(p["actions"]), p["d_num"], p["d_den"], p["core"])


METHODS = [("planner", lambda m: PL.ModelProbs(m)), ("planner-stub", lambda m: Stub()), ("RAS", lambda m: None), ("RPS", lambda m: None)]


# ------------------------------------------------------------------------------------------------ nesting
@pytest.mark.parametrize("name,mk", METHODS)
def test_n8_is_the_first_8_of_n32_which_is_the_first_32_of_n128(items, model, name, mk):
    method = "planner" if name.startswith("planner") else name
    rec, layout = items[0]
    pf = mk(model)
    ph = "x"
    # three INDEPENDENT caches (not prefixes of one list): the plans must match anyway because plan k never depends on N
    p8 = _plans(method, rec, layout, 3, 8, pf, ph)
    p32 = _plans(method, rec, layout, 3, 32, pf, ph)
    p128 = _plans(method, rec, layout, 3, 128, pf, ph)
    assert [_sig(p) for p in p8] == [_sig(p) for p in p32[:8]]
    assert [_sig(p) for p in p32] == [_sig(p) for p in p128[:32]]
    # best-of-N is non-decreasing in N, and the chosen plan is the max of its N
    best = []
    for n in PL.N_VALUES:
        ps = p128[:n]
        r = [score_ratio(p["d_num"], p["d_den"], p["core"]) for p in ps]
        i = PL.choose(ps)
        assert all(not PL.better(x, r[i]) for x in r)
        best.append(r[i])
    assert not PL.better(best[0], best[1]) and not PL.better(best[1], best[2])


def test_plans_differ_between_plans_and_between_seeds(items, model):
    rec, layout = items[0]
    a = _plans("RPS", rec, layout, 1, 8)
    b = _plans("RPS", rec, layout, 2, 8)
    assert len({tuple(p["actions"]) for p in a}) > 1 and [_sig(p) for p in a] != [_sig(p) for p in b]


def test_shared_cache_cells_equal_independent_runs(items, model):
    rec, layout = items[1]
    cells = PL.make_cells("planner", 5, probs_fn=PL.ModelProbs(model))
    got = [E.evaluate(c, "TUNE-val", "tuning", items=[(rec, layout)])["rows"][0] for c in cells]       # N = 8, 32, 128 in order, one cache
    assert cells[0].cache.n_sim_calls == 128                                                          # sampled once, not 8 + 32 + 128
    for c, row in zip(cells, got):
        solo = PL.Planner(PL.ModelProbs(model), c.N, 5)
        row2 = E.evaluate(solo, "TUNE-val", "tuning", items=[(rec, layout)])["rows"][0]
        assert row["score"] == row2["score"] and json.loads(row["extra"])["chosen_index"] == json.loads(row2["extra"])["chosen_index"]


# ------------------------------------------------------------------------------------------------ determinism, tie-break
@pytest.mark.parametrize("name,mk", METHODS)
def test_deterministic_given_layout_seed_and_policy(items, model, name, mk):
    method = "planner" if name.startswith("planner") else name
    rec, layout = items[2]
    pf = mk(model)
    runs = []
    for _ in range(2):
        cell = PL.make_cells(method, 11, (32,), probs_fn=pf, policy_hash="h")[0]
        r = E.evaluate(cell, "TUNE-val", "tuning", items=[(rec, layout)])
        runs.append((r["rows"][0]["score"], r["rows"][0]["extra"], r["episodes"][rec["id"]][0]["actions"]))
    a, b = runs
    assert a[0] == b[0] and a[2] == b[2] and json.loads(a[1])["chosen_index"] == json.loads(b[1])["chosen_index"]


def test_tie_break_is_the_lowest_plan_index_with_exact_ratios():
    mk = lambda n, d, c=False: {"d_num": n, "d_den": d, "core": c}
    assert PL.choose([mk(1, 2), mk(2, 4), mk(3, 6)]) == 0                       # equal ratios with different denominators: lowest index
    assert PL.choose([mk(1, 3), mk(1, 2), mk(2, 4)]) == 1                       # first of the tied best
    assert PL.choose([mk(0, 5), mk(0, 5, True)]) == 1                           # core destroyed beats equal D
    assert PL.choose([mk(5, 5)]) == 0
    # float round-off cannot decide: 1/3 vs 2/6 are equal exactly
    assert PL.choose([mk(1, 3), mk(2, 6)]) == 0


def test_documented_tie_break_text_and_exact_comparison_in_source():
    src = (ROOT / "rl" / "planner.py").read_text()
    assert "lowest plan index" in src.lower() or "LOWEST plan index" in src
    assert "def better" in src and "a[0] * b[1] > b[0] * a[1]" in src


# ------------------------------------------------------------------------------------------------ N = 1, masks
def test_n1_equals_one_independent_sampled_rollout(items, model):
    rec, layout = items[0]
    pf = PL.ModelProbs(model)
    plan = _plans("planner", rec, layout, 7, 1, pf, "x")[0]
    env = BreachEnv(layouts=[layout], role="train")                                  # independent rollout, written out by hand
    rng = PL.plan_rng(7, rec["id"], "planner", 0)
    obs, _ = env.reset(seed=123, options={"layout_index": 0})
    done, actions = False, []
    while not done:
        mask = env.action_masks()
        a = PL.sample_action(pf(obs, mask), mask, rng)
        actions.append(a)
        obs, _, done, _, info = env.step(a)
    assert plan["actions"] == actions
    assert score_value(plan["d_num"], plan["d_den"], plan["core"]) == info["score"]
    one = PL.Planner(pf, 1, 7, policy_hash="x")
    ep = E.evaluate(one, "TUNE-val", "tuning", items=[(rec, layout)])["episodes"][rec["id"]][0]
    assert ep["actions"] == actions and ep["chosen_index"] == 0 and ep["n_sims"] == 1


def test_masks_are_respected_even_if_the_policy_puts_mass_on_illegal_actions(items):
    rec, layout = items[0]
    for stub in (Stub(illegal_mass=50.0), Stub(nan=True)):
        plans = _plans("planner", rec, layout, 2, 16, stub, "stub")                  # BreachEnv raises on any illegal action
        assert len(plans) == 16 and all(p["n_deployed"] >= 1 for p in plans)
    env = BreachEnv(layouts=[layout], role="train")
    env.reset(seed=0, options={"layout_index": 0})
    mask = env.action_masks()
    for s in range(200):
        assert mask[PL.sample_action(np.ones(N_ACTIONS), mask, np.random.default_rng(s))]
    with pytest.raises(ValueError):
        PL.sample_action(np.where(mask, 0.0, 1.0), mask, np.random.default_rng(0))     # no probability on any legal action


def test_sampling_follows_the_policy_probabilities():
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[[3, 5, 9]] = True
    p = np.zeros(N_ACTIONS)
    p[[3, 5, 9]] = [0.2, 0.5, 0.3]
    p[7] = 9.0                                                                      # illegal, ignored
    rng = np.random.default_rng(0)
    c = np.bincount([PL.sample_action(p, mask, rng) for _ in range(20000)], minlength=N_ACTIONS) / 20000
    assert abs(c[3] - 0.2) < 0.02 and abs(c[5] - 0.5) < 0.02 and abs(c[9] - 0.3) < 0.02 and c[7] == 0


# ------------------------------------------------------------------------------------------------ budget = simulated plans
def test_every_method_uses_exactly_n_simulator_calls(items, model, monkeypatch):
    import breachlens.env as BE
    rec, layout = items[0]
    calls = {"n": 0}
    real_planner_sim, real_env_sim = PL.simulate, BE.simulate

    def counting(real):
        def f(*a, **k):
            calls["n"] += 1
            return real(*a, **k)
        return f
    monkeypatch.setattr(PL, "simulate", counting(real_planner_sim))
    monkeypatch.setattr(BE, "simulate", counting(real_env_sim))
    for n in (1, 8, 32, 128):
        for method, pf in (("planner", PL.ModelProbs(model)), ("RAS", None), ("RPS", None)):
            calls["n"] = 0
            cell = PL.make_cells(method, 4, (n,), probs_fn=pf, policy_hash="h")[0]
            res = E.evaluate(cell, "TUNE-val", "tuning", items=[(rec, layout)])
            assert calls["n"] == n, (method, n, calls["n"])
            assert json.loads(res["rows"][0]["extra"])["n_sims"] == n and cell.cache.n_sim_calls == n


# ------------------------------------------------------------------------------------------------ scores and provenance
def test_scores_only_through_breachlens_score():
    src = (ROOT / "rl" / "planner.py").read_text()
    assert not re.search(r"0\.6\s*\*|0\.4\s*\*|6\s*\*\s*d_num", src)
    assert "from breachlens.score import score_ratio" in src


def test_rows_carry_b8_provenance_plus_n_method_and_policy_hash(items, model):
    h = PL.state_dict_hash(model.policy)
    assert re.fullmatch(r"[0-9a-f]{64}", h)
    cells = [*PL.make_cells("planner", 0, (8,), probs_fn=PL.ModelProbs(model)), *PL.make_cells("RAS", 0, (8,)), *PL.make_cells("RPS", 0, (8,))]
    for cell, method, ph in zip(cells, ("planner", "RAS", "RPS"), (h, "none", "none")):
        res = E.evaluate(cell, "TUNE-val", "tuning", items=items[:2])
        for row in res["rows"] + res["summary"]:
            for k in E.REQUIRED_META:
                assert row[k] not in (None, "")
            assert row["N"] == 8 and row["method"] == method and row["policy_hash"] == ph and row["plan_seed"] == 0
            assert row["role"] == "tuning" and row["set"] == "TUNE-val"
        for row in res["rows"]:
            ep = res["episodes"][row["id"]][0]
            assert row["score"] == score_value(ep["d_num"], ep["d_den"], ep["core"])
    other = _random_init_model(1)
    assert PL.state_dict_hash(other.policy) != h and PL.state_dict_hash(_random_init_model(0).policy) == h        # seeded: same weights, same hash
    with pytest.raises(ValueError):
        PL.Planner(Stub(), 8, 0)                                                      # a planner without a checkpoint hash is refused


# ------------------------------------------------------------------------------------------------ FINAL
def _final_rec(items):
    fin = [r for r in json.loads(P.POOL_PATH.read_text())["records"] if r["split"] == "FINAL-familiar"][0]
    return {"id": fin["id"], "family": fin["family"], "seed": fin["seed"]}, items[0][1]                   # record metadata only; layout content is a TUNE-val layout


def _cells(model, **kw):
    return [PL.Planner(PL.ModelProbs(model), 8, 0, **kw), PL.RAS(8, 0, **kw), PL.RPS(8, 0, **kw)]


def test_planner_and_controls_refuse_final_unless_role_eval_and_unlock(items, model, tmp_path, monkeypatch):
    monkeypatch.setattr(P, "LEDGER_PATH", tmp_path / "ledger.log")
    rec, layout = _final_rec(items)
    assert P.is_final(rec["family"], rec["seed"])
    for kw in ({}, {"role": "validation"}, {"role": "tuning", "unlock": True}, {"role": "eval", "unlock": False}):
        for cell in _cells(model, **kw):
            with pytest.raises(PermissionError):
                cell.episodes(rec, layout, None)                                      # direct call, no evaluator in between
    for cell in _cells(model):                                                        # through the evaluator
        with pytest.raises(PermissionError):
            E.evaluate(cell, "FINAL-familiar", "eval", unlock=False, items=[(rec, layout)])
        with pytest.raises(PermissionError):
            E.evaluate(cell, "TUNE-val", "tuning", items=[(rec, layout)])             # a dev-side role never scores FINAL ids
        with pytest.raises(PermissionError):
            E.evaluate(cell, "FINAL-familiar", "eval", unlock=True, items=[(rec, layout)])     # unlocked evaluator, but the object itself is not unlocked
    assert P.read_ledger() == []                                                      # every refusal happened before any ledger write


def test_unlocked_eval_on_synthetic_final_layouts_writes_only_the_temp_ledger(items, model, tmp_path, monkeypatch):
    fake = tmp_path / "ledger.log"
    monkeypatch.setattr(P, "LEDGER_PATH", fake)
    rec, layout = _final_rec(items)
    for cell in _cells(model, role="eval", unlock=True):
        res = E.evaluate(cell, "FINAL-familiar", "eval", unlock=True, items=[(rec, layout)])
        assert res["rows"][0]["role"] == "eval" and res["rows"][0]["method"] in ("planner", "RAS", "RPS")
    led = P.read_ledger()
    assert len(led) == 3 and all(e["purpose"] == "eval" for e in led)
    monkeypatch.undo()
    led = P.read_ledger()
    assert sum(e["purpose"] == "build" for e in led) == 5 and sum(e["purpose"] == "eval" for e in led) == 0


def test_no_dev_side_planner_run_touches_final(items, model):
    src = (ROOT / "rl" / "planner.py").read_text()
    assert "load_pool_records" not in src and "load_layout" not in src and "load_set" not in src                      # the module loads no layout itself


# ------------------------------------------------------------------------------------------------ cost projection and fallback selector
def test_full_session_plan_counts_equal_the_spec_table():
    c = PL.cost_cells()
    assert c["planner_plain"] + c["planner_d4"] == 307_200 and c["RAS"] == 153_600 and c["RPS"] == 153_600


def test_wall_projection_matches_a_hand_calculation():
    t = {"planner": 0.1, "RAS": 0.05, "RPS": 0.05}
    pr = PL.project_wall_hours(t, PL.cost_cells())
    assert pr["total_cpu_hours"] == pytest.approx((307_200 * 0.1 + 153_600 * 0.05 * 2) / 3600)
    assert pr["total_wall_hours"] == pytest.approx(pr["total_cpu_hours"] / 7)


@pytest.mark.parametrize("t,step,fits", [(0.10, 0, True), (0.15, 1, True), (0.20, 2, True), (0.25, 3, True), (5.0, 3, False)])
def test_fallback_selector_picks_the_pre_registered_option_for_synthetic_timings(t, step, fits):
    sel = PL.select_fallback({"planner": t, "RAS": t, "RPS": t})
    assert sel["step"] == step and sel["fits"] is fits
    assert sel["fits"] == (sel["projection"]["total_wall_hours"] <= 3.0) or not fits
    if step > 0:                                                                   # the previous option (applied in order) did not fit
        prev = PL.select_fallback({"planner": t, "RAS": t, "RPS": t}, limit_h=0.0)
        assert prev["step"] == 3                                                   # nothing fits a zero limit: all options applied, in order
    cfgs = {0: (10, 10, 120), 1: (5, 5, 120), 2: (5, 0, 120), 3: (5, 0, 30)}[step]
    c = sel["config"]
    assert (c["plain_128_learners"], c["d4_128_learners"], c["layouts_128"]) == cfgs


def test_fallback_controls_follow_the_plain_planner_cell():
    c = PL.cost_cells(plain_128_learners=5, d4_128_learners=0, layouts_128=30)
    assert c["RAS"] == c["RPS"] == c["planner_plain"]
