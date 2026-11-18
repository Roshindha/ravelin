"""B17: duel core (breachlens/duel.py, scripts/build_duel_cache.py, Gate changelog row (kk)). DEV-validation only; nothing is trained or evaluated for FINAL."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from breachlens import duel as D
from breachlens import evaluator as EV
from breachlens import pool as P
from breachlens.constants import END_BUY, UNIT_TYPES, WAIT
from breachlens.env import BreachEnv
from breachlens.score import result_score
from breachlens.sim import Prepared, simulate

ROOT = Path(__file__).resolve().parents[1]
LEDGER_SHA = "20a8b2877c91c9cf45c3692617ad3bd08d860ef0f7783ca3610a9a3fc44c9dbd"
_spec = importlib.util.spec_from_file_location("build_duel_cache", ROOT / "scripts" / "build_duel_cache.py")
B = importlib.util.module_from_spec(_spec)
sys.modules["build_duel_cache"] = B
_spec.loader.exec_module(B)

IDX = (0, 10, 20)                                   # first DEV-validation layout of each familiar family


def ledger_sha() -> str:
    return hashlib.sha256(P.LEDGER_PATH.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def cache():
    return D.load_cache()


@pytest.fixture(scope="module")
def models():
    return B.load_models()


def duel(index=0, **kw):
    kw.setdefault("human_runs_path", None)
    return D.Duel(index=index, **kw)


def play_single(d, unit="sniper", n=8, start=0):
    for _ in range(n):
        d.buy(unit)
    d.end_buy()
    tiles = [b for b in range(52) if d.env._st.border_free[b]][start:start + n]
    for t in tiles:
        d.deploy(unit, t)
    return tiles


# ------------------------------------------------------------------------------------------------------------------------ cache
def test_ledger_sha_stays():
    assert ledger_sha() == LEDGER_SHA
    assert P.count_eval_sessions() == 1


def test_cache_content_and_provenance(cache):
    ids = [r["id"] for r in P.load_pool_records("DEV-validation", "validation")]
    assert len(ids) == 30 and set(cache["layouts"]) == set(ids)
    assert {v["family"] for v in cache["layouts"].values()} == {"spine_pods", "archipelago", "bunker_nest"}
    for lid, v in cache["layouts"].items():
        assert set(v["contestants"]) == set(D.CONTESTANTS), lid
        for k, r in v["contestants"].items():
            assert set(r) >= {"score", "d", "breach", "bought", "n_bought", "n_deployed", "deploy"} and 0.0 <= r["score"] <= 1.0
            assert sum(r["bought"].values()) == r["n_bought"] and len(r["deploy"]) == r["n_deployed"]
    m = cache["meta"]
    for key in ("code_commit", "git_dirty", "models", "sim_version", "score_version", "generator_hash", "pool_sha256", "oracle", "best_single", "split"):
        assert key in m
    assert m["split"] == "DEV-validation" and m["oracle"]["rng"].endswith("4242])") and m["oracle"]["n_random"] == m["oracle"]["n_refine"] == 200
    assert m["generator_hash"] == json.loads(P.GENERATOR_HASH_PATH.read_text())["generator_hash"] and m["pool_sha256"] == P.pool_sha256()
    assert set(m["models"]) == {"ppo_plain_tta", "ppo_d4_tta", "ddqn_tta"} and all(len(x["sha256"]) == 64 for x in m["models"].values())
    assert "search baseline, not the optimum" in m["oracle"]["label"]
    assert not any("time" in k.lower() or k.lower().endswith("_at") for k in m)          # no timestamp field: the file is deterministic


@pytest.mark.parametrize("idx", IDX)
def test_cache_matches_fresh_recomputation(cache, models, idx):
    rec, layout = EV.load_dev_layout("DEV-validation", idx)
    fresh = B.contestants_for(rec, layout, models, B.best_single_map())
    assert fresh == cache["layouts"][rec["id"]]["contestants"]                      # all five opponent types, Oracle included, exactly


def test_cache_models_are_the_recorded_checkpoints(cache):
    from rl import explain as X
    for arm, (run, algo) in X.DISPLAY_MODELS.items():
        f = X.MODELS / run / ("final.zip" if algo == "ppo" else "final.pt")
        assert cache["meta"]["models"][arm + "_tta"]["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()


def test_cache_refuses_non_dev_validation_ids(cache, tmp_path):
    recs = json.loads(P.POOL_PATH.read_text())["records"]
    bad = {"final": next(r["id"] for r in recs if r["split"] == "FINAL-familiar"), "transfer": next(r["id"] for r in recs if r["split"] == "FINAL-transfer"),
           "dev_train": next(r["id"] for r in recs if r["split"] == "DEV-train")}
    before = ledger_sha()
    for name, lid in bad.items():
        with pytest.raises(PermissionError):
            D.cache_lookup(cache, lid)
        with pytest.raises(PermissionError):
            D.Duel(layout_id=lid, human_runs_path=None)
        with pytest.raises(PermissionError):
            B.build(tmp_path / "x.json", ids=[lid], allow_dirty=True)
    assert not (tmp_path / "x.json").exists()
    tampered = json.loads(json.dumps(cache))                                        # a hand-edited cache that contains a FINAL id is still refused
    tampered["layouts"][bad["final"]] = next(iter(cache["layouts"].values()))
    with pytest.raises(PermissionError):
        D.cache_lookup(tampered, bad["final"])
    assert set(cache["layouts"]) .isdisjoint(bad.values())
    assert ledger_sha() == before == LEDGER_SHA


def test_cache_agent_actions_equal_b15_eval_path(cache, models, monkeypatch):
    """The cached TTA rows are the B15 evaluator path's: `scripts_b15_final._task` (policy built there) is run with env.step wrapped to record actions;
    the cache row's bought counts, deploy list in order and score must equal what those actions produce, and our chooser's actions must be identical."""
    import scripts_b15_final as F
    from rl import explain as X
    orig = BreachEnv.step
    for arm, key in (("ppo_plain", "ppo_plain_tta"), ("ppo_d4", "ppo_d4_tta"), ("ddqn", "ddqn_tta")):
        run, algo = X.DISPLAY_MODELS[arm]
        m = models[arm][0]
        for idx in IDX:
            rec, layout = EV.load_dev_layout("DEV-validation", idx)
            taken = []

            def rec_step(self, action):
                taken.append(int(action))
                return orig(self, action)
            with monkeypatch.context() as mp:
                mp.setattr(BreachEnv, "step", rec_step)
                out = F._task(({"kind": "learner", "run": run, "algo": algo, "tta": True, "name": key}, rec, layout, {"models_dir": str(X.MODELS)}))
            row_b15 = out[0]
            _, mine = B.tta_episode(m, algo, rec, layout)
            assert mine == taken, (arm, rec["id"])
            c = cache["layouts"][rec["id"]]["contestants"][key]
            from breachlens.actions import decode_action
            dec = [decode_action(a) for a in taken]
            assert c["deploy"] == [[UNIT_TYPES[u], b] for k, u, b in dec if k == "deploy"]
            for u, name in enumerate(UNIT_TYPES):
                assert c["bought"][name] == sum(1 for k, uu, _ in dec if k == "buy" and uu == u)
            assert c["score"] == row_b15["score"] and c["breach"] == bool(row_b15["breach"]) and c["n_bought"] == row_b15["n_bought"]


def test_bestsingle_cache_equals_b8_policy(cache):
    from breachlens import baselines as BL
    pol = BL.BestSingle(B.best_single_map()["by_family"], pooled=B.best_single_map()["pooled"])
    for idx in IDX:
        rec, layout = EV.load_dev_layout("DEV-validation", idx)
        r = EV._work((pol, rec, layout))
        c = cache["layouts"][rec["id"]]["contestants"]["best_single"]
        assert c["score"] == r["score"] and c["breach"] == bool(r["breach"]) and c["bought"]["sniper"] == r["n_bought"]


# ------------------------------------------------------------------------------------------------------------------------ engine
def test_same_env_masks_and_observation():
    d = duel()
    env = BreachEnv(layouts=[d.layout], role="train")
    obs, _ = env.reset(seed=D.reset_seed(d.layout_id), options={"layout_index": 0})
    assert np.array_equal(d.obs["grid"], obs["grid"]) and np.array_equal(d.obs["state"], obs["state"])
    assert d.legal_actions() == [int(a) for a in np.flatnonzero(env.action_masks())]
    assert d.env.gold == 240


def test_human_outcome_matches_simulator():
    d = duel()
    tiles = play_single(d)
    assert d.finished
    row = d.human_row()
    army = place_on_border_like(d, tiles)
    r = simulate(Prepared(d.layout), army)
    assert row["score"] == result_score(r) and row["breach"] == bool(r.core_destroyed) and row["n_bought"] == 8 and row["bought"]["sniper"] == 8
    assert row["deploy"] == [["sniper", t] for t in tiles]


def place_on_border_like(d, tiles):
    return [(UNIT_TYPES.index("sniper"), t) for t in tiles]


def test_illegal_actions_rejected_with_clear_message():
    d = duel()
    n0 = list(d.actions)
    cases = [(lambda: d.deploy("sniper", 3), "cannot deploy during the BUY phase"),
             (lambda: d.end_buy(), "buy at least one unit"),
             (lambda: d.wait(), "cannot start the battle during the BUY phase"),
             (lambda: d.buy("tank"), "unknown unit"),
             (lambda: d.deploy("sniper", 99), "out of range"),
             (lambda: d.deploy("sniper", "x"), "integer border index"),
             (lambda: d.act(9999), "out of range")]
    for fn, msg in cases:
        with pytest.raises(D.DuelError, match=msg):
            fn()
    assert d.actions == n0 and d.env.gold == 240 and d.env.bought == 0          # nothing changed
    for _ in range(8):
        d.buy("blaster") if d.env.gold >= 36 else None
    with pytest.raises(D.DuelError, match="cannot afford blaster"):
        while True:
            d.buy("blaster")
    d.end_buy()
    with pytest.raises(D.DuelError, match="cannot buy during the DEPLOY phase"):
        d.buy("scout")
    with pytest.raises(D.DuelError, match="no sniper left to deploy"):
        d.deploy("sniper", 3)
    t = next(b for b in range(52) if d.env._st.border_free[b])
    d.deploy("blaster", t)
    with pytest.raises(D.DuelError, match="already occupied"):
        d.deploy("blaster", t)
    with pytest.raises(D.DuelError, match="no scout left"):
        d.deploy("scout", t)


def test_wall_tile_and_unit_cap_rejected():
    for i in range(30):                                                          # a layout with a wall on the border, if the pool has one
        d = duel(i)
        walls = [b for b in range(52) if not d.env._st.border_free[b]]
        if walls:
            d.buy("scout")
            d.end_buy()
            with pytest.raises(D.DuelError, match="wall tile"):
                d.deploy("scout", walls[0])
            break
    d = duel()
    for _ in range(20):
        d.buy("scout")
    with pytest.raises(D.DuelError, match="unit cap reached"):
        d.buy("scout")
    with pytest.raises(D.DuelError, match="over"):
        d2 = duel()
        play_single(d2)
        d2.buy("scout")


def test_opponents_hidden_until_end(monkeypatch):
    calls = []
    real = D.load_cache
    monkeypatch.setattr(D, "load_cache", lambda *a, **k: calls.append(1) or real(*a, **k))
    d = duel()
    for fn in (d.reveal, d.human_row):
        with pytest.raises(D.DuelNotFinished):
            fn()
    d.buy("sniper")
    d.buy("sniper")
    d.end_buy()
    with pytest.raises(D.DuelNotFinished):
        d.reveal()
    assert calls == [] and not hasattr(d, "opponents") and not hasattr(d, "cache")          # the cache file is not even opened before the end
    d.deploy("sniper", next(b for b in range(52) if d.env._st.border_free[b]))
    with pytest.raises(D.DuelNotFinished):
        d.reveal()
    d.wait()
    rows = d.reveal()
    assert calls == [1] and [r["contestant"] for r in rows] == ["human", *D.CONTESTANTS]
    assert "search baseline, not the optimum" in rows[-1]["label"]
    opp = D.cache_lookup(D.load_cache(), d.layout_id)
    assert all(r["score"] == opp[r["contestant"]]["score"] for r in rows[1:])


def test_undo_only_before_end_buy():
    d = duel()
    with pytest.raises(D.DuelError, match="nothing to undo"):
        d.undo()
    for u in ("sniper", "scout", "bulwark"):
        d.buy(u)
    ref = duel()
    ref.buy("sniper")
    ref.buy("scout")
    assert d.undo() == D.encode_buy(UNIT_TYPES.index("bulwark"))
    assert d.env.gold == ref.env.gold and d.env.owned == ref.env.owned and d.env.bought == 2 and d.actions == ref.actions
    assert np.array_equal(d.obs["state"], ref.obs["state"]) and np.array_equal(d.obs["grid"], ref.obs["grid"])
    d.end_buy()
    with pytest.raises(D.DuelError, match="only possible during the BUY phase"):
        d.undo()


def test_human_runs_file_is_gitignored_and_local(tmp_path):
    r = subprocess.run(["git", "check-ignore", "-q", str(D.HUMAN_RUNS_PATH.relative_to(ROOT))], cwd=ROOT)
    assert r.returncode == 0, "results_v2/duel_human_runs.jsonl must be gitignored"
    assert "results_v2/duel_human_runs.jsonl" in (ROOT / ".gitignore").read_text().splitlines()
    tracked = subprocess.run(["git", "ls-files", "results_v2"], cwd=ROOT, capture_output=True, text=True).stdout
    assert "duel_human_runs" not in tracked
    p = tmp_path / "runs.jsonl"
    d = duel(human_runs_path=p)
    play_single(d)
    line = json.loads(p.read_text().splitlines()[0])
    assert line["layout_id"] == d.layout_id and "never a result" in line["kind"] and line["row"]["contestant"] == "human"
    d2 = duel()                                                                  # no path -> nothing written
    play_single(d2)


def test_cli_session_and_labels(capsys):
    out = []
    lines = ["buy sniper", "buy sniper", "undo", "buy foo", "deploy sniper 3", "buy scout", "end", "deploy sniper 3", "deploy sniper 3", "show", "deploy scout 4"]
    d = duel()
    rows = D.run_cli(d, lines + ["wait"], out=out.append)
    text = "\n".join(out)
    assert rows is not None and len(rows) == 6
    assert "search baseline, not the optimum" in text and "demo" in text.lower() and "Comparison" in text
    assert "! unknown unit" in text and "! illegal action" in text and "undid" in text
    d2 = duel()                                                                  # opponents never printed when the run is abandoned
    out2 = []
    assert D.run_cli(d2, ["buy sniper", "quit"], out=out2.append) is None
    assert D.LABELS["oracle"] not in "\n".join(out2) and "PPO plain" not in "\n".join(out2) and "BestSingle (Sniper" not in "\n".join(out2)


def test_ledger_unchanged_after_everything():
    assert ledger_sha() == LEDGER_SHA
