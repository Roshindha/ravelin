"""B13c: the SPEC 7.4 shaped-reward pilot (prepared, never launched here). Evaluation ignores shaping; the launcher refuses without Gate changelog (z);
the shaped report is a separate section that never extends. TUNE sets only; FINAL ledger unchanged. Tiny CPU runs."""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from breachlens import evaluator as EV
from breachlens import g7 as G
from breachlens import pool as P
from breachlens.env import BreachEnv
from breachlens.score import result_score
from rl import eval_ppo_v2 as E
from rl import train_ppo_v2 as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts_b13_pilot as L  # noqa: E402
import scripts_g7_report as R  # noqa: E402
from tests.test_g7_report import curve, good_manifest, make_run, write_run  # noqa: E402

TINY = ["--n-envs", "2", "--n-steps", "16", "--batch-size", "16", "--n-epochs", "2", "--val-freq", "320", "--checkpoint-freq", "160", "--phase-a-steps", "200"]


@pytest.fixture(autouse=True)
def _ledger_unchanged():
    before = P.LEDGER_PATH.read_bytes()
    yield
    assert P.LEDGER_PATH.read_bytes() == before


@pytest.fixture
def clean(monkeypatch, build_only_ledger):
    monkeypatch.setattr(L, "LEDGER", build_only_ledger)                       # B15d: synthetic build-only ledger for "would start" paths
    monkeypatch.setattr(L, "dirty_tracked", lambda: [])
    monkeypatch.setattr(L, "ps_text", lambda: "  102 /usr/bin/vim notes.txt\n")
    for k in L.FORBIDDEN_ENV:
        monkeypatch.delenv(k, raising=False)


# ------------------------------------------------------------------------------------------------ evaluation ignores shaping
def _rollout(model, layout, shaping: bool):
    env = BreachEnv(layouts=[layout], role="train", shaping=shaping)
    obs, _ = env.reset(seed=zlib.crc32(b"x"), options={"layout_index": 0})
    rewards, done = [], False
    while not done:
        a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, r, done, _, info = env.step(int(a))
        rewards.append(r)
    return env, rewards, info


def test_shaped_and_unshaped_env_give_identical_validation_scores_for_the_same_policy():
    model = T.build_model(T.build_venv("TUNE-train", "none", 0, 1, None), 3, T.PPO_CONFIG)          # seeded, untrained: any fixed policy will do
    items = EV.load_set("TUNE-val", "validation", entry_point="tests")[::5]                          # 12 layouts over all three families
    policy = E.ModelPolicy(model)
    shaped_nonzero_steps = 0
    for rec, layout in items:
        env_u, r_u, info_u = _rollout(model, layout, False)
        env_s, r_s, info_s = _rollout(model, layout, True)
        for k in ("d_num", "d_den", "core_destroyed", "ticks", "n_units"):
            assert info_u[k] == info_s[k], k
        assert info_u["score"] == info_s["score"] == result_score(env_s.result)                      # the TRUE score is bit-identical
        assert r_u[-1] == info_u["score"] and sum(r_u[:-1]) == 0                                     # unshaped: terminal reward only
        assert abs(sum(r_s) - info_s["score"]) < 1e-9                                                # shaped return still sums to the true score
        shaped_nonzero_steps += sum(1 for r in r_s[:-1] if r != 0)
        ep = policy.episodes(rec, layout, None)[0]                                                    # the evaluator's own path (unshaped BreachEnv)
        assert EV.score_value(ep["d_num"], ep["d_den"], ep["core"]) == info_s["score"]
    assert shaped_nonzero_steps > 0                                                                   # shaping was really active (the test is not vacuous)


def test_validation_envs_never_shape_even_when_training_does(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "MODELS_DIR", tmp_path)
    seen = {"eval": [], "train": []}

    def recorder(kind):
        def make(*a, **k):                                   # a factory (not a subclass): the real BreachEnv class stays picklable for checkpoints
            env = BreachEnv(*a, **k)
            seen[kind].append(env.shaping)
            return env
        return make

    monkeypatch.setattr(E, "BreachEnv", recorder("eval"))
    monkeypatch.setattr(T, "BreachEnv", recorder("train"))
    run = T.main(["--run-name", "sh", "--timesteps", "320", "--shaping", "on", *TINY])
    assert seen["train"] and all(seen["train"]) and seen["eval"] and not any(seen["eval"])
    ident = json.loads((run / "model_identity.json").read_text())
    assert ident["shaping"] == "on" and ident["config"]["shaping"] == "on"


def test_first_validation_eval_is_identical_for_a_shaped_and_an_unshaped_run(tmp_path, monkeypatch):
    """Same seed -> same initial weights; at t = 0 nothing has been learned, so the validation eval must not depend on the training reward at all."""
    import csv
    rows = {}
    for mode in ("off", "on"):
        monkeypatch.setattr(T, "MODELS_DIR", tmp_path / mode)
        run = T.main(["--run-name", "r", "--timesteps", "320", "--seed", "5", "--shaping", mode, *TINY])
        with (run / "val_eval.csv").open(newline="") as fh:
            rows[mode] = [{k: v for k, v in r.items() if k not in ("eval_seconds",)} for r in csv.DictReader(fh) if r["timestep"] == "0"]
        assert json.loads((run / "model_identity.json").read_text())["shaping"] == mode
    assert rows["off"] and rows["off"] == rows["on"]


def test_default_trainer_shaping_is_off_and_the_frozen_plain_pilot_is_unchanged():
    assert T.parse_args(["--run-name", "x"]).shaping == "off"
    assert L.frozen()["shaping"] == "off"
    assert L.config_sha(L.frozen()) == "ef0867e601a74ca8aef16400d377f6ed3ed36373d09d049b3b89a0e0f40497e9"        # row (v): not edited by B13c


# ------------------------------------------------------------------------------------------------ launcher
def test_shaped_preflight_passes_on_the_committed_state_and_plain_is_unaffected(clean):
    assert L.preflight(shaped=True) == []
    assert L.preflight() == []


def test_shaped_preflight_refuses_on_the_real_ledger_permanently_after_b15(clean, monkeypatch):
    monkeypatch.setattr(L, "LEDGER", L.ROOT / "data" / "v2" / "final_access.log")
    assert any("eval entr" in p for p in L.preflight(shaped=True)) and any("eval entr" in p for p in L.preflight())


@pytest.mark.parametrize("marker", ["(y) **B13c", "(z) **B13c"])
def test_shaped_launcher_refuses_when_the_new_changelog_rows_are_missing(clean, monkeypatch, tmp_path, marker, capsys):
    f = tmp_path / "spec.md"
    f.write_text(L.SPEC.read_text().replace(marker, "(zz) **gone"))
    monkeypatch.setattr(L, "SPEC", f)
    assert any(marker in p for p in L.preflight(shaped=True))
    started = []
    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: started.append(a))
    assert L.main(["--shaped"]) == 3 and not started
    assert "REFUSED" in capsys.readouterr().err
    assert L.preflight() == []                                                   # the plain pilot launcher does not depend on the shaped rows


def test_shaped_launcher_refuses_when_the_shaped_config_hash_is_not_in_the_changelog(clean, monkeypatch, tmp_path):
    f = tmp_path / "spec.md"
    f.write_text(L.SPEC.read_text().replace(L.config_sha(L.frozen(True)), "0" * 64))
    monkeypatch.setattr(L, "SPEC", f)
    assert any("shaped-frozen-config" in p for p in L.preflight(shaped=True))


def test_shaped_launcher_refuses_extension_and_overrides_and_a_dirty_tree(clean, monkeypatch, capsys):
    assert L.main(["--shaped", "--extend"]) == 3
    assert L.main(["--shaped", "--steps", "5000"]) == 3
    assert L.main(["--shaped", "--plain-seeds", "1,2,3"]) == 3
    monkeypatch.setattr(L, "dirty_tracked", lambda: [" M rl/train_ppo_v2.py"])
    assert L.main(["--shaped"]) == 3
    assert "REFUSED" in capsys.readouterr().err


def test_shaped_command_list(clean, capsys):
    assert L.main(["--shaped", "--print-only"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln]
    assert len(lines) == 3
    cfg = L.frozen(True)
    for ln, seed in zip(lines, (9201, 9202, 9203)):
        assert "--aug none" in ln and f"--seed {seed}" in ln and f"--run-name shaped_plain_s{seed}" in ln and "--shaping on" in ln
        assert "--timesteps 300000" in ln and "--phase-a-steps 100000" in ln and "--curriculum on" in ln and "--train-set TUNE-train" in ln
        assert "--threads 1" in ln and "--val-freq 25000" in ln and "--checkpoint-freq 50000" in ln and "--extend-to" not in ln and "d4" not in ln
        for k, v in cfg["ppo"].items():
            if k != "net_arch":
                assert f"--{k.replace('_', '-')} {v}" in ln, k
    assert L.main(["--print-only"]) == 0
    plain = capsys.readouterr().out.splitlines()
    assert len(plain) == 6 and not any("--shaping" in ln for ln in plain)         # the plain pilot's commands are untouched


def test_shaped_config_is_the_plain_config_plus_exactly_the_pinned_differences():
    plain, sh = L.frozen(), L.frozen(True)
    assert sh["ppo"] == plain["ppo"] == json.loads(json.dumps(T.PPO_CONFIG))
    for k in ("timesteps", "val_freq", "checkpoint_freq", "torch_threads", "train_set", "val_set", "curriculum", "eval_policy", "features_extractor"):
        assert sh[k] == plain[k], k
    assert sh["shaping"].startswith("on") and sh["plain_seeds"] == [9201, 9202, 9203] and sh["d4_seeds"] == [] and sh["extension_timesteps"] is None
    assert sh["run_names"]["plain"] == "shaped_plain_s{seed}"
    used = set(range(10)) | set(plain["plain_seeds"]) | set(plain["d4_seeds"])
    assert not used & set(sh["plain_seeds"])
    spec = L.SPEC.read_text()
    assert L.config_sha(sh) in spec and "0.4369" in spec and "(z) **B13c" in spec and "(y) **B13c" in spec
    row = next(ln for ln in spec.splitlines() if "(z) **B13c" in ln)
    for needle in ("Φ(s) = 0", "tick cap 120", "γ = 1", "9201, 9202, 9203", "No extension", "0.436862743301702", "true score", "never the shaped return"):
        assert needle in row, needle


# ------------------------------------------------------------------------------------------------ report
def _shaped_models(tmp_path, scores):
    md = tmp_path / "models"
    for i, sc in enumerate(scores):
        d = md / f"shaped_plain_s{9201 + i}"
        write_run(d, make_run(curve(lambda t, sc=sc: sc)), good_manifest())
        ident = json.loads((d / "model_identity.json").read_text())
        ident["shaping"] = "on"
        (d / "model_identity.json").write_text(json.dumps(ident))
    return md


def test_shaped_report_is_a_separate_section_and_uses_the_same_bar(tmp_path, capsys):
    md = _shaped_models(tmp_path, (0.50, 0.48, 0.45))
    rc = R.main(["--shaped", "--models-dir", str(md), "--smoke", "--out", str(tmp_path / "o.json")])
    out = capsys.readouterr().out
    assert rc == 0 and "SHAPED-REWARD PILOT" in out and "TRUE score" in out and "VERDICT: PASS" in out and "bar=0.4369" in out
    assert "D4 diagnostics" not in out
    md2 = _shaped_models(tmp_path / "b", (0.43, 0.43, 0.43))
    R.main(["--shaped", "--models-dir", str(md2), "--smoke", "--out", str(tmp_path / "o2.json")])
    assert "FAIL  mean_ge_bar" in capsys.readouterr().out
    assert R.FROZEN_SHAPED != R.FROZEN and str(tmp_path) not in str(R.ROOT / "logs_b13" / "g7_shaped_decision.json")


def test_shaped_report_never_says_extend_even_when_the_slope_licenses_it():
    rising = make_run(curve(lambda t: 0.15 + 0.25 * (t / 300_000)))          # slope far above +0.02 per 100k, still below the bar
    plain = {9201: rising, 9202: rising, 9203: rising}
    assert G.decide(plain, 0.4369, 300_000, 25_000)["verdict"] == "EXTEND"        # the plain pilot would extend
    sh = G.decide(plain, 0.4369, 300_000, 25_000, allow_extension=False)
    assert sh["verdict"] == "FAIL" and sh["slope_ge_threshold"] is True           # the slope is reported, it licenses nothing
    with pytest.raises(ValueError):
        G.decide(plain, 0.4369, 600_000, 25_000, allow_extension=False)


def test_shaped_report_flags_runs_trained_without_shaping(tmp_path, capsys):
    md = _shaped_models(tmp_path, (0.5, 0.5, 0.5))
    d = md / "shaped_plain_s9201"
    ident = json.loads((d / "model_identity.json").read_text())
    ident["shaping"] = "off"
    (d / "model_identity.json").write_text(json.dumps(ident))
    rc = R.main(["--shaped", "--models-dir", str(md), "--smoke", "--out", str(tmp_path / "o.json")])
    assert rc == 2 and "trained with shaping 'off'" in capsys.readouterr().out
    np.testing.assert_equal(rc, 2)


# ------------------------------------------------------------------------------------------------ Gate changelog (aa): Oracle-deploy diagnostic
def test_oracle_deploy_search_is_deterministic_budgeted_and_never_below_scripted():
    import scripts_b13_diag as D
    from breachlens import oracle as O
    from breachlens.sim import Prepared
    items = EV.load_set("TUNE-val", "validation", entry_point="tests")[::20]
    for rec, layout in items:
        Pp = Prepared(layout)
        army = O.army_search(layout, O.oracle_rng(rec["seed"], O.ARMY_SEARCH_TAG), n_random=8, n_refine=8, prepared=Pp)
        a = D.oracle_deploy_search(layout, Pp, army.types, rec["seed"])
        b = D.oracle_deploy_search(layout, Pp, army.types, rec["seed"])
        assert (a.key, a.types, a.start, a.sims) == (b.key, b.types, b.start, b.sims)
        assert a.sims <= D.ORACLE_DEPLOY_SIMS and sorted(a.types) == sorted(army.types)
        assert a.score >= D.scripted(layout, Pp, D.counts_of(army.types))["score"]


def test_oracle_deploy_is_not_part_of_the_default_run_and_row_aa_gates_the_shaped_launcher(clean, monkeypatch, tmp_path):
    src = (ROOT / "scripts_b13_diag.py").read_text()
    assert src.count("run_oracle_deploy(") == 2 and "if a.oracle_deploy:" in src            # definition + the one flagged call
    spec = L.SPEC.read_text()
    assert "(aa) **B13c" in spec and "7.4 skip clause NOT invoked" in spec and "[layout seed, 4444]" in spec
    f = tmp_path / "s.md"
    f.write_text(spec.replace("(aa) **B13c", "(zz) **gone"))
    monkeypatch.setattr(L, "SPEC", f)
    assert any("(aa) **B13c" in p for p in L.preflight(shaped=True))
