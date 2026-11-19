"""B13e: `--shaping` for the DDQN trainer (row cc). Shaping is the SAME BreachEnv implementation as the PPO shaped pilot (row z), training envs only.
TUNE sets only; FINAL ledger unchanged. Tiny CPU runs."""
from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from breachlens import evaluator as EV
from breachlens import pool as P
from breachlens.env import BreachEnv
from rl import ddqn as D
from rl import train_ddqn_v2 as T
from rl import train_ppo_v2 as TP

ROOT = Path(__file__).resolve().parents[1]
TINY = ["--timesteps", "1600", "--n-envs", "4", "--buffer-size", "2000", "--batch-size", "16", "--learning-starts", "200",
        "--target-update-interval", "50", "--val-freq", "800", "--checkpoint-freq", "400"]


@pytest.fixture(autouse=True)
def _models_dir(tmp_path, monkeypatch):
    before = P.LEDGER_PATH.read_bytes()
    monkeypatch.setattr(T, "MODELS_DIR", tmp_path)
    yield
    assert P.LEDGER_PATH.read_bytes() == before


def run(name, extra=(), base=TINY):
    return T.main(["--run-name", name, *base, *extra])


def ident(name):
    return json.loads((T.MODELS_DIR / name / "model_identity.json").read_text())


def val_rows(name):
    with (T.MODELS_DIR / name / "val_eval.csv").open(newline="") as fh:
        return [{k: v for k, v in r.items() if k != "eval_seconds"} for r in csv.DictReader(fh)]


def test_ddqn_shaping_flag_defaults_off_and_reuses_the_ppo_env_implementation():
    assert T.parse_args(["--run-name", "x"]).shaping == "off"
    assert T.build_venv is TP.build_venv                                   # one implementation: BreachEnv(shaping=...) via the PPO builder
    v = T.build_venv("TUNE-train", "none", 0, 2, None, shaping=True)
    assert all(e.unwrapped.shaping for e in v.envs)
    assert not any(e.unwrapped.shaping for e in T.build_venv("TUNE-train", "none", 0, 2, None).envs)
    src = (ROOT / "rl" / "train_ddqn_v2.py").read_text()
    assert "_phi" not in src and "shaping_reward" not in src and "BreachEnv(" not in src    # no copy of the formula; the env is only built via the PPO builder


# ------------------------------------------------------------------------------------------------ validation never shapes
def test_shaped_and_unshaped_validation_give_identical_true_scores_for_the_same_q_network():
    items = EV.load_set("TUNE-val", "validation", entry_point="tests")[::5]
    out = {}
    for shaping in (False, True):
        venv = T.build_venv("TUNE-train", "none", 0, 2, None, shaping=shaping)
        hp = dict(D.DDQN_CONFIG, n_envs=2)
        tr = T.Trainer(hp, venv, 3, 1000, 0)                                  # seeded init: the same Q-network in both
        res = EV.evaluate(D.DDQNPolicy(tr.online, name="q"), "TUNE-val", "validation", items=items)
        out[shaping] = (tr.sha(), [(r["id"], r["score"], r["d"], r["breach"]) for r in res["rows"]])
    assert out[False][0] == out[True][0] and out[False][1] == out[True][1] and out[False][1]


def test_validation_envs_never_shape_when_training_does(monkeypatch):
    E = D                                                                   # the DDQN evaluation policy builds its own BreachEnv (rl/ddqn.py)
    seen = {"eval": [], "train": []}

    def recorder(kind):
        def make(*a, **k):
            env = BreachEnv(*a, **k)
            seen[kind].append(env.shaping)
            return env
        return make
    monkeypatch.setattr(E, "BreachEnv", recorder("eval"))
    monkeypatch.setattr(TP, "BreachEnv", recorder("train"))
    d = run("v", ["--shaping", "on"])
    assert seen["train"] and all(seen["train"]) and seen["eval"] and not any(seen["eval"])
    i = ident("v")
    assert i["shaping"] == "on" and i["config"]["shaping"] == "on"
    assert {r["timestep"] for r in val_rows("v")} == {"0", "800", "1600"} and d.name == "v"


def test_first_validation_eval_is_identical_for_shaped_and_unshaped_runs(tmp_path, monkeypatch):
    rows = {}
    for mode in ("off", "on"):
        monkeypatch.setattr(T, "MODELS_DIR", tmp_path / mode)
        run("r", ["--seed", "5", "--shaping", mode])
        rows[mode] = [r for r in val_rows("r") if r["timestep"] == "0"]
        assert ident("r")["shaping"] == mode
    assert rows["off"] and rows["off"] == rows["on"]


# ------------------------------------------------------------------------------------------------ replay buffer carries the shaped reward
def _collect(shaping: bool, steps: int = 1200):
    venv = T.build_venv("TUNE-train", "none", 0, 2, None, shaping=shaping)
    scores = [[], []]
    real = venv.step

    def spy(a):
        o, r, d, infos = real(a)
        for i, dn in enumerate(d):
            if dn:
                scores[i].append(float(infos[i]["score"]))
        return o, r, d, infos
    venv.step = spy
    hp = dict(D.DDQN_CONFIG, n_envs=2, buffer_size=4000, learning_starts=10 ** 9)          # no gradient updates: only the data path is under test
    tr = T.Trainer(hp, venv, 1, steps, 0)
    while tr.t < steps:
        tr.vector_step()
    return tr, scores


def test_shaped_transitions_in_the_buffer_carry_the_shaped_reward_and_sum_to_the_true_terminal_score():
    tr, scores = _collect(True)
    tr_u, scores_u = _collect(False)
    buf, n = tr.buffer, tr.buffer.n
    assert n * 2 == 1200 and buf.n_envs == 2
    nonterminal_nonzero = 0
    for e in range(2):
        r, d = buf.reward[e, :n].astype(np.float64), buf.done[e, :n]
        ends = np.flatnonzero(d)
        assert len(ends) >= 3 and len(ends) == len(scores[e])
        start = 0
        for k, end in enumerate(ends):
            assert abs(r[start:end + 1].sum() - scores[e][k]) < 1e-5, (e, k)             # gamma 1, Phi(terminal) = 0: sum of shaped rewards = true score
            nonterminal_nonzero += int((r[start:end] != 0).sum())
            start = end + 1
    assert nonterminal_nonzero > 0                                                          # shaping really put reward on non-terminal steps
    bu = tr_u.buffer
    for e in range(2):                                                                      # unshaped control: reward only at the terminal step
        r, d = bu.reward[e, :bu.n], bu.done[e, :bu.n]
        assert (r[~d] == 0).all() and d.any()


# ------------------------------------------------------------------------------------------------ determinism / resume / SIGKILL with shaping on
def test_shaped_runs_are_bit_identical_and_differ_from_unshaped():
    run("s1", ["--shaping", "on"]), run("s2", ["--shaping", "on"]), run("u", ["--shaping", "off"])
    a, b, u = ident("s1"), ident("s2"), ident("u")
    for k in ("params_sha256", "params_optimizer_sha256", "updates"):
        assert a[k] == b[k]
    assert a["params_optimizer_sha256"] != u["params_optimizer_sha256"]
    assert (T.MODELS_DIR / "s1" / "val_layouts.jsonl").read_text() == (T.MODELS_DIR / "s2" / "val_layouts.jsonl").read_text()


@pytest.mark.parametrize("at", [400, 1200])
def test_shaped_resume_is_bit_identical_to_an_uninterrupted_run(at):
    run("full", ["--shaping", "on"])
    with pytest.raises(T.Interrupted):
        run("part", ["--shaping", "on", "--debug-interrupt-at", str(at)])
    run("part", ["--resume"])                                                               # shaping comes back from the recorded identity
    f, p = ident("full"), ident("part")
    assert p["shaping"] == "on" and p["resumes"][0]["at_timestep"] >= at
    for k in ("params_sha256", "params_optimizer_sha256", "updates", "timesteps_trained"):
        assert f[k] == p[k]
    assert val_rows("full") == val_rows("part")
    assert (T.MODELS_DIR / "full" / "val_layouts.jsonl").read_text() == (T.MODELS_DIR / "part" / "val_layouts.jsonl").read_text()


def test_shaped_resume_after_sigkill_is_bit_identical(tmp_path):
    base = ["--timesteps", "3200", "--n-envs", "4", "--buffer-size", "2000", "--batch-size", "16", "--learning-starts", "200",
            "--target-update-interval", "50", "--val-freq", "800", "--checkpoint-freq", "400", "--shaping", "on"]
    run("kfull", base=base)
    env = dict(os.environ, BREACHLENS_MODELS_DIR=str(tmp_path), OMP_NUM_THREADS="1", PYTHONPATH=str(ROOT))
    cmd = [sys.executable, "-m", "rl.train_ddqn_v2", "--run-name", "kpart", *base, "--resume"]
    pr = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ck = tmp_path / "kpart" / "checkpoints"
    deadline = time.time() + 300
    while time.time() < deadline and pr.poll() is None:
        if any((d / "state.pkl").exists() and (d / "buffer" / "meta.json").exists() for d in ck.glob("ckpt_*") if not d.name.endswith(".tmp")) \
                and (tmp_path / "kpart" / "val_eval.csv").exists() and len(val_rows("kpart")) > 4:
            break
        time.sleep(0.05)
    assert pr.poll() is None, "the run finished before it could be killed"
    os.kill(pr.pid, signal.SIGKILL)
    pr.wait()
    assert not (tmp_path / "kpart" / "final.pt").exists()
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    f, p = ident("kfull"), ident("kpart")
    assert p["shaping"] == "on" and p["status"] == "finished" and p["resumes"]
    for k in ("params_sha256", "params_optimizer_sha256", "updates", "timesteps_trained"):
        assert f[k] == p[k]
    assert val_rows("kfull") == val_rows("kpart")


# ------------------------------------------------------------------------------------------------ launcher + report (scripts)
sys.path.insert(0, str(ROOT))
from scripts import b13_ddqn_pilot as L  # noqa: E402
from scripts import b13_ddqn_report as R  # noqa: E402


@pytest.fixture
def clean(monkeypatch, build_only_ledger):
    monkeypatch.setattr(L.B, "LEDGER", build_only_ledger)                     # B15d: synthetic build-only ledger for "would start" paths
    monkeypatch.setattr(L.B, "dirty_tracked", lambda: [])
    monkeypatch.setattr(L.B, "ps_text", lambda: "  102 /usr/bin/vim notes.txt\n")
    for k in L.B.FORBIDDEN_ENV:
        monkeypatch.delenv(k, raising=False)


def test_ddqn_config_sha_is_recorded_in_row_cc_and_matches_the_code(clean):
    cfg = L.frozen()
    assert L.B.config_sha(cfg) in L.B.SPEC.read_text()
    assert L.config_matches_code(cfg) == [] and L.preflight() == []
    assert cfg["seeds"] == [9301, 9302] and cfg["timesteps"] == 300000 and cfg["val_freq"] == 25000 and cfg["torch_threads"] == 1


def test_ddqn_launcher_commands_pin_every_hyperparameter_and_shaping(clean):
    cfg = L.frozen()
    cmds = L.build_commands(cfg, steps=300000, val_freq=25000, checkpoint_freq=50000, learning_starts=5000, seeds=[9301, 9302])
    assert [n for n, _ in cmds] == ["ddqn_pilot_f4_s9301", "ddqn_pilot_f4_s9302", "ddqn_pilot_f8_s9301", "ddqn_pilot_f8_s9302"]
    for name, c in cmds:
        args = T.parse_args(c[c.index("rl.train_ddqn_v2") + 1:])
        assert args.shaping == "on" and args.timesteps == 300000 and args.phase_a_steps == 100000 and args.threads == 1 and args.val_freq == 25000
        assert args.train_set == "TUNE-train" and args.curriculum == "on"
        for k, v in D.DDQN_CONFIG.items():
            if k not in ("net_arch", "train_every"):
                assert getattr(args, k) == v, k
        assert args.train_every == (4 if "_f4_" in name else 8)


@pytest.mark.parametrize("what", ["row_p", "row_cc", "dirty", "running", "ledger", "env", "hash"])
def test_ddqn_launcher_refusals(clean, monkeypatch, tmp_path, capsys, what):
    if what in ("row_p", "row_cc", "hash"):
        txt = L.B.SPEC.read_text()
        txt = {"row_p": txt.replace("(p) **B12 item 0", "(zz) gone"), "row_cc": txt.replace("(cc) **B13d", "(zz) gone"),
               "hash": txt.replace(L.B.config_sha(L.frozen()), "0" * 64)}[what]
        f = tmp_path / "spec.md"
        f.write_text(txt)
        monkeypatch.setattr(L.B, "SPEC", f)
    elif what == "dirty":
        monkeypatch.setattr(L.B, "dirty_tracked", lambda: [" M rl/ddqn.py"])
    elif what == "running":
        monkeypatch.setattr(L.B, "ps_text", lambda: "  555 /x/python -m rl.train_ddqn_v2 --run-name a\n")
    elif what == "ledger":
        monkeypatch.setattr(L.B, "eval_entries", lambda ledger=None: 1)
    elif what == "env":
        monkeypatch.setenv("BREACHLENS_GEN_PARAMS", "x")
    started = []
    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: started.append(a))
    assert L.preflight() and L.main([]) == 3 and not started
    assert "REFUSED" in capsys.readouterr().err


def test_ddqn_launcher_refuses_overrides_without_smoke_and_smoke_in_real_models_dir(clean, monkeypatch, capsys):
    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: pytest.fail("started"))
    assert L.main(["--steps", "1000"]) == 3
    assert L.main(["--smoke", "--steps", "1000"]) == 3
    assert L.main(["--print-only"]) == 0


def test_ddqn_launcher_refuses_on_the_real_ledger_permanently_after_b15(clean, monkeypatch, capsys):
    monkeypatch.setattr(L.B, "LEDGER", L.B.ROOT / "data" / "v2" / "final_access.log")
    assert any("eval entr" in p for p in L.preflight())
    assert L.main(["--print-only"]) == 3 and "REFUSED" in capsys.readouterr().err


def test_ddqn_cfg_tampering_changes_the_hash(clean):
    cfg = L.frozen()
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["ddqn"]["gamma"] = 0.99
    assert L.B.config_sha(cfg2) != L.B.config_sha(cfg) and L.config_matches_code(cfg2)


def write_run(models, freq, seed, scores, breach=0.3, d=0.2, bought=7.0, n_armies=12):
    d_ = models / f"ddqn_pilot_f{freq}_s{seed}"
    d_.mkdir(parents=True)
    with (d_ / "val_eval.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestep", "group", "score_mean", "breach_mean", "d_mean", "mean_n_bought"])
        for t, sc in scores.items():
            w.writerow([t, "ALL", sc, breach, d, bought])
            w.writerow([t, "spine_pods", 99.0, 99.0, 99.0, 99.0])                  # non-ALL rows must be ignored
    with (d_ / "val_layouts.jsonl").open("w") as fh:
        for t in scores:
            for i in range(60):
                fh.write(json.dumps({"timestep": t, "id": f"x{i}", "army": [i % n_armies, 1, 0, 0, 0]}) + "\n")


def curve(last3, early=0.1):
    out = {t: early for t in range(0, 250_000, 25_000)}
    out.update(dict(zip((250_000, 275_000, 300_000), last3)))
    return out


def test_report_selection_uses_last_three_evals_and_both_seeds_and_the_tie_rule(tmp_path, capsys):
    for freq, vals in ((4, [(0.40, 0.41, 0.42), (0.44, 0.45, 0.46)]), (8, [(0.40, 0.40, 0.40), (0.40, 0.40, 0.40)])):
        for seed, v in zip((9301, 9302), vals):
            write_run(tmp_path, freq, seed, curve(v, early=0.9 if freq == 4 else 0.0))        # early evals must not matter
    txt = R.report(tmp_path, [9301, 9302], 300_000, 25_000)
    assert "arm mean (last 3 evals, both seeds) = 0.4300" in txt and "arm mean (last 3 evals, both seeds) = 0.4000" in txt
    assert "CHOSEN train_freq = 4" in txt and "PASS" not in txt and "FAIL" not in txt and "distinct_armies" in txt
    assert R.main(["--models-dir", str(tmp_path)]) == 0 and "CHOSEN train_freq = 4" in capsys.readouterr().out


@pytest.mark.parametrize("m4,m8,want", [(0.43, 0.42, 4), (0.42, 0.43, 8), (0.4301, 0.42, 4), (0.42, 0.4301, 8), (0.50, 0.40, 4), (0.4099, 0.42, 8)])
def test_choose_tie_rule(m4, m8, want):
    assert R.choose(m4, m8)["train_freq"] == want


def test_report_tie_prints_8_and_incomplete_logs_print_no_selection(tmp_path, capsys):
    for freq, v in ((4, 0.455), (8, 0.450)):
        for seed in (9301, 9302):
            write_run(tmp_path, freq, seed, curve((v, v, v)), n_armies=7)
    txt = R.report(tmp_path, [9301, 9302], 300_000, 25_000)
    assert "CHOSEN train_freq = 8" in txt and "tie" in txt and " 7" in txt
    (tmp_path / "ddqn_pilot_f8_s9302" / "val_eval.csv").write_text("timestep,group,score_mean,breach_mean,d_mean,mean_n_bought\n0,ALL,0.1,0.1,0.1,1\n")
    assert R.main(["--models-dir", str(tmp_path)]) == 2 and "INCOMPLETE" in capsys.readouterr().out
