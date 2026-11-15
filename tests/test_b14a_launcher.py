"""B14a: grid launcher (scripts_b14_grid.py): frozen configs, queue order, commands, preflight refusals, disk rule, scheduler (slots, never two copies, kill +
resume, retention, end-of-job manifest checker, restart limit) and status. No real training: the scheduler tests drive a fake trainer process.
Nothing is launched; FINAL ledger unchanged."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts_b14_grid as G  # noqa: E402
from breachlens import manifest as MF  # noqa: E402
from breachlens import pool as P  # noqa: E402
from rl import train_ddqn_v2 as TD  # noqa: E402
from rl import train_ppo_v2 as TP  # noqa: E402

SPEC = (ROOT / "docs" / "SPEC_V2.md").read_text()
CFGS = {c: G.load_config(c) for c in G.CONFIG_FILES}


@pytest.fixture(autouse=True)
def _ledger_unchanged():
    before = P.LEDGER_PATH.read_bytes()
    yield
    assert P.LEDGER_PATH.read_bytes() == before


# ------------------------------------------------------------------------------------------------ frozen configs
def test_config_shas_are_recorded_in_row_ee_and_files_equal_the_code_constants():
    for k, s in G.config_shas().items():
        assert s in SPEC, k
    row = next(ln for ln in SPEC.splitlines() if ln.startswith("| 2026-11-15 | (ee) "))
    for k, s in G.config_shas().items():
        assert s in row, k
    assert G.configs_vs_code(CFGS, G.load_queue()) == []
    assert all(r in SPEC for r in G.ROWS)


def test_the_three_conditions_differ_only_where_they_must():
    a, b, c = CFGS["ppo_plain"], CFGS["ppo_d4"], CFGS["ddqn"]
    common = ("train_set", "val_set", "timesteps", "curriculum", "val_freq", "checkpoint_freq", "checkpoint_keep", "torch_threads", "shaping", "seeds", "eval_policy", "selection")
    for k in common:
        assert a[k] == b[k] == c[k], k
    assert a["ppo"] == b["ppo"] and a["aug"] == "none" and b["aug"] == "d4"
    assert c["ddqn"]["train_every"] == 8
    assert (a["timesteps"], a["curriculum"]["phase_a_steps"], a["val_freq"], a["seeds"]) == (1_500_000, 500_000, 25_000, list(range(10)))
    assert json.loads(G.STEP_BUDGET.read_text())["T"] == a["timesteps"]
    assert "no best checkpoint" in a["selection"]


# ------------------------------------------------------------------------------------------------ queue order and commands
def test_queue_order_ppo_first_seed_ascending_plain_then_d4_then_ddqn():
    jobs = G.job_list(CFGS)
    names = [j.name for j in jobs]
    assert len(jobs) == 30 and len(set(names)) == 30
    assert names[:4] == ["grid_ppo_plain_s0", "grid_ppo_d4_s0", "grid_ppo_plain_s1", "grid_ppo_d4_s1"]
    assert [j.algo for j in jobs] == ["ppo"] * 20 + ["ddqn"] * 10
    assert names[20:] == [f"grid_ddqn_s{s}" for s in range(10)]


def test_every_command_parses_with_its_trainer_and_matches_the_frozen_config():
    for j in G.job_list(CFGS):
        argv = G.build_argv(j, CFGS)
        cfg = CFGS[j.cond]
        assert argv[1] == "-m" and argv[2] == G.TRAINER[j.algo]
        a = (TP if j.algo == "ppo" else TD).parse_args(argv[3:])
        assert (a.train_set, a.seed, a.run_name, a.timesteps, a.val_freq, a.checkpoint_freq) == ("DEV-train", j.seed, j.name, 1_500_000, 25_000, 50_000)
        assert (a.curriculum, a.phase_a_steps, a.shaping, a.threads, a.resume) == ("on", 500_000, "on", 1, True)
        hp = cfg["ppo" if j.algo == "ppo" else "ddqn"]
        for k, v in hp.items():
            if k != "net_arch":
                assert getattr(a, k) == v, (j.name, k)
        if j.algo == "ppo":
            assert a.aug == cfg["aug"]
        else:
            assert a.train_every == 8


def test_dry_run_command_list_is_pinned():
    cmds = {j.name: G.build_argv(j, CFGS) for j in G.job_list(CFGS)}
    s = " ".join(cmds["grid_ppo_d4_s3"][1:])
    assert s.startswith("-m rl.train_ppo_v2 --train-set DEV-train --seed 3 --run-name grid_ppo_d4_s3 --timesteps 1500000 --val-freq 25000 --checkpoint-freq 50000 "
                        "--curriculum on --phase-a-steps 500000 --shaping on --threads 1 --resume --aug d4 --speed-freq 10000 ")
    d = " ".join(cmds["grid_ddqn_s9"][1:])
    assert d.startswith("-m rl.train_ddqn_v2 --train-set DEV-train --seed 9 --run-name grid_ddqn_s9 --timesteps 1500000 ") and "--train-every 8" in d and "--aug" not in d


def test_overrides_only_touch_known_hyperparameters_and_scale_phase_a():
    ov = G.Overrides(steps=5000, val_freq=1000, checkpoint_freq=1000, speed_freq=1000, hp=(("n_steps", 50), ("not_a_key", 1)))
    a = G.build_argv(G.job_list(CFGS)[0], CFGS, ov)
    assert "--n-steps" in a and a[a.index("--n-steps") + 1] == "50" and "not-a-key" not in " ".join(a)
    assert a[a.index("--phase-a-steps") + 1] == str(5000 // 3) and a[a.index("--timesteps") + 1] == "5000"


# ------------------------------------------------------------------------------------------------ preflight
@pytest.fixture
def clean(monkeypatch, tmp_path):
    monkeypatch.setattr(G.B, "dirty_tracked", lambda: [])
    monkeypatch.setattr(G.B, "ps_text", lambda: "  101 /bin/zsh -c eval 'python -m rl.train_ppo_v2 --run-name zzz'\n  102 /usr/bin/vim notes.txt\n")
    for k in G.B.FORBIDDEN_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(G, "REAL_LOGS", tmp_path / "logs_b14")
    monkeypatch.setattr(G.B, "REAL_MODELS", tmp_path / "models_v2")
    return tmp_path


def pf(tmp, **kw):
    jobs = G.job_list(CFGS)
    kw.setdefault("check_disk", False)
    return G.preflight(tmp / "models_v2", tmp / "logs_b14", jobs, G.Overrides(), False, **kw)


def test_clean_preflight_passes_on_the_committed_state(clean):
    assert pf(clean) == []


def test_refuses_dirty_tree(clean, monkeypatch):
    monkeypatch.setattr(G.B, "dirty_tracked", lambda: [" M rl/ddqn.py"])
    assert any("modified tracked" in p for p in pf(clean))


def _gates(tmp_path, monkeypatch, mutate):
    g = json.loads(G.GATES.read_text())
    mutate(g["g7"])
    f = tmp_path / "gates.json"
    f.write_text(json.dumps(g))
    monkeypatch.setattr(G, "GATES", f)


def test_refuses_unless_g7_passed_and_report_hashes_match(clean, monkeypatch):
    _gates(clean, monkeypatch, lambda g: g.update({"pass": False}))
    assert any("g7.pass" in p for p in pf(clean))
    _gates(clean, monkeypatch, lambda g: g.pop("pass"))
    assert any("g7.pass" in p for p in pf(clean))
    _gates(clean, monkeypatch, lambda g: g["reports"].update({"results/b13/g7_report_300k.txt": "0" * 64}))
    assert any("does not match the hash" in p for p in pf(clean))
    _gates(clean, monkeypatch, lambda g: g.update({"reports": {}}))
    assert any("no report hashes" in p for p in pf(clean))
    _gates(clean, monkeypatch, lambda g: g.update({"shaping_adopted_for_grid": False}))
    assert any("shaping_adopted" in p for p in pf(clean))
    monkeypatch.setattr(G, "GATES", clean / "missing.json")
    assert any("no readable g7" in p for p in pf(clean))


def test_refuses_when_a_row_or_a_config_sha_is_missing(clean, monkeypatch):
    for marker in G.ROWS:
        f = clean / "spec.md"
        f.write_text(SPEC.replace(marker, "(zz) **gone"))
        monkeypatch.setattr(G.B, "SPEC", f)
        assert any(marker in p for p in pf(clean)), marker
    for k, s in G.config_shas().items():
        f.write_text(SPEC.replace(s, "0" * 64))
        assert any(k in p and "not recorded" in p for p in pf(clean)), k


def test_refuses_foreign_trainers_pilots_and_leaves_own_orphans_alone(clean, monkeypatch):
    for line in (f"  555 {sys.executable} -m rl.train_ppo_v2 --run-name pilot_plain_s9001", f"  556 {sys.executable} -m rl.train_ddqn_v2 --run-name other",
                 f"  557 {sys.executable} scripts_b13_pilot.py --wait", f"  558 {sys.executable} scripts_b13_ddqn_pilot.py"):
        monkeypatch.setattr(G.B, "ps_text", lambda line=line: line + "\n")
        assert any("foreign trainer" in p for p in pf(clean)), line
    monkeypatch.setattr(G.B, "ps_text", lambda: f"  559 {sys.executable} -m rl.train_ppo_v2 --run-name grid_ppo_plain_s0 --resume\n")
    assert pf(clean) == []                                                       # an orphan of THIS grid is adopted by the scheduler, not a refusal
    monkeypatch.setattr(G.B, "ps_text", lambda: f"  560 {sys.executable} -m rl.train_ppo_v2 --run-name smoke_grid_ppo_plain_s0\n")
    assert any("foreign trainer" in p for p in pf(clean))                        # a smoke run is foreign to the real grid


def test_refuses_ledger_eval_entries_and_env_overrides(clean, monkeypatch):
    led = clean / "ledger.log"
    led.write_text(json.dumps({"purpose": "build"}) + "\n" + json.dumps({"purpose": "eval"}) + "\n")
    monkeypatch.setattr(G.B, "LEDGER", led)
    assert any("eval entr" in p for p in pf(clean))
    monkeypatch.undo()
    monkeypatch.setattr(G.B, "dirty_tracked", lambda: [])
    monkeypatch.setattr(G.B, "ps_text", lambda: "")
    monkeypatch.setenv("BREACHLENS_STATS_PATH", "/x")
    assert any("environment override" in p for p in G.preflight(G.B.REAL_MODELS, G.REAL_LOGS, G.job_list(CFGS), G.Overrides(), False, check_disk=False))


def test_real_mode_takes_no_overrides_and_smoke_never_the_real_dirs(clean):
    jobs = G.job_list(CFGS)
    assert any("no overrides" in p for p in G.preflight(clean / "models_v2", clean / "logs_b14", jobs, G.Overrides(steps=5000), False, check_disk=False))
    assert any("no overrides" in p for p in G.preflight(clean / "elsewhere", clean / "logs_b14", jobs, G.Overrides(), False, check_disk=False))
    assert any("--smoke must not use" in p for p in G.preflight(clean / "models_v2", clean / "logs_b14", jobs, G.Overrides(steps=5000), True, check_disk=False))
    assert not any("no overrides" in p for p in G.preflight(clean / "m", clean / "l", jobs, G.Overrides(steps=5000), True, check_disk=False))


def test_refuses_when_code_changed_since_the_first_launch(clean):
    logs = clean / "logs_b14"
    assert pf(clean) == []
    G.save_state(logs, {"first_launch": {"commit": "899cf82d1feaf5233a2d9874b0feac27b5d3a209"}})            # rl/ differs from that commit (B14a changed it)
    assert any("code changed since the first launch" in p for p in pf(clean))
    assert not any("code changed" in p for p in pf(clean, accept_code_change=True))
    G.save_state(logs, {"first_launch": {"commit": G.head_commit()}})
    assert pf(clean) == []


def test_refuses_when_free_disk_is_below_1_5x_the_remaining_peak(clean, monkeypatch):
    pk = G.peak_bytes({"ppo": 20, "ddqn": 10})["total"]
    monkeypatch.setattr(G.shutil, "disk_usage", lambda p: type("U", (), {"free": int(1.5 * pk) - 1})())
    assert any("free disk" in p for p in pf(clean, check_disk=True))
    monkeypatch.setattr(G.shutil, "disk_usage", lambda p: type("U", (), {"free": int(1.5 * pk) + 1})())
    assert pf(clean, check_disk=True) == []


# ------------------------------------------------------------------------------------------------ disk model
def test_peak_formula_and_the_numbers_for_the_real_grid():
    assert G.peak_bytes({"ppo": 0, "ddqn": 0})["total"] == 0
    c = G.SIZES["ddqn_ckpt"]
    assert G.peak_bytes({"ddqn": 10}, slots=6)["ddqn"] == (2 * 10 + 6) * c + 10 * G.SIZES["ddqn_other"]       # 4 finished x 2 + 6 running x 3 = 26 dirs
    assert G.peak_bytes({"ddqn": 4}, slots=6)["ddqn"] == (2 * 4 + 4) * c + 4 * G.SIZES["ddqn_other"]
    pk = G.peak_bytes({"ppo": 20, "ddqn": 10})
    assert 10.0e9 < pk["total"] < 11.0e9 and pk["ppo"] < 1e9
    assert G.ckpt_bytes("ddqn", 50_000) == G.SIZES["ddqn_ckpt"] and G.ckpt_bytes("ddqn", 2_000) < 0.1 * G.SIZES["ddqn_ckpt"]
    assert G.disk_problem(10_000, 0, 14_999) and not G.disk_problem(10_000, 0, 15_000)
    assert G.disk_problem(10_000, 4_000, 8_999) and not G.disk_problem(10_000, 4_000, 9_000)                  # resume: only the remaining growth counts
    t = G.time_estimate()
    assert 7.5 < t["ppo_h"] < 8.5 and 9 < t["ddqn_h"] < 10.5 and t["total_low_h"] < t["total_h"] < t["total_high_h"]


# ------------------------------------------------------------------------------------------------ process table, lock
def test_running_jobs_matches_run_names_only():
    ps = (f"  10 {sys.executable} -m rl.train_ppo_v2 --run-name grid_ppo_plain_s1 --resume\n  11 {sys.executable} -m rl.train_ppo_v2 --run-name grid_ppo_plain_s10\n"
          f"  12 /bin/zsh -c python -m rl.train_ppo_v2 --run-name grid_ppo_plain_s2\n  13 vim grid_ppo_plain_s3\n")
    assert G.running_jobs(ps, {"grid_ppo_plain_s1", "grid_ppo_plain_s2", "grid_ppo_plain_s3"}) == {"grid_ppo_plain_s1": 10}


def test_scheduler_lock_is_exclusive(tmp_path):
    a, b = G.SchedulerLock(tmp_path), G.SchedulerLock(tmp_path)
    assert a.held_by_other() is None
    assert a.acquire() and not b.acquire()
    assert a.held_by_other() == os.getpid()
    os.close(a.fd)
    assert b.acquire()


# ------------------------------------------------------------------------------------------------ scheduler with a fake trainer
FAKE = textwrap.dedent('''
    import json, os, shutil, signal, sys, time
    from pathlib import Path
    a = sys.argv
    name = a[a.index("--run-name") + 1]
    T = int(a[a.index("--timesteps") + 1])
    root = Path(os.environ["BREACHLENS_MODELS_DIR"]) / name
    cfg = json.loads(os.environ["FAKE_CFG"])
    root.mkdir(parents=True, exist_ok=True)
    ck = root / "checkpoints"
    ck.mkdir(exist_ok=True)
    note = Path(cfg["notes"]) / (name + ".log")
    note.parent.mkdir(exist_ok=True)
    def say(s):
        with note.open("a") as fh:
            fh.write(f"{time.time()} {s}\\n")
    done = sorted(int(c.name.split("_")[1]) for c in ck.glob("ckpt_*") if c.is_dir() and not c.name.endswith(".tmp"))
    t = done[-1] if done else 0
    say(f"start t={t} pid={os.getpid()}")
    if (root / "final.zip").exists():
        sys.exit(0)
    from breachlens import pool as P
    tr = [r["id"] for r in P.dev_records("train")]; va = [r["id"] for r in P.dev_records("validation")]
    step = T // 6
    die_at = cfg.get("die_at", {}).get(name)
    marker = Path(cfg["notes"]) / (name + ".died")
    while t < T:
        time.sleep(cfg.get("sleep", 0.05))
        t += step
        (ck / f"ckpt_{t}").mkdir()
        (ck / f"ckpt_{t}" / "state.pkl").write_text("x"); (ck / f"ckpt_{t}" / "model.zip").write_text("x")
        for c in sorted(ck.glob("ckpt_*"), key=lambda p: int(p.name.split("_")[1]))[:-2]:
            shutil.rmtree(c)
        with (root / "val_eval.csv").open("a") as fh:
            if t == step:
                fh.write("timestep,group,score_mean\\n")
            fh.write(f"{t},ALL,0.5\\n")
        if die_at and t >= die_at and not marker.exists():
            marker.write_text("died")
            os.kill(os.getpid(), signal.SIGKILL)
        if cfg.get("always_die") and name in cfg["always_die"]:
            sys.exit(3)
    final_ids = list(tr)
    if name in cfg.get("leak", []):
        final_ids.append(next(r["id"] for r in json.loads(P.POOL_PATH.read_text())["records"] if r["split"] == "FINAL-familiar"))
    (root / "manifest.json").write_text(json.dumps({"run_name": name, "train_set": "DEV-train", "val_set": "DEV-validation", "train_ids": final_ids, "val_ids": va}))
    (root / "val_layouts.jsonl").write_text("".join(json.dumps({"timestep": T, "id": i}) + "\\n" for i in va))
    if name in cfg.get("extra_ckpt", []):
        for k in (1, 2, 3):
            (ck / f"ckpt_{k}").mkdir(exist_ok=True); (ck / f"ckpt_{k}" / "state.pkl").write_text("x"); (ck / f"ckpt_{k}" / "model.zip").write_text("x")
    (root / "model_identity.json").write_text(json.dumps({"status": "finished", "timesteps_trained": T, "git_commit": "c", "git_dirty": False, "resumes": []}))
    (root / "final.zip").write_text("x"); (root / "final.pt").write_text("x")
    say("done")
''')


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "fake_trainer.py"
    script.write_text(FAKE)
    models, logs, notes = tmp_path / "models", tmp_path / "logs", tmp_path / "notes"
    models.mkdir(), logs.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    jobs = [G.Job(f"smoke_j{i}", "ppo_plain" if i % 2 == 0 else "ddqn", i) for i in range(5)]

    def make(cfg=None, slots=2, max_restarts=3):
        monkeypatch.setenv("FAKE_CFG", json.dumps({"notes": str(notes), **(cfg or {})}))
        argv_fn = lambda j: [sys.executable, str(script), "rl.train_ppo_v2", "--run-name", j.name, "--timesteps", "600"]  # noqa: E731
        return G.Scheduler(models, logs, CFGS, jobs, G.Overrides(steps=600), slots=slots, poll=0.05, max_restarts=max_restarts, argv_fn=argv_fn)
    return make, models, logs, notes, jobs


def starts(notes, name):
    f = notes / f"{name}.log"
    return [ln for ln in f.read_text().splitlines() if " start t=" in ln] if f.exists() else []


def test_scheduler_runs_the_queue_in_order_within_the_slot_limit_and_checks_every_job(fake):
    make, models, logs, notes, jobs = fake
    s = make(slots=2)
    assert s.run() == 0
    for j in jobs:
        res = json.loads((logs / f"{j.name}.result.json").read_text())
        assert res["checks"]["ok"] and res["checks"]["manifest_ok"] and res["checks"]["retention_ok"] and res["checks"]["val_ids_in_manifest"]
        assert len(starts(notes, j.name)) == 1                                           # never started twice
        assert G.job_state(models, logs, j, {}, set()) == "finished"
    t0 = [float(starts(notes, j.name)[0].split()[0]) for j in jobs]
    assert t0 == sorted(t0)                                                              # started in queue order
    ends = [float((notes / f"{j.name}.log").read_text().splitlines()[-1].split()[0]) for j in jobs]
    for i, t in enumerate(t0):
        assert sum(1 for k in range(len(jobs)) if t0[k] <= t < ends[k]) <= 2, i          # at most 2 slots busy at any job start


def test_a_killed_job_is_restarted_with_resume_and_continues_from_its_checkpoint(fake):
    make, models, logs, notes, jobs = fake
    s = make({"die_at": {"smoke_j1": 300}}, slots=3)
    assert s.run() == 0
    st = starts(notes, "smoke_j1")
    assert len(st) == 2 and "t=0" in st[0] and "t=300" in st[1]                          # second start resumed at the latest checkpoint
    assert json.loads((logs / "smoke_j1.result.json").read_text())["checks"]["ok"]
    assert G.load_state(logs)["attempts"]["smoke_j1"] == 1


def test_a_job_that_keeps_dying_is_marked_failed_and_the_rest_still_finish(fake):
    make, models, logs, notes, jobs = fake
    s = make({"always_die": ["smoke_j2"]}, slots=2, max_restarts=2)
    assert s.run() == 1
    assert "smoke_j2" in s.failed and len(starts(notes, "smoke_j2")) == 2
    assert G.job_state(models, logs, jobs[2], {}, s.failed) == "FAILED"
    for j in jobs:
        if j.name != "smoke_j2":
            assert G.job_state(models, logs, j, {}, set()) == "finished"


def test_manifest_checker_runs_at_the_end_of_every_job_and_fails_on_a_final_id(fake):
    make, models, logs, notes, jobs = fake
    s = make({"leak": ["smoke_j3"]}, slots=3)
    assert s.run() == 1
    res = json.loads((logs / "smoke_j3.result.json").read_text())["checks"]
    assert res["manifest_ok"] is False and "FINAL" in res["manifest_error"] and res["ok"] is False
    assert G.job_state(models, logs, jobs[3], {}, set()) == "CHECK-FAILED"
    assert json.loads((logs / "smoke_j0.result.json").read_text())["checks"]["ok"]


def test_retention_check_fails_with_more_than_two_checkpoints_or_a_leftover_tmp(fake, tmp_path):
    make, models, logs, notes, jobs = fake
    s = make({"extra_ckpt": ["smoke_j0"]}, slots=3)
    s.run()
    r = json.loads((logs / "smoke_j0.result.json").read_text())["checks"]
    assert r["n_ckpt"] > 2 and r["retention_ok"] is False and r["ok"] is False
    d = models / "smoke_j1"
    assert G.check_job_outputs(models, jobs[1], 600)["retention_ok"]
    (d / "checkpoints" / "ckpt_999.tmp").mkdir()
    assert G.check_job_outputs(models, jobs[1], 600)["retention_ok"] is False
    G.clean_tmp(models, jobs[1])
    assert G.check_job_outputs(models, jobs[1], 600)["retention_ok"]


def test_a_running_orphan_is_adopted_and_never_started_a_second_time(fake):
    make, models, logs, notes, jobs = fake
    s = make({"sleep": 0.3}, slots=2)
    orphan = subprocess.Popen(s.argv_fn(jobs[0]), env=os.environ.copy() | {"BREACHLENS_MODELS_DIR": str(models)}, start_new_session=True)
    try:
        time.sleep(0.5)
        assert s.start(jobs[0]) is False                                                 # the process-table scan finds it
        assert s.run() == 0
        assert len(starts(notes, "smoke_j0")) == 1                                       # only the orphan ever ran it
        assert json.loads((logs / "smoke_j0.result.json").read_text())["checks"]["ok"]
    finally:
        orphan.kill()
        orphan.wait()


def test_status_rows_and_rendering(fake):
    make, models, logs, notes, jobs = fake
    G.save_state(logs, {"target_steps": 600})
    rows = G.status_rows(models, logs, jobs, 600, "")
    assert [r["state"] for r in rows] == ["queued"] * 5
    make(slots=3).run()
    rows = G.status_rows(models, logs, jobs, 600, "")
    assert all(r["state"] == "finished" and r["t"] == 600 and r["pct"] == 100.0 and r["check"] is True and r["val"] == 0.5 for r in rows)
    txt = G.render_status(rows, 6, None)
    assert "smoke_j0" in txt and "finished" in txt and "NOT running" in txt and "100.0%" in txt
    (models / "smoke_j4" / "final.zip").unlink()
    (logs / "smoke_j4.result.json").unlink()
    assert G.status_rows(models, logs, jobs, 600, "")[4]["state"] == "interrupted"


def test_a_stopped_scheduler_starts_nothing_new_and_leaves_running_trainers_alone(fake):
    make, models, logs, notes, jobs = fake
    s = make({"sleep": 0.3}, slots=3)
    assert s.start(jobs[0])
    s.stop = True
    assert s.tick() is True                                                              # j0 still running, nothing else started
    assert [j.name for j in jobs if (models / j.name).exists()] == ["smoke_j0"]
    s.procs["smoke_j0"].wait(timeout=20)
    s.stop = False
    assert s.run() == 0                                                                  # a fresh scheduler checks the job that finished unobserved, then runs the rest
    assert len(starts(notes, "smoke_j0")) == 1
    assert json.loads((logs / "smoke_j0.result.json").read_text())["checks"]["ok"]
