"""B13a: pilot launcher preflight / commands and the frozen config. Nothing is launched here."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import b13_pilot as L  # noqa: E402
from rl.train_ppo_v2 import PPO_CONFIG  # noqa: E402


@pytest.fixture
def clean(monkeypatch, build_only_ledger):
    monkeypatch.setattr(L, "LEDGER", build_only_ledger)                       # B15d: the real ledger now has a finished eval session; "would start" paths use a synthetic one
    monkeypatch.setattr(L, "dirty_tracked", lambda: [])
    monkeypatch.setattr(L, "ps_text", lambda: "  101 /bin/zsh -c eval 'python -m rl.train_ppo_v2 scripts/b13_pilot.py'\n  102 /usr/bin/vim notes.txt\n")
    for k in L.FORBIDDEN_ENV:
        monkeypatch.delenv(k, raising=False)


def test_clean_preflight_passes_on_the_committed_tree_state(clean):
    assert L.preflight() == []                                    # real SPEC rows, synthetic 5 build / 0 eval ledger, shell text is not a running copy


def test_refuses_on_the_real_ledger_permanently_after_b15(clean, monkeypatch, capsys):
    monkeypatch.setattr(L, "LEDGER", L.ROOT / "data" / "v2" / "final_access.log")             # the real, post-B15 ledger
    assert L.eval_entries() == 238
    assert any("238 eval entries" in p for p in L.preflight())
    assert L.main(["--print-only"]) == 3 and "REFUSED" in capsys.readouterr().err


def test_refuses_dirty_tracked_tree(clean, monkeypatch):
    monkeypatch.setattr(L, "dirty_tracked", lambda: [" M rl/planner.py"])
    assert any("modified tracked" in p for p in L.preflight())


def test_refuses_when_an_item0_row_or_the_config_hash_is_missing(clean, monkeypatch, tmp_path):
    spec = L.SPEC.read_text()
    for marker in L.ROWS:
        f = tmp_path / "spec.md"
        f.write_text(spec.replace(marker, "(zz) **gone"))
        monkeypatch.setattr(L, "SPEC", f)
        assert any(marker in p for p in L.preflight()), marker
    f.write_text(spec.replace(L.config_sha(L.frozen()), "0" * 64))
    assert any("frozen-config" in p for p in L.preflight())


def test_refuses_when_any_copy_is_already_running(clean, monkeypatch):
    own = os.getpid()
    for line in (f"  555 caffeinate -dimsu {sys.executable} -m rl.train_ppo_v2 --seed 1", f"  556 {sys.executable} -m rl.train_ppo_v2 --seed 2",
                 f"  557 {sys.executable} scripts/b13_pilot.py --wait"):
        monkeypatch.setattr(L, "ps_text", lambda line=line: line + "\n")
        assert any("already running" in p for p in L.preflight()), line
    monkeypatch.setattr(L, "ps_text", lambda: f"  {own} {sys.executable} scripts/b13_pilot.py\n")
    assert L.preflight() == []                                    # the launcher itself is not "another copy"


def test_refuses_when_the_ledger_has_an_eval_entry(clean, monkeypatch, tmp_path):
    led = tmp_path / "ledger.log"
    led.write_text(json.dumps({"purpose": "build"}) + "\n" + json.dumps({"purpose": "eval", "session_id": "x"}) + "\n")
    monkeypatch.setattr(L, "LEDGER", led)
    assert any("eval entr" in p for p in L.preflight())
    assert L.eval_entries(led) == 1


@pytest.mark.parametrize("var", L.FORBIDDEN_ENV)
def test_refuses_when_a_stats_or_generator_override_is_set(clean, monkeypatch, var):
    monkeypatch.setenv(var, "x")
    assert any(var in p for p in L.preflight())


def test_main_exits_3_with_reasons_and_starts_nothing(clean, monkeypatch, capsys):
    monkeypatch.setattr(L, "dirty_tracked", lambda: [" M x"])
    started = []
    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: started.append(a))
    assert L.main([]) == 3 and not started
    assert "REFUSED" in capsys.readouterr().err


def test_real_mode_takes_no_overrides_and_smoke_cannot_touch_models_v2(clean, capsys):
    assert L.main(["--steps", "5000"]) == 3
    assert L.main(["--plain-seeds", "1,2,3"]) == 3
    assert L.main(["--smoke", "--steps", "5000"]) == 3            # --smoke with the default models dir
    assert "REFUSED" in capsys.readouterr().err


def test_command_list_for_the_real_pilot(clean, capsys):
    assert L.main(["--print-only"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln]
    assert len(lines) == 6
    cfg = L.frozen()
    for ln, (aug, seed) in zip(lines, [("none", 9001), ("none", 9002), ("none", 9003), ("d4", 9101), ("d4", 9102), ("d4", 9103)]):
        assert " caffeinate -dimsu " in ln and "-m rl.train_ppo_v2" in ln and f"--aug {aug}" in ln and f"--seed {seed}" in ln
        assert "--threads 1" in ln and "--timesteps 300000" in ln and "--phase-a-steps 100000" in ln and "--curriculum on" in ln and "--train-set TUNE-train" in ln
        assert "--val-freq 25000" in ln and "--checkpoint-freq 50000" in ln and "--speed-freq 10000" in ln and "--resume" in ln
        assert f"pilot_{'plain' if aug == 'none' else 'd4'}_s{seed}" in ln and f"> {ROOT}/logs_b13/" in ln
        for k, v in cfg["ppo"].items():
            if k != "net_arch":
                assert f"--{k.replace('_', '-')} {v}" in ln, k
    assert "BREACHLENS_MODELS_DIR=" in lines[0]


def test_extension_command_continues_with_the_same_config(clean, capsys):
    assert L.main(["--print-only", "--extend"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 6 and all("--extend-to 600000" in ln and "--timesteps" not in ln and "--phase-a-steps" not in ln for ln in lines)


# ------------------------------------------------------------------------------------------------ the frozen config
def test_frozen_config_matches_the_code_the_changelog_and_the_launcher():
    cfg = L.frozen()
    assert json.loads(json.dumps(PPO_CONFIG)) == cfg["ppo"]
    assert (cfg["ppo"]["n_envs"], cfg["ppo"]["n_steps"], cfg["torch_threads"], cfg["timesteps"], cfg["extension_timesteps"]) == (10, 250, 1, 300_000, 600_000)
    assert cfg["curriculum"]["phase_a_steps"] == cfg["timesteps"] // 3 and cfg["val_freq"] == 25_000 and cfg["shaping"] == "off"
    assert len(cfg["plain_seeds"]) == 3 and len(cfg["d4_seeds"]) == 3 and not set(cfg["plain_seeds"] + cfg["d4_seeds"]) & set(range(10))
    spec = L.SPEC.read_text()
    assert L.config_sha(cfg) in spec                              # the changelog row (v) quotes the hash of this exact file
    for m in L.ROWS:
        assert m in spec
    assert "0.4369" in spec and "(≥ 10 distinct army compositions" in spec
