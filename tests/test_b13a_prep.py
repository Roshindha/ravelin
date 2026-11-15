"""B13a: pilot preparation. Manifest checker, `--extend-to`, speed log and validation provenance (tiny CPU runs, TUNE sets only, ledger unchanged)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from breachlens import evaluator as EV
from breachlens import manifest as MF
from breachlens import pool as P
from rl import train_ppo_v2 as T

ROOT = Path(__file__).resolve().parents[1]
TINY = ["--n-envs", "2", "--n-steps", "16", "--batch-size", "16", "--n-epochs", "2", "--val-freq", "320", "--checkpoint-freq", "160", "--phase-a-steps", "200"]


@pytest.fixture(autouse=True)
def _models_dir(tmp_path, monkeypatch):
    before = P.LEDGER_PATH.read_bytes()
    monkeypatch.setattr(T, "MODELS_DIR", tmp_path)
    yield
    assert P.LEDGER_PATH.read_bytes() == before


def _rows(path):
    with Path(path).open(newline="") as fh:
        return [{k: v for k, v in r.items() if k != "eval_seconds"} for r in csv.DictReader(fh)]


def _good_manifest():
    tr = [r["id"] for r in P.design_records("TUNE-train", "train")]
    va = [r["id"] for r in P.design_records("TUNE-val", "validation")]
    return MF.build_manifest("t", "TUNE-train", "TUNE-val", tr, va)


# ------------------------------------------------------------------------------------------------ manifest checker
def test_good_manifests_pass_for_tune_and_dev():
    MF.check_manifest(_good_manifest())
    MF.check_manifest(MF.build_manifest("d", "DEV-train", "DEV-validation", [r["id"] for r in P.dev_records("train")], [r["id"] for r in P.dev_records("validation")]))


def _final_id():
    return next(r["id"] for r in json.loads(P.POOL_PATH.read_text())["records"] if r["split"] == "FINAL-familiar")


def _transfer_id():
    return next(r["id"] for r in json.loads(P.POOL_PATH.read_text())["records"] if r["split"] == "FINAL-transfer")


def _lab_id():
    return P.design_records("LAB-DESIGN", "design")[0]["id"]


@pytest.mark.parametrize("make_bad", [
    lambda m: m["val_ids"].append(_final_id()),
    lambda m: m["train_ids"].append(_final_id()),
    lambda m: m["val_ids"].append(_transfer_id()),
    lambda m: m["train_ids"].append(_lab_id()),
    lambda m: m["val_ids"].append(_lab_id()),
])
def test_manifest_checker_fails_on_final_transfer_and_lab_design_ids(make_bad):
    m = _good_manifest()
    make_bad(m)
    with pytest.raises(MF.ManifestError):
        MF.check_manifest(m)
    with pytest.raises(MF.ManifestError):
        MF.check_manifest(m, strict_sets=False)                # also without the exact-set comparison: the id / stream rules alone catch it


def test_manifest_checker_fails_on_wrong_stream_overlap_duplicates_and_set_mismatch():
    dev_id = P.dev_records("train")[0]["id"]
    for mut in (lambda m: m["train_ids"].append(dev_id),                                   # stream F id in a TUNE manifest
                lambda m: m["val_ids"].append(m["train_ids"][0]),                         # overlap
                lambda m: m["train_ids"].pop(),                                           # not exactly the declared set
                lambda m: m.update(val_set="DEV-validation"),                             # mismatched pair
                lambda m: m.update(train_ids=m["train_ids"] + [m["train_ids"][0]])):      # duplicate
        m = _good_manifest()
        mut(m)
        with pytest.raises(MF.ManifestError):
            MF.check_manifest(m)


def test_trainer_aborts_before_training_if_a_final_id_reaches_the_validation_set(monkeypatch):
    real = EV.load_set
    fin = {"id": _final_id(), "family": "spine_pods", "seed": 1}

    def bad(set_name, role, **k):
        return real(set_name, role, **k) + [(fin, None)]
    monkeypatch.setattr(T.EV, "load_set", bad)
    with pytest.raises(MF.ManifestError):
        T.main(["--run-name", "bad", "--timesteps", "320", *TINY])
    assert not (T.MODELS_DIR / "bad" / "val_eval.csv").exists()


# ------------------------------------------------------------------------------------------------ outputs of a tiny run
def test_tiny_run_writes_manifest_speed_log_and_provenance(tmp_path):
    run = T.main(["--run-name", "w", "--timesteps", "640", "--speed-freq", "160", *TINY])
    m = MF.check_manifest_file(run / "manifest.json")
    assert len(m["train_ids"]) == 180 and len(m["val_ids"]) == 60 and m["train_set"] == "TUNE-train"
    sp = _rows(run / "speed.csv")
    assert [int(r["timestep"]) for r in sp] == [160 * k for k in range(1, 5) if 160 * k % 32 == 0 and 160 * k <= 640][:len(sp)] and len(sp) >= 3
    assert all(float(r["steps_per_s_window"]) > 0 for r in sp)
    val = _rows(run / "val_eval.csv")
    for r in val:
        for k in ("score_version", "sim_version", "generator_hash", "pool_sha256", "git_commit", "git_dirty", "seed", "role", "set"):
            assert r[k] not in ("", None), k
        assert r["role"] == "validation" and r["set"] == "TUNE-val"


# ------------------------------------------------------------------------------------------------ extension
def test_extend_to_equals_an_uninterrupted_longer_run_bit_for_bit(tmp_path):
    T.main(["--run-name", "full", "--timesteps", "960", *TINY])
    short = T.main(["--run-name", "ext", "--timesteps", "640", *TINY])
    id0 = json.loads((short / "model_identity.json").read_text())
    T.main(["--run-name", "ext", "--extend-to", "960"])
    a, b = json.loads((T.MODELS_DIR / "full" / "model_identity.json").read_text()), json.loads((T.MODELS_DIR / "ext" / "model_identity.json").read_text())
    assert a["params_sha256"] == b["params_sha256"] and a["params_optimizer_sha256"] == b["params_optimizer_sha256"]
    assert _rows(T.MODELS_DIR / "full" / "val_eval.csv") == _rows(T.MODELS_DIR / "ext" / "val_eval.csv")
    assert (T.MODELS_DIR / "ext" / "final_640.zip").exists() and b["timesteps_trained"] == 960 and b["extensions"][0]["to_timesteps"] == 960
    assert (T.MODELS_DIR / "ext" / "curriculum.log").read_text() == (T.MODELS_DIR / "full" / "curriculum.log").read_text()    # boundary did not move
    assert id0["config"]["phase_a_steps"] == 200


def test_extend_to_refuses_without_explicit_phase_a_or_unfinished_run_or_shorter_target():
    with pytest.raises(SystemExit):
        T.main(["--run-name", "nothing", "--extend-to", "960"])
    T.main(["--run-name", "np", "--timesteps", "640", "--n-envs", "2", "--n-steps", "16", "--batch-size", "16", "--n-epochs", "2", "--val-freq", "320",
            "--checkpoint-freq", "160"])
    with pytest.raises(SystemExit):
        T.main(["--run-name", "np", "--extend-to", "960"])                    # phase A was left at its default (steps // 3): the boundary would move
    T.main(["--run-name", "ok", "--timesteps", "640", *TINY])
    with pytest.raises(SystemExit):
        T.main(["--run-name", "ok", "--extend-to", "320"])
