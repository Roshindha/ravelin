"""B14a item 1: every trainer's validation CSV carries the same provenance columns, through one shared writer. TUNE sets only; ledger unchanged."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from breachlens import pool as P
from rl import train_ddqn_v2 as TD
from rl import train_ppo_v2 as TP
from rl import valrows as VR

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["score_version", "sim_version", "generator_hash", "pool_sha256", "git_commit", "git_dirty", "seed", "role", "set"]
PPO_TINY = ["--timesteps", "640", "--n-envs", "2", "--n-steps", "16", "--batch-size", "16", "--n-epochs", "2", "--val-freq", "320", "--checkpoint-freq", "160"]
DDQN_TINY = ["--timesteps", "800", "--n-envs", "4", "--buffer-size", "1000", "--batch-size", "16", "--learning-starts", "200",
             "--target-update-interval", "50", "--val-freq", "400", "--checkpoint-freq", "400"]


@pytest.fixture(autouse=True)
def _models_dir(tmp_path, monkeypatch):
    before = P.LEDGER_PATH.read_bytes()
    monkeypatch.setattr(TP, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(TD, "MODELS_DIR", tmp_path)
    yield
    assert P.LEDGER_PATH.read_bytes() == before


def test_required_fields_are_part_of_the_shared_schema():
    assert VR.PROVENANCE_FIELDS == REQUIRED
    assert all(f in VR.FIELDS for f in REQUIRED)
    assert TP.ValidationCallback.FIELDS is VR.FIELDS and TD.Validator.FIELDS is VR.FIELDS      # one schema, no per-trainer copy


def test_trainers_write_validation_only_through_the_shared_writer():
    for f in ("train_ppo_v2.py", "train_ddqn_v2.py"):
        src = (ROOT / "rl" / f).read_text()
        assert "VR.write_validation(" in src
        assert "DictWriter(fh, fieldnames=self.FIELDS" not in src                                # no private val_eval.csv writer


@pytest.mark.parametrize("algo", ["ppo", "ddqn"])
def test_validation_csv_of_every_trainer_has_every_provenance_field(algo, tmp_path):
    if algo == "ppo":
        run = TP.main(["--run-name", "prov_ppo", *PPO_TINY])
    else:
        run = TD.main(["--run-name", "prov_ddqn", *DDQN_TINY])
    path = run / "val_eval.csv"
    assert VR.missing_fields(path) == []
    ident = json.loads((run / "model_identity.json").read_text())
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 2 * 4                                                                    # >= 2 evals x (ALL + 3 families)
    for r in rows:
        for f in REQUIRED:
            assert r[f] not in ("", "unknown"), f
        assert r["git_commit"] == ident["git_commit"] and r["pool_sha256"] == ident["pool_sha256"] and r["generator_hash"] == ident["generator_hash"]
        assert r["role"] == "validation" and r["set"] == "TUNE-val" and r["score_version"] == ident["score_version"]


def test_missing_fields_detects_a_csv_without_provenance(tmp_path):
    p = tmp_path / "val_eval.csv"
    cols = VR.METRIC_FIELDS                                                                      # the old DDQN header
    p.write_text(",".join(cols) + "\n" + ",".join("0" for _ in cols) + "\n")
    assert set(VR.missing_fields(p)) == set(REQUIRED)
    p.write_text(",".join(VR.FIELDS) + "\n" + ",".join("" if f == "git_commit" else "x" for f in VR.FIELDS) + "\n")
    assert VR.missing_fields(p) == ["git_commit (empty)"]


def test_writer_refuses_a_summary_row_without_provenance(tmp_path):
    row = {k: 0 for k in VR.METRIC_FIELDS} | {"group": "ALL"}
    res = {"summary": [row], "rows": [], "episodes": {}}
    with pytest.raises(ValueError, match="lacks provenance"):
        VR.write_validation(tmp_path / "v.csv", tmp_path / "v.jsonl", 0, res, 1.0)
    assert not (tmp_path / "v.csv").exists()
