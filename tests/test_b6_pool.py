"""B6: the frozen 300-layout pool, the FINAL-access ledger, role guards, cross-set D4 disjointness."""
import hashlib
import json
import os
import random
import stat
from collections import Counter
from pathlib import Path

import pytest

from breachlens import layouts as Y
from breachlens import pool as P
from breachlens.acceptance import stat_check
from breachlens.config import sim_version

ROOT = Path(__file__).resolve().parents[1]
DOC = json.loads(P.POOL_PATH.read_text())
RECS = DOC["records"]
DESIGN = Y.load_design_sets_meta()
DESIGN_RECS = [r for v in DESIGN["sets"].values() for r in v]


@pytest.fixture
def tmp_ledger(tmp_path, monkeypatch):
    """Tests that load FINAL layouts write to a throwaway ledger, never to data/v2/final_access.log."""
    p = tmp_path / "final_access.log"
    monkeypatch.setenv("BREACHLENS_FINAL_LEDGER", str(p))
    return p


# ------------------------------------------------------------------ frozen generators / stats
def test_generators_and_stats_match_the_b5_record():
    rec = json.loads((ROOT / "data/v2/generator_hash.json").read_text())
    assert P.check_frozen()["generator_hash"] == rec["generator_hash"] == DOC["generator_hash"]
    assert DOC["stats_version"] == sim_version() == DESIGN["sim_version"]
    assert DOC["generator_version"] == Y.GENERATOR_VERSION
    assert not Y.gen_override_active() and not os.environ.get("BREACHLENS_STATS_PATH")


def test_check_frozen_fails_when_the_hash_differs(tmp_path, monkeypatch):
    bad = tmp_path / "gh.json"
    bad.write_text(json.dumps({"generator_hash": "0" * 64, "commit": "x"}))
    monkeypatch.setattr(P, "GENERATOR_HASH_PATH", bad)
    with pytest.raises(RuntimeError, match="generator hash"):
        P.check_frozen()


# ------------------------------------------------------------------ counts, ids, ranges
def test_counts_per_group_and_family_match_spec_5_2():
    c = Counter((r["family"], r["split"]) for r in RECS)
    for fam in Y.FAMILIAR:
        assert (c[(fam, "DEV-train")], c[(fam, "DEV-validation")], c[(fam, "FINAL-familiar")]) == (50, 10, 20)
        assert c[(fam, "FINAL-transfer")] == 0
    assert c[("labyrinth", "FINAL-transfer")] == 60 and sum(v for (f, s), v in c.items() if f == "labyrinth") == 60
    assert Counter(r["split"] for r in RECS) == {"DEV-train": 150, "DEV-validation": 30, "FINAL-familiar": 60, "FINAL-transfer": 60}
    assert len(RECS) == 300 and DOC["split_sizes"] == {"DEV-train": 150, "DEV-validation": 30, "FINAL-familiar": 60, "FINAL-transfer": 60}


def test_ids_seeds_hashes_unique_and_in_the_reserved_stream():
    assert len({r["id"] for r in RECS}) == len({r["seed"] for r in RECS}) == len({r["content_hash"] for r in RECS}) == 300
    for r in RECS:
        assert r["id"] == f"{r['family']}-{r['seed']}"
        assert Y.stream_of(r["seed"]) == "F"                       # never T / L
        lo, hi = Y.stream_range("F", r["family"])
        assert lo <= r["seed"] < hi
        if r["split"].startswith("DEV"):
            assert r["family"] != "labyrinth" and P.dev_range(r["family"])[0] <= r["seed"] < P.dev_range(r["family"])[1] and not P.is_final(r["family"], r["seed"])
        else:
            assert P.final_range(r["family"])[0] <= r["seed"] < P.final_range(r["family"])[1] and P.is_final(r["family"], r["seed"])
        assert r["pool"] == ("DEV" if r["split"].startswith("DEV") else "FINAL")
    for fam in Y.FAMILIAR:                                         # within a family the walk order assigns train, then validation (DEV), by seed
        d = sorted((r for r in RECS if r["family"] == fam and r["pool"] == "DEV"), key=lambda r: r["seed"])
        assert [r["split"] for r in d] == ["DEV-train"] * 50 + ["DEV-validation"] * 10


# ------------------------------------------------------------------ D4 duplicates across ALL sets
def test_no_d4_overlap_between_pool_and_every_design_set_and_inside_the_pool():
    pool_h = [r["content_hash"] for r in RECS]
    all_h = pool_h + [r["content_hash"] for r in DESIGN_RECS]
    assert len(all_h) == len(set(all_h)) == 300 + 280
    assert not ({r["seed"] for r in RECS} & {r["seed"] for r in DESIGN_RECS})
    assert not ({r["id"] for r in RECS} & {r["id"] for r in DESIGN_RECS})
    lab_design = {r["content_hash"] for r in DESIGN["sets"]["LAB-DESIGN"]}
    assert not (lab_design & {r["content_hash"] for r in RECS if r["split"] == "FINAL-transfer"})
    assert DOC["design_set_hashes_checked"] == 280


def test_stored_hashes_regenerate_for_dev_and_design_sets_and_pairwise_d4_images_differ():
    dev = [r for r in RECS if r["pool"] == "DEV"]
    seen = {}
    for r in dev + DESIGN_RECS:
        L = Y.generate(r["family"], r["seed"])
        assert Y.content_hash(L) == r["content_hash"]
        for k in range(8):                                         # every D4 image of every layout hashes to the same canonical value
            assert Y.content_hash(Y.transform(L, k)) == r["content_hash"]
        assert r["content_hash"] not in seen
        seen[r["content_hash"]] = r["id"]


def test_walk_redraws_on_a_d4_collision_and_counts_it():
    fam = "spine_pods"
    base, rep0 = P.walk("DEV", fam, 2, set())
    assert rep0["redraws_d4_duplicate"] == 0 and [r["seed"] for r in base] == [r["seed"] for r in RECS if r["family"] == fam and r["pool"] == "DEV"][:2]
    taken = {base[0]["content_hash"]}                             # the hash is D4-canonical: a mirrored / rotated copy collides as well
    again, rep = P.walk("DEV", fam, 1, taken)
    assert rep["redraws_d4_duplicate"] == 1 and again[0]["seed"] == base[1]["seed"]
    assert ["d4_duplicate"] in [b[2] for b in rep["bits"]]
    assert DOC["walk"]["DEV/spine_pods"]["redraws_d4_duplicate"] == 0                 # recorded redraw counts of the real build
    assert all("redraws_d4_duplicate" in w for w in DOC["walk"].values())


# ------------------------------------------------------------------ acceptance under the current simulator
def test_every_accepted_layout_is_still_accepted_and_final_stores_only_bits(tmp_ledger):
    for r in RECS:
        rec = r
        L = P.load_layout(rec, "build", purpose="build", entry_point="tests/test_b6_pool.py") if r["pool"] == "FINAL" else Y.generate(r["family"], r["seed"])
        assert Y.check(L, r["family"]) == []
        assert stat_check(L, r["seed"])["reasons"] == []           # result is used as a bit only; nothing is printed or stored
    assert all(json.loads(x)["purpose"] == "build" for x in tmp_ledger.read_text().splitlines())
    assert len(tmp_ledger.read_text().splitlines()) == 120       # one entry per FINAL load


def test_final_candidate_log_holds_only_bits_and_rule_ids():
    for k, w in DOC["walk"].items():
        assert set(w) == {"candidates", "accepted", "structural", "structural_rejected", "redraws_d4_duplicate", "stat_evaluated", "single_type_trivial",
                          "too_hard", "stat_rejected", "seeds_consumed", "seed_lo", "candidate_bits"}
        assert w["candidates"] == len(w["candidate_bits"]) and w["accepted"] == sum(b[1] for b in w["candidate_bits"])
        for seed, bit, rules in w["candidate_bits"]:
            assert bit in (0, 1) and (bit == 1) == (rules == [])
        if k.startswith("FINAL"):
            fam = k.split("/")[1]
            assert {rec["seed"] for rec in RECS if rec["family"] == fam and rec["pool"] == "FINAL"} == {b[0] for b in w["candidate_bits"] if b[1]}
    for r in RECS:                                                  # records carry no score-like field
        assert set(r) == {"id", "family", "seed", "content_hash", "split", "pool"}
    text = json.dumps({k: v for k, v in DOC.items() if k != "note"})
    assert not any(k in text for k in ("score", "oracle", "baseline", "best_single_mean"))
    src = (ROOT / "scripts_b6_build_pool.py").read_text().split('"""')[2]          # code after the docstring
    assert "score" not in src and "oracle" not in src.lower() and "stat_check" not in src   # the build script never touches a score


# ------------------------------------------------------------------ ledger
def test_ledger_has_build_entries_only_and_zero_eval():
    entries = P.read_ledger(P.LEDGER_PATH)
    assert len(entries) >= 4
    assert all(e["purpose"] == "build" for e in entries) and sum(e["purpose"] == "eval" for e in entries) == 0
    assert all({"time", "commit", "entry_point", "purpose"} <= set(e) for e in entries)
    assert {e["family"] for e in entries if e.get("action") == "walk"} == set(Y.FAMILIES)


def test_log_rejects_unknown_purpose(tmp_ledger):
    with pytest.raises(ValueError):
        P.log_final_access("peek", "x")
    assert not tmp_ledger.exists()


# ------------------------------------------------------------------ role guards
FINAL_REC = next(r for r in RECS if r["split"] == "FINAL-familiar")
TRANSFER_REC = next(r for r in RECS if r["split"] == "FINAL-transfer")


@pytest.mark.parametrize("role", ["train", "validation", "tuning"])
def test_training_validation_and_tuning_loaders_raise_on_final_ids(role, tmp_ledger):
    for rec in (FINAL_REC, TRANSFER_REC):
        with pytest.raises(PermissionError):
            P.load_layout(rec, role)
        with pytest.raises(PermissionError):
            P.load_layout(rec, role, purpose="build")
        with pytest.raises(PermissionError):
            P.assert_dev_only([rec["id"]], role)
    for split in P.FINAL_SPLITS:
        with pytest.raises(PermissionError):
            P.load_pool_records(split, role, purpose="build")
    assert not tmp_ledger.exists()                                 # a refused load leaves no ledger entry and no layout


def test_train_loader_raises_on_a_final_id_inside_a_mixed_list():
    dev_id = next(r["id"] for r in RECS if r["split"] == "DEV-train")
    with pytest.raises(PermissionError):
        P.assert_dev_only([dev_id, FINAL_REC["id"]], "train")
    P.assert_dev_only([dev_id], "train")


def test_role_split_matrix(tmp_ledger):
    assert {r["split"] for r in P.dev_records("train")} == {"DEV-train"}
    assert {r["split"] for r in P.dev_records("validation")} == {"DEV-validation"}
    assert {r["split"] for r in P.dev_records("tuning")} == {"DEV-train", "DEV-validation"}
    with pytest.raises(PermissionError):
        P.load_pool_records("DEV-validation", "train")
    with pytest.raises(PermissionError):
        P.load_pool_records("DEV-train", "validation")
    with pytest.raises(PermissionError):
        P.dev_records("eval")
    with pytest.raises(PermissionError):
        P.dev_records("build")


def test_eval_needs_unlock_and_matching_purpose_and_build_cannot_eval(tmp_ledger):
    with pytest.raises(PermissionError):
        P.load_layout(FINAL_REC, "eval", purpose="eval")                       # no unlock
    with pytest.raises(PermissionError):
        P.load_layout(FINAL_REC, "build", purpose="eval", unlock=True)         # wrong role
    with pytest.raises(PermissionError):
        P.load_layout(FINAL_REC, "eval", purpose="build", unlock=True)
    with pytest.raises(PermissionError):
        P.load_layout(FINAL_REC, "build")                                      # no purpose
    assert not tmp_ledger.exists()
    L = P.load_layout(FINAL_REC, "build", purpose="build", entry_point="t")
    assert Y.check(L, FINAL_REC["family"]) == []
    assert [e["purpose"] for e in P.read_ledger(tmp_ledger)] == ["build"]


def test_dev_sampler_never_yields_final_or_labyrinth():
    rng = random.Random(0)
    for role in ("train", "validation", "tuning"):
        pool = P.dev_records(role)
        picks = [rng.choice(pool) for _ in range(2000)]
        assert not any(P.is_final_id(r["id"]) or r["family"] == "labyrinth" or r["pool"] == "FINAL" for r in picks)
    assert all(P.is_final_id(r["id"]) for r in RECS if r["pool"] == "FINAL")
    assert not any(P.is_final_id(r["id"]) for r in RECS if r["pool"] == "DEV")


def test_training_and_tuning_modules_do_not_import_the_final_loaders():
    for p in list((ROOT / "rl").glob("*.py")) + [ROOT / "breachlens" / n for n in ("gates.py", "acceptance.py", "oracle.py", "armies.py")]:
        src = p.read_text()
        assert "load_pool_records" not in src and "load_layout(" not in src.replace("load_layout_pool(", "") and "breachlens.pool" not in src and "import pool" not in src, p


# ------------------------------------------------------------------ freeze
def test_pool_file_is_read_only_and_sha256_matches():
    mode = P.POOL_PATH.stat().st_mode
    assert not mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    assert not P.POOL_SHA_PATH.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    sha = hashlib.sha256(P.POOL_PATH.read_bytes()).hexdigest()
    assert P.POOL_SHA_PATH.read_text().split()[0] == sha == P.pool_sha256()
    assert json.loads(P.POOL_PATH.read_text()) == DOC


def test_id_and_split_lists_rederive_from_the_walk_log():
    """Re-derive ids and splits from the stored candidate log (accepted seeds in walk order) and the split sizes."""
    for fam in Y.FAMILIAR:
        dev = [b[0] for b in DOC["walk"][f"DEV/{fam}"]["candidate_bits"] if b[1]]
        want = [(f"{fam}-{s}", "DEV-train" if i < 50 else "DEV-validation") for i, s in enumerate(dev)]
        assert [(r["id"], r["split"]) for r in sorted((r for r in RECS if r["family"] == fam and r["pool"] == "DEV"), key=lambda r: r["seed"])] == want
        fin = [f"{fam}-{b[0]}" for b in DOC["walk"][f"FINAL/{fam}"]["candidate_bits"] if b[1]]
        assert sorted(r["id"] for r in RECS if r["split"] == "FINAL-familiar" and r["family"] == fam) == sorted(fin) and len(fin) == 20
    lab = [f"labyrinth-{b[0]}" for b in DOC["walk"]["FINAL/labyrinth"]["candidate_bits"] if b[1]]
    assert sorted(r["id"] for r in RECS if r["split"] == "FINAL-transfer") == sorted(lab) and len(lab) == 60


def test_build_refuses_to_overwrite_a_frozen_pool():
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(ROOT / "scripts_b6_build_pool.py")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "frozen" in (r.stdout + r.stderr)
