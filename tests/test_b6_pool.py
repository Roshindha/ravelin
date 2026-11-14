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
    """Throwaway ledger for REFUSAL tests and synthetic-record tests only: no test loads FINAL layout content (that goes to the real ledger)."""
    p = tmp_path / "final_access.log"
    monkeypatch.setattr(P, "LEDGER_PATH", p)
    return p


# ------------------------------------------------------------------ frozen generators / stats
GENERATOR_HASH_LITERAL = "49b8a18c53228c6ec9271f0082bbf6d6ec0d6fdbc3c5755721bfa0f36ec42095"


def test_generator_hash_equals_the_literal_frozen_value():
    from breachlens.gates import generator_hash
    assert generator_hash() == GENERATOR_HASH_LITERAL
    assert json.loads((ROOT / "data/v2/generator_hash.json").read_text())["generator_hash"] == GENERATOR_HASH_LITERAL
    assert DOC["generator_hash"] == GENERATOR_HASH_LITERAL


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
def test_every_dev_layout_is_still_accepted():
    """DEV only. The FINAL re-check is the manual script scripts_b6_verify_final.py (it writes to the real ledger; not part of the suite)."""
    for r in RECS:
        if r["pool"] == "DEV":
            L = Y.generate(r["family"], r["seed"])
            assert Y.check(L, r["family"]) == []
            assert stat_check(L, r["seed"])["reasons"] == []


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
    entries = P.read_ledger()
    assert {e["purpose"] for e in entries} <= {"build"}
    assert sum(e["purpose"] == "eval" for e in entries) == 0
    assert sum(e["purpose"] == "build" for e in entries) == 5
    assert all({"time", "commit", "entry_point", "purpose"} <= set(e) for e in entries)
    assert {e["family"] for e in entries if e.get("action") == "walk"} == set(Y.FAMILIES)


def test_ledger_path_cannot_be_redirected_in_non_test_code(monkeypatch, tmp_path):
    import inspect
    assert not hasattr(P, "ledger_path")
    assert "path" not in inspect.signature(P.read_ledger).parameters
    assert "path" not in inspect.signature(P.log_final_access).parameters
    monkeypatch.setenv("BREACHLENS_FINAL_LEDGER", str(tmp_path / "x.log"))          # an env var must have no effect
    before = P.LEDGER_PATH.read_bytes()
    assert P.LEDGER_PATH == ROOT / "data" / "v2" / "final_access.log"
    assert "BREACHLENS_FINAL_LEDGER" not in "".join(p.read_text() for p in _non_test_py())
    assert P.LEDGER_PATH.read_bytes() == before


def _non_test_py():
    return [p for p in ROOT.rglob("*.py") if "tests" not in p.relative_to(ROOT).parts and ".venv" not in p.parts and "venv" not in p.parts
            and "site-packages" not in p.parts]


def _call_names(path):
    import ast
    names = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Call):
            f = n.func
            names.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return names


def test_no_code_outside_tests_loads_final_layouts_without_the_ledger():
    import ast
    allowed_generators = {"pool.py", "layouts.py", "gates.py", "acceptance.py", "scripts_b5r1_sweep.py", "scripts_b5r1b_sweep.py", "scripts_b5r3_sweep.py", "scripts_b8_checks.py"}
    pool_users = {"pool.py", "env.py", "scripts_b6_build_pool.py", "scripts_b6_verify_final.py",
                 "evaluator.py", "scripts_b8_checks.py"}   # env.py: BreachEnv, role-guarded (tests/test_b7_env.py); evaluator.py: role-guarded loaders; scripts_b8_checks.py: DEV walks only (tests/test_b8_baselines.py)
    for p in _non_test_py():
        src = p.read_text()
        calls = _call_names(p)
        if calls & {"generate", "candidate"}:
            assert p.name in allowed_generators, f"{p} generates layouts but is not on the audited list"
        if p.name not in pool_users:
            assert not (calls & {"load_layout", "load_pool_records", "build_pool", "walk", "verify_final_acceptance", "final_range"}), p
            assert "pool_v1" not in src and "POOL_PATH" not in src and "FINAL_OFFSET" not in src, p
            if p.name != "pool.py":
                assert 'stream_range("F"' not in src and "stream_range('F'" not in src, p      # only pool.py walks stream F
    # inside pool.py: every function that generates layouts logs first (a walk is preceded by a ledger entry in build_pool)
    tree = ast.parse((ROOT / "breachlens" / "pool.py").read_text())
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("load_layout", "verify_final_acceptance")]:
        calls = [(c.lineno, c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", "")) for c in ast.walk(fn) if isinstance(c, ast.Call)]
        log = min(l for l, n in calls if n == "log_final_access")
        gen = min(l for l, n in calls if n == "generate")
        assert log < gen, fn.name
    bp = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_pool")
    walks = sorted(c.lineno for c in ast.walk(bp) if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "walk")
    logs = sorted(c.lineno for c in ast.walk(bp) if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "log_final_access")
    assert len(walks) == 3 and len(logs) == 2          # DEV walk (no ledger), FINAL-familiar loop walk, labyrinth walk
    assert logs[0] < walks[1] < logs[1] < walks[2]     # each FINAL walk is preceded by its ledger entry


def test_final_load_writes_the_ledger_before_any_layout_content_exists(monkeypatch):
    """Runtime: the ledger call precedes generation. The ledger call is stubbed to raise, so no FINAL layout is ever generated by this test."""
    class Logged(Exception):
        pass

    def stub(*a, **k):
        raise Logged

    def no_gen(*a, **k):
        raise AssertionError("layout content generated before the ledger entry")
    monkeypatch.setattr(P, "log_final_access", stub)
    monkeypatch.setattr(Y, "generate", no_gen)
    monkeypatch.setattr(Y, "candidate", no_gen)
    with pytest.raises(Logged):
        P.load_layout(FINAL_REC, "build", purpose="build")
    with pytest.raises(Logged):
        P.verify_final_acceptance()
    with pytest.raises(Logged):
        P.load_pool_records("FINAL-familiar", "build", purpose="build")


def test_log_writes_one_line_to_the_module_ledger_path(tmp_ledger):
    e = P.log_final_access("build", "tests", {"id": "synthetic-0"})      # synthetic id: no layout content involved
    assert P.read_ledger() == [e] and tmp_ledger.read_text().count("\n") == 1


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
    assert not tmp_ledger.exists()                                            # refusals leave no entry; the allowed path is covered with a stub below


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


# ------------------------------------------------------------------ B7.1: LAB-DESIGN guard and the "design" role
LAB_RECS = P.design_records("LAB-DESIGN", "design")
DESIGN_BY_SPLIT = {s: P.design_records(s, "design") for s in ("TUNE-train", "TUNE-val", "LAB-DESIGN")}


def test_design_role_loads_tune_and_lab_design_and_nothing_from_the_pool(tmp_ledger):
    assert [len(DESIGN_BY_SPLIT[s]) for s in ("TUNE-train", "TUNE-val", "LAB-DESIGN")] == [180, 60, 40]
    for recs in DESIGN_BY_SPLIT.values():
        for r in recs[:3]:
            P.load_layout(r, "design")
    for rec in RECS:                                            # every pool id: DEV-train, DEV-validation, FINAL-familiar, FINAL-transfer
        with pytest.raises(PermissionError):
            P.load_layout(rec, "design")
        with pytest.raises(PermissionError):
            P.load_layout(rec, "design", purpose="build")
        with pytest.raises(PermissionError):
            P.assert_dev_only([rec["id"]], "design")
    for split in P.SPLITS:
        with pytest.raises(PermissionError):
            P.load_pool_records(split, "design", purpose="build")
    with pytest.raises(PermissionError):
        P.dev_records("design")
    assert not tmp_ledger.exists()


@pytest.mark.parametrize("role", ["train", "validation", "tuning"])
def test_dev_side_roles_raise_on_every_lab_design_id(role, tmp_ledger):
    assert len(LAB_RECS) == 40
    for rec in LAB_RECS:
        with pytest.raises(PermissionError):
            P.load_layout(rec, role)
        with pytest.raises(PermissionError):
            P.load_layout(dict(rec, split="TUNE-train"), role)         # a relabelled record is still refused (stream L)
        with pytest.raises(PermissionError):
            P.assert_dev_only([rec["id"]], role)
    with pytest.raises(PermissionError):
        P.design_records("LAB-DESIGN", role)
    with pytest.raises(PermissionError):
        P.assert_dev_only([LAB_RECS[0]["id"], RECS[0]["id"]], role)
    assert not tmp_ledger.exists()


def test_design_set_role_matrix():
    allowed = {"train": {"TUNE-train"}, "validation": {"TUNE-val"}, "tuning": {"TUNE-train", "TUNE-val"},
               "design": {"TUNE-train", "TUNE-val", "LAB-DESIGN"}}
    for role, ok in allowed.items():
        for split in ("TUNE-train", "TUNE-val", "LAB-DESIGN"):
            if split in ok:
                assert {r["split"] for r in P.design_records(split, role)} == {split}
                P.load_layout(P.design_records(split, role)[0], role)
            else:
                with pytest.raises(PermissionError):
                    P.design_records(split, role)
                with pytest.raises(PermissionError):
                    P.load_layout(DESIGN_BY_SPLIT[split][0], role)
    for role in ("build", "eval"):
        for split in ("TUNE-train", "TUNE-val", "LAB-DESIGN"):
            with pytest.raises(PermissionError):
                P.design_records(split, role)


def test_gate_and_balance_code_does_not_use_pool_roles_and_lab_design_stays_out_of_rl():
    # Gate / design scripts read design sets through breachlens.layouts (never through pool.py), so no pool role applies to them.
    for p in [ROOT / "breachlens" / n for n in ("gates.py", "acceptance.py", "oracle.py")] + list(ROOT.glob("scripts_b[45]*.py")) + [ROOT / "scripts_gates.py"]:
        if p.exists():
            assert "breachlens.pool" not in p.read_text() and "import pool" not in p.read_text(), p
    for p in (ROOT / "rl").glob("*.py"):
        assert "LAB-DESIGN" not in p.read_text(), p


# ------------------------------------------------------------------ B9 housekeeping: eval sessions, git_dirty
def test_count_eval_sessions_synthetic(tmp_ledger):
    P.log_final_access("build", "t")
    assert P.count_eval_sessions() == 0
    with P.eval_session("s1"):
        P.log_final_access("eval", "t", {"split": "FINAL-familiar"})
        P.log_final_access("eval", "t", {"split": "FINAL-transfer"})
        with P.eval_session("ignored"):                    # re-entrant: same session
            P.log_final_access("eval", "t", {"action": "evaluate"})
    with P.eval_session("s2"):
        P.log_final_access("eval", "t")
    P.log_final_access("eval", "t")                        # outside a session: its own session
    es = P.read_ledger()
    assert [e.get("session_id") for e in es if e["purpose"] == "eval"][:3] == ["s1"] * 3
    assert P.count_eval_sessions() == 3 and P.count_eval_sessions(es[:1]) == 0
    assert "session_id" not in es[0]


def test_real_ledger_still_5_build_0_eval():
    es = P.read_ledger()
    assert sum(e["purpose"] == "build" for e in es) == 5 and sum(e["purpose"] == "eval" for e in es) == 0 and P.count_eval_sessions() == 0


def test_git_dirty_ignores_gitignored_untracked_files(tmp_path, monkeypatch):
    import subprocess
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    run("init", "-q"); run("config", "user.email", "a@b"); run("config", "user.name", "t")
    (tmp_path / ".gitignore").write_text("logs_*/\n")
    run("add", "."); run("commit", "-qm", "x")
    monkeypatch.setattr(P, "ROOT", tmp_path)
    assert P.git_dirty() is False
    (tmp_path / "logs_x").mkdir(); (tmp_path / "logs_x" / "a.log").write_text("1")
    assert P.git_dirty() is False                           # ignored untracked file
    (tmp_path / "new.py").write_text("1")
    assert P.git_dirty() is True                            # non-ignored untracked file


def test_g7_bar_recomputes_from_b8_layout_rows():
    import csv, json
    rows = [r for r in csv.DictReader(open(ROOT / "data/v2/baselines/b8_layout_rows.csv")) if r["set"] == "TUNE-val" and r["policy"] in ("BestSingle", "Greedy")]
    by = {}
    for r in rows:
        by.setdefault(r["id"], {})[r["policy"]] = float(r["score"])
    assert len(by) == 60 and all(len(v) == 2 for v in by.values())
    base = sum(max(v.values()) for v in by.values()) / 60
    rec = json.loads((ROOT / "data/v2/g7_bar.json").read_text())
    assert abs(base + 0.05 - rec["g7_bar"]) < 1e-12 and round(rec["g7_bar"], 4) == 0.4369
