"""B15a tests: the session runner (scripts_b15_final.py), the cell registry and the analysis script (scripts_b15_analysis.py).

No test loads a FINAL layout and none writes the real ledger (P.LEDGER_PATH is monkeypatched to a temp copy wherever a ledger could be written).
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

import scripts_b15_analysis as A
import scripts_b15_cells as C
import scripts_b15_final as F
from breachlens import evaluator as EV
from breachlens import pool as P

ROOT = Path(__file__).resolve().parents[1]
HAVE_MODELS = (F.MODELS / "grid_ppo_plain_s0" / "final.zip").exists() and F.AUDIT_JSON.exists()
needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="grid models not present locally")


@pytest.fixture
def temp_ledger(tmp_path, monkeypatch):
    p = tmp_path / "final_access.log"
    shutil.copy(P.LEDGER_PATH, p)
    monkeypatch.setattr(P, "LEDGER_PATH", p)
    return p


# ------------------------------------------------------------------------------------------------------------ registry
def test_registry_order_and_size():
    cells = C.all_cells()
    assert len(cells) == 116 and len({c["name"] for c in cells}) == 116
    kinds = [c["kind"] for c in cells]
    assert kinds == ["baseline"] * 6 + ["learner"] * 60 + ["control"] * 30 + ["planner"] * 20
    assert [c["name"] for c in cells[:6]] == ["random", "greedy", "bestsingle", "bestsingle_perlayout", "bestsingle_hindsight", "oracle"]
    assert [c["seed"] for c in cells[6:16]] == list(range(10)) and cells[6]["cond"] == "ppo_plain" and cells[36]["tta"] is True
    assert [c["cond"] for c in cells[66:96:10]] == ["ras", "rps", "rpsfull"]
    assert all(c["cond"] != "planner_ddqn" for c in cells) and not any(c["tta"] and c["kind"] != "learner" for c in cells if "tta" in c)
    assert len(C.all_conditions()) == 6 + 6 + 9 + 6


# ------------------------------------------------------------------------------------------------------------ exclusive create, manifest, crash rule
def test_exclusive_create_and_manifest(tmp_path):
    ctx = F.Ctx(mode="rehearse", out=tmp_path)
    rows = [{k: "" for k in C.RAW_COLUMNS} | {"cell": "x", "id": "a"}]
    e = F.write_cell(ctx, "x", rows, "c" * 40, "sid")
    assert e["rows"] == 1 and len(F.read_manifest(ctx)) == 1
    with pytest.raises(FileExistsError):
        F.write_cell(ctx, "x", rows, "c" * 40, "sid")                               # a second write of a cell fails
    assert len(F.read_manifest(ctx)) == 1                                           # and adds no manifest line
    F.verify_entry(ctx, e)


def test_plan_cells_keeps_verified_discards_partial_and_stops_on_corruption(tmp_path):
    cells = C.all_cells()[:3]
    ctx = F.Ctx(mode="rehearse", out=tmp_path, cells=cells)
    row = lambda n: [{k: "" for k in C.RAW_COLUMNS} | {"cell": n}]
    F.write_cell(ctx, cells[0]["name"], row(cells[0]["name"]), "c" * 40, "s")
    (ctx.raw / f"{cells[1]['name']}.csv").write_text("cell\nhalf")                  # half-written file, no manifest line
    kept, pending = F.plan_cells(ctx)
    assert [c["name"] for c in kept] == [cells[0]["name"]] and [c["name"] for c in pending] == [cells[1]["name"], cells[2]["name"]]
    assert not (ctx.raw / f"{cells[1]['name']}.csv").exists() and list((tmp_path / "discarded").glob("greedy.*"))
    assert "discarded and recomputed" in (tmp_path / "crash_log.md").read_text()
    (ctx.raw / f"{cells[0]['name']}.csv").write_text("cell\ntampered\n")            # a manifest line that no longer verifies: stop, never recompute
    with pytest.raises(F.CorruptCell):
        F.plan_cells(ctx)


# ------------------------------------------------------------------------------------------------------------ refusals
def test_real_mode_refuses_without_unlock(tmp_path, capsys):
    assert F.main([]) == 2                                                           # CLI: no --unlock-final
    assert F.main(["--rehearse", "--unlock-final"]) == 2
    with pytest.raises(F.Refusal):
        F.run_session(F.Ctx(mode="real", unlock=False, out=tmp_path))
    assert not (tmp_path / "session.json").exists()


def test_real_mode_refuses_with_eval_entries_or_session_json_or_dirty(tmp_path, temp_ledger):
    ctx = F.Ctx(mode="real", unlock=True, out=tmp_path)
    assert "5 build / 0 eval" in F.check_ledger(ctx)
    with open(temp_ledger, "a") as f:
        f.write(json.dumps({"purpose": "eval", "session_id": "x"}) + "\n")
    with pytest.raises(F.Refusal, match="eval entries"):
        F.check_ledger(ctx)
    with pytest.raises(F.Refusal, match="outside session"):
        F.check_ledger(ctx, restart=True, session_id="y")
    F.check_ledger(ctx, restart=True, session_id="x")
    (tmp_path / "session.json").write_text("{}")
    with pytest.raises(F.Refusal, match="session.json"):
        F.check_session_json(ctx)
    with pytest.raises(F.Refusal, match="dirty"):
        F.check_clean_tree(F.Ctx(mode="real", unlock=True, out=tmp_path, git_dirty=lambda: True))
    with pytest.raises(F.Refusal, match="exit code 3"):
        F.check_suite(F.Ctx(mode="real", unlock=True, out=tmp_path, suite_cmd=[sys.executable, "-c", "import sys; sys.exit(3)"]))


@needs_models
def test_model_and_hash_refusals(tmp_path):
    ctx = F.Ctx(mode="real", unlock=True, out=tmp_path)
    assert "30 model files" in F.check_models(ctx)
    sub = [c for c in C.all_cells() if c.get("run") == "grid_ppo_plain_s0"]
    md = tmp_path / "models" / "grid_ppo_plain_s0"
    md.mkdir(parents=True)
    with pytest.raises(F.Refusal, match="missing model"):
        F.check_models(F.Ctx(mode="real", unlock=True, out=tmp_path, cells=sub, models_dir=tmp_path / "models"))
    b = bytearray((F.MODELS / "grid_ppo_plain_s0" / "final.zip").read_bytes())
    b[100] ^= 1
    (md / "final.zip").write_bytes(bytes(b))
    with pytest.raises(F.Refusal, match="differs from the grid audit"):
        F.check_models(F.Ctx(mode="real", unlock=True, out=tmp_path, cells=sub, models_dir=tmp_path / "models"))
    a = json.loads(F.AUDIT_JSON.read_text())
    a["n_runs"] = 29
    (tmp_path / "audit.json").write_text(json.dumps(a))
    with pytest.raises(F.Refusal, match="audit"):
        F.check_audit(F.Ctx(mode="real", unlock=True, out=tmp_path, audit_path=tmp_path / "audit.json"), rerun=False)


def test_pinned_values_match_repo_files():
    pn = F.pinned()
    ctx = F.Ctx(mode="real", unlock=True, out=Path("/nonexistent"))
    assert pn["audit_sha256"].startswith("7cbe0d69") and pn["seen_rows_sha256"].startswith("7278b43c")
    assert "row typo" in F.check_pool_and_generator(ctx)                           # row (gg) generator-hash suffix is a typo; flagged, not silent
    F.check_seen_rows(ctx)
    F.check_best_single(ctx)
    if F.AUDIT_JSON.exists():
        F.check_audit(ctx, rerun=False)


@needs_models
def test_audit_rerun_is_read_only():
    import scripts_b14_audit as AU
    before = F.sha256_file(F.AUDIT_JSON), F.sha256_file(F.AUDIT_JSON.with_suffix(".md"))
    out, _, dev = AU.run(write=False)
    assert not dev and out["n_runs"] == 30
    assert (F.sha256_file(F.AUDIT_JSON), F.sha256_file(F.AUDIT_JSON.with_suffix(".md"))) == before


def test_group_ids_are_asserted():
    items = EV.load_set("DEV-validation", "validation")
    spec = {"FINAL-familiar": {"set": "DEV-validation", "n": 30, "families": {"spine_pods": 10, "archipelago": 10, "bunker_nest": 10}}}
    ctx = F.Ctx(mode="rehearse", out=Path("/x"), groups=spec)
    F.verify_group(ctx, "FINAL-familiar", items)
    with pytest.raises(F.Refusal):
        F.verify_group(ctx, "FINAL-familiar", items[:29])
    with pytest.raises(F.Refusal):
        F.verify_group(ctx, "FINAL-familiar", items[:20] + items[:10])               # duplicates
    bad = [dict(r, split="DEV-train") for r, _ in items]
    with pytest.raises(F.Refusal):
        F.verify_group(ctx, "FINAL-familiar", list(zip(bad, [l for _, l in items])))
    real = F.Ctx(mode="real", unlock=True, out=Path("/x"), groups=spec)
    with pytest.raises(F.Refusal, match="not a FINAL"):
        F.verify_group(real, "FINAL-familiar", items)                                  # real mode wants FINAL ids only


# ------------------------------------------------------------------------------------------------------------ a whole (small) real-mode session on DEV layouts
def _fake_loader(ctx_ledger_note: str):
    def load(set_name: str):
        role = "validation"
        items = EV.load_set(set_name, role)
        per = {}
        out = []
        for r, l in items:
            if per.get(r["family"], 0) < (2 if set_name == "DEV-validation" else 1):
                per[r["family"]] = per.get(r["family"], 0) + 1
                out.append((r, l))
        P.log_final_access("eval", "test-loader", {"split": set_name, "n": len(out)})    # what the real loaders write: eval entries inside the session
        return out
    return load


SMALL_GROUPS = {"FINAL-familiar": {"set": "DEV-validation", "n": 6, "families": {"spine_pods": 2, "archipelago": 2, "bunker_nest": 2}},
                "FINAL-transfer": {"set": "TUNE-val", "n": 3, "families": {"spine_pods": 1, "archipelago": 1, "bunker_nest": 1}}}


@needs_models
def test_real_mode_writes_exactly_one_eval_session_and_restart_keeps_it(tmp_path, temp_ledger, monkeypatch):
    names = ["random", "greedy", "bestsingle", "bestsingle_hindsight", "ppo_plain__s0", "ddqn_tta__s1", "ras__s0"]
    cells = [c for c in C.all_cells() if c["name"] in names]
    assert len(cells) == len(names)
    monkeypatch.setattr(F, "run_preconditions", lambda ctx, restart=False, session_id=None: [{"check": "stub", "ok": True, "detail": ""}])
    killed = {"done": False}

    def hook(event, name, ctx):
        if event == "before_write" and name == "ppo_plain__s0" and not killed["done"]:
            killed["done"] = True
            with open(ctx.raw / f"{name}.csv", "x") as f:
                f.write("cell\npartial")
            raise F.SimulatedKill(name)
    mk = lambda **kw: F.Ctx(mode="real", unlock=True, out=tmp_path / "b15", groups=SMALL_GROUPS, cells=cells, procs=1, loader=_fake_loader(""),
                            require_final_ids=False, log=lambda *_: None, **kw)
    with pytest.raises(F.SimulatedKill):
        F.run_session(mk(crash_hook=hook))
    sess = json.loads((tmp_path / "b15" / "session.json").read_text())
    assert [e["cell"] for e in F.read_manifest(mk()).values()] == names[:4]                      # cells before the kill are complete
    with pytest.raises(F.Refusal, match="session.json"):
        F.check_session_json(mk())                                                               # a plain second start is refused
    with pytest.raises(F.Refusal, match="--cause"):
        F.run_session(mk(), restart=True, cause=None)
    summary = F.run_session(mk(), restart=True, cause="machine")
    assert summary["kept"] == 4 and summary["computed_this_run"] == 3 and summary["eval_sessions"] == 1
    assert json.loads((tmp_path / "b15" / "session.json").read_text())["session_id"] == sess["session_id"]      # same session, not a second one
    led = P.read_ledger()
    ev = [e for e in led if e["purpose"] == "eval"]
    assert P.count_eval_sessions(led) == 1 and {e["session_id"] for e in ev} == {sess["session_id"]}
    assert sum(e.get("action") == "cell" for e in ev) == len(names) and sum(e["purpose"] == "build" for e in led) == 5
    log = (tmp_path / "b15" / "crash_log.md").read_text()
    assert "process died" in log and "ppo_plain__s0" in log
    assert list((tmp_path / "b15" / "discarded").glob("ppo_plain__s0.*"))
    # rows: required metadata, every cell has rows for every layout (and every N)
    rows = list(csv.DictReader(open(tmp_path / "b15" / "raw" / "ras__s0.csv")))
    assert len(rows) == 9 * 3 and {r["N"] for r in rows} == {"8", "32", "128"} and {r["session_id"] for r in rows} == {sess["session_id"]}
    assert all(r["role"] == "eval" and r["seed"] == "8080" for r in rows)
    with pytest.raises(F.Refusal, match="complete"):
        F.run_session(mk(), restart=True, cause="machine")                                       # finished: no second session exists


@needs_models
def test_cells_do_not_depend_on_worker_count_and_rows_are_reproducible(tmp_path):
    import multiprocessing as mp
    items = F.OrderedDict(g=[(r, l) for r, l in EV.load_set("DEV-validation", "validation")[:4]])
    ctx = F.Ctx(mode="rehearse", out=tmp_path, groups={"g": {"set": "DEV-validation", "n": 4, "families": {}}})
    ctx.groups = {"g": {"set": "DEV-validation"}}
    for name in ("ras__s3", "ppo_d4_tta__s0"):
        cell = next(c for c in C.all_cells() if c["name"] == name)
        a = F.compute_cell(ctx, cell, items, None, "sid", {})
        with mp.get_context("spawn").Pool(2, initializer=F._init_worker) as pool:
            b = F.compute_cell(ctx, cell, items, pool, "sid", {})
        c = F.compute_cell(ctx, cell, items, None, "sid", {})
        assert a == b == c


def test_hindsight_selection_is_per_group_and_family_with_low_index_ties():
    rec = lambda i, f: ({"id": i, "family": f}, None)
    items = {"g1": [rec("a", "x"), rec("b", "x")], "g2": [rec("c", "x")]}
    five = lambda s: [{"single_type": t, "score": s[t]} for t in range(5)]
    w = {"g1": [five([0.1, 0.5, 0.5, 0.2, 0.0]), five([0.1, 0.5, 0.5, 0.2, 0.0])], "g2": [five([0.9, 0.0, 0.0, 0.0, 0.0])]}
    out = F.select_hindsight(items, w)
    assert [(g, x["single_type"]) for g, x in out] == [("g1", 1), ("g1", 1), ("g2", 0)]


# ------------------------------------------------------------------------------------------------------------ statistics
def test_draw_order_is_the_pinned_one():
    R = 40
    counts, la = A._draws((5, 5), (7,), R, 8080)
    rng = np.random.default_rng(8080)
    for b in range(R):
        sa, sb, li = rng.integers(0, 5, size=5), rng.integers(0, 5, size=5), rng.integers(0, 7, size=7)       # seeds A, seeds B, ONE layout vector
        assert (np.bincount(sa, minlength=5) == counts[0][b]).all() and (np.bincount(sb, minlength=5) == counts[1][b]).all() and (la[0][b] == li).all()


def test_paired_ci_known_answer():
    rng = np.random.default_rng(3)
    n, S = 40, 10
    base = rng.normal(0.5, 0.1, (S, n))
    cols = np.arange(n)
    # shared seeds, constant difference: the CI collapses on the true difference
    r = A.interval([(base + 0.05, False), (base, False)], [1, -1], cols, [0, 0])
    assert r["mean"] == pytest.approx(0.05) and r["lo95"] == pytest.approx(0.05) and r["hi95"] == pytest.approx(0.05)
    # a learner whose seeds differ by a constant per seed vs a deterministic baseline: only the seed level is random (layouts constant)
    mu = np.linspace(0.40, 0.60, S)
    learner = np.repeat(mu[:, None], n, axis=1)
    r = A.interval([(learner, False), (np.full((1, n), 0.3), True)], [1, -1], cols)
    sd = mu.std() / np.sqrt(S)
    assert r["mean"] == pytest.approx(mu.mean() - 0.3)
    assert (r["hi95"] - r["lo95"]) == pytest.approx(2 * 1.96 * sd, rel=0.2) and r["lo95"] < r["mean"] < r["hi95"]
    # independent seeds are wider than shared seeds for the same models
    noisy = base + rng.normal(0, 0.05, (S, n))
    shared = A.interval([(noisy, False), (base, False)], [1, -1], cols, [0, 0])
    indep = A.interval([(noisy, False), (base, False)], [1, -1], cols, [0, 1])
    assert shared["hi95"] - shared["lo95"] < indep["hi95"] - indep["lo95"]
    # a small true effect inside the noise does not 'hold' (lower bound <= 0); a large one does
    tiny = A.interval([(base[:, :10] + 0.001, False), (rng.normal(0.5, 0.1, (S, 10)), False)], [1, -1], np.arange(10))
    big = A.interval([(base + 0.3, False), (rng.normal(0.5, 0.1, (S, n)), False)], [1, -1], cols)
    assert tiny["lo95"] <= 0 < big["lo95"]
    # layout level: with S = 1 identical arms the interval is the plain layout bootstrap of the mean
    x = rng.normal(0, 1, (1, n))
    r = A.interval([(x, True)], [1], cols)
    assert r["lo95"] < x.mean() < r["hi95"] and r["mean"] == pytest.approx(x.mean())


def test_secondary_interval_ignores_layouts_and_two_set_gap():
    rng = np.random.default_rng(4)
    M = rng.normal(0.5, 0.1, (10, 30))
    p = A.interval([(M, False)], [1], np.arange(30), level="hier")
    s = A.interval([(M, False)], [1], np.arange(30), level="seeds")
    assert s["mean"] == p["mean"] and (s["hi95"] - s["lo95"]) < (p["hi95"] - p["lo95"])                  # the layout level adds width
    d = A.interval([(np.full((1, 30), 0.4), True)], [1], np.arange(30), level="seeds")
    assert d["lo95"] == d["hi95"] == d["mean"]                                                           # nothing random: point value
    g = A.interval_two_sets(M, np.arange(30), M - 0.1, np.arange(30), False)
    assert g["mean"] == pytest.approx(0.1) and g["lo95"] < 0.1 < g["hi95"]


# ------------------------------------------------------------------------------------------------------------ the analysis on a synthetic complete session
def synth_session(root: Path, planner_bonus: float = 0.1, seed: int = 0) -> tuple:
    """A complete 116-cell session on 12 + 8 invented layouts, plus matching seen / B8 files. Returns (session dir, seen path, b8 path)."""
    ctx = F.Ctx(mode="rehearse", out=root / "s")
    fams = {"FINAL-familiar": ["spine_pods", "archipelago", "bunker_nest"], "FINAL-transfer": ["labyrinth"]}
    lay = [(g, f, f"{f}-{i}") for g, fs in fams.items() for f in fs for i in range(4 if g == "FINAL-familiar" else 8)]
    base_score = {"random": 0.1, "greedy": 0.4, "bestsingle": 0.45, "bestsingle_perlayout": 0.6, "bestsingle_hindsight": 0.5, "oracle": 0.9,
                  "ppo_plain": 0.55, "ppo_d4": 0.6, "ddqn": 0.5, "ras": 0.5, "rps": 0.3, "rpsfull": 0.4}

    def val(base, n, s, i):
        r = np.random.default_rng([seed, zlib.crc32(f"{base}{n}{s}{i}".encode())])
        lay_eff = np.random.default_rng([seed, zlib.crc32(lay[i][2].encode())]).normal(0, 0.1)
        if base.endswith("_tta"):
            return val(base[:-4], n, s, i) + 0.05
        if base.startswith("planner_"):
            return val(base[len("planner_"):], n, s, i) + planner_bonus + 0.02 * np.log2(n)
        if base in ("ras", "rps", "rpsfull"):
            return base_score[base] + lay_eff + 0.02 * np.log2(n) + r.normal(0, 0.02)
        return base_score[base] + lay_eff + (r.normal(0, 0.03) if base in ("ppo_plain", "ppo_d4", "ddqn") else 0.0)

    for cell in C.all_cells():
        rows = []
        for ns in (cell.get("ns") or [None]):
            for i, (g, f, lid) in enumerate(lay):
                sc = float(np.clip(val(cell["cond"].replace("planner_", "planner_"), ns or 0, cell["seed"] or 0, i), 0, 1))
                rows.append({k: "" for k in C.RAW_COLUMNS} | {
                    "cell": cell["name"], "condition": cell["cond"], "N": "" if ns is None else ns, "learner_seed": "" if cell["seed"] is None else cell["seed"],
                    "tta": int(bool(cell.get("tta"))), "group": g, "set": "synthetic", "family": f, "id": lid, "layout_seed": i, "n_episodes": 1, "score": sc,
                    "d": sc / 2, "breach": float(sc > 0.5), "n_bought": 8.0, "n_deployed": 8.0, "session_id": "synthetic-session", "git_commit": "c" * 40,
                    **({} if cell["cond"] in ("random", "oracle") else {a: 1.0 + (j == i % 5) for j, a in enumerate(C.ARMY_COLS)})})
        F.write_cell(ctx, cell["name"], rows, "c" * 40, "synthetic-session")
    ids = [x[2] for x in lay if x[0] == "FINAL-familiar"]
    dev = [f"dev-{i}" for i in range(15)]
    with open(root / "seen.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["run", "cond", "seed", "set", "id", "family", "score", "breach", "d", "n_bought"])
        w.writeheader()
        for c in ("ppo_plain", "ppo_d4", "ddqn"):
            for s in range(10):
                for i, d in enumerate(dev):
                    w.writerow({"run": f"grid_{c}_s{s}", "cond": c, "seed": s, "set": "DEV-train", "id": d, "family": "spine_pods", "score": 0.6 + 0.01 * s + 0.005 * i, "breach": 1, "d": 0.5, "n_bought": 8})
    with open(root / "b8.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["id", "policy", "set", "score"])
        w.writeheader()
        for pol in ("Random", "Greedy", "BestSingle", "Oracle"):
            for i, d in enumerate(dev):
                w.writerow({"id": d, "policy": pol, "set": "DEV-train", "score": 0.3 + 0.01 * i})
    return root / "s", root / "seen.csv", root / "b8.csv"


def test_analysis_is_deterministic_correct_and_raw_only(tmp_path):
    sess, seen, b8 = synth_session(tmp_path)
    r1 = A.analyze(sess, tmp_path / "t1", label="TEST", seen_path=seen, b8_path=b8)
    r2 = A.analyze(sess, tmp_path / "t2", label="TEST", seen_path=seen, b8_path=b8)
    assert F._tree_hash(tmp_path / "t1") == F._tree_hash(tmp_path / "t2")                              # same inputs, same bytes
    prim = r1["primary"]
    assert len(prim) == 16 * 2 * 2 and {p["id"] for p in prim} == {"P1", "P2", "P3", "P4", "P5"}
    assert {p["id"]: sum(q["id"] == p["id"] for q in prim) for p in prim} == {"P1": 12, "P2": 12, "P3": 4, "P4": 24, "P5": 12}
    # known answer: TTA was built as exactly +0.05 on the same seeds, so the shared-seed paired CI is the point 0.05 and the comparison holds on score
    p5 = [p for p in prim if p["id"] == "P5" and p["metric"] == "score"]
    assert all(p["diff"] == pytest.approx(0.05, abs=0.011) for p in p5) and all(p["verdict"] == "holds" for p in p5)
    # planner = control + 0.1 + same N effect on the same layouts; vs the strongest control (RAS, 0.5) it must hold on score
    p4 = [p for p in prim if p["id"] == "P4" and p["metric"] == "score"]
    assert all(p["verdict"] == "holds" for p in p4) and all("RAS" in p["comparison"] for p in p4)
    assert [x["strongest"] for x in r1["strongest"]] == ["ras"] * 6
    assert all(p["holds"] in (0, 1) for p in prim) and all("learner_only_lo95" in p for p in prim)
    # exploratory rows are labelled and carry no 'holds'
    assert all(e["status"] == "EXPLORATORY" and e.get("holds", "") == "" for e in r1["exploratory"])
    # gap: DEV-train minus FINAL-familiar, learners and baselines
    assert {g["condition"] for g in r1["gap"]} == {"ppo_plain", "ppo_d4", "ddqn", "random", "greedy", "bestsingle", "oracle"}
    assert (tmp_path / "t1" / "tables.md").read_text().startswith("# B15 tables (TEST, never results)")
    # per-group / per-family / pooled rows exist for every condition
    groups = {c["group"] for c in r1["conditions"]}
    assert {"FINAL-familiar", "FINAL-transfer", "POOLED (descriptive)", "FINAL-familiar / spine_pods", "FINAL-transfer / labyrinth"} <= groups


def test_analysis_refuses_tampered_or_mixed_raw_files(tmp_path):
    sess, seen, b8 = synth_session(tmp_path)
    p = sess / "raw" / "greedy.csv"
    p.write_text(p.read_text().replace("0.4", "0.9", 1))
    with pytest.raises(A.AnalysisError, match="does not hash"):
        A.read_session(sess)
    sess2, _, _ = synth_session(tmp_path / "second")
    lines = (sess2 / "manifest.jsonl").read_text().splitlines()
    (sess2 / "manifest.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(A.AnalysisError, match="every cell"):
        A.read_session(sess2)


def test_analysis_imports_no_simulator_model_or_pool():
    code = ("import sys, scripts_b15_analysis, scripts_b15_cells\n"
            "bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'sb3_contrib', 'stable_baselines3', 'gymnasium') or m in ('breachlens.sim', 'breachlens.pool', 'breachlens.env', 'breachlens.evaluator')]\n"
            "assert not bad, bad\n")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_cells_module_constants_match_the_project():
    from breachlens.constants import UNIT_TYPES
    assert tuple(UNIT_TYPES) == C.UNIT_TYPES
    assert C.EVAL_SEED == EV.EVAL_SEED


@needs_models
def test_restart_preconditions_accept_the_sessions_own_eval_entries(tmp_path, temp_ledger):
    """The first rehearsal died here: the audit's own 'ledger is 5 build / 0 eval' check fired on a restart. check_ledger owns that rule on a restart."""
    with open(temp_ledger, "a") as f:
        f.write(json.dumps({"purpose": "eval", "session_id": "sid", "action": "cell"}) + "\n")
    (tmp_path / "session.json").write_text("{}")
    mk = lambda: F.Ctx(mode="rehearse", out=tmp_path, run_suite=False, require_clean=False)
    log = F.run_preconditions(mk(), restart=True, session_id="sid")
    assert all(x["ok"] for x in log) and any("re-run 30/30" in x["detail"] for x in log)
    (tmp_path / "session.json").unlink()
    with pytest.raises(F.Refusal, match="eval entries"):
        F.run_preconditions(mk(), restart=False)                                                   # a fresh start with eval entries in the ledger is refused
    assert json.loads((tmp_path / "preconditions.json").read_text())["checks"][3]["ok"] is False
