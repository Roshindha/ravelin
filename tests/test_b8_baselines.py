"""B8: evaluator + baselines. Only DEV / TUNE layouts and synthetic layouts are loaded; the real ledger is never written."""
import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest

from breachlens import baselines as B
from breachlens import evaluator as E
from breachlens import pool as P
from breachlens.armies import greedy_types, place_on_border, single_unit_types
from breachlens.score import SCORE_VERSION, score_value
from breachlens.sim import Prepared, simulate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2" / "baselines"


@pytest.fixture(scope="module")
def items():
    return E.load_set("TUNE-train", "train")[:6]


def test_score_version_and_formula_only_in_score_module():
    assert SCORE_VERSION == "b7-single-rounding"
    for f in ("breachlens/evaluator.py", "breachlens/baselines.py"):
        src = (ROOT / f).read_text()
        assert not re.search(r"0\.6\s*\*|0\.4\s*\*|6\s*\*\s*d_num", src), f


def test_every_row_has_required_meta(items):
    r = E.evaluate(B.Greedy(), "TUNE-train", "train", items=items)
    assert r["rows"] and r["summary"]
    for row in r["rows"] + r["summary"]:
        for k in E.REQUIRED_META:
            assert row[k] not in (None, "")
    assert r["summary"][-1]["group"] == "ALL"


@pytest.mark.parametrize("field", E.REQUIRED_META)
def test_evaluator_fails_if_any_meta_field_missing(items, monkeypatch, field):
    real = E.run_meta

    def broken(*a, **k):
        m = real(*a, **k)
        m.pop(field)
        return m
    monkeypatch.setattr(E, "run_meta", broken)
    with pytest.raises(ValueError, match="missing required metadata"):
        E.evaluate(B.Greedy(), "TUNE-train", "train", items=items)


def test_writers_refuse_rows_without_meta(tmp_path, items):
    r = E.evaluate(B.Greedy(), "TUNE-train", "train", items=items)
    bad = [dict(r["rows"][0])]
    bad[0].pop("pool_sha256")
    with pytest.raises(ValueError):
        E.write_csv(bad, tmp_path / "x.csv")
    with pytest.raises(ValueError):
        E.write_json(bad, tmp_path / "x.json")
    E.write_csv(r["rows"], tmp_path / "ok.csv")
    assert set(E.REQUIRED_META) <= set(next(csv.DictReader(open(tmp_path / "ok.csv"))))


def test_scores_match_direct_simulation(items):
    for pol, types_fn in ((B.Greedy(), greedy_types), (B.SingleUnit(2), lambda: single_unit_types(2))):
        res = E.evaluate(pol, "TUNE-train", "train", items=items)
        for row, (rec, L) in zip(res["rows"], items):
            r = simulate(Prepared(L), place_on_border(L, types_fn()))
            assert row["score"] == score_value(r.d_num, r.d_den, r.core_destroyed)
            assert row["breach"] == float(r.core_destroyed)


def test_random_is_deterministic_uses_env_and_derived_seeds(items):
    a = E.evaluate(B.RandomPolicy(), "TUNE-train", "train", items=items[:3])
    b = E.evaluate(B.RandomPolicy(), "TUNE-train", "train", items=items[:3])
    assert [r["score"] for r in a["rows"]] == [r["score"] for r in b["rows"]]
    assert all(r["n_episodes"] == 5 for r in a["rows"])
    assert B.random_seeds("x-1") == B.random_seeds("x-1") and B.random_seeds("x-1") != B.random_seeds("x-2")
    assert len(set(B.random_seeds("x-1"))) == 5
    assert a["summary"][-1]["mean_n_bought"] > 0 and a["summary"][-1]["mean_n_deployed"] > 0


def test_select_best_single_uses_given_rows_only_and_ties_to_lowest():
    rows = []
    for f, vals in (("a", [0.1, 0.5, 0.5, 0.2, 0.0]), ("b", [0.3, 0.1, 0.2, 0.9, 0.0])):
        for t, v in enumerate(vals):
            rows.append({"family": f, "policy": f"Single-{['scout', 'sniper', 'bulwark', 'blaster', 'mender'][t]}", "score": v})
    ch = B.select_best_single(rows)
    assert ch["by_family"] == {"a": 1, "b": 3}            # tie in family a -> lowest index
    assert ch["pooled"] == int(np.argmax(np.array([0.2, 0.3, 0.35, 0.55, 0.0])))


def test_perlayout_dominates_every_single_and_oracle_dominates_all(items):
    sub = items[:4]
    pols = [B.SingleUnit(t) for t in range(5)] + [B.Greedy(), B.BestSinglePerLayout(), B.RandomPolicy(), B.Oracle(30, 30)]
    sc = {p.name: [r["score"] for r in E.evaluate(p, "TUNE-train", "train", items=sub)["rows"]] for p in pols}
    for t in range(5):
        assert all(a >= b for a, b in zip(sc["BestSingle-perlayout"], sc[B.SingleUnit(t).name]))
    for name, v in sc.items():
        assert all(o >= x for o, x in zip(sc["Oracle"], v)), name


def test_oracle_cache_roundtrip(items):
    pol = B.Oracle(20, 20)
    r = E.evaluate(pol, "TUNE-train", "train", items=items[:2])
    cache = {k: v[0] for k, v in r["episodes"].items()}
    r2 = E.evaluate(B.Oracle(20, 20, cache=cache), "TUNE-train", "train", items=items[:2])
    assert [x["score"] for x in r["rows"]] == [x["score"] for x in r2["rows"]]


def test_bootstrap_by_group_reproducible(items):
    a = E.evaluate(B.Greedy(), "TUNE-train", "train", items=items)["summary"]
    b = E.evaluate(B.Greedy(), "TUNE-train", "train", items=items)["summary"]
    assert a == b
    for g in a:
        assert g["score_lo95"] <= g["score_mean"] <= g["score_hi95"] and g["breach_lo95"] <= g["breach_mean"] <= g["breach_hi95"]


def test_role_guards_without_loading_final():
    with pytest.raises(PermissionError):
        E.load_set("FINAL-familiar", "validation")                 # refused before any ledger write
    with pytest.raises(PermissionError):
        E.load_set("DEV-train", "eval")
    with pytest.raises(PermissionError):
        E.evaluate(B.Greedy(), "FINAL-familiar", "eval", unlock=False, items=[])
    es = P.read_ledger()
    assert len(es) == 243 and sum(e["purpose"] == "eval" for e in es) == 238                  # refused loads wrote nothing: still the frozen post-B15 ledger


def test_final_path_writes_eval_entry_only_to_monkeypatched_ledger(tmp_path, monkeypatch, items):
    real_before = P.LEDGER_PATH.read_text()
    fake = tmp_path / "ledger.log"
    monkeypatch.setattr(P, "LEDGER_PATH", fake)
    synth = [({"id": f"synthetic-{i}", "family": rec["family"], "seed": rec["seed"]}, L) for i, (rec, L) in enumerate(items[:2])]
    res = E.evaluate(B.Greedy(), "FINAL-familiar", "eval", unlock=True, items=synth)
    assert res["meta"]["role"] == "eval" and res["meta"]["set"] == "FINAL-familiar"
    led = P.read_ledger()
    assert [e["purpose"] for e in led] == ["eval"] and led[0]["policy"] == "Greedy" and led[0]["set"] == "FINAL-familiar"
    monkeypatch.undo()
    assert P.LEDGER_PATH.read_text() == real_before                 # the real ledger is untouched
    led = P.read_ledger()
    assert sum(e["purpose"] == "build" for e in led) == 5 and sum(e["purpose"] == "eval" for e in led) == 238   # frozen post-B15 state


def test_non_eval_role_refuses_final_ids_in_items(items):
    fin = [r for r in json.loads(P.POOL_PATH.read_text())["records"] if r["split"] == "FINAL-familiar"][0]
    with pytest.raises(PermissionError):
        E.evaluate(B.Greedy(), "TUNE-train", "validation", items=[(fin, items[0][1])])


def test_b8_outputs_exist_with_meta_and_precede_pilot():
    rows = json.loads((OUT / "b8_summary.json").read_text())
    assert rows and all(k in r for r in rows for k in E.REQUIRED_META)
    pols = {r["policy"] for r in rows}
    assert {"Random", "Greedy", "BestSingle", "BestSingle-hindsight", "BestSingle-perlayout", "Oracle"} <= pols
    assert {r["set"] for r in rows} == {"DEV-train", "DEV-validation", "TUNE-train", "TUNE-val"}
    assert not (ROOT / "data" / "v2" / "gates_g7.json").exists()      # no G7 pilot result exists yet
    chk = json.loads((OUT / "b8_checks.json").read_text())
    assert chk["boundary_dev_acceptance"]["bit_mismatches"] == 0 and chk["ledger_after"]["eval"] == 0


def test_b8_scripts_touch_only_dev_and_design_sets():
    import ast
    chk = ast.parse((ROOT / "scripts/b8_checks.py").read_text())
    walks = [c for c in ast.walk(chk) if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "walk"]
    assert walks and all(isinstance(c.args[0], ast.Constant) and c.args[0].value == "DEV" for c in walks)
    for f in ("scripts/b8_baselines.py", "scripts/b8_checks.py"):
        tree = ast.parse((ROOT / f).read_text())
        for n in ast.walk(tree):                                   # string constants other than the module docstring / comments
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value != ast.get_docstring(tree):
                assert not n.value.startswith("FINAL-") and n.value != "LAB-DESIGN" or f == "scripts/b8_checks.py", (f, n.value)
        names = {getattr(c.func, "attr", getattr(c.func, "id", "")) for c in ast.walk(tree) if isinstance(c, ast.Call)}
        assert not names & {"log_final_access", "verify_final_acceptance", "build_pool", "load_pool_records"}, f
        assert not any(k.arg == "unlock" for c in ast.walk(tree) if isinstance(c, ast.Call) for k in c.keywords), f
