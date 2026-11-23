"""B16b: self-contained HTML replays (scripts/make_replays.py, Gate changelog row (jj)). DEV-validation only; nothing is trained or evaluated for the FINAL set."""
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
from tests import _ckpt as CK

from breachlens import pool as P

ROOT = Path(__file__).resolve().parents[1]
LEDGER_SHA = "20a8b2877c91c9cf45c3692617ad3bd08d860ef0f7783ca3610a9a3fc44c9dbd"
_spec = importlib.util.spec_from_file_location("make_replays", ROOT / "scripts" / "make_replays.py")
M = importlib.util.module_from_spec(_spec)
sys.modules["make_replays"] = M
_spec.loader.exec_module(M)
from rl import explain as X  # noqa: E402


def ledger_sha() -> str:
    return hashlib.sha256(P.LEDGER_PATH.read_bytes()).hexdigest()


def page_data(path: Path) -> dict:
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', path.read_text(), re.S)
    return json.loads(m.group(1))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    CK.require()
    a, b = tmp_path_factory.mktemp("run_a"), tmp_path_factory.mktemp("run_b")
    rows_a = M.write_replays(a, allow_dirty=True)
    rows_b = M.write_replays(b, allow_dirty=True)
    return a, b, rows_a, rows_b


def test_ledger_sha_before_and_after(runs):
    assert ledger_sha() == LEDGER_SHA
    assert P.count_eval_sessions() == 1


def test_generator_refuses_non_dev_splits(tmp_path):
    before = ledger_sha()
    for split in ("FINAL-familiar", "FINAL-transfer", "LAB-DESIGN", "TUNE-val", "TUNE-train", "nonsense"):
        with pytest.raises(PermissionError):
            M.write_replays(tmp_path / "x", split, allow_dirty=True)
        with pytest.raises(PermissionError):
            M.main(["--out", str(tmp_path / "x"), "--split", split])
    with pytest.raises(PermissionError):                       # DEV-train is a DEV split but not in the replay set of row (jj)
        M.write_replays(tmp_path / "x", "DEV-train", allow_dirty=True)
    assert not (tmp_path / "x").exists()
    assert ledger_sha() == before == LEDGER_SHA


def test_replay_set_fixed_by_index(runs):
    _, _, rows, _ = runs
    assert len(rows) == 18 and len({r["file"] for r in rows}) == 18
    assert [(r["family"], r["arm"], r["tta"]) for r in rows] == [(f, a, t) for f in M.FAMILIES for a in M.ARMS for t in (False, True)]
    recs = P.load_pool_records("DEV-validation", "validation")
    first = {}
    for i, r in enumerate(recs):
        first.setdefault(r["family"], (i, r["id"]))
    assert {r["family"]: (r["position"], r["layout_id"]) for r in rows} == {f: first[f] for f in M.FAMILIES}
    assert [first[f][0] for f in M.FAMILIES] == [0, 10, 20]


def _b15_episode(arm, tta, rec, layout, monkeypatch):
    """Actions and the evaluation row of the B15 evaluator's own code path (`scripts.b15_final._task`); nothing from rl.explain / make_replays."""
    from scripts import b15_final as F
    from breachlens.env import BreachEnv
    run, algo = X.DISPLAY_MODELS[arm]
    taken, orig = [], BreachEnv.step

    def rec_step(self, action):
        taken.append(int(action))
        return orig(self, action)
    with monkeypatch.context() as mp:
        mp.setattr(BreachEnv, "step", rec_step)
        rows = F._task(({"kind": "learner", "run": run, "algo": algo, "tta": tta, "name": f"{arm}{'_tta' if tta else ''}"}, rec, layout,
                        {"models_dir": str(X.MODELS)}))
    assert len(rows) == 1 and rows[0]["n_episodes"] == 1
    return taken, rows[0]


def test_actions_and_outcome_equal_fresh_b15_evaluation(runs, monkeypatch):
    out, _, rows, _ = runs
    layouts = {}
    for i, r in enumerate(P.load_pool_records("DEV-validation", "validation")):
        layouts.setdefault(r["family"], (r, P.load_layout(r, "validation")))
    n = 0
    for r in rows:
        rec, layout = layouts[r["family"]]
        taken, ev = _b15_episode(r["arm"], r["tta"], rec, layout, monkeypatch)
        d = page_data(out / r["file"])
        assert [s["action"] for s in d["steps"]] == taken, r["file"]                 # the replay's chosen actions = the B15 eval-path actions
        assert d["outcome"]["score"] == ev["score"], r["file"]                       # the outcome in the HTML = a fresh evaluation of that episode
        assert d["outcome"]["d"] == pytest.approx(ev["d"], abs=1e-12)
        assert float(d["outcome"]["core"]) == ev["breach"]
        assert f"{ev['score']:.4f}" in (out / r["file"]).read_text() and (out / "index.html").read_text().count(r["file"]) == 1
        n += 1
    assert n == 18


def test_outcomes_shown_whatever_they_are(runs):
    out, _, rows, _ = runs
    assert any(not r["outcome"]["core"] for r in rows)         # failed breaches are included, not filtered
    for r in rows:
        t = (out / r["file"]).read_text()
        assert "Episode outcome (true result of this replay)" in t
        assert ('core <span class="pill breach">breach</span>' in t) == r["outcome"]["core"]        # B18c: pill instead of YES / NO


def test_byte_identical_across_two_runs(runs):
    a, b, rows_a, rows_b = runs
    names_a, names_b = sorted(p.name for p in a.iterdir()), sorted(p.name for p in b.iterdir())
    assert names_a == names_b and len(names_a) == 19 and "index.html" in names_a
    for n in names_a:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n
    assert rows_a == rows_b


def test_size_caps(runs):
    a, *_ = runs
    sizes = [p.stat().st_size for p in a.iterdir()]
    assert max(sizes) <= M.FILE_CAP == int(1.5 * 1024 * 1024) and sum(sizes) <= M.TOTAL_CAP == 15 * 1024 * 1024


def test_self_contained_no_network(runs):
    a, *_ = runs
    for p in a.iterdir():
        t = p.read_text()
        t = re.sub(r"<!--.*?-->", "", t, flags=re.S)             # an optional comment may mention a URL
        assert "http" not in t.lower(), p.name
        t = t.replace("url(data:font/woff2;base64,", "")         # B18c: the only url() allowed is an embedded base64 font (data URI, no network)
        assert not re.search(r"url\((?!#)", t), p.name
        for bad in ("src=", "@import", "<link", "fetch(", "XMLHttpRequest", "WebSocket", "sendBeacon", "import(", "//cdn"):
            assert bad not in t, (p.name, bad)
        assert t.count("<script") == 2 or p.name == "index.html"


def test_labels_and_provenance_persistent(runs):
    a, _, rows, _ = runs
    prov = M.provenance()
    assert len(prov["code_commit"]) >= 40 and prov["sim_version"] and prov["score_version"]
    for r in rows:
        t = (a / r["file"]).read_text()
        run = X.DISPLAY_MODELS[r["arm"]][0]
        assert f"<b>{r['arm']}</b>" in t and f"<b>{run}</b>" in t and M.SPLIT_LABEL in t and r["layout_id"] in t
        assert ("TTA (8 views)" if r["tta"] else "plain") in t
        for k in ("code_commit", "model id", "layout id", "split", "sim_version", "SCORE_VERSION", "battle frames"):
            assert k in t
        assert prov["code_commit"] in t and prov["sim_version"] in t and prov["score_version"] in t
        assert "SHAPED" in t
        assert ("NOT a policy" in re.sub(r"<script.*?</script>", "", t, flags=re.S)) == (r["arm"] == "ddqn")      # visible page text, not the JS branch
        assert not re.search(r"20\d\d-\d\d-\d\d", t)             # no timestamp / date
        assert f"every {r['rate']} tick" in t


def test_page_contents(runs):
    a, _, rows, _ = runs
    for r in rows:
        d = page_data(a / r["file"])
        assert len(d["structs"]) >= 5 and len(d["steps"]) == r["n_steps"] >= 3
        assert d["steps"][-1]["action"] in (266,) or d["steps"][-1]["phase"] == "DEPLOY"
        for s in d["steps"]:
            assert len(s["top"]) == min(5, s["n_legal"]) and s["top"][0]["a"] == s["action"]
            assert len(s["heat"]) == 5 and all(len(x) == 52 for x in s["heat"]) and len(s["heat_tile"]) == 52
            assert abs(sum(s["heat_tile"]) - s["heat_total"]) < 1e-2 and s["value"] is not None
            if d["algo"] == "ddqn":
                assert all(t["q"] is not None for t in s["top"])
        fr = d["battle"]["frames"]
        assert fr[0]["t"] == 0 and fr[-1]["t"] == d["outcome"]["ticks"] and len(fr[0]["u"]) == d["outcome"]["n_units"]
        assert all(len(f["s"]) == len(d["structs"]) for f in fr)
        # the replay's final structure state is the simulator's: breach <=> core hp 0 on the last frame
        core = [i for i, s in enumerate(d["structs"]) if s[0] == "c"][0]
        assert (fr[-1]["s"][core] == 0) == d["outcome"]["core"]
    idx = (a / "index.html").read_text()
    assert idx.count("<tr><td>") == 18 and all(r["file"] in idx for r in rows)


def test_committed_replays_equal_fresh_generation(runs):
    """Drift guard (row (jj) addendum): docs/replays must be byte-identical to a fresh generation. The header carries code_commit (last commit
    touching the generator, rl/ or breachlens/), so replays must be regenerated after any such commit or this test fails."""
    a, _, rows, _ = runs
    d = ROOT / "docs" / "replays"
    assert d.exists(), "docs/replays missing: run scripts/make_replays.py"
    committed = sorted(f.name for f in d.iterdir())
    fresh = sorted(f.name for f in a.iterdir())
    assert committed == fresh and len(fresh) == 19
    stale = [n for n in fresh if (d / n).read_bytes() != (a / n).read_bytes()]
    assert not stale, f"committed replays are stale (regenerate with scripts/make_replays.py): {stale[:3]}"
    assert max(f.stat().st_size for f in d.iterdir()) <= M.FILE_CAP and sum(f.stat().st_size for f in d.iterdir()) <= M.TOTAL_CAP


# ------------------------------------------------------------------------------------------------ B18e-a: provenance must be clean
def test_generator_refuses_a_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "tree_dirty", lambda: True)
    with pytest.raises(SystemExit, match="dirty tree"):
        M.write_replays(tmp_path / "x")
    assert not (tmp_path / "x").exists()                                  # refused before anything was written or any model loaded


def test_no_committed_replay_provenance_is_dirty():
    files = sorted((ROOT / "docs" / "replays").glob("*.html"))
    assert files
    bad = [f.name for f in files if "+dirty" in f.read_text()]
    assert not bad, f"replays built from a dirty tree: {bad[:3]}"
