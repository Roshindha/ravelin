"""B13a: G7 decision logic (breachlens/g7.py, scripts/g7_report.py) on synthetic logs: pass, fail, per-seed, collapse (both kinds), extension branch."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from breachlens import g7 as G
from breachlens import manifest as MF
from breachlens import pool as P

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import g7_report as R  # noqa: E402

BAR = 0.4369
FAMS = ("spine_pods", "archipelago", "bunker_nest")


def diverse_armies(n_distinct=40):
    """60 layout armies (20 per family), `n_distinct` different 5-vectors, no single type above ~50 %."""
    rng = np.random.default_rng(0)
    pool = []
    while len(pool) < n_distinct:
        v = tuple(int(x) for x in rng.integers(0, 5, 5))
        if 4 <= sum(v) and v not in pool:
            pool.append(v)
    return [pool[i % n_distinct] for i in range(60)]


def make_run(scores: dict, armies=None, d=0.5, breach=0.3):
    """scores {t: mean score}; armies = list of 60 5-vectors used at every t (layout i -> family i // 20)."""
    armies = armies or diverse_armies()
    return {"score": dict(scores), "d": {t: d for t in scores}, "breach": {t: breach for t in scores}, "n_bought": {t: 10.0 for t in scores},
            "layouts": {t: [{"timestep": t, "id": f"{FAMS[i // 20]}-{i}", "family": FAMS[i // 20], "army": list(a)} for i, a in enumerate(armies)] for t in scores}}


def curve(f, horizon=300_000, freq=25_000):
    return {t: float(f(t)) for t in range(0, horizon + 1, freq)}


def seeds(*curves, armies=None):
    return {9001 + i: make_run(c, armies) for i, c in enumerate(curves)}


def test_metric_is_the_mean_of_the_last_three_evals_only():
    c = curve(lambda t: 0.9 if t == 225_000 else 0.5)                       # a spike at 225k is outside the window
    assert G.seed_metric(make_run(c), 300_000, 25_000)["score"] == pytest.approx(0.5)
    c[250_000], c[275_000], c[300_000] = 0.40, 0.50, 0.60
    assert G.seed_metric(make_run(c), 300_000, 25_000)["score"] == pytest.approx(0.50)
    assert G.window(300_000, 25_000) == [250_000, 275_000, 300_000] and G.window(600_000, 25_000) == [550_000, 575_000, 600_000]


def test_pass_when_mean_and_two_seeds_clear_the_bar_without_collapse():
    d = G.decide(seeds(curve(lambda t: 0.50), curve(lambda t: 0.48), curve(lambda t: 0.40)), BAR, 300_000, 25_000)
    assert d["verdict"] == "PASS" and all(d["first_look"]["criteria"].values()) and d["first_look"]["seeds_above_bar"] == [9001, 9002]


def test_fail_when_the_mean_clears_the_bar_but_only_one_seed_does():
    d = G.decide(seeds(curve(lambda t: 0.95), curve(lambda t: 0.40), curve(lambda t: 0.40)), BAR, 300_000, 25_000)
    c = d["first_look"]["criteria"]
    assert c["mean_ge_bar"] and not c["two_of_three_above_bar"] and d["verdict"] == "FAIL"


def test_fail_when_two_seeds_clear_but_the_mean_does_not():
    d = G.decide(seeds(curve(lambda t: 0.44), curve(lambda t: 0.44), curve(lambda t: 0.10)), BAR, 300_000, 25_000)
    c = d["first_look"]["criteria"]
    assert c["two_of_three_above_bar"] and not c["mean_ge_bar"] and d["verdict"] == "FAIL"


def test_bar_is_inclusive_for_the_mean_and_strict_per_seed():
    d = G.judge(seeds(curve(lambda t: BAR), curve(lambda t: BAR), curve(lambda t: BAR)), BAR, 300_000, 25_000)
    assert d["criteria"]["mean_ge_bar"] and not d["criteria"]["two_of_three_above_bar"]


def test_collapse_fewer_than_10_distinct_armies_fails_even_with_a_high_score():
    arm = diverse_armies(9)
    d = G.decide(seeds(*[curve(lambda t: 0.6)] * 3, armies=arm), BAR, 300_000, 25_000)
    assert d["first_look"]["collapse"][9001]["distinct_armies"] == 9
    assert not d["first_look"]["criteria"]["distinct_armies_ge_10_every_seed"] and d["verdict"] == "FAIL"
    assert G.collapse(make_run(curve(lambda t: .5), diverse_armies(10)), 300_000)["distinct_ok"]        # exactly 10 is enough


def _with_family_armies(per_family: dict):
    """60 diverse armies, then overwrite the 20 layouts of the given families (index 0 spine_pods, 1 archipelago, 2 bunker_nest)."""
    arm = diverse_armies(40)
    for k, v in per_family.items():
        arm[20 * k:20 * k + 20] = v
    return arm


SNIPER, SCOUT = 1, 0


def test_collapse_90_percent_rule_is_strict_and_uses_units_bought():
    one = lambda t, n_t, n_other: [tuple(n_t if j == t else (n_other if j == 2 else 0) for j in range(5))] * 20
    arm = _with_family_armies({0: one(SNIPER, 9, 1)})                                # exactly 90 %: not above
    assert G.collapse(make_run(curve(lambda t: .5), arm), 300_000)["max_type_share_per_family"]["spine_pods"] == pytest.approx(0.9)
    sh = lambda arm: G.collapse(make_run(curve(lambda t: .5), arm), 300_000)
    all_fam = lambda n_t, n_o: _with_family_armies({k: one(SNIPER, n_t, n_o) for k in range(3)})
    assert sh(all_fam(9, 1))["share_ok"]                                              # 90 % in every family: not strictly above -> no collapse
    assert not sh(all_fam(19, 1))["share_ok"]                                         # 95 % in every family -> collapse


def test_over_90_percent_sniper_in_ONE_family_only_is_not_a_collapse():
    arm = _with_family_armies({0: [(0, 19, 0, 0, 1)] * 20})                           # spine_pods: 95 % Sniper; the other families stay diverse
    c = G.collapse(make_run(curve(lambda t: .5), arm), 300_000)
    assert c["max_type_share_per_family"]["spine_pods"] > 0.9 and c["max_type_per_family"]["spine_pods"] == SNIPER
    assert c["share_ok"] and c["single_type_in_every_family_above_90pct"] == []
    d = G.decide(seeds(*[curve(lambda t: 0.6)] * 3, armies=arm), BAR, 300_000, 25_000)
    assert d["first_look"]["criteria"]["no_single_type_above_90pct_in_every_family_every_seed"] and d["verdict"] == "PASS"
    two = _with_family_armies({0: [(0, 19, 0, 0, 1)] * 20, 1: [(0, 19, 0, 0, 1)] * 20})        # two of three families: still not "every family"
    assert G.collapse(make_run(curve(lambda t: .5), two), 300_000)["share_ok"]


def test_over_90_percent_the_same_type_in_every_family_is_a_collapse():
    arm = _with_family_armies({k: [(0, 19, 0, 0, 1)] * 20 for k in range(3)})
    c = G.collapse(make_run(curve(lambda t: .5), arm), 300_000)
    assert not c["share_ok"] and c["single_type_in_every_family_above_90pct"] == [SNIPER]
    d = G.decide(seeds(*[curve(lambda t: 0.6)] * 3, armies=arm), BAR, 300_000, 25_000)
    assert not d["first_look"]["criteria"]["no_single_type_above_90pct_in_every_family_every_seed"] and d["verdict"] == "FAIL"


def test_different_dominant_types_in_different_families_is_not_a_collapse():
    arm = _with_family_armies({0: [(0, 19, 0, 0, 1)] * 20, 1: [(19, 0, 0, 0, 1)] * 20, 2: [(0, 0, 19, 0, 1)] * 20})     # >90 % each, but not ONE type
    c = G.collapse(make_run(curve(lambda t: .5), arm), 300_000)
    assert c["share_ok"] and all(v > 0.9 for v in c["max_type_share_per_family"].values())


def test_collapse_in_a_single_seed_fails_g7():
    good, bad = diverse_armies(40), [(20, 0, 0, 0, 0)] * 60
    runs = {9001: make_run(curve(lambda t: .6), good), 9002: make_run(curve(lambda t: .6), good), 9003: make_run(curve(lambda t: .6), bad)}
    d = G.decide(runs, BAR, 300_000, 25_000)
    assert d["verdict"] == "FAIL" and not d["first_look"]["criteria"]["distinct_armies_ge_10_every_seed"]


def test_slope_is_least_squares_over_the_last_five_evals_in_score_per_100k():
    run = make_run(curve(lambda t: 0.3 + 0.01 * (t / 25_000)))                       # +0.01 per 25k = +0.04 per 100k
    assert G.slope_per_100k(run, 300_000, 25_000) == pytest.approx(0.04)
    assert G.slope_ts(300_000, 25_000) == [200_000, 225_000, 250_000, 275_000, 300_000]
    noisy = curve(lambda t: 0.3)
    noisy[0] = 5.0                                                                  # evals before t = 200k are ignored
    assert G.slope_per_100k(make_run(noisy), 300_000, 25_000) == pytest.approx(0.0, abs=1e-12)


def test_extension_branch_extend_then_pass_or_fail_or_invalid():
    rise = lambda t: 0.20 + 0.04 * (t / 100_000) if t <= 300_000 else 0.20 + 0.04 * 3 + 0.10 * ((t - 300_000) / 100_000)     # 0.32 at 300k (below the bar), slope 0.04; steeper afterwards
    s300 = {9001 + i: make_run(curve(rise, 300_000)) for i in range(3)}
    d = G.decide(s300, BAR, 300_000, 25_000)
    assert d["verdict"] == "EXTEND" and d["mean_slope_per_100k"] == pytest.approx(0.04) and not d["first_look"]["pass"]
    s600 = {9001 + i: make_run(curve(rise, 600_000)) for i in range(3)}
    d6 = G.decide(s600, BAR, 600_000, 25_000)
    assert d6["verdict"] == "PASS" and d6["second_look"]["window"] == [550_000, 575_000, 600_000]
    flat_after = lambda t: rise(min(t, 300_000))
    assert G.decide({9001 + i: make_run(curve(flat_after, 600_000)) for i in range(3)}, BAR, 600_000, 25_000)["verdict"] == "FAIL"       # no second extension
    low_slope = {9001 + i: make_run(curve(lambda t: 0.30 + 0.001 * t / 25_000, 600_000)) for i in range(3)}
    assert G.decide(low_slope, BAR, 300_000, 25_000)["verdict"] == "FAIL"           # slope < 0.02: no extension
    assert G.decide(low_slope, BAR, 600_000, 25_000)["verdict"].startswith("INVALID")  # 600k look without a licence
    passing = {9001 + i: make_run(curve(lambda t: 0.6, 600_000)) for i in range(3)}
    assert G.decide(passing, BAR, 600_000, 25_000)["verdict"].startswith("INVALID")   # G7 already passed at 300k: no extension


def test_mean_slope_is_averaged_over_seeds_not_per_seed_thresholded():
    c = [curve(lambda t, k=k: 0.2 + k * (t / 100_000)) for k in (0.05, 0.03, -0.0)]      # slopes .05, .03, 0 -> mean .0267 >= .02
    assert G.decide(seeds(*c), BAR, 300_000, 25_000)["verdict"] == "EXTEND"
    c = [curve(lambda t, k=k: 0.2 + k * (t / 100_000)) for k in (0.05, 0.0, -0.0)]       # mean .0167 < .02
    assert G.decide(seeds(*c), BAR, 300_000, 25_000)["verdict"] == "FAIL"


def test_missing_evals_raise_instead_of_guessing():
    c = curve(lambda t: .5)
    del c[275_000]
    with pytest.raises(ValueError):
        G.seed_metric(make_run(c), 300_000, 25_000)


def test_bar_file_is_the_pinned_one():
    assert round(json.loads(G.BAR_PATH.read_text())["g7_bar"], 4) == BAR


# ------------------------------------------------------------------------------------------------ the CLI on synthetic run directories
def write_run(d: Path, run: dict, manifest: dict, status="finished", commit="abc", dirty="False"):
    d.mkdir(parents=True)
    fields = ["timestep", "group", "n_layouts", "score_mean", "d_mean", "breach_mean", "mean_n_bought", "score_version", "sim_version", "generator_hash", "pool_sha256",
              "git_commit", "git_dirty", "seed", "role", "set"]
    with (d / "val_eval.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for t in sorted(run["score"]):
            w.writerow({"timestep": t, "group": "ALL", "n_layouts": 60, "score_mean": run["score"][t], "d_mean": run["d"][t], "breach_mean": run["breach"][t],
                        "mean_n_bought": run["n_bought"][t], "score_version": "s", "sim_version": "v", "generator_hash": "g", "pool_sha256": "p", "git_commit": commit,
                        "git_dirty": dirty, "seed": 8080, "role": "validation", "set": "TUNE-val"})
    (d / "val_layouts.jsonl").write_text("".join(json.dumps(r) + "\n" for t in sorted(run["layouts"]) for r in run["layouts"][t]))
    (d / "manifest.json").write_text(json.dumps(manifest))
    (d / "model_identity.json").write_text(json.dumps({"status": status, "train_set": "TUNE-train", "val_set": "TUNE-val", "environment": {"torch_threads": 1}}))


def good_manifest():
    return MF.build_manifest("t", "TUNE-train", "TUNE-val", [r["id"] for r in P.design_records("TUNE-train", "train")], [r["id"] for r in P.design_records("TUNE-val", "validation")])


def synth_dir(tmp_path, scores=(0.5, 0.48, 0.45), d4=0.3, manifest=None):
    md = tmp_path / "models"
    for i, sc in enumerate(scores):
        write_run(md / f"pilot_plain_s{9001 + i}", make_run(curve(lambda t, sc=sc: sc)), manifest or good_manifest())
    for i in range(3):
        write_run(md / f"pilot_d4_s{9101 + i}", make_run(curve(lambda t: d4)), good_manifest())
    return md


def test_cli_prints_pass_fail_per_criterion_and_the_verdict(tmp_path, capsys):
    md = synth_dir(tmp_path)
    rc = R.main(["--models-dir", str(md), "--smoke", "--out", str(tmp_path / "o.json")])
    out = capsys.readouterr().out
    assert rc == 0 and "PASS  mean_ge_bar" in out and "PASS  two_of_three_above_bar" in out and "VERDICT: PASS" in out and "D4 diagnostics (non-gating)" in out
    assert out.count("max single-type share per family (diagnostic, non-gating)") == 3 and "spine_pods" in out and "(type " in out
    assert json.loads((tmp_path / "o.json").read_text())["decision"]["verdict"] == "PASS"
    md2 = synth_dir(tmp_path / "b", scores=(0.30, 0.30, 0.30))
    R.main(["--models-dir", str(md2), "--smoke", "--out", str(tmp_path / "o2.json")])
    out = capsys.readouterr().out
    assert "FAIL  mean_ge_bar" in out and "VERDICT: FAIL" in out


def test_cli_reports_a_manifest_with_a_final_id_as_a_problem(tmp_path, capsys):
    bad = good_manifest()
    bad["val_ids"].append(next(r["id"] for r in json.loads(P.POOL_PATH.read_text())["records"] if r["split"] == "FINAL-familiar"))
    md = synth_dir(tmp_path, manifest=bad)
    rc = R.main(["--models-dir", str(md), "--smoke", "--out", str(tmp_path / "o.json")])
    out = capsys.readouterr().out
    assert rc == 2 and "PROBLEM" in out and "FINAL" in out and "PROVENANCE PROBLEMS" in out


def test_cli_refuses_to_call_a_two_seed_or_unfinished_run_a_valid_g7(tmp_path, capsys):
    md = synth_dir(tmp_path)
    R.main(["--models-dir", str(md), "--plain-seeds", "9001,9002", "--smoke", "--out", str(tmp_path / "o.json")])
    assert "NOT A VALID G7" in capsys.readouterr().out
    assert not G.decide({9001: make_run(curve(lambda t: .5))}, BAR, 300_000, 25_000)["first_look"]["valid_g7"]


def test_the_report_never_writes_the_gate_file():
    assert "gates.json" not in (ROOT / "scripts/g7_report.py").read_text().replace("never writes data/v2/gates.json", "")
