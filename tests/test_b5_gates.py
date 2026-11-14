"""B5: gate machinery, thresholds file vs SPEC_V2, stat-dependent acceptance, seed-stream disjointness, stat pins, B5 report fields."""
import copy
import json
import re
from pathlib import Path

import numpy as np
import pytest

import scripts_gates as SG
from breachlens import acceptance as A
from breachlens import config as K
from breachlens import gates as G
from breachlens import layouts as Y
from breachlens import oracle as O
from breachlens.constants import UNIT_TYPES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v2"
SPEC = (ROOT / "docs" / "SPEC_V2.md").read_text()
TH = json.loads((DATA / "gate_thresholds.json").read_text())
META = Y.load_design_sets_meta()


# ------------------------------------------------------------------ paired bootstrap helper on synthetic data
def test_bootstrap_known_mean_difference_and_lower_bound_sign():
    rng = np.random.default_rng(0)
    pos = rng.normal(0.10, 0.05, size=80)
    b = G.bootstrap_mean(pos)
    assert abs(b["mean"] - pos.mean()) < 1e-12 and 0.08 < b["mean"] < 0.12
    assert b["lo95"] > 0 and b["lo_1s"] > 0 and b["lo_1s"] <= b["lo95"] <= b["mean"] <= b["hi95"]
    zero = rng.normal(0.0, 0.05, size=80)
    z = G.bootstrap_mean(zero)
    assert z["lo95"] < 0 < z["hi95"] and z["lo_1s"] < 0
    neg = G.bootstrap_mean(-pos)
    assert neg["hi95"] < 0 and neg["lo95"] < 0


def test_bootstrap_is_deterministic_percentiles_follow_the_settings_and_one_sided_is_stricter():
    x = np.linspace(-0.1, 0.3, 60)
    a, b = G.bootstrap_mean(x), G.bootstrap_mean(x)
    assert a == b
    # independent re-derivation of the percentiles from the declared settings
    s = TH["bootstrap"]
    means = x[np.random.default_rng(s["seed"]).integers(0, len(x), size=(s["n_resamples"], len(x)))].mean(axis=1)
    assert a["lo95"] == pytest.approx(np.percentile(means, 2.5)) and a["lo_1s"] == pytest.approx(np.percentile(means, 100 * (1 - 0.9875)))
    assert a["lo_1s"] < a["lo95"]
    assert s["one_sided_level"] == pytest.approx(1 - s["familywise_alpha"] / s["bonferroni_families"])


def test_g2_g3_require_the_one_sided_bound_in_addition():
    # a family where mean >= 0.02 and lo95 > 0 but lo_1s <= 0 must NOT pass
    n = 80
    d = np.zeros(n); d[:12] = 0.2          # mean 0.03, heavy-tailed: percentile bounds straddle zero at the strict level
    b = G.bootstrap_mean(d)
    assert b["mean"] >= 0.02
    assert (b["lo95"] > 0 and b["lo_1s"] > 0) == (b["lo_1s"] > 0)       # the conjunction equals the stricter condition
    src = (ROOT / "breachlens" / "gates.py").read_text()
    assert src.count('bs["lo_1s"] > 0') >= 1                                # used in g2 (G3 is pooled per-layout since B5-R1: no one-sided bound)
    assert "lo_1s" not in src.split("def g3")[1].split("def g4")[0]
    assert "lo_1s" not in src.split("def g4")[1].split("def g5")[0]       # G4 keeps the 95 % rules


# ------------------------------------------------------------------ thresholds file == SPEC_V2 text
def _row(gate):
    return next(l for l in SPEC.splitlines() if l.startswith(f"| **{gate}**"))


def _nums(s):
    return [float(x) for x in re.findall(r"\d+\.\d+|\d+", s.replace(" %", "").replace("%", ""))]


def test_thresholds_file_equals_the_spec():
    r1 = _row("G1")
    assert TH["G1"]["max_best_single_share"] == 0.60 and "> 60 %" in r1
    assert TH["G1"]["max_ban_loss_in_every_family"] == 0.25 and "> 0.25" in r1
    assert TH["G2"]["min_ban_loss"] == 0.02 and "≥ 0.02" in _row("G2") and "98.75" in _row("G2")
    assert TH["G3"]["min_template_margin"] == 0.03 and "≥ 0.03" in _row("G3") and "98.75" not in _row("G3")
    assert TH["G3"]["min_layout_share"] == 0.08 and "≥ 8 %" in _row("G3") and "per-layout niche" in _row("G3") and "bulwark_mender" in _row("G3")
    assert TH["G3"]["bonferroni_applies"] is False and "B5-R1 clarification of G3" in SPEC
    r4 = _row("G4")
    assert TH["G4"]["min_mender_gain"] == 0.02 and "≥ 0.02" in r4 and TH["G4"]["mender_only_max_mean"] == 0.01 and "< 0.01" in r4 and TH["G4"]["mender_only_max_upper95"] == 0.02 and "< 0.02" in r4
    r5 = _row("G5")
    assert TH["G5"]["best_single_mean_range"] == [0.25, 0.65] and "[0.25, 0.65]" in r5 and TH["G5"]["oracle_score_range"] == [0.80, 0.97] and "[0.80, 0.97]" in r5 and TH["G5"]["oracle_breach_min"] == 0.6 and "≥ 0.6" in r5 and "0.95" not in r5
    r6 = _row("G6")
    assert TH["G6"]["min_units_over"] == 12 and "> 12 units" in r6 and TH["G6"]["min_layout_share"] == 0.30 and "30 %" in r6 and TH["G6"]["blocking"] is False and "diagnostic" in r6
    r7 = _row("G7")
    assert TH["G7"]["margin"] == 0.05 and "0.05" in r7 and "≥ 2 of 3" in r7 and "≥ 10 distinct" in r7 and "> 90 %" in r7 and "+0.02" in r7 and "600k" in r7
    assert TH["G7"]["min_seeds_above"] == 2 and TH["G7"]["of_seeds"] == 3 and TH["G7"]["extension_timesteps"] == 600000 and TH["G7"]["extension_slope_per_100k"] == 0.02
    r8 = _row("G8")
    assert TH["G8"]["army_search_margin"] == 0.10 and "≥ 0.10" in r8 and TH["G8"]["army_search_layout_share"] == 0.60 and "60 %" in r8 and TH["G8"]["oracle_margin"] == 0.20 and "≥ 0.20" in r8
    r9 = _row("G9")
    assert TH["G9"]["oracle_mean_min"] == 0.60 and "≥ 0.60" in r9 and TH["G9"]["greedy_mean_range"] == [0.10, 0.65] and "≥ 0.10" in r9 and "≤ 0.65" in r9 and "≤ 0.45" not in r9
    assert TH["G9"]["greedy_min_layout_score"] == 0.05 and "0.05" in r9 and TH["G9"]["greedy_min_layout_share"] == 0.50 and "50 %" in r9
    assert TH["bootstrap"]["n_resamples"] == 10000 and "10 000 resamples" in SPEC and TH["bootstrap"]["seed"] == 8080 and "seed 8080" in SPEC
    assert TH["bootstrap"]["one_sided_level"] == 0.9875 and "98.75 %" in SPEC and TH["bootstrap"]["bonferroni_families"] == 4
    assert TH["acceptance"]["single_type_trivial_best_single_gt"] == 0.99 and "> 0.99" in SPEC
    assert TH["search"]["oracle_n_random"] == O.N_RANDOM == 200 and TH["search"]["oracle_n_refine"] == O.N_REFINE
    assert TH["search"]["army_search_n_random"] == O.AS_RANDOM == 128 and TH["search"]["army_search_n_refine"] == O.AS_REFINE
    assert (TH["search"]["oracle_tag"], TH["search"]["army_search_tag"], TH["search"]["acceptance_tag"]) == (O.ORACLE_TAG, O.ARMY_SEARCH_TAG, O.ACCEPT_TAG)
    assert "no_attackers" in SPEC
    assert TH["G2"]["estimator"] == TH["G4"]["estimator"] == "monotone" and "monotone estimator" in _row("G2") and "monotone estimator" in _row("G4")
    assert "Gate changelog" in SPEC and all(f"({c})" in SPEC.split("Gate changelog (post-B5-R1")[1] for c in "abcd")


# ------------------------------------------------------------------ stats pinned by test (every unit, defender, structure)
def test_every_unit_defender_and_structure_stat_is_pinned():
    raw = json.loads(K.STATS_PATH.read_text())
    units = {k: (v["cost"], v["hp"], v["damage"], v["range_x10"], v["speed_x10"], v["armor"], tuple(v["pref"]), v["splash_r_x10"], v["splash_pct"], v["heal"], v["heal_range_x10"])
             for k, v in raw["units"].items()}
    assert units == {
        "scout": (8, 16, 6, 12, 15, 0, ("building", "turret", "wall"), 0, 0, 0, 0),
        "sniper": (28, 28, 12, 50, 10, 0, ("turret", "building", "wall"), 0, 0, 0, 0),   # revision 1: cost 24 -> 28
        "bulwark": (30, 170, 16, 14, 7, 4, ("wall", "turret", "building"), 0, 0, 0, 0),
        "blaster": (36, 60, 26, 25, 7, 0, ("building", "wall", "turret"), 15, 50, 0, 0),
        "mender": (28, 40, 0, 0, 10, 0, (), 0, 0, 8, 20)}
    d = {k: (v["hp"], v["damage"], v["damage_neighbour"], v["min_range_x10"], v["max_range_x10"], v["period"], v["area"], v["splash_r_x10"]) for k, v in raw["defenders"].items()}
    assert d == {"gun": (126, 12, 0, 0, 40, 1, False, 0), "lancer": (98, 18, 0, 20, 60, 1, False, 0), "flak": (140, 14, 7, 20, 50, 2, True, 15)}
    assert raw["structures"] == {"wall_hp": 80, "core_hp": 840, "building_hp_default": 84} and raw["fixed_point_scale"] == 840   # revision 1
    assert raw["units"]["sniper"]["class_damage_pct"] == {"building": 40, "wall": 40, "turret": 100} and raw["stats_version"] == "v2-rev1"
    assert raw["rules"] == {"stalemate_ticks": 12, "min_hit_hp": 1}


# ------------------------------------------------------------------ stat-dependent acceptance rules
def test_trivial_rule_flags_an_easy_layout_and_accepts_the_design_sets(monkeypatch):
    easy = {"buildings": [{"cell": (1, 3), "value": 1}], "core": {"cell": (1, 4)}, "walls": [], "defenders": []}
    st = A.stat_check(easy, 1, n_random=2, n_refine=0)
    assert st["trivial"] and "single_type_trivial" in st["reasons"] and st["best_single"] > 0.99


def test_too_hard_rule_fires_when_the_acceptance_oracle_cannot_breach(monkeypatch):
    L = Y.load_design_set("TUNE-train", "spine_pods")[0][1]
    class R: breach = False; score = 0.1
    monkeypatch.setattr(O, "oracle_search", lambda *a, **k: R())
    st = A.stat_check(L, 1)
    assert st["too_hard"] and "too_hard" in st["reasons"]


def test_acceptance_stream_uses_a_different_rng_than_the_gate_oracle():
    assert not (O.oracle_rng(5, O.ACCEPT_TAG).random(4) == O.oracle_rng(5, O.ORACLE_TAG).random(4)).all()


def test_design_sets_file_records_stat_acceptance_and_is_consistent():
    sa = META["stat_acceptance"]
    assert set(sa) == set(Y.FAMILIES)
    for fam, s in sa.items():
        assert s["stat_rejected"] <= s["single_type_trivial"] + s["too_hard"] and s["stat_rejected"] >= max(s["single_type_trivial"], s["too_hard"])
        assert s["stat_rejected"] == len(s["stat_rejected_seeds"]) and s["stat_rejection_rate"] == pytest.approx(s["stat_rejected"] / s["stat_evaluated"])
        assert s["flag_above_50pct"] == (s["stat_rejection_rate"] > 0.5)
        n_acc = sum(1 for name in META["sets"] for r in META["sets"][name] if r["family"] == fam)
        assert s["stat_evaluated"] == n_acc + s["stat_rejected"]
        lo, hi = Y.stream_range("L" if fam == "labyrinth" else "T", fam)
        assert all(lo <= x < lo + META["stream_use"][fam]["seeds_consumed"] <= hi for x in s["stat_rejected_seeds"])
    assert META["sim_version"] == K.sim_version(), "stats or simulator changed: rebuild the design sets (scripts_b5_design_sets.py)"


def test_accepted_records_pass_and_rejected_seeds_fail_the_stat_rules():
    rng = np.random.default_rng(77)
    for fam in Y.FAMILIES:
        recs = [r for name in META["sets"] for r in META["sets"][name] if r["family"] == fam]
        for r in [recs[int(i)] for i in rng.choice(len(recs), size=4, replace=False)]:
            assert A.stat_check(Y.generate(fam, r["seed"]), r["seed"])["reasons"] == []
        rej = META["stat_acceptance"][fam]["stat_rejected_seeds"]
        for sd in [rej[int(i)] for i in rng.choice(len(rej), size=min(3, len(rej)), replace=False)] if rej else []:
            assert A.stat_check(Y.candidate(fam, sd), sd)["reasons"] != []


def test_file_is_the_structural_stream_minus_exactly_the_stat_rejected_seeds_and_a_small_rebuild_is_a_prefix():
    for fam in Y.FAMILIES:
        stream = "L" if fam == "labyrinth" else "T"
        lo = Y.stream_range(stream, fam)[0]
        used = META["stream_use"][fam]["seeds_consumed"]
        struct_ok = [sd for sd in range(lo, lo + used) if not Y.check(Y.candidate(fam, sd), fam)]
        recs = [r for name in META["sets"] for r in META["sets"][name] if r["family"] == fam]
        got = {r["seed"] for r in recs}
        rejected = set(META["stat_acceptance"][fam]["stat_rejected_seeds"])
        assert got.isdisjoint(rejected) and (got | rejected) <= set(struct_ok)
        hashes = {r["content_hash"] for r in recs}
        for sd in set(struct_ok) - got - rejected:       # skipped by the builder: only D4-duplicates of an already accepted layout
            assert Y.content_hash(Y.candidate(fam, sd)) in hashes
    small = A.build_accepted_design_sets(sizes={"TUNE-train": 2, "TUNE-val": 1}, lab_size=2, procs=1)
    for fam in Y.FAMILIAR:
        tr = [r["seed"] for r in small["sets"]["TUNE-train"] if r["family"] == fam] + [r["seed"] for r in small["sets"]["TUNE-val"] if r["family"] == fam]
        full = [r["seed"] for r in META["sets"]["TUNE-train"] if r["family"] == fam]
        assert tr[:2] == full[:2]
    assert [r["seed"] for r in small["sets"]["LAB-DESIGN"]] == [r["seed"] for r in META["sets"]["LAB-DESIGN"][:2]]


# ------------------------------------------------------------------ seed streams and the FINAL ledger
def test_design_sets_are_disjoint_from_stream_F_every_pool_and_the_ledger_is_empty():
    all_recs = [r for name in META["sets"] for r in META["sets"][name]]
    assert not any(Y.stream_of(r["seed"]) == "F" for r in all_recs)
    assert len({r["content_hash"] for r in all_recs}) == len(all_recs) == 280 and len({r["seed"] for r in all_recs}) == 280
    assert META["counts"] == {"TUNE-train": 180, "TUNE-val": 60, "LAB-DESIGN": 40}
    pool_seeds = set()
    for p in sorted((ROOT / "results").glob("layouts*.json")):
        d = json.loads(p.read_text())
        if isinstance(d, list):
            pool_seeds |= {r["seed"] for r in d}
    assert not ({r["seed"] for r in all_recs} & pool_seeds)
    v2 = DATA / "layouts_v2.json"
    if v2.exists():
        recs = json.loads(v2.read_text())
        assert not ({r["seed"] for r in recs} & {r["seed"] for r in all_recs})
    ledger = DATA / "final_access.log"
    assert (not ledger.exists()) or ledger.read_text().strip() == ""
    src = (ROOT / "breachlens" / "acceptance.py").read_text() + (ROOT / "breachlens" / "gates.py").read_text()
    assert 'assert stream in ("T", "L")' in src


def test_defender_mix_acceptance_rules_hold_on_the_generated_sets():
    mix = G.defender_mix()
    rules = G.defender_mix_rules(mix)
    assert all(rules.values()), rules


# ------------------------------------------------------------------ the gate script is deterministic (small, fast, no files written)
def test_gate_evaluation_is_deterministic_and_pure():
    budgets = {"oracle_n_random": 6, "oracle_n_refine": 6, "as_n_random": 6, "as_n_refine": 6}
    tasks = G.design_tasks(limit=2, budgets=budgets)
    assert {t["family"] for t in tasks} == set(Y.FAMILIES)
    a = G.run_raw(tasks, procs=1)
    b = G.run_raw(tasks, procs=1)
    assert a == b
    ga, gb = G.compute_gates(a), G.compute_gates(b)
    assert ga == gb and set(ga["gates"]) == {"G1", "G2", "G3", "G4", "G5", "G6", "G8", "G9"}
    json.dumps(ga)
    rep = SG.build_report(a, record_hash=False)
    md = SG.markdown(rep)
    assert "G1-G9 summary" in md and "Wall behaviour" in md


# ------------------------------------------------------------------ B5 report and gates.json on disk
def test_gates_json_and_report_exist_with_every_required_field():
    gj = json.loads((DATA / "gates.json").read_text())
    rep = json.loads((DATA / "b5_report.json").read_text())
    assert (DATA / "b5_report.md").exists() and (DATA / "b5_raw.json").exists()
    assert gj["sim_version"] == K.sim_version() and gj["git"]["commit"] and gj["seed_sets"] == {"TUNE-train": 180, "TUNE-val": 60, "LAB-DESIGN": 40}
    assert set(gj["gates"]) == {"G1", "G2", "G3", "G4", "G5", "G6", "G8", "G9"} and gj["all_pass"] == (not gj["failed"])
    assert rep["thresholds"]["bootstrap"] == TH["bootstrap"]     # the stored revision-0 report predates the gate changelog thresholds
    for u in UNIT_TYPES:
        s = rep["unit_static"][u]
        assert set(s["ehp_per_gold_by_defender"]) == {"gun", "lancer", "flak"} and "ehp_per_gold_mean_of_three" in s and "ehp_per_gold_flak_neighbour_hit" in s
        assert "dps_per_gold" in s and "threatened_by_defenders_(<=3_hits_and_max_range>=unit_range)" in s and "outranged_by" in s
        for col in (*Y.FAMILIAR, "LAB-DESIGN"):
            c = rep["single_unit_table"][col]["units"][u]
            assert {"score", "D", "breach", "units", "gold", "best"} <= set(c) and all("lo95" in c[k] and "hi95" in c[k] for k in ("score", "D", "breach"))
        for fam in (*Y.FAMILIAR, "LAB-DESIGN"):
            w = rep["wall_metrics"][fam]
            assert "walls_attacked_per_battle" in w["single_unit_armies"][u] and "wall_share_of_alive_unit_ticks" in w["single_unit_armies"][u]
            assert "walls_attacked_per_battle" in w["oracle_armies"][u]
            assert "wall_share_of_alive_attacker_unit_ticks" in w["family_totals"]["single_unit_armies"]
    for col in (*Y.FAMILIAR, "LAB-DESIGN"):
        assert sum(c["best"] for c in rep["single_unit_table"][col]["units"].values()) == 1
    assert set(rep["stat_acceptance"]) == set(Y.FAMILIES) and set(rep["defender_mix"]) == set(Y.FAMILIES)
    assert rep["meta"]["layouts_per_set"] == {"TUNE-train": 180, "TUNE-val": 60, "LAB-DESIGN": 40}


def test_report_is_reproducible_from_raw_and_matches_the_files():
    raw = json.loads((DATA / "b5_raw.json").read_text())["layouts"]
    assert len(raw) == 280
    th = json.loads((DATA / "gate_thresholds.json").read_text())
    gj = json.loads((DATA / "gates.json").read_text())
    # gates.json is the stored revision-1 record (R2a, B5-R2b): every gate reproduces exactly from the stored raw data (which has the monotone fields)
    assert gj["revision"] == 1
    again = G.compute_gates(raw, th)
    for k in ("G1", "G2", "G3", "G4", "G5", "G6", "G8", "G9"):
        assert again["gates"][k] == gj["gates"][k], k


def test_generator_hash_is_recorded_only_if_all_gates_pass():
    gj = json.loads((DATA / "gates.json").read_text())
    f = DATA / "generator_hash.json"
    if gj["all_pass"]:
        assert f.exists() and json.loads(f.read_text())["generator_hash"] == G.generator_hash()
    else:
        assert not f.exists()


# ------------------------------------------------------------------ B5-R1: per-layout niche G3 on synthetic data
def _fake_row(i, scores, fam="spine_pods"):
    """scores = [scout, sniper, bulwark, blaster, bulwark_mender, mender_only]"""
    return {"id": f"x{i}", "family": fam, "set": "TUNE-train", "single": [{"score": s} for s in (*scores[:4], scores[5])], "template_bulwark_mender": {"score": scores[4]}}


def test_g3_per_layout_niche_counts_margin_and_share_threshold():
    rows = []
    for i in range(100):
        sc = [0.20, 0.20, 0.20, 0.20, 0.20, 0.0]
        if i < 10:   sc[0] = 0.40    # scout wins by 0.10 on 10 layouts
        elif i < 20: sc[1] = 0.45    # sniper wins by 0.15 on 10
        elif i < 30: sc[2] = 0.50    # bulwark on 10
        elif i < 38: sc[3] = 0.50    # blaster on only 8 -> share exactly 8 %
        elif i < 45: sc[4] = 0.50    # bulwark_mender (Mender's template) on 7 -> 7 % < 8 %
        elif i < 60: sc[0] = 0.215                  # scout wins by only 0.015: no niche for anyone
        rows.append(_fake_row(i, sc))
    g = G.g3(rows, TH)
    ns = g["niche_share"]
    assert ns["sniper"]["layouts"] == 10 and ns["bulwark"]["layouts"] == 10 and ns["scout"]["layouts"] == 10
    assert ns["blaster"]["layouts"] == 8 and ns["blaster"]["mean"] == pytest.approx(0.08) and ns["blaster"]["pass"]
    assert ns["mender"]["layouts"] == 7 and not ns["mender"]["pass"] and not g["pass"] and g["units_without_niche"] == ["mender"]
    assert G.TEMPLATES[:5] == ("scout", "sniper", "bulwark", "blaster", "bulwark_mender")


def test_g3_passes_when_every_unit_has_a_niche_and_lower_bound_is_positive():
    rows = []
    for i in range(100):
        sc = [0.10] * 5 + [0.0]
        sc[i % 5] = 0.40 if i < 90 else 0.10
        rows.append(_fake_row(i, sc))
    g = G.g3(rows, TH)
    assert g["pass"] and all(v["mean"] >= 0.08 and v["lo95"] > 0 for v in g["niche_share"].values())


def test_g3_ties_are_not_niches():
    rows = [_fake_row(i, [0.5, 0.5, 0.1, 0.1, 0.1, 0.0]) for i in range(50)]
    g = G.g3(rows, TH)
    assert all(v["layouts"] == 0 for v in g["niche_share"].values()) and not g["pass"]
