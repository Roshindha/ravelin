"""B5-R1: generator parameters (Lancer share, defender count), sweep arm definitions and the pre-registered selection rule."""
import json
import os
from pathlib import Path

import pytest

import scripts_b5r1_sweep as SW
from breachlens import config as K
from breachlens import layouts as Y

ROOT = Path(__file__).resolve().parents[1]


def test_no_override_is_active_in_the_test_process():
    assert not K.stats_override_active() and not Y.gen_override_active()
    assert K.STATS_PATH == ROOT / "data" / "v2" / "unit_stats.json"


def _stream(fam, n=120):
    lo = Y.stream_range("L" if fam == "labyrinth" else "T", fam)[0]
    return [(lo + i, Y.candidate(fam, lo + i)) for i in range(n)]


def _defs(fam, n=120):
    out = []
    for sd, L in _stream(fam, n):
        if not Y.check(L, fam):
            out.append([d["type"] for d in L["defenders"]])
    return out


def test_lancer_share_param_gives_about_forty_percent_and_keeps_gun_majority(monkeypatch):
    monkeypatch.setitem(Y.GEN_PARAMS, "lancer_share", 0.4)
    for fam in ("spine_pods", "bunker_nest"):
        d = _defs(fam)
        assert len(d) > 80
        share = sum(t.count("lancer") for t in d) / sum(len(t) for t in d)
        assert 0.33 <= share <= 0.45, (fam, share)
        assert all(t.count("gun") >= 0.5 * len(t) for t in d)
    d = _defs("labyrinth")
    assert all({"gun", "lancer", "flak"} <= set(t) for t in d)


def test_defender_count_mult_scales_counts_and_the_accepted_range(monkeypatch):
    base = {f: _defs(f) for f in Y.FAMILIES}
    monkeypatch.setitem(Y.GEN_PARAMS, "defender_count_mult", 1.25)
    for fam, b in base.items():
        d = _defs(fam)
        ratio = (sum(map(len, d)) / len(d)) / (sum(map(len, b)) / len(b))
        assert 1.15 <= ratio <= 1.35, (fam, ratio)
        lo, hi = Y.COUNT_RANGES[fam]["defenders"]
        assert max(map(len, d)) <= -(-hi * 5 // 4)


def test_default_generator_params_are_the_current_revision_and_layouts_unchanged_when_neutral(monkeypatch):
    meta = Y.load_design_sets_meta()
    rec = meta["sets"]["TUNE-train"][0]
    h = Y.content_hash(Y.generate(rec["family"], rec["seed"]))
    monkeypatch.setitem(Y.GEN_PARAMS, "defender_count_mult", 1.0)
    assert Y.content_hash(Y.generate(rec["family"], rec["seed"])) == h == rec["content_hash"]


def test_arm_stats_change_only_hp_structures_and_sniper_cost():
    base = json.loads(SW.BASE_STATS.read_text())
    for name, arm in SW.ARMS.items():
        s = SW.arm_stats(arm)
        assert s["structures"]["wall_hp"] == base["structures"]["wall_hp"] == 80
        for u in base["units"]:        # attack stats, ranges, speeds, armor, hp of units never change
            for k, v in base["units"][u].items():
                if not (u == "sniper" and k == "cost" and arm["sniper_cost"]):
                    assert s["units"][u][k] == v, (name, u, k)
        for d, v in base["defenders"].items():
            assert {k: x for k, x in s["defenders"][d].items() if k != "hp"} == {k: x for k, x in v.items() if k != "hp"}
            assert s["defenders"][d]["hp"] == round(v["hp"] * arm["hp"])
        assert s["structures"]["core_hp"] == round(base["structures"]["core_hp"] * arm["hp"])
        assert s["structures"]["building_hp_default"] == round(base["structures"]["building_hp_default"] * arm["hp"])
        assert s["rules"] == base["rules"] and s["fixed_point_scale"] == base["fixed_point_scale"]
    assert SW.arm_stats(SW.ARMS["C"])["units"]["sniper"]["cost"] == 28 and SW.arm_stats(SW.ARMS["B"])["units"]["sniper"]["cost"] == 24
    assert SW.arm_stats(SW.ARMS["E"])["defenders"]["gun"]["hp"] == 162 and SW.arm_stats(SW.ARMS["A"])["defenders"]["gun"]["hp"] == 126
    assert SW.arm_gen(SW.ARMS["D"]) == {"lancer_share": 0.4, "defender_count_mult": 1.25} and SW.arm_gen(SW.ARMS["A"])["lancer_share"] is None


def _sm(breach=0.9, score=0.9, single=0.5, greedy=0.3, rej=0.2, sniper=0.5, bad=()):
    bf = {f: {"oracle_breach": breach, "oracle_score": score, "best_single_mean": single, "greedy_score": greedy, "rejection_rate": rej} for f in SW.FAMILIES}
    return {"by_family": bf, "sniper_share_best_single_pooled_familiar": sniper}


def test_selection_rule_counts_criteria_breaks_ties_by_fewest_changes_and_requires_four():
    all_ok = _sm()
    assert set(SW.criteria(all_ok).values()) == {True}
    assert not SW.criteria(_sm(breach=0.96))["1_oracle_not_saturated"] and not SW.criteria(_sm(score=0.98))["1_oracle_not_saturated"]
    assert not SW.criteria(_sm(single=0.7))["2_best_single_in_0.25_0.65"] and not SW.criteria(_sm(single=0.2))["2_best_single_in_0.25_0.65"]
    assert not SW.criteria(_sm(sniper=0.61))["3_sniper_best_single_share_le_0.60"] and not SW.criteria(_sm(rej=0.41))["4_rejection_rate_le_0.40"]
    assert not SW.criteria(_sm(greedy=0.46))["5_greedy_labyrinth_le_0.45"]
    sums = {"A": _sm(), "B": _sm(), "C": _sm(), "D": _sm(), "E": _sm()}
    assert SW.select_arm(sums, SW.ARMS)["chosen"] == "A"                         # all tie at 5/5 -> fewest changes (A = 1)
    sums["A"] = _sm(sniper=0.9)                                                    # A 4/5
    assert SW.select_arm(sums, SW.ARMS)["ranking"][0] == "B"                      # B, D, E are 5/5 with 2 changes -> earliest letter
    sums = {k: _sm(sniper=0.9, rej=0.9) for k in "ABCDE"}                          # everyone 3/5
    assert SW.select_arm(sums, SW.ARMS)["chosen"] is None
    sums["D"] = _sm(sniper=0.9)                                                    # 4/5: applied
    r = SW.select_arm(sums, SW.ARMS)
    assert r["chosen"] == "D" and r["best_n_met"] == 4
    one = _sm(); one["by_family"]["labyrinth"]["rejection_rate"] = 0.5            # a single family over 40 % fails criterion 4 ("per family")
    assert not SW.criteria(one)["4_rejection_rate_le_0.40"] and SW.criteria(one, SW.FAMILIAR)["4_rejection_rate_le_0.40"]


def test_selection_rule_text_is_in_the_spec():
    spec = (ROOT / "docs" / "SPEC_V2.md").read_text()
    assert "B5-R1 exploration sweep and selection rule" in spec and "fewest changed parameters" in spec and "at least 4 of 5" in spec
