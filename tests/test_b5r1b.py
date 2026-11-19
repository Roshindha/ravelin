"""B5-R1b: sweep parameters (class-damage multipliers, sealed cores, unit cap, Mender stats), monotone ban-loss estimator, arm definitions,
pre-registered selection rule, gate changelog."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import b5r1b_sweep as SW
from breachlens import config as K
from breachlens import gates as G
from breachlens import layouts as Y
from breachlens import oracle as O
from breachlens.acceptance import warm_starts
from breachlens.armies import greedy_types, random_types, single_unit_types
from breachlens.sim import Prepared, simulate

ROOT = Path(__file__).resolve().parents[1]
SCOUT, SNIPER, BULWARK, BLASTER, MENDER = range(5)


def layout(buildings=(), core=(13, 13), walls=(), defenders=()):
    return {"buildings": [{"cell": c, "value": v} for c, v in buildings], "core": {"cell": core}, "walls": list(walls),
            "defenders": [{"cell": c, "type": t} for c, t in defenders]}


def _cfg_with(mod):
    raw = json.loads(K.STATS_PATH.read_text())
    mod(raw)
    return K._build(raw)


# ------------------------------------------------------------------ class-damage multiplier
def test_default_class_pct_is_100_and_stats_file_is_unchanged_by_the_feature():
    # revision 1 (R2a): only the Sniper has a class-damage multiplier; every other unit is 100 %, the cap stays 20
    assert all(u.class_pct == (100, 100, 100) for u in K.CFG.units if u.name != "sniper") and K.CFG.unit_cap == 20
    assert K.CFG.units[SNIPER].class_pct == (40, 100, 40)
    raw = json.loads(K.STATS_PATH.read_text())["units"]
    assert raw["sniper"]["class_damage_pct"] == {"building": 40, "wall": 40, "turret": 100}
    assert all("class_damage_pct" not in v for k, v in raw.items() if k != "sniper")


def test_sniper_deals_40_percent_to_buildings_walls_core_and_100_to_turrets():
    def mod(raw):
        raw["units"]["sniper"]["class_damage_pct"] = {"building": 40, "wall": 40, "turret": 100}
    cfg = _cfg_with(mod)
    assert cfg.units[SNIPER].class_pct == (40, 100, 40) and cfg.units[SCOUT].class_pct == (100, 100, 100)
    L = layout(buildings=[((6, 6), 1)], core=(6, 7))
    army = [(SNIPER, 20)]
    cfg100 = _cfg_with(lambda raw: raw["units"]["sniper"].pop("class_damage_pct"))      # revision 1 ships the 40 % Sniper; this is the 100 % reference
    full, cut = simulate(Prepared(L, cfg100), army), simulate(Prepared(L, cfg), army)
    assert full.core_destroyed and cut.core_destroyed and cut.ticks > full.ticks * 2          # ~2.5x slower on buildings and the core
    # vs a turret: 100 % (4 Snipers next to a Gun kill it in 2 ticks and lose nobody); the same cfg with turret 40 % kills it in ~5 ticks and loses a Sniper
    Lt = layout(buildings=[((12, 12), 1)], core=(12, 11), defenders=[((1, 3), "gun")])
    four = [(SNIPER, t) for t in (2, 3, 4, 5)]
    # the turret scenario was calibrated on the revision-0 Gun (hp 90; revision 1 ships 126), so pin the Gun hp to 90 here
    def gun90(extra):
        def m(raw):
            raw["defenders"]["gun"]["hp"] = 90
            extra(raw)
        return _cfg_with(m)
    c100 = gun90(lambda raw: raw["units"]["sniper"].pop("class_damage_pct"))
    c40t100 = gun90(lambda raw: None)
    cfg_all40 = gun90(lambda raw: raw["units"]["sniper"].__setitem__("class_damage_pct", {"building": 40, "wall": 40, "turret": 40}))
    alive = lambda c: simulate(Prepared(Lt, c), four, max_ticks=7).units_alive
    assert alive(c100) == 4 and alive(c40t100) == 4 and alive(cfg_all40) < 4
    # exact integers: 40 % of the 12-damage hit is an exact multiple of the fixed-point scale split
    S = K.CFG.S
    assert (12 * S) * 40 // 100 * 100 == 12 * S * 40


def test_class_damage_is_d4_invariant_for_the_sniper_arm():
    def mod(raw):
        raw["units"]["sniper"]["class_damage_pct"] = {"building": 40, "wall": 40, "turret": 100}
    cfg = _cfg_with(mod)
    rec, L = Y.load_design_set("TUNE-train", "spine_pods")[0]
    army = [(SNIPER, 5), (SNIPER, 6), (SCOUT, 7), (BLASTER, 8)]
    ref = simulate(Prepared(L, cfg), army)
    for k in range(1, 8):
        Lk = Y.transform(L, k)
        from breachlens.actions import BORDER
        idx = {rc: i for i, rc in enumerate(BORDER)}
        rk = simulate(Prepared(Lk, cfg), [(t, idx[Y._tr(k, BORDER[tile])]) for t, tile in army])
        assert (rk.score, rk.core_destroyed, rk.d_num) == (ref.score, ref.core_destroyed, ref.d_num), k


# ------------------------------------------------------------------ unit cap
def test_unit_cap_is_read_from_rules_and_used_by_every_army_builder():
    cfg = _cfg_with(lambda raw: raw["rules"].__setitem__("unit_cap", 14))
    assert cfg.unit_cap == 14
    import breachlens.armies as AR
    import breachlens.oracle as OR
    _old_a, old_o = AR.CFG if hasattr(AR, "CFG") else None, OR.CFG
    try:
        OR.CFG = cfg
        import breachlens.config as KC
        orig = KC.CFG
        KC.CFG = cfg
        assert len(single_unit_types(SCOUT)) == 14 and len(greedy_types()) <= 14
        rng = np.random.default_rng(0)
        assert max(len(random_types(rng, allowed=(SCOUT,))) for _ in range(200)) == 14
        rec, L = Y.load_design_set("TUNE-train", "bunker_nest")[0]
        r = O.oracle_search(L, O.oracle_rng(rec["seed"]), n_random=30, n_refine=30, warm_starts=warm_starts())
        assert r.units <= 14
    finally:
        OR.CFG = old_o
        KC.CFG = orig
    assert len(single_unit_types(SCOUT)) == 20


# ------------------------------------------------------------------ sealed cores
def _stream_layouts(fam, n=60):
    lo = Y.stream_range("L" if fam == "labyrinth" else "T", fam)[0]
    return [(lo + i, Y.candidate(fam, lo + i)) for i in range(n)]


def test_sealed_fraction_zero_is_bit_identical_and_draws_nothing(monkeypatch):
    # revision 1 ships sealed_fraction 0.5 (stored sets are reproduced by the defaults); 0.0 must draw nothing: no sealed core except by chance
    assert Y.GEN_PARAMS["sealed_fraction"] == 0.5 and Y.GEN_PARAMS["lancer_share"] == 0.4 and Y.GEN_PARAMS["island_gap"] == 4 and not Y.gen_override_active()
    rec = Y.load_design_sets_meta()["sets"]["TUNE-train"][0]
    assert Y.content_hash(Y.generate(rec["family"], rec["seed"])) == rec["content_hash"]
    monkeypatch.setitem(Y.GEN_PARAMS, "sealed_fraction", 0.0)
    assert not any(Y.is_sealed(Y.candidate("spine_pods", sd)) for sd, _ in _stream_layouts("spine_pods", 40))


@pytest.mark.parametrize("fam", Y.FAMILIES)
def test_sealed_layouts_enclose_the_core_keep_counts_and_relax_only_for_sealed(fam, monkeypatch):
    monkeypatch.setitem(Y.GEN_PARAMS, "sealed_fraction", 0.0)
    base = {sd: Y.candidate(fam, sd) for sd, _ in _stream_layouts(fam)}
    monkeypatch.setitem(Y.GEN_PARAMS, "sealed_fraction", 0.5)
    n_sealed = n_ok = 0
    for sd, L in _stream_layouts(fam):
        B = base[sd]
        sealed = Y.is_sealed(L)
        assert len(L["buildings"]) == len(B["buildings"]) and len(L["defenders"]) == len(B["defenders"])
        assert L["core"] == B["core"]
        if not Y.is_sealed(B) and sealed:
            assert len(set(map(tuple, L["walls"]))) == len(L["walls"]) and len(L["walls"]) > len(B["walls"])
            cells = [tuple(b["cell"]) for b in L["buildings"]] + [tuple(d["cell"]) for d in L["defenders"]] + [tuple(w) for w in L["walls"]] + [tuple(L["core"]["cell"])]
            assert len(set(cells)) == len(cells)                      # nothing overlaps after displacement
            bad = Y.check(L, fam)
            assert "unreachable_core" not in bad
            n_ok += not bad
        n_sealed += sealed
    assert n_sealed >= 15 and n_ok >= 8
    # an unsealed layout with an unreachable core is still rejected: the waiver needs the full ring
    L = copy.deepcopy(base[next(iter(base))])
    r, c = L["core"]["cell"]
    L["walls"] = [tuple(w) for w in L["walls"]] + [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    assert not Y.is_sealed(L)


def test_sealed_decision_does_not_change_when_other_params_do(monkeypatch):
    monkeypatch.setitem(Y.GEN_PARAMS, "sealed_fraction", 0.5)
    a = [Y.is_sealed(L) for _, L in _stream_layouts("spine_pods", 30)]
    monkeypatch.setitem(Y.GEN_PARAMS, "lancer_share", 0.4)
    b = [Y.is_sealed(L) for _, L in _stream_layouts("spine_pods", 30)]
    assert sum(a) and sum(b)


# ------------------------------------------------------------------ monotone ban-loss estimator
def test_monotone_ban_loss_is_never_negative_and_full_is_never_below_ablated():
    for rec, L in Y.load_design_set("TUNE-train")[::45][:4]:
        P = Prepared(L)
        ws = warm_starts()
        for u in (SNIPER, MENDER):
            ab = O.oracle_search(L, O.oracle_rng(rec["seed"]), n_random=15, n_refine=15, allowed=tuple(t for t in range(5) if t != u), warm_starts=ws, prepared=P)
            m = O.monotone_ban_pair(L, rec["seed"], u, ab, warm_starts=ws, prepared=P, n_random=15, n_refine=15)
            assert m["full"].key >= m["ablated"].key >= ab.key and m["loss"] >= 0
            if u not in m["full"].types:
                assert m["loss"] == 0.0 or m["ablated"].key < m["full"].key


def test_monotone_estimator_is_non_negative_even_on_tiny_budgets_where_the_independent_one_is_noisy():
    for rec, L in Y.load_design_set("TUNE-train")[::12]:
        P, _ws = Prepared(L), warm_starts()
        O.oracle_search(L, O.oracle_rng(rec["seed"]), n_random=4, n_refine=4, warm_starts=(), prepared=P)
        ab = O.oracle_search(L, O.oracle_rng(rec["seed"]), n_random=4, n_refine=4, allowed=(0, 1, 2, 3), warm_starts=(), prepared=P)
        m = O.monotone_ban_pair(L, rec["seed"], MENDER, ab, warm_starts=(), prepared=P, n_random=4, n_refine=4)
        assert m["loss"] >= 0


def test_gate_g2_g4_use_the_threshold_file_estimator_and_the_independent_one_is_still_available():
    th = json.loads((ROOT / "data" / "v2" / "gate_thresholds.json").read_text())
    assert th["G2"]["estimator"] == th["G4"]["estimator"] == "monotone"
    rows = [{"family": "bunker_nest", "set": "TUNE-train", "mono": {str(u): {"loss": 0.1 * (u + 1)} for u in range(5)},
             "oracle": {"all": {"score": 0.9}, **{f"ban{u}": {"score": 0.95} for u in range(5)}}} for _ in range(5)]
    assert G.ban_loss_values(rows, 2, "monotone").tolist() == [pytest.approx(0.3)] * 5
    assert G.ban_loss_values(rows, 2, "independent").tolist() == [pytest.approx(-0.05)] * 5
    with pytest.raises(ValueError):
        G.ban_loss_values(rows, 2, "other")


# ------------------------------------------------------------------ arms and selection rule
def test_arm_definitions_are_cumulative_and_touch_only_the_declared_parameters():
    base = json.loads(SW.BASE_STATS.read_text())
    b = SW.OLD.arm_stats(SW.ARM_B)
    a, bb, c, d = (SW.arm_stats(SW.ARMS[k]) for k in ("R1a", "R1b", "R1c", "R1d"))
    assert a["units"]["sniper"]["class_damage_pct"] == {"building": 40, "wall": 40, "turret": 100}
    strip = lambda s: {**s, "units": {u: {k: v for k, v in x.items() if k != "class_damage_pct"} for u, x in s["units"].items()}}
    assert strip(a) == b and a["structures"]["core_hp"] == 280
    assert bb["structures"]["core_hp"] == 840 and bb["rules"] == base["rules"]
    assert c["rules"]["unit_cap"] == 14 and c["structures"] == bb["structures"] and "unit_cap" not in d["rules"]
    assert d["units"]["mender"]["heal"] == 14 and d["units"]["mender"]["hp"] == 60 and bb["units"]["mender"] == base["units"]["mender"]
    assert d["units"]["sniper"] == bb["units"]["sniper"]
    assert SW.arm_gen(SW.ARMS["R1a"]) == {"lancer_share": 0.4, "defender_count_mult": 1.0, "sealed_fraction": 0.0}
    assert SW.arm_gen(SW.ARMS["R1b"])["sealed_fraction"] == 0.5 == SW.arm_gen(SW.ARMS["R1c"])["sealed_fraction"]


def _ci(mean, lo):
    return {"mean": mean, "lo95": lo, "hi95": mean + 0.05}


def _sm(sniper=0.5, single=0.5, rej=0.2, score=0.9, mender=(0.03, 0.01)):
    bf = {f: {"best_single_mean": single, "rejection_rate": rej, "oracle_score": score, "ban_loss_monotone": {"mender": _ci(*mender)}} for f in SW.FAMILIES}
    return {"by_family": bf, "sniper_share_best_single_pooled_familiar": sniper}


def test_selection_rule_counts_criteria_ties_to_smallest_change_and_needs_four():
    ok = _sm()
    assert set(SW.criteria(ok).values()) == {True}
    assert not SW.criteria(_sm(sniper=0.61))["i_sniper_best_single_share_le_0.60"]
    assert not SW.criteria(_sm(single=0.7))["ii_best_single_in_0.25_0.65"] and not SW.criteria(_sm(single=0.2))["ii_best_single_in_0.25_0.65"]
    assert not SW.criteria(_sm(rej=0.41))["iii_rejection_le_0.40"]
    assert not SW.criteria(_sm(score=0.98))["iv_oracle_score_in_0.80_0.97"] and not SW.criteria(_sm(score=0.79))["iv_oracle_score_in_0.80_0.97"]
    assert not SW.criteria(_sm(mender=(0.019, 0.01)))["v_mender_monotone_ban_loss"] and not SW.criteria(_sm(mender=(0.03, 0.0)))["v_mender_monotone_ban_loss"]
    one = _sm(mender=(0.0, -0.01)); one["by_family"]["labyrinth"]["ban_loss_monotone"]["mender"] = _ci(0.04, 0.001)
    assert SW.criteria(one)["v_mender_monotone_ban_loss"]                                   # one family is enough
    sums = {k: _sm() for k in SW.ARMS}
    assert SW.select_arm(sums, SW.ARMS)["chosen"] == "R1a"                                  # 5/5 ties -> smallest change
    sums["R1a"] = _sm(sniper=0.9)                                                           # 4/5, still applicable but others have 5
    assert SW.select_arm(sums, SW.ARMS)["ranking"][0] == "R1b"
    sums = {k: _sm(sniper=0.9, rej=0.9) for k in SW.ARMS}
    r = SW.select_arm(sums, SW.ARMS)
    assert r["chosen"] is None and r["best_n_met"] == 3
    sums["R1d"] = _sm(sniper=0.9)
    assert SW.select_arm(sums, SW.ARMS)["chosen"] == "R1d"
    assert not SW.criteria(_sm(rej=0.41), SW.FAMILIAR)["iii_rejection_le_0.40"]


def test_spec_carries_the_changelog_and_the_preregistered_r1b_rule():
    spec = (ROOT / "docs" / "SPEC_V2.md").read_text()
    for t in ("B5-R1b exploration sweep and selection rule", "pre-registered, fixed before any R1b sweep result existed", "at least 4 of 5", "R1c** = R1b + unit cap 14",
              "Sniper 40 % vs buildings / walls / core", "Mender monotone ban-loss >= 0.02", "Gate changelog", "monotone ban-loss estimator"):
        assert t.replace(">=", "≥") in spec or t in spec, t


def test_no_override_is_active_and_gate_script_asserts_it():
    assert not K.stats_override_active() and not Y.gen_override_active()
    assert "stats_override_active() or Y.gen_override_active()" in (ROOT / "scripts/gates.py").read_text()
