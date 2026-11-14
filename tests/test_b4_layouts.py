"""B4: layout generators, structural acceptance, design sets (TUNE / LAB-DESIGN), seed-stream separation, labyrinth exactness."""
import copy
import json
import re
from pathlib import Path

import numpy as np
import pytest

from breachlens import _d4check as D
from breachlens import config
from breachlens import layouts as Y
from breachlens.armies import random_army, scripted_army
from breachlens.constants import GRID
from breachlens.sim import Prepared, simulate

ROOT = Path(__file__).resolve().parents[1]
META = Y.load_design_sets_meta()
SETS = {n: Y.load_design_set(n) for n in ("TUNE-train", "TUNE-val", "LAB-DESIGN")}
ALL = [(n, rec, L) for n, v in SETS.items() for rec, L in v]


def _fam(name, fam):
    return [L for rec, L in SETS[name] if rec["family"] == fam]


# ------------------------------------------------------------------ determinism, sizes
@pytest.mark.parametrize("fam", Y.FAMILIES)
def test_generation_is_deterministic_from_family_and_seed(fam):
    lo = Y.stream_range("L" if fam == "labyrinth" else "T", fam)[0]
    for s in range(lo, lo + 15):
        a, b = Y.candidate(fam, s), Y.candidate(fam, s)
        assert a == b
    assert any(Y.candidate(fam, lo) != Y.candidate(fam, lo + k) for k in range(1, 6))


def test_design_set_sizes_and_family_mix():
    assert {k: len(v) for k, v in SETS.items()} == {"TUNE-train": 180, "TUNE-val": 60, "LAB-DESIGN": 40}
    for fam in Y.FAMILIAR:
        assert len(_fam("TUNE-train", fam)) == 60 and len(_fam("TUNE-val", fam)) == 20
    assert all(rec["family"] == "labyrinth" for rec, _ in SETS["LAB-DESIGN"])
    ids = [rec["id"] for _, rec, _ in ALL]
    assert len(ids) == len(set(ids)) == 280
    assert META["counts"] == {"TUNE-train": 180, "TUNE-val": 60, "LAB-DESIGN": 40}


def test_design_sets_file_matches_regeneration():
    """B5: the committed sets are the structural stream walk minus the stat-dependent rejections (tests/test_b5_gates.py checks that
    difference exactly); every record regenerates from (family, seed) with the recorded content hash."""
    for name, rec, L in ALL:
        assert Y.candidate(rec["family"], rec["seed"]) == L and Y.content_hash(L) == rec["content_hash"]
    assert META["note"].startswith("B5")


def test_no_labyrinth_in_tune_and_no_familiar_in_lab_design():
    assert all(rec["family"] != "labyrinth" for rec, _ in SETS["TUNE-train"] + SETS["TUNE-val"])


# ------------------------------------------------------------------ acceptance rules
def test_every_layout_passes_every_acceptance_rule():
    for n, rec, L in ALL:
        assert Y.check(L, rec["family"]) == [], (rec["id"], Y.check(L, rec["family"]))


def test_core_and_every_building_reachable_without_breaking_walls_all_families():
    for n, rec, L in ALL:
        reach = Y.passable_reach(L)
        assert tuple(L["core"]["cell"]) in reach or Y.is_sealed(L), rec["id"]     # revision 1: ~50 % of layouts have a sealed core (the only waiver)
        assert all(tuple(b["cell"]) in reach for b in L["buildings"]), rec["id"]


def test_labyrinth_core_is_deep_and_the_path_is_long():
    from collections import deque
    for rec, L in SETS["LAB-DESIGN"]:
        if Y.is_sealed(L):      # revision 1: sealing only adds ring walls to the unsealed twin (same seed); the depth rule is measured on the twin
            old = Y.GEN_PARAMS["sealed_fraction"]
            Y.GEN_PARAMS["sealed_fraction"] = 0.0
            try:
                L = Y.candidate("labyrinth", rec["seed"])
            finally:
                Y.GEN_PARAMS["sealed_fraction"] = old
        walls = set(map(tuple, L["walls"]))
        ok = lambda r, c: 0 <= r < GRID and 0 <= c < GRID and (r, c) not in walls
        dist = {}
        dq = deque()
        for r in range(GRID):
            for c in range(GRID):
                if (r in (0, GRID - 1) or c in (0, GRID - 1)) and ok(r, c):
                    dist[(r, c)] = 0; dq.append((r, c))
        while dq:
            r, c = dq.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    n = (r + dr, c + dc)
                    if (dr or dc) and ok(*n) and n not in dist and (not dr or not dc or (ok(r + dr, c) and ok(r, c + dc))):
                        dist[n] = dist[(r, c)] + 1; dq.append(n)
        assert dist[tuple(L["core"]["cell"])] >= 12, rec["id"]     # needs real pathing, not a straight walk


def test_counts_value_weights_and_defender_mix_in_range():
    for n, rec, L in ALL:
        rc = dict(Y.COUNT_RANGES[rec["family"]])
        if Y.is_sealed(L):                    # revision 1: sealed layouts may carry the 8 ring walls on top of the family maximum
            rc["walls"] = (rc["walls"][0], rc["walls"][1] + 8)
        for key, v in (("buildings", len(L["buildings"])), ("defenders", len(L["defenders"])), ("walls", len(L["walls"])),
                       ("value_total", sum(b["value"] for b in L["buildings"]))):
            assert rc[key][0] <= v <= rc[key][1], (rec["id"], key, v)
        assert all(b["value"] in (1, 2, 3) for b in L["buildings"])
        assert L["core"] is not None and len(L["core"]["cell"]) == 2


@pytest.mark.parametrize("name", ["TUNE-train", "TUNE-val"])
def test_defender_mix_requirements_spec_3_1b(name):
    def share(fam, typ):
        Ls = _fam(name, fam)
        return sum(any(d["type"] == typ for d in L["defenders"]) for L in Ls) / len(Ls)
    # spec asks >= 70 % / >= 50 % on the layouts; the 20-layout val sets get the same bar
    assert share("spine_pods", "lancer") >= 0.7 and share("bunker_nest", "lancer") >= 0.7
    assert share("archipelago", "flak") >= 0.5
    for fam in Y.FAMILIAR:
        for L in _fam(name, fam):
            g = sum(d["type"] == "gun" for d in L["defenders"])
            assert g >= 0.5 * len(L["defenders"])


def test_labyrinth_uses_all_three_defender_types_everywhere():
    for rec, L in SETS["LAB-DESIGN"]:
        assert {"gun", "lancer", "flak"} <= {d["type"] for d in L["defenders"]}


def _good():
    return Y.candidate("bunker_nest", Y.stream_range("T", "bunker_nest")[0] + 5)


def test_rules_reject_bad_layouts(monkeypatch):
    monkeypatch.setitem(Y.GEN_PARAMS, "sealed_fraction", 0.0)    # the sealed-core waiver is off: a walled-in core must be rejected
    L = _good()
    assert Y.check(L, "bunker_nest") == []
    E = {"buildings": [], "core": {"cell": (6, 6)}, "walls": [], "defenders": []}
    assert "empty" in Y.check(E, "bunker_nest")
    # core walled in completely -> unreachable
    W = copy.deepcopy(L)
    cr, cc = W["core"]["cell"]
    W["walls"] = [w for w in W["walls"]] + [(cr + dr, cc + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                                            if (dr or dc) and (cr + dr, cc + dc) not in set(map(tuple, W["walls"]))
                                            and (cr + dr, cc + dc) not in {tuple(b["cell"]) for b in W["buildings"]}
                                            and (cr + dr, cc + dc) not in {tuple(d["cell"]) for d in W["defenders"]}]
    W["buildings"] = [b for b in W["buildings"] if max(abs(b["cell"][0] - cr), abs(b["cell"][1] - cc)) > 1]
    W["defenders"] = [d for d in W["defenders"] if max(abs(d["cell"][0] - cr), abs(d["cell"][1] - cc)) > 1]
    assert "unreachable_core" in Y.check(W, "bunker_nest")
    O = copy.deepcopy(L); O["walls"].append(tuple(O["core"]["cell"]))
    assert "overlap" in Y.check(O, "bunker_nest")
    G = copy.deepcopy(L)
    for d in G["defenders"]:
        d["type"] = "lancer"
    assert "gun_majority" in Y.check(G, "bunker_nest")
    B = copy.deepcopy(L); B["buildings"][0]["value"] = 5
    assert "value_weight" in Y.check(B, "bunker_nest")


def test_generate_raises_on_rejected_seed_and_the_stream_skips_it():
    rej = [s for s in range(Y.stream_range("T", "archipelago")[0], Y.stream_range("T", "archipelago")[0] + 120)
           if Y.check(Y.candidate("archipelago", s), "archipelago")]
    assert rej, "expected some rejected candidates in the archipelago stream"
    with pytest.raises(ValueError):
        Y.generate("archipelago", rej[0])
    assert rej[0] not in {r["seed"] for r in META["sets"]["TUNE-train"] + META["sets"]["TUNE-val"]}


def test_layout_module_is_stat_independent():
    src = (ROOT / "breachlens" / "layouts.py").read_text()
    code = "\n".join(l for l in src.splitlines() if l.lstrip().startswith(("import ", "from ")))
    assert "config" not in code and "sim" not in code and "unit_stats" not in src


def test_acceptance_does_not_change_when_stats_change():
    raw = json.loads(config.STATS_PATH.read_text())
    for u in raw["units"].values():
        u["hp"] *= 3
    before = Y.build_design_sets()
    config._build(raw)                      # a different stat set exists; the structural generators never look at it
    assert Y.build_design_sets() == before


# ------------------------------------------------------------------ seed streams, FINAL separation
def test_streams_T_L_F_are_disjoint_and_beyond_the_prototype_seeds():
    ranges = []
    for s in ("T", "L", "F"):
        for f in Y.FAMILIES:
            lo, hi = Y.stream_range(s, f)
            assert lo > Y.PROTOTYPE_SEED_MAX
            ranges.append((lo, hi, s, f))
    ranges.sort()
    for a, b in zip(ranges, ranges[1:]):
        assert a[1] <= b[0], (a, b)
    assert META["streams"]["T"]["ranges"]["spine_pods"] == list(Y.stream_range("T", "spine_pods"))
    assert {s: META["streams"][s]["base"] for s in "TLF"} == Y.STREAM_BASE


def test_design_seeds_lie_in_their_own_streams_and_never_in_F():
    for n, rec, L in ALL:
        want = "L" if n == "LAB-DESIGN" else "T"
        assert Y.stream_of(rec["seed"]) == want
        lo, hi = Y.stream_range(want, rec["family"])
        assert lo <= rec["seed"] < hi
    assert not any(Y.stream_of(rec["seed"]) == "F" for _, rec, _ in ALL)


def test_design_set_ids_seeds_and_content_hashes_are_unique_and_disjoint_from_every_pool_record():
    seeds = [rec["seed"] for _, rec, _ in ALL]
    hashes = [rec["content_hash"] for _, rec, _ in ALL]
    assert len(set(seeds)) == len(seeds) and len(set(hashes)) == len(hashes)    # incl. D4-mirrored duplicates
    pool_ids, pool_seeds, pool_hashes = set(), set(), set()
    for p in sorted((ROOT / "results").glob("layouts*.json")):           # the prototype pools (their seeds are < PROTOTYPE_SEED_MAX)
        d = json.loads(p.read_text())
        if isinstance(d, list):
            for r in d:
                pool_ids.add(r["id"]); pool_seeds.add(r["seed"])
    v2 = ROOT / "data" / "v2" / "layouts_v2.json"                         # the v2 pool, once B6 builds it
    if v2.exists():
        for r in json.loads(v2.read_text()):
            pool_ids.add(r["id"]); pool_seeds.add(r["seed"])
            if r["family"] in Y.FAMILIES and Y.stream_of(r["seed"]) == "F":
                pool_hashes.add(Y.content_hash(Y.candidate(r["family"], r["seed"])))
    assert not ({rec["id"] for _, rec, _ in ALL} & pool_ids)
    assert not (set(seeds) & pool_seeds)
    assert not (set(hashes) & pool_hashes)


def test_content_hash_is_d4_canonical_and_catches_a_regenerated_duplicate():
    L = _good()
    assert len({Y.content_hash(Y.transform(L, k)) for k in range(8)}) == 1
    h = {rec["content_hash"] for _, rec, _ in ALL}
    dup = Y.transform(SETS["TUNE-train"][0][1], 3)       # same layout, different frame, "different seed"
    assert Y.content_hash(dup) in h


def test_final_access_ledger_has_no_entries_from_this_step(tmp_path):
    ledger = ROOT / "data" / "v2" / "final_access.log"
    assert (not ledger.exists()) or ledger.read_text().strip() == ""
    bad = copy.deepcopy(META)                              # the loader refuses a stream-F seed
    rec = bad["sets"]["TUNE-train"][0]
    rec["seed"] = Y.stream_range("F", "spine_pods")[0]
    p = tmp_path / "ds.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(PermissionError):
        Y.load_design_set("TUNE-train", "spine_pods", path=p)
    assert (not ledger.exists()) or ledger.read_text().strip() == ""


# ------------------------------------------------------------------ contact sheets
def test_contact_sheets_render_for_every_family(tmp_path, monkeypatch):
    import scripts_b4_contact_sheets as C
    monkeypatch.setattr(C, "OUT", tmp_path)
    for fam in Y.FAMILIAR:
        C.sheet(fam, Y.load_design_set("TUNE-train", fam), f"{fam}.png")
    C.sheet("labyrinth", Y.load_design_set("LAB-DESIGN"), "labyrinth.png")
    for f in [*Y.FAMILIES]:
        assert (tmp_path / f"{f}.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    for f in Y.FAMILIES:
        assert (ROOT / "results" / "design" / f"contact_sheet_{f}.png").exists()


# ------------------------------------------------------------------ labyrinth D4-exactness and no-deadlock (spec: movement tests, B4)
def test_labyrinth_d4_exact_2000_cases_x_8_symmetries_incl_positions_and_shuffles():
    a = D.run_parallel_real("labyrinth", 2000, 4401, False, with_trace=True)
    assert a["n"] == 2000 and a["sims"] == 16000
    assert a["bad"] == [] and a["trace_bad"] == [], (a["bad"][:3], a["trace_bad"][:3])
    assert a["menders"] >= 0.5 * a["n"]
    assert a["fallback_calls"] > 0                          # the tie path is genuinely exercised
    b = D.run_parallel_real("labyrinth", 500, 4402, True, with_trace=True)    # adversarial id / order shuffles
    assert b["bad"] == [] and b["trace_bad"] == [] and b["fallback_calls"] > 0


@pytest.mark.parametrize("fam", Y.FAMILIAR)
def test_familiar_families_d4_exact_on_real_layouts(fam):
    a = D.run_parallel_real(fam, 300, 4410 + Y.FAMILIES.index(fam), False, with_trace=True)
    assert a["bad"] == [] and a["trace_bad"] == []
    b = D.run_parallel_real(fam, 100, 4420 + Y.FAMILIES.index(fam), True, with_trace=True)
    assert b["bad"] == [] and b["trace_bad"] == []


def test_labyrinth_no_deadlock_random_and_scripted_armies():
    rng = np.random.default_rng(4403)
    P = [(rec, L, Prepared(L)) for rec, L in SETS["LAB-DESIGN"]]
    stale_live_attacker = idle = 0
    for i in range(2000):
        rec, L, pre = P[int(rng.integers(len(P)))]
        army = random_army(rng, L, min_units=1, max_units=20, n_types=5)
        r = simulate(pre, army)
        idle += r.max_idle_run > 2          # global invariant: no unit with a live target idles > 2 consecutive ticks
        alive_attacker = any(t != 4 and hp > 0 for (t, _), hp in zip(army, r.unit_hp))
        stale_live_attacker += (r.reason == "stalemate" and alive_attacker and not r.core_destroyed)
    for rec, L, pre in P:
        r = simulate(pre, scripted_army(L))
        idle += r.max_idle_run > 2
        stale_live_attacker += (r.reason == "stalemate" and any(t != 4 and hp > 0 for (t, _), hp in zip(scripted_army(L), r.unit_hp)) and not r.core_destroyed)
    assert idle == 0 and stale_live_attacker == 0


def test_every_labyrinth_layout_can_be_breached_by_some_single_unit_walk():
    """Sanity (not a balance gate): a lone Bulwark-type fixed army makes progress, i.e. some building dies or a wall breaks."""
    for rec, L in SETS["LAB-DESIGN"]:
        r = simulate(L, [(2, i) for i in (0, 1, 2)] + [(1, 3), (1, 4)])
        assert r.ticks > 0 and (r.d_num > 0 or r.walls_destroyed > 0 or r.defenders_destroyed > 0)


# ------------------------------------------------------------------ B4 report file (full-scale run, scripts_b4_report.py)
def test_b4_report_is_current_and_meets_the_step_requirements():
    rep = json.loads((ROOT / "data" / "v2" / "b4_report.json").read_text())
    assert rep["sim_version"] == config.sim_version(), "stats or simulator changed: rerun scripts_b4_report.py"
    plain = [r for r in rep["d4"] if r["mode"] == "plain"]
    shuf = [r for r in rep["d4"] if r["mode"] == "shuffle"]
    assert sum(r["battles"] for r in plain) >= 2000 and all(r["symmetries"] == 8 for r in rep["d4"])
    assert sum(r["battles"] for r in plain if r["family"] == "labyrinth") >= 500
    assert sum(r["battles"] for r in shuf) >= 500
    assert sum(r["armies_with_mender"] for r in plain) >= 0.5 * sum(r["battles"] for r in plain)
    assert all(r["mismatches"] == 0 and r["trace_mismatches"] == 0 for r in rep["d4"])
    assert set(rep["battle_lengths"]) == {f"{f}/{p}" for f in Y.FAMILIES for p in ("random", "scripted")}


def test_speed_json_was_measured_on_the_real_layouts():
    d = json.loads((ROOT / "data" / "v2" / "speed.json").read_text())
    assert "real design layouts" in d["layouts"] and d["passed"] is True
