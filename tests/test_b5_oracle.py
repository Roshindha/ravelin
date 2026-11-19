"""B5: breachlens/oracle.py (one implementation for B5 and B8), scripted deployer helpers."""
import numpy as np
import pytest

from breachlens import layouts as Y
from breachlens import oracle as O
from breachlens.acceptance import warm_starts
from breachlens.actions import BORDER
from breachlens.armies import (army_cost, default_start, free_tiles, greedy_rank, greedy_types, place_on_border, random_types, single_unit_types,
                               sort_by_rank)
from breachlens.config import CFG
from breachlens.constants import BUDGET, MAX_UNITS
from breachlens.sim import Prepared, simulate

SETS = [(rec, L) for rec, L in Y.load_design_set("TUNE-train")][::36] + Y.load_design_set("LAB-DESIGN")[:2]


def test_place_on_border_is_consecutive_clockwise_from_the_start_tile():
    rec, L = SETS[0]
    s = 10
    army = place_on_border(L, [0, 1, 2, 3, 4], s)
    assert [t for _, t in army] == [10, 11, 12, 13, 14] and [u for u, _ in army] == [0, 1, 2, 3, 4]
    wrap = place_on_border(L, [0] * 4, 50)
    assert [t for _, t in wrap] == [50, 51, 0, 1]
    d = default_start(L)
    core = L["core"]["cell"]
    best = min(((BORDER[i][0] - core[0]) ** 2 + (BORDER[i][1] - core[1]) ** 2) for i in free_tiles(L))
    assert (BORDER[d][0] - core[0]) ** 2 + (BORDER[d][1] - core[1]) ** 2 == best


def test_single_unit_greedy_and_random_armies_are_legal():
    for t in range(5):
        a = single_unit_types(t)
        assert len(a) <= MAX_UNITS and army_cost(a) <= BUDGET and len(a) == min(MAX_UNITS, BUDGET // CFG.units[t].cost)
    assert army_cost(greedy_types()) <= BUDGET and len(greedy_types()) <= MAX_UNITS
    assert greedy_rank()[-1] == 4                     # a Mender (damage 0) ranks last
    rng = np.random.default_rng(1)
    for _ in range(200):
        a = random_types(rng)
        assert 1 <= len(a) <= MAX_UNITS and army_cost(a) <= BUDGET
        assert set(random_types(rng, allowed=(1, 3))) <= {1, 3}


def _legal(res, L):
    assert len(res.army) <= MAX_UNITS and res.gold <= BUDGET and len({t for _, t in res.army}) == len(res.army)
    assert all(t in free_tiles(L) for _, t in res.army)


def test_oracle_is_deterministic_legal_and_reproduces_its_own_score():
    for rec, L in SETS:
        P = Prepared(L)
        a = O.oracle_search(L, O.oracle_rng(rec["seed"]), prepared=P, warm_starts=warm_starts())
        b = O.oracle_search(L, O.oracle_rng(rec["seed"]), warm_starts=warm_starts())
        assert (a.score, a.types, a.start, a.sims) == (b.score, b.types, b.start, b.sims)
        _legal(a, L)
        assert simulate(P, list(a.army)).score == pytest.approx(a.score, abs=1e-12)   # sim float vs exact-integer key / den: may differ by 1 ulp
        assert a.sims <= O.N_RANDOM + O.N_REFINE + len(warm_starts())


def test_oracle_never_below_any_warm_start_and_banned_type_never_used():
    for rec, L in SETS:
        P = Prepared(L)
        res = O.oracle_search(L, O.oracle_rng(rec["seed"]), prepared=P, warm_starts=warm_starts())
        for types, start in warm_starts():
            army = place_on_border(L, list(types), start)
            assert res.score >= simulate(P, army).score - 1e-12     # exact ranking is by integer key; floats may differ by 1 ulp
        for ban in range(5):
            allowed = tuple(t for t in range(5) if t != ban)
            rb = O.oracle_search(L, O.oracle_rng(rec["seed"]), allowed=allowed, warm_starts=warm_starts(), prepared=P)
            assert ban not in rb.types
            _legal(rb, L)


def test_same_stream_pairing_and_distinct_tags():
    a, b = O.oracle_rng(7).random(5), O.oracle_rng(7).random(5)
    assert (a == b).all()
    assert not (O.oracle_rng(7).random(5) == O.oracle_rng(7, O.ARMY_SEARCH_TAG).random(5)).all()
    assert len({O.ORACLE_TAG, O.ARMY_SEARCH_TAG, O.ACCEPT_TAG}) == 3


def test_army_search_uses_scripted_deployment_and_its_budget():
    for rec, L in SETS:
        res = O.army_search(L, O.oracle_rng(rec["seed"], O.ARMY_SEARCH_TAG))
        assert res.start == default_start(L) and list(res.types) == sort_by_rank(res.types)
        assert res.sims <= O.AS_RANDOM + O.AS_REFINE
        _legal(res, L)
        res2 = O.army_search(L, O.oracle_rng(rec["seed"], O.ARMY_SEARCH_TAG))
        assert (res.score, res.types) == (res2.score, res2.types)


def test_best_of_search_is_monotone_in_budget_on_nested_streams():
    rec, L = SETS[1]
    small = O.oracle_search(L, O.oracle_rng(rec["seed"]), n_random=20, n_refine=0)
    big = O.oracle_search(L, O.oracle_rng(rec["seed"]), n_random=200, n_refine=0)
    assert big.score >= small.score          # the first 20 random draws are a prefix of the first 200


def test_oracle_module_is_the_only_search_implementation():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    hits = [p.name for p in list((root / "breachlens").glob("*.py")) + list(root.glob("scripts/*.py"))
            if "def oracle_search" in p.read_text() or "def army_search" in p.read_text()]
    assert hits == ["oracle.py"]
