"""B1: scaffold, constants, action codec, border tiles (docs/SPEC_V2.md section 11, B1)."""
import re
from pathlib import Path

import pytest

from breachlens import actions, constants as C

SPEC = Path(__file__).resolve().parents[1] / "docs" / "SPEC_V2.md"


def _spec_param(name: str) -> str:
    text = SPEC.read_text()
    m = re.search(rf"^\| {re.escape(name)} \| (.+?) \|$", text, re.M)
    assert m, f"parameter {name!r} not found in SPEC_V2.md section 2"
    return m.group(1)


def test_border_has_52_unique_tiles_all_on_the_edge():
    b = actions.BORDER
    assert len(b) == 52 == C.N_BORDER
    assert len(set(b)) == 52
    for r, c in b:
        assert 0 <= r < C.GRID and 0 <= c < C.GRID
        assert r in (0, C.GRID - 1) or c in (0, C.GRID - 1)
    # every edge cell is present
    edge = {(r, c) for r in range(C.GRID) for c in range(C.GRID) if r in (0, C.GRID - 1) or c in (0, C.GRID - 1)}
    assert set(b) == edge


def test_border_is_clockwise_from_top_left_and_contiguous():
    b = actions.BORDER
    assert b[0] == (0, 0)
    assert b[C.GRID - 1] == (0, C.GRID - 1)            # end of top row
    assert b[2 * C.GRID - 2] == (C.GRID - 1, C.GRID - 1)  # bottom-right corner
    assert b[3 * C.GRID - 3] == (C.GRID - 1, 0)        # bottom-left corner
    assert b[-1] == (1, 0)                              # last tile just below top-left
    # consecutive tiles (cyclically) are 4-neighbours
    for i in range(len(b)):
        (r1, c1), (r2, c2) = b[i], b[(i + 1) % len(b)]
        assert abs(r1 - r2) + abs(c1 - c2) == 1
    # clockwise: signed area (row down, col right) is positive with y=row, x=col
    area = sum(b[i][1] * b[(i + 1) % 52][0] - b[(i + 1) % 52][1] * b[i][0] for i in range(52))
    assert area > 0


def test_action_constants_layout():
    assert C.N_ACTIONS == 267 == 5 + 5 * 52 + 1 + 1
    assert (C.DEPLOY_BASE, C.END_BUY, C.WAIT) == (5, 265, 266)
    assert C.N_UNIT_TYPES == 5 and C.UNIT_TYPES == ("scout", "sniper", "bulwark", "blaster", "mender")


def test_codec_is_a_bijection_over_all_267_actions():
    seen = set()
    for a in range(C.N_ACTIONS):
        kind, u, t = actions.decode_action(a)
        if kind == "buy":
            assert actions.encode_buy(u) == a and t is None
        elif kind == "deploy":
            assert actions.encode_deploy(u, t) == a
        elif kind == "end_buy":
            assert a == C.END_BUY
        else:
            assert kind == "wait" and a == C.WAIT
        seen.add((kind, u, t))
    assert len(seen) == 267
    kinds = [k for k, _, _ in (actions.decode_action(a) for a in range(267))]
    assert kinds.count("buy") == 5 and kinds.count("deploy") == 260
    # every (unit, tile) pair is reachable exactly once
    pairs = {(u, t) for k, u, t in (actions.decode_action(a) for a in range(267)) if k == "deploy"}
    assert pairs == {(u, t) for u in range(5) for t in range(52)}


@pytest.mark.parametrize("bad", [-1, 267, 1000])
def test_decode_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        actions.decode_action(bad)


@pytest.mark.parametrize("u,t", [(-1, 0), (5, 0), (0, -1), (0, 52)])
def test_encode_deploy_rejects_out_of_range(u, t):
    with pytest.raises(ValueError):
        actions.encode_deploy(u, t)


@pytest.mark.parametrize("u", [-1, 5])
def test_encode_buy_rejects_out_of_range(u):
    with pytest.raises(ValueError):
        actions.encode_buy(u)


def test_constants_match_spec_document():
    assert _spec_param("Grid") == f"{C.GRID} × {C.GRID}"
    assert int(re.match(r"(\d+) gold", _spec_param("Budget")).group(1)) == C.BUDGET == 240
    assert int(re.match(r"(\d+) ticks", _spec_param("Battle horizon")).group(1)) == C.MAX_BATTLE_TICKS == 240
    assert int(_spec_param("Max units")) == C.MAX_UNITS == 20
    assert int(re.match(r"(\d+)", _spec_param("Border tiles")).group(1)) == C.N_BORDER
    assert int(_spec_param("Unit types")) == C.N_UNIT_TYPES
    assert f"Discrete({C.N_ACTIONS})" in SPEC.read_text()
    assert (C.N_GRID_CHANNELS, C.N_STATE) == (9, 11)
