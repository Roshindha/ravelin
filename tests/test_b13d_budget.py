"""B13d: the 12.1 step-budget rule on synthetic slopes (curve table, max over conditions, cap, rounding, stop rule)."""
import pytest

from rl.budget import step_budget, t_cap, t_curve


@pytest.mark.parametrize("s,t", [(-0.01, 1_000_000), (0.0, 1_000_000), (0.00499, 1_000_000), (0.005, 1_500_000), (0.00829, 1_500_000), (0.01499, 1_500_000),
                                 (0.015, 2_000_000), (0.02999, 2_000_000), (0.03, 3_000_000), (0.5, 3_000_000)])
def test_t_curve_edges(s, t):
    assert t_curve(s) == t


def test_max_over_conditions_and_cap_and_rounding():
    assert step_budget(0.0083, 0.0030, 184.3)["T"] == 1_500_000          # the B13d inputs
    assert step_budget(0.0030, 0.0083, 184.3)["s"] == 0.0083              # D4 slope can be the larger one
    b = step_budget(0.05, 0.0, 100.0)                                     # cap binds: 1.8M -> 1.75M
    assert b["T_cap"] == pytest.approx(1_800_000) and b["T"] == 1_750_000 and b["binding"] == "cap"
    assert t_cap(110) == pytest.approx(1_980_000)                         # spec example
    assert step_budget(0.05, 0.0, 50.0)["stop_for_user_decision"] is True   # cap 0.9M < 1.0M
    assert step_budget(0.05, 0.0, 56.0)["stop_for_user_decision"] is False  # 1.008M -> 1.0M
