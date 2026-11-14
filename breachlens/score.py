"""The ONE canonical score (docs/SPEC_V2.md 1, 3.4): score = 0.6 * D + 0.4 * C.

D = value-weighted share of non-core buildings destroyed (d_num / d_den, integers), C = 1 if the core is destroyed else 0.
The score is an exact integer ratio

    score = (6 * d_num + 4 * d_den * C) / (10 * d_den)          (d_den = 0 is read as 1: D = 0)

converted to float exactly once (Python's int / int true division is correctly rounded). The simulator (`BattleResult.score`), the env's
terminal reward, the Oracle / army-search ranking keys, the evaluator and every CSV / JSON writer call these functions; no other module
may contain the formula (a test greps for it).
"""
from __future__ import annotations


def score_ratio(d_num: int, d_den: int, core_destroyed: bool) -> tuple[int, int]:
    """Exact (numerator, denominator) of the score; the denominator is 10 * max(d_den, 1) (not reduced, so keys compare within one layout)."""
    den = max(int(d_den), 1)
    return 6 * int(d_num) + 4 * den * (1 if core_destroyed else 0), 10 * den


def ratio_to_float(num: int, den: int) -> float:
    """The single int -> float conversion (correctly rounded)."""
    return num / den


def score_value(d_num: int, d_den: int, core_destroyed: bool) -> float:
    return ratio_to_float(*score_ratio(d_num, d_den, core_destroyed))


def result_score(r) -> float:
    """Score of a BattleResult (or anything with d_num, d_den, core_destroyed)."""
    return score_value(r.d_num, r.d_den, r.core_destroyed)


def result_key(r) -> tuple[int, int]:
    """(numerator, denominator): an exact integer ranking key within one layout; score = key / den."""
    return score_ratio(r.d_num, r.d_den, r.core_destroyed)
