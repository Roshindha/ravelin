"""B10 -> B13: the throughput side of the step-budget rule (docs/SPEC_V2.md 12.1 items 3-4 and the 12.2 sensitivity note)."""
from __future__ import annotations

WALL_CAP_H = 20          # W, hours for the PPO grid (12.1 item 3)
RUNS = 20                # R, PPO plain + PPO+D4 runs
PARALLEL = 6             # P
MIN_STEPS_PER_S = 100    # B2 derivation (section 11): >= 100 steps/s per PPO job at 6 parallel
T_ROUND = 250_000


def t_cap(r: float, wall_h: float = WALL_CAP_H, runs: int = RUNS, parallel: int = PARALLEL) -> float:
    """T_cap = W * 3600 * r / ceil(R / P)  (= 18 000 * r for the spec constants)."""
    waves = -(-runs // parallel)
    return wall_h * 3600.0 * r / waves


def throughput_verdict(r: float) -> dict:
    """Compare a measured per-job steps/s at P = 6 parallel with the spec: minimum 100 (B2), T_cap, and the stop rule (rounded T_cap < 1.0M)."""
    cap = t_cap(r)
    cap_rounded = int(cap // T_ROUND * T_ROUND)
    return {"r_per_job_at_6_parallel": r, "spec_minimum": MIN_STEPS_PER_S, "meets_spec_minimum": r >= MIN_STEPS_PER_S, "t_cap": cap,
            "t_cap_rounded_250k": cap_rounded, "below_90_cap_binds_early": r < 90,
            "stop_for_user_decision": cap_rounded < 1_000_000}
