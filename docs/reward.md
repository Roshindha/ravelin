# Reward specification

The reward is computed from the deterministic battle simulator after the army is fully deployed or when the agent explicitly waits during the battle phase.

The score is a normalized destruction measure:

- `reward = destroyed_building_value / total_building_value`
- clipped to the range `[0.0, 1.0]`

This means a full clear yields exactly `1.0` and an empty or already-destroyed base yields `0.0`.

Additional environment behavior:

- invalid actions are treated as safe no-ops so the Gymnasium checker can execute them without crashing
- legal action masks remain the policy-facing source of truth for gameplay decisions
- each environment step produces a finite scalar reward and cannot exceed a single clear-state reward
