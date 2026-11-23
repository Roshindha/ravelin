"""Skip helpers for tests that need the three seed-0 checkpoints (not tracked in git; ``python scripts/fetch_checkpoints.py`` downloads and verifies them).
A test is skipped ONLY when a file is absent; with the files present it runs exactly as before."""
import pytest

from rl import explain as X

REASON = "checkpoints missing: run python scripts/fetch_checkpoints.py"
FILES = [X.MODELS / run / ("final.zip" if algo == "ppo" else "final.pt") for run, algo in X.DISPLAY_MODELS.values()]
GRID_ONLY = X.MODELS / "grid_ppo_plain_s0" / "model_identity.json"          # written by the training grid; not part of the release assets
GRID_REASON = "the full training grid (model_identity.json and the other 57 runs) is not part of the release: only the three seed-0 checkpoints are published"


def present() -> bool:
    return all(f.exists() for f in FILES)


needs_checkpoints = pytest.mark.skipif(not present(), reason=REASON)
needs_full_grid = pytest.mark.skipif(not (present() and GRID_ONLY.exists()), reason=GRID_REASON)


def require() -> None:
    """For fixtures: skip the dependent tests when a checkpoint is absent."""
    if not present():
        pytest.skip(REASON)
