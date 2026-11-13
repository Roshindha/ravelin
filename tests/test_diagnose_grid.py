import numpy as np
import pandas as pd
import pytest

from rl import diagnose_grid as D


def test_curve_stats_detects_rise_and_flat():
    ts = np.arange(0, 1_000_001, 25_000)
    rows = []
    for aug, slope in (("none", 0.0), ("d4", 0.4)):
        for s in range(3):
            rows.append(pd.DataFrame({"timestep": ts, "overall": 0.2 + slope * ts / 1e6, "aug": aug, "seed": s}))
    cs = D.curve_stats(pd.concat(rows)).set_index("aug")
    assert cs.loc["none", "mean_change_last_250k"] == pytest.approx(0, abs=1e-9)
    assert 0.08 < cs.loc["d4", "mean_change_last_250k"] < 0.12 and cs.loc["d4", "seeds_rising"] == 3


def test_provenance_matches_files():
    pv = D.provenance(pd.read_csv(D.RES / "final_eval.csv"))
    assert pv["training_commit_ok"] and pv["eval_commit_ok"] and pv["sha_ok"] and pv["n_models"] == 20


def test_per_seed_flags_resumed():
    fe = pd.read_csv(D.RES / "final_eval.csv")
    lc = D.learning_curves()
    ps = D.per_seed_table(fe, lc)
    assert len(ps) == 20 and set(ps[ps.resumed_twice].seed) == {13, 14, 15}
