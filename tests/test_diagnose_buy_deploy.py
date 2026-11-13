import json

import numpy as np
import pandas as pd
import pytest

from rl import diagnose_buy_deploy as B
from rl import eval_all as EA

MODEL = EA.MODELS_DIR / "ppo_none_s10" / "final.zip"


def test_layout_selection_never_touches_final_and_is_stratified():
    recs = B.select_layouts()
    final_ids = {r["id"] for r in EA.load_group_records(EA.HEADLINE_GROUPS)}
    assert not ({r["id"] for r in recs} & final_ids)
    assert all(r["pool"] == "DEV" and r["group"] in B.ALLOWED_GROUPS for r in recs)
    assert sum(r["group"] == "validation" for r in recs) == 20 and len(recs) == 60
    for fam in B.FAMILIES:
        assert sum(r["group"] == "DEV-train" and r["family"] == fam for r in recs) == 20
        assert not any(r["family"] == "ring_defence" for r in recs)
    assert [r["id"] for r in recs] == [r["id"] for r in B.select_layouts()]  # seeded


def test_loading_final_groups_is_not_requested_by_this_module():
    import inspect
    src = inspect.getsource(B)
    assert "FINAL-familiar" not in src.replace("FINAL-familiar / transfer layouts are NEVER", "") and '"transfer"' not in src
    assert "HEADLINE_GROUPS" not in src
    assert all("ALLOWED_GROUPS" in l for l in src.splitlines() if "EA.load_group_records(" in l)


def test_army_helpers():
    assert B.greedy_counts() == [2, 0, 6, 0]
    assert B.single_counts("ranger") == [0, 8, 0, 0] and B.single_counts("swarm") == [20, 0, 0, 0]
    assert B.names_to_counts(B.counts_to_names([3, 1, 2, 0])) == [3, 1, 2, 0]


def test_scripted_cells_match_existing_baselines():
    recs = B.select_layouts()
    bl = pd.read_csv(EA.RESULTS / "final_eval_tables" / "baselines_per_layout.csv").set_index("layout_id")
    from siege.baselines import _layout_for_record
    best = EA.best_single_troops(EA.load_group_records(B.ALLOWED_GROUPS))
    for r in recs[:12]:
        lay = _layout_for_record(r)
        g = B.scripted_cell(B.greedy_counts(), lay, r)["destruction"]
        s = B.scripted_cell(B.single_counts(best[r["family"]]), lay, r)["destruction"]
        assert g == pytest.approx(bl.loc[r["id"], "Greedy"]) and s == pytest.approx(bl.loc[r["id"], "BestSingle"])


@pytest.mark.skipif(not MODEL.exists(), reason="no model")
def test_forced_buy_matches_policy_own_buy_control_and_prebought_obs():
    from sb3_contrib import MaskablePPO
    from siege.env import SiegeEnv
    model = MaskablePPO.load(MODEL, device="cpu")
    env = SiegeEnv()
    rec = [r for r in B.select_layouts() if r["group"] == "validation"][0]
    own = B.ppo_episode(model, env, rec, None)
    again = B.ppo_episode(model, env, rec, own["bought"])  # force the same purchases
    assert again["destruction"] == pytest.approx(own["destruction"])
    # forced greedy buys give the same observation the phase-A prebought env showed (same state, grid)
    B.ppo_episode  # noqa
    e1, e2 = SiegeEnv(), SiegeEnv()
    o1, _ = e1.reset(seed=1, options={"layout_record": rec, "prebought": B.greedy_counts()})
    o2, _ = e2.reset(seed=1, options={"layout_record": rec, "prebought": False})
    for k, n in enumerate(B.greedy_counts()):
        for _ in range(n):
            o2, *_ = e2.step(k)
    o2, *_ = e2.step(244)
    assert np.array_equal(o1["grid"], o2["grid"]) and np.array_equal(o1["state"], o2["state"])
