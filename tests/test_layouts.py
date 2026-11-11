import numpy as np

from siege.bases import FAMILIES, generate_base_layout
from siege.env import SiegeEnv
from siege.layouts import (
    TRANSFER_FAMILY,
    acceptance_reason,
    build_layout_pool,
    generate_and_score,
    is_reachable,
    load_layout_pool,
)


def test_frozen_layouts_json_is_241_split_as_designed(tmp_path=None):
    records = load_layout_pool()
    assert len(records) == 241
    by_pool = {"DEV": 0, "FINAL": 0}
    by_tag = {"familiar": 0, "transfer": 0}
    for r in records:
        by_pool[r["pool"]] += 1
        by_tag[r["tag"]] += 1
    assert by_pool == {"DEV": 160, "FINAL": 81}
    assert by_tag == {"familiar": 200, "transfer": 41}
    ids = [r["id"] for r in records]
    assert sorted(ids) == list(range(241))


def test_transfer_family_is_entirely_final_and_never_dev():
    records = load_layout_pool()
    transfer_records = [r for r in records if r["family"] == TRANSFER_FAMILY]
    assert len(transfer_records) == 41
    assert all(r["pool"] == "FINAL" and r["tag"] == "transfer" for r in transfer_records)
    assert all(r["family"] != TRANSFER_FAMILY for r in load_layout_pool(pool="DEV"))


def test_dev_familiar_sampler_never_leaks_final_or_transfer_ids():
    # Guards the DEV/FINAL split against regressions: sampling many episodes from the
    # DEV-familiar pool must never produce a layout id that belongs to FINAL or to the
    # held-out transfer family.
    all_records = {r["id"]: r for r in load_layout_pool()}
    dev_ids = {r["id"] for r in load_layout_pool(pool="DEV", tag="familiar")}
    final_or_transfer_ids = {r["id"] for r in all_records.values() if r["id"] not in dev_ids}
    assert dev_ids and final_or_transfer_ids  # sanity: both sets are non-empty

    dev_records = [all_records[i] for i in dev_ids]
    env = SiegeEnv(layout_pool=dev_records)
    seen_ids = set()
    for seed in range(50):
        env.reset(seed=seed)
        record = env.layout_record
        assert record["id"] in dev_ids
        assert record["id"] not in final_or_transfer_ids
        assert record["pool"] == "DEV" and record["tag"] == "familiar"
        seen_ids.add(record["id"])
    assert seen_ids <= dev_ids


def test_env_with_layout_pool_reproduces_the_exact_recorded_layout():
    records = load_layout_pool(pool="FINAL")
    record = records[0]
    env = SiegeEnv(layout_pool=[record])
    env.reset(seed=0)
    expected = generate_base_layout(np.random.default_rng(record["seed"]), family=record["family"])
    assert env.layout["buildings"] == expected["buildings"]
    assert env.layout["turrets"] == expected["turrets"]
    assert env.layout["walls"] == expected["walls"]


def test_acceptance_rules_reject_empty_unreachable_too_hard_and_ranger_trivial():
    empty_layout = {"buildings": [], "turrets": [], "walls": {}}
    assert acceptance_reason(empty_layout, {"swarm": 0.0}) == "empty"

    overlapping_layout = {"buildings": [{"x": 5, "y": 5, "hp": 10, "value": 10}], "turrets": [], "walls": {(5, 5): 10}}
    assert is_reachable(overlapping_layout) is False
    assert acceptance_reason(overlapping_layout, {"swarm": 1.0}) == "unreachable"

    valid_layout = {"buildings": [{"x": 5, "y": 5, "hp": 10, "value": 10}], "turrets": [], "walls": {}}
    assert acceptance_reason(valid_layout, {name: 0.05 for name in ("swarm", "ranger", "tank", "splash", "mixed_balanced", "mixed_breach")}) == "too_hard"
    assert acceptance_reason(valid_layout, {"swarm": 0.05, "ranger": 1.0, "tank": 0.05, "splash": 0.05, "mixed_balanced": 0.05, "mixed_breach": 0.05}) == "ranger_trivial"
    assert acceptance_reason(valid_layout, {"swarm": 0.05, "ranger": 0.5, "tank": 0.05, "splash": 0.05, "mixed_balanced": 0.05, "mixed_breach": 0.05}) is None


def test_build_layout_pool_is_deterministic_given_same_seed_base():
    counts = {"ring_defence": 3, "clustered_core": 3, "scattered_outposts": 3}
    r1, rej1 = build_layout_pool(target_counts=counts, seed_base=42)
    r2, rej2 = build_layout_pool(target_counts=counts, seed_base=42)
    assert r1 == r2 and rej1 == rej2


def test_generate_and_score_covers_every_family():
    for family in FAMILIES:
        layout, scores = generate_and_score(0, family)
        assert layout["family"] == family
        assert set(scores) == {"swarm", "ranger", "tank", "splash", "mixed_balanced", "mixed_breach"}
        assert all(0.0 <= v <= 1.0 for v in scores.values())
