"""B2 simulator speed gate: speed.json (written by scripts/speed.py) is asserted against the SPEC_V2 gate."""
import json
import re
import time
from pathlib import Path

import numpy as np

from breachlens import config
from breachlens._bench_layouts import FAMILIES, make_layout, random_army
from breachlens.sim import simulate

ROOT = Path(__file__).resolve().parents[1]
SPEED = ROOT / "data" / "v2" / "speed.json"
SPEC = (ROOT / "docs" / "SPEC_V2.md").read_text()


def _spec_gate():
    m = re.search(r"mean battle time \*\*≤ (\d+) ms\*\* and p95 \*\*≤ (\d+) ms\*\*.*?\*\*≥ (\d+) steps/s\*\*", SPEC, re.S)
    assert m, "speed gate not found in SPEC_V2.md"
    return tuple(int(x) for x in m.groups())


def test_gate_thresholds_equal_the_spec():
    mean_ms, p95_ms, sps = _spec_gate()
    g = json.loads(SPEED.read_text())["gate"]
    assert (g["mean_ms_max"], g["p95_ms_max"], g["steps_per_s_min"]) == (mean_ms, p95_ms, sps)


def test_speed_json_is_current_and_meets_the_gate():
    d = json.loads(SPEED.read_text())
    assert d["sim_version"] == config.sim_version(), "stats or simulator changed: rerun scripts/speed.py"
    mean_ms, p95_ms, sps = _spec_gate()
    assert d["battle_warm"]["n"] >= 200 and d["battle_warm"]["mean_units"] >= 10
    assert d["battle_warm"]["mean_ms"] <= mean_ms
    assert d["battle_warm"]["p95_ms"] <= p95_ms
    assert d["battle_cold"]["mean_ms"] <= mean_ms          # even with cold per-layout caches
    assert d["env_1proc"]["steps_per_s"] >= sps
    assert d["env_6proc"]["steps_per_s"] >= 600           # SPEC: >= 100 steps/s per PPO job x 6
    assert d["passed"] is True


def test_live_battle_time_under_gate():
    mean_ms, _, _ = _spec_gate()
    rng = np.random.default_rng(11)
    ts = []
    for s in range(40):
        L = make_layout(FAMILIES[s % 4], s)
        a = random_army(rng, L, min_units=10)
        t0 = time.perf_counter()
        simulate(L, a)
        ts.append((time.perf_counter() - t0) * 1000)
    assert np.mean(ts) <= mean_ms
