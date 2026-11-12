#!/usr/bin/env python3
"""Status table of PPO grid runs: state, current timestep, latest + best validation score (full task).

    .venv/bin/python scripts/status.py [--prefix ppo] [--seeds 10-19] [--timesteps 1000000]
"""
import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def running_run_names() -> set[str]:
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    names = set()
    for line in out.splitlines():
        if "rl.train_ppo" in line and "--run-name" in line:
            toks = line.split()
            names.add(toks[toks.index("--run-name") + 1])
    return names


def run_info(name: str, running: set[str], target: int) -> dict:
    d = MODELS / name
    row = {"run": name, "state": "not started", "timestep": 0, "last_val": None, "best_val": None, "resumes": 0, "target": target}
    if not d.exists():
        return row
    ident = json.loads((d / "model_identity.json").read_text()) if (d / "model_identity.json").exists() else {}
    row["resumes"] = len(ident.get("resumes", []))
    target = row["target"] = ident.get("timesteps_requested", target)
    ts = [0]
    if (d / "val_eval.csv").exists():
        v = pd.read_csv(d / "val_eval.csv")
        v = v[v["mode"] == "full"].sort_values("timestep")
        if len(v):
            row["last_val"], row["best_val"] = float(v["overall"].iloc[-1]), float(v["overall"].max())
            ts.append(int(v["timestep"].iloc[-1]))
    ck = sorted(int(f.stem.split("_")[1]) for f in (d / "checkpoints").glob("ckpt_*_steps.zip")) if (d / "checkpoints").exists() else []
    ts += ck[-1:]
    row["timestep"] = max(ts)
    if (d / "final.zip").exists():
        row["state"], row["timestep"] = "finished", ident.get("timesteps_trained", target)
    elif name in running:
        row["state"] = "running"
    else:
        row["state"] = "interrupted"  # started, not running, no final.zip -> rerun the grid script to resume
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="ppo")
    p.add_argument("--seeds", default="10-19", help="e.g. 10-19 or 900,901")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    a = p.parse_args()
    if "-" in a.seeds:
        lo, hi = a.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in a.seeds.split(",")]
    running = running_run_names()
    rows = [run_info(f"{a.prefix}_{aug}_s{s}", running, a.timesteps) for s in seeds for aug in ("none", "d4")]
    df = pd.DataFrame(rows)
    df["progress"] = (df["timestep"] / df["target"] * 100).clip(upper=100).round(0).astype(int).astype(str) + "%"
    show = df[["run", "state", "timestep", "progress", "last_val", "best_val", "resumes"]].copy()
    for c in ("last_val", "best_val"):
        show[c] = show[c].map(lambda x: "-" if pd.isna(x) else f"{x:.3f}")
    print(show.to_string(index=False))
    print("\n" + ", ".join(f"{k}: {v}" for k, v in df["state"].value_counts().items()))


if __name__ == "__main__":
    main()
