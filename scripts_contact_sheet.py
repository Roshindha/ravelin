"""Render a contact sheet: 12 sample layouts per family from the frozen pool (results/layouts.json).

Usage: .venv/bin/python scripts_contact_sheet.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from siege.bases import FAMILIES, generate_base_layout
from siege.layouts import load_layout_pool
from siege.render import render_board

N_SAMPLES = 12
OUT_PATH = "results/layout_contact_sheet.png"


def layout_to_frame(layout: dict) -> dict:
    return {
        "tick": 0,
        "troops": [],
        "buildings": layout["buildings"],
        "turrets": layout["turrets"],
        "walls": [{"x": x, "y": y, "hp": hp} for (x, y), hp in layout["walls"].items()],
    }


def main() -> None:
    records = load_layout_pool()
    by_family = {family: [r for r in records if r["family"] == family][:N_SAMPLES] for family in FAMILIES}

    fig, axes = plt.subplots(len(FAMILIES), N_SAMPLES, figsize=(2.2 * N_SAMPLES, 2.2 * len(FAMILIES)))
    for row, family in enumerate(FAMILIES):
        for col, record in enumerate(by_family[family]):
            layout = generate_base_layout(np.random.default_rng(record["seed"]), family=family)
            ax = axes[row, col]
            render_board(layout_to_frame(layout), ax=ax, title=f"{family}\nid={record['id']} pool={record['pool']}")
            ax.set_title(f"id={record['id']} {record['pool']}", fontsize=7)
            if col == 0:
                ax.set_ylabel(family, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=110)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
