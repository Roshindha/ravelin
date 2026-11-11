from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


def render_board(state, ax=None, title: str = "Base Siege"):
    """Render a board snapshot from either a frame dict or a BattleResult."""
    if hasattr(state, "frames"):
        frame = state.frames[-1]
    else:
        frame = state

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    ax.set_facecolor("#eaf5ec")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 16)
    ax.set_aspect("equal")
    ax.set_xticks(range(16))
    ax.set_yticks(range(16))
    ax.grid(True, color="#d8e3da", linewidth=0.5)

    for wall in frame.get("walls", []):
        x, y = int(wall["x"]), int(wall["y"])
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#9c7b63", edgecolor="black", linewidth=0.6))
        ax.text(x + 0.5, y + 0.5, "W", ha="center", va="center", fontsize=8, color="white")

    for building in frame.get("buildings", []):
        x, y = int(building["x"]), int(building["y"])
        color = "#6b7280" if building.get("value", 1) > 20 else "#4b5563"
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="black", linewidth=0.8))
        ax.text(x + 0.5, y + 0.5, str(int(building["hp"])), ha="center", va="center", fontsize=8, color="white")

    for turret in frame.get("turrets", []):
        x, y = int(turret["x"]), int(turret["y"])
        ax.add_patch(Circle((x + 0.5, y + 0.5), 0.32, facecolor="#fbbf24", edgecolor="black"))

    colors = {"swarm": "#22c55e", "ranger": "#60a5fa", "tank": "#f97316", "splash": "#a78bfa"}
    for troop in frame.get("troops", []):
        x, y = int(troop["x"]), int(troop["y"])
        ax.add_patch(Circle((x + 0.5, y + 0.5), 0.28, facecolor=colors.get(troop["type"], "#1f2937"), edgecolor="black"))
        ax.text(x + 0.5, y + 0.5, troop["type"][0].upper(), ha="center", va="center", fontsize=8, color="white")

    ax.set_title(f"{title} (tick {frame.get('tick', 0)})")
    return ax


def save_battle_frames(battle, output_dir: str | os.PathLike[str], fps: int = 5):
    """Render a battle sequence and save one image per frame in a directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = getattr(battle, "frames", battle)
    paths = []
    for idx, frame in enumerate(frames):
        fig, ax = plt.subplots(figsize=(6, 6))
        render_board(frame, ax=ax, title=f"Battle frame {idx}")
        path = out_dir / f"frame_{idx:03d}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths
