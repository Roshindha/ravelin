"""B4: contact sheets, one PNG per family (6 layouts each, with a legend) -> results/design/. TUNE-train / LAB-DESIGN only."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # the repo root is importable when run as `python scripts/<name>.py`
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from breachlens import layouts as Y
from breachlens.constants import GRID

OUT = Path(__file__).resolve().parents[1] / "results" / "design"
N = 6


def draw(ax, L, title):
    ax.set_xlim(-0.5, GRID - 0.5); ax.set_ylim(GRID - 0.5, -0.5); ax.set_aspect("equal")
    for r in range(GRID):
        for c in range(GRID):
            edge = r in (0, GRID - 1) or c in (0, GRID - 1)
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fc="#dbe7f3" if edge else "#f7f7f2", ec="#c9c9c9", lw=.4))
    for r, c in L["walls"]:
        ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fc="#4a4a4a", ec="#222", lw=.4))
    for b in L["buildings"]:
        r, c = b["cell"]
        ax.add_patch(Rectangle((c - .38, r - .38), .76, .76, fc=["#a9cdf0", "#5fa1de", "#1f6fbf"][b["value"] - 1], ec="#14406f", lw=.8))
        ax.text(c, r, str(b["value"]), ha="center", va="center", fontsize=7, color="white" if b["value"] == 3 else "black")
    r, c = L["core"]["cell"]
    ax.add_patch(Rectangle((c - .45, r - .45), .9, .9, fc="#f2b705", ec="black", lw=1.5))
    ax.text(c, r, "C", ha="center", va="center", fontsize=8, fontweight="bold")
    for d in L["defenders"]:
        r, c = d["cell"]
        m, col = {"gun": ("o", "#d62728"), "lancer": ("^", "#8e44ad"), "flak": ("D", "#e67e22")}[d["type"]]
        ax.plot(c, r, m, ms=8, mfc=col, mec="black", mew=.8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=7)


def pick(items, n=N, n_sealed=3):
    """B5-R2b: 6 layouts per sheet = the first `n_sealed` sealed-core layouts + the first unsealed ones (order kept within each group)."""
    sealed = [it for it in items if Y.is_sealed(it[1])][:n_sealed]
    rest = [it for it in items if not Y.is_sealed(it[1])]
    out = sealed + rest[:n - len(sealed)]
    return out + [it for it in items if it not in out][:n - len(out)]


def sheet(family, items, name):
    items = pick(items)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.6))
    for ax, (rec, L) in zip(axes.flat, items[:N]):
        draw(ax, L, f"{rec['id']}  [{rec['set']}]{'  SEALED CORE' if Y.is_sealed(L) else ''}")
    h = [Rectangle((0, 0), 1, 1, fc="#dbe7f3", ec="#c9c9c9"), Rectangle((0, 0), 1, 1, fc="#4a4a4a"),
         Rectangle((0, 0), 1, 1, fc="#f2b705", ec="black"),
         Rectangle((0, 0), 1, 1, fc="#a9cdf0", ec="#14406f"), Rectangle((0, 0), 1, 1, fc="#5fa1de", ec="#14406f"), Rectangle((0, 0), 1, 1, fc="#1f6fbf", ec="#14406f"),
         Line2D([], [], marker="o", ls="", mfc="#d62728", mec="k", ms=8), Line2D([], [], marker="^", ls="", mfc="#8e44ad", mec="k", ms=8),
         Line2D([], [], marker="D", ls="", mfc="#e67e22", mec="k", ms=8)]
    t = ["border (deploy tiles)", "wall (no score value)", "core", "building value 1", "building value 2", "building value 3",
         "Gun turret (r<=4)", "Lancer (2<=r<=6)", "Flak (2<=r<=5, area)"]
    fig.legend(h, t, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"{family}  -  {N} layouts of {items[0][0]['set']}, up to 3 sealed-core (design sets only; no FINAL)", fontsize=11)
    fig.tight_layout(rect=(0, .07, 1, .96))
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fam in Y.FAMILIAR:
        sheet(fam, Y.load_design_set("TUNE-train", fam), f"contact_sheet_{fam}.png")
    sheet("labyrinth", Y.load_design_set("LAB-DESIGN"), "contact_sheet_labyrinth.png")
    print(sorted(p.name for p in OUT.glob("*.png")))
