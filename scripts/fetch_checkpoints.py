"""Download the three seed-0 checkpoints of the v1.0 release, verify them against the release's SHA256SUMS and place them where the code expects them.

    .venv/bin/python scripts/fetch_checkpoints.py            # -> models_v2/grid_ppo_plain_s0/final.zip, grid_ppo_d4_s0/final.zip, grid_ddqn_s0/final.pt

The only network use is downloading the four release assets (three checkpoints + SHA256SUMS) from github.com/Roshindha/ravelin. Nothing is placed unless every
hash matches. ``--src-dir`` uses already downloaded assets (no network); ``--dest`` changes the target directory (default ``models_v2/``, which is gitignored).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://github.com/Roshindha/ravelin/releases/download/v1.0/"
SUMS = "SHA256SUMS"
# release asset -> path under the models directory (rl/explain.py DISPLAY_MODELS / load_display_model)
ASSETS = {"grid_ppo_plain_s0.zip": "grid_ppo_plain_s0/final.zip", "grid_ppo_d4_s0.zip": "grid_ppo_d4_s0/final.zip", "grid_ddqn_s0.pt": "grid_ddqn_s0/final.pt"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sums(text: str) -> dict:
    out = {}
    for ln in text.splitlines():
        if ln.strip():
            digest, name = ln.split(None, 1)
            out[name.strip().lstrip("*")] = digest.lower()
    return out


def fetch(dest: Path, src_dir: Path | None = None) -> list:
    """Download (or read from ``src_dir``), verify, place. Returns the placed paths. Raises SystemExit on any hash mismatch before placing anything."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(src_dir) if src_dir else Path(tmp)
        if not src_dir:
            for name in (SUMS, *ASSETS):
                print(f"downloading {name}")
                urllib.request.urlretrieve(BASE + name, work / name)
        sums = parse_sums((work / SUMS).read_text())
        for name in ASSETS:
            if name not in sums:
                sys.exit(f"{SUMS} has no entry for {name}")
            got = sha256(work / name)
            if got != sums[name]:
                sys.exit(f"SHA256 mismatch for {name}: expected {sums[name]}, got {got}; nothing was placed")
            print(f"{name}: sha256 OK")
        placed = []
        for name, rel in ASSETS.items():
            target = Path(dest) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(work / name, target)
            placed.append(target)
            print(f"placed {target}")
        return placed


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", default=str(ROOT / "models_v2"))
    ap.add_argument("--src-dir", default=None, help="use already downloaded release assets instead of downloading")
    a = ap.parse_args(argv)
    fetch(Path(a.dest), Path(a.src_dir) if a.src_dir else None)


if __name__ == "__main__":
    main()
