"""B21: build data/app_bundle/ (docs/SPEC_V2.md, Gate changelog row (qq)). Refuses a dirty tree (provenance would be wrong), like the replay and duel-cache builders.

For the 30 DEV-validation layouts x the 3 seed-0 arms x {plain, TTA}: the live ``run_episode`` data of "Watch the AI" (per decision step top 5, chosen action,
heatmap, value, labels; battle frames; structures; outcome), one gzip JSON per layout plus ``manifest.json`` (provenance: code commit, model run names and
checkpoint sha256, sim_version, SCORE_VERSION; sha256 and size of every file). Needs torch and the models; the deployed app does not.

    .venv/bin/python scripts/build_app_bundle.py
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import make_replays as M  # noqa: E402
from app import bundle as B  # noqa: E402
from rl import explain as X  # noqa: E402


def live_episode(loaded: dict, arm: str, split: str, layout_id: str, tta: bool) -> dict:
    """The app's own ``episode()`` computation (app/streamlit_app.py), without streamlit."""
    rec, layout = X.load_dev_layout(split, layout_id=layout_id)
    model, algo, run = loaded[arm]
    ep = M.run_episode(model, algo, arm, run, rec, layout, tta)
    ep.update(arm=arm, run=run, algo=algo, tta=tta, layout_id=rec["id"], family=rec["family"], split=split)
    return ep


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out: Path = B.BUNDLE_DIR, allow_dirty: bool = False) -> dict:
    if M.tree_dirty() and not allow_dirty:
        raise SystemExit("refusing to build the app bundle from a dirty tree (provenance would be wrong); commit first or pass --allow-dirty for a scratch build")
    from breachlens import evaluator as EV
    out.mkdir(parents=True, exist_ok=True)
    ids = EV.dev_layout_ids(B.BUNDLE_SPLIT, entry_point="scripts.build_app_bundle")
    loaded = {arm: X.load_display_model(arm) for arm in X.DISPLAY_MODELS}
    models = {}
    for arm, (run, algo) in X.DISPLAY_MODELS.items():
        ck = X.MODELS / run / ("final.zip" if algo == "ppo" else "final.pt")
        models[arm] = {"run": run, "algo": algo, "checkpoint": ck.name, "sha256": _sha(ck)}
    files = {}
    for lid in ids:
        eps = {B.episode_key(arm, tta): live_episode(loaded, arm, B.BUNDLE_SPLIT, lid, tta) for arm in X.DISPLAY_MODELS for tta in (False, True)}
        f = B.layout_file(lid, out)
        with open(f, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=9) as gz:
            gz.write(json.dumps({"layout_id": lid, "episodes": eps}, separators=(",", ":"), sort_keys=True).encode())
        files[lid] = {"file": f.name, "bytes": f.stat().st_size, "sha256": _sha(f)}
        print(lid, f.stat().st_size, flush=True)
    manifest = {"version": B.BUNDLE_VERSION, "split": B.BUNDLE_SPLIT, "provenance": M.provenance(), "models": models, "arms": list(X.DISPLAY_MODELS),
                "views": ["plain", "tta"], "layouts": files, "total_bytes": sum(v["bytes"] for v in files.values())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=str(B.BUNDLE_DIR))
    p.add_argument("--allow-dirty", action="store_true", help="scratch builds only; the provenance then carries +dirty and must not be committed")
    a = p.parse_args(argv)
    m = build(Path(a.out), a.allow_dirty)
    print(f"bundle: {len(m['layouts'])} layouts, {m['total_bytes']} bytes (gzip), code {m['provenance']['code_commit'][:10]}")
    return m


if __name__ == "__main__":
    main()
