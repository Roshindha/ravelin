"""B10 item 0: v2 modules never import the prototype-only files rl/augment_v1.py and rl/train_ppo.py."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTOTYPE_ONLY = {"augment_v1", "train_ppo"}            # module basenames of the prototype files
PROTOTYPE_FILES = {ROOT / "rl" / "augment_v1.py", ROOT / "rl" / "train_ppo.py"}


def _imported_modules(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = ("." * node.level) + (node.module or "")
            out.add(base)
            out |= {f"{base}.{a.name}" if base and not base.endswith(".") else f"{base}{a.name}" for a in node.names}
    return out


def _is_prototype(name: str) -> bool:
    return name.split(".")[-1] in PROTOTYPE_ONLY and (name.startswith("rl.") or name.startswith(".") or name in PROTOTYPE_ONLY)


def test_no_v2_module_imports_the_prototype_only_rl_files():
    v2 = [p for d in ("breachlens", "rl") for p in sorted((ROOT / d).glob("*.py")) if p not in PROTOTYPE_FILES]
    assert len(v2) > 20
    bad = {str(p.relative_to(ROOT)): sorted(m for m in _imported_modules(p) if _is_prototype(m)) for p in v2}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, f"v2 modules import prototype-only files: {bad}"


def test_the_checker_itself_detects_each_import_form(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("import rl.augment_v1\nfrom rl import train_ppo\nfrom .train_ppo import main\nfrom rl.augment_v1 import D4Wrapper\n")
    hits = sorted(m for m in _imported_modules(f) if _is_prototype(m))
    assert any(m == "rl.augment_v1" for m in hits) and any(m == "rl.train_ppo" for m in hits) and any(m == ".train_ppo" for m in hits)
