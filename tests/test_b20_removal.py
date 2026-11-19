"""B20: the prototype (siege/, rl/augment_v1.py, rl/train_ppo.py and the prototype rl/ modules, scripts and tests) is gone and nothing imports it.
It lives at tag prototype-v1 (ef43f63 before the 2026-11-19 history rewrite; see docs/commit_map.tsv). Names are matched whole, so rl.train_ppo_v2 does NOT trigger this."""
from __future__ import annotations

import ast
import importlib
import pkgutil
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()
GUARDS = {SELF, ROOT / "tests" / "test_b10_hygiene.py"}          # these name the removed modules on purpose

REMOVED = ["siege", "rl/augment_v1.py", "rl/train_ppo.py", "rl/eval.py", "rl/eval_all.py", "rl/eval_ppo.py", "rl/evaluation.py",
           "rl/policy_io.py", "rl/split.py", "rl/train.py", "rl/diagnose_buy_deploy.py", "rl/diagnose_grid.py",
           "scripts/balance.py", "scripts/contact_sheet.py", "scripts/diag_deploy.py", "scripts/rebuild_pool.py", "scripts/smoke.py",
           "scripts/sweep_splash.py", "scripts/tuning.py", "scripts/run_one.sh", "scripts/run_ppo_grid.sh"]
REMOVED_RL = {"augment_v1", "train_ppo", "eval", "eval_all", "eval_ppo", "evaluation", "policy_io", "split", "train",
              "diagnose_buy_deploy", "diagnose_grid"}
TEXT_PAT = re.compile(r"\b(rl\.train_ppo|rl\.augment_v1)\b(?!_)")


def _py_files():
    return [p for p in sorted(ROOT.rglob("*.py")) if not {".venv", "__pycache__", ".git"} & set(p.relative_to(ROOT).parts)]


def _is_removed(mod: str, level: int = 0) -> bool:
    parts = mod.split(".") if mod else []
    if level == 0:
        return parts[:1] == ["siege"] or (parts[:1] == ["rl"] and len(parts) > 1 and parts[1] in REMOVED_RL)
    return False


def _imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, 0
        elif isinstance(node, ast.ImportFrom):
            yield node.module or "", node.level
            if node.level == 0 and node.module == "rl":
                for a in node.names:
                    yield f"rl.{a.name}", 0
            if node.level == 1 and path.parent.name == "rl":           # from .train_ppo import x / from . import train_ppo
                names = [node.module] if node.module else [a.name for a in node.names]
                for n in names:
                    yield f"rl.{n}", 0


@pytest.mark.parametrize("rel", REMOVED)
def test_removed_paths_do_not_exist(rel):
    assert not (ROOT / rel).exists(), f"{rel} must stay removed (see tag prototype-v1)"


def test_no_python_file_imports_removed_modules():
    files = [p for p in _py_files() if p not in GUARDS]
    assert len(files) > 50
    bad = {str(p.relative_to(ROOT)): sorted({m for m, lv in _imports(p) if _is_removed(m, lv)}) for p in files}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, f"imports of removed prototype modules: {bad}"
    text_bad = [str(p.relative_to(ROOT)) for p in files if TEXT_PAT.search(p.read_text())]
    assert not text_bad, f"textual references to removed modules: {text_bad}"


def test_checker_matches_whole_names_only(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("import siege.env\nfrom rl import train_ppo\nfrom rl.train_ppo import main\nimport rl.augment_v1\n")
    assert {m for m, lv in _imports(f) if _is_removed(m, lv)} == {"siege.env", "rl.train_ppo", "rl.augment_v1"}
    g = tmp_path / "y.py"
    g.write_text("from rl import train_ppo_v2\nfrom rl.train_ppo_v2 import PPO_CONFIG\nimport siegel\n")
    assert not {m for m, lv in _imports(g) if _is_removed(m, lv)}
    assert TEXT_PAT.search("rl.train_ppo --aug") and not TEXT_PAT.search("python -m rl.train_ppo_v2 --seed 1")


def _modules(pkg: str):
    mod = importlib.import_module(pkg)
    return [pkg] + [m.name for m in pkgutil.walk_packages(mod.__path__, prefix=pkg + ".")]


def test_every_remaining_module_imports():
    names = _modules("breachlens") + [n for n in _modules("rl")] + (_modules("app") if (ROOT / "app" / "__init__.py").exists() else [])
    app_files = [p for p in (ROOT / "app").glob("*.py") if p.name != "__init__.py"] if (ROOT / "app").is_dir() else []
    names += [f"app.{p.stem}" for p in app_files if not (ROOT / "app" / "__init__.py").exists()]
    assert any(n.startswith("rl.") and n.endswith("_v2") for n in names)
    for n in sorted(set(names)):
        importlib.import_module(n)
