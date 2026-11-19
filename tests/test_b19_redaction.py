"""B19 (Gate changelog row (rr)): two G7 reports had one local absolute path replaced by the repo-relative `models_v2`; nothing else changed.

The test pins the OLD sha256 (the file at tag b15-final), the NEW sha256, and that new == old with only that one prefix replaced.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOME_PREFIX = "/" + "Users" + "/" + "roshinisudhakar" + "/" + "base" + "-" + "siege" + "/models_v2"   # built at runtime: the literal never appears
FILES = {
    "results/b13/g7_report_300k.txt": ("d01e6ff86ad7f7fa31e00751203b9cf0b5489f9042786a1a0628079f0eb0d3c1",
                                       "7effc86b8dd4c049165f7109fe3bbb170e4ab1ba6536c2e7e5f93691cd31eb7d"),
    "results/b13/g7_report_shaped_300k.txt": ("75baf01cc0a5937cde6d8ddfb4965ca9f8dc83b5981f5e6b7835eb33af416fb3",
                                              "496e66f6c148041e4204cbdb8e1d97e2940b186dae8214c4a9caba5ee6b22f49"),
}


def _tagged(path: str) -> bytes:
    try:
        return subprocess.run(["git", "show", f"b15-final:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:       # shallow clone without the tag
        pytest.skip("tag b15-final not available")


@pytest.mark.parametrize("path", sorted(FILES))
def test_tagged_file_has_the_old_sha_and_the_new_file_differs_only_by_the_path_prefix(path):
    old_sha, new_sha = FILES[path]
    old = _tagged(path)
    assert hashlib.sha256(old).hexdigest() == old_sha
    new = (ROOT / path).read_bytes()
    assert hashlib.sha256(new).hexdigest() == new_sha
    assert old.count(HOME_PREFIX.encode()) == 1 and old.replace(HOME_PREFIX.encode(), b"models_v2") == new


def test_gates_json_pins_the_new_shas():
    pins = {}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in FILES and isinstance(v, str):
                    pins[k] = v
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(json.loads((ROOT / "data/v2/gates.json").read_text()))
    assert {k: v for k, v in pins.items()} == {k: v[1] for k, v in FILES.items()}
