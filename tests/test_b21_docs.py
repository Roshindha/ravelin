"""B21: README and docs polish (docs/SPEC_V2.md, Gate changelog row (qq)). Pure file checks; no training, no evaluation."""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def tracked_md() -> list:
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    return sorted(set(out) | {"README.md"})


def test_readme_has_no_inspired_by_placeholders_or_old_name():
    """Owner decision (row (qq)): no 'Inspired by' line anywhere; the placeholders are gone; the old name does not appear in the README."""
    for bad in ("AUTHOR_NAME", "AUTHOR_URL", "Inspired by", "inspired by", "base" + " " + "siege", "troops.md"):
        assert bad.lower() not in README.lower(), bad


def test_readme_content_rules():
    lines = README.splitlines()
    assert lines[0] == "# Ravelin"
    assert "Play it: https://ravelin.streamlit.app" in README
    for shot in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", README):
        assert not shot.startswith(("http", "/")) and (ROOT / shot).exists(), shot
    assert len(re.findall(r"!\[[^\]]*\]\(", README)) == 2
    for needle in ("Fraunces", "Atkinson Hyperlegible", "SIL Open Font License", "docs/RESULTS.md", "docs/LIMITATIONS.md", "requirements-dev.txt",
                   "No learned policy beats a fixed single-type army on breach rate", "bootstrap", "BestSingle", "Oracle", "TTA", "DDQN", "PPO+D4"):
        assert needle in README, needle
    assert not re.search(r"[\U0001F000-\U0001FAFF☀-➿⬀-⯿️]", README), "emoji"
    assert "The Python package keeps its original internal name (breachlens) because frozen provenance hashes reference it." in README
    assert "ravelin_1_placing" in README and "ravelin_2_battle" in README
    assert not (ROOT / "LICENSE").exists() and "License:" not in README and "MIT License" not in README     # owner decision (B19): no licence file, no licence section


def test_readme_table_numbers_come_from_results_md():
    results = (ROOT / "docs" / "RESULTS.md").read_text()
    sec = results[results.index("## Headline conditions"):results.index("All other conditions")]
    src = {}
    for ln in sec.splitlines():
        if ln.startswith("|") and not ln.startswith("|---") and "Condition" not in ln:
            c = [x.strip() for x in ln.strip("|").split("|")]
            src[c[0].split(" (")[0]] = [re.match(r"[0-9.]+", x).group(0) for x in c[1:]]
    tab = README[README.index("| Condition |"):README.index("Intervals, the planner")]
    rows = [r for r in tab.splitlines() if r.startswith("|") and not r.startswith("|---") and "Condition" not in r]
    assert len(rows) == 8
    for r in rows:
        c = [x.strip() for x in r.strip("|").split("|")]
        assert src[c[0].split(" (")[0]] == c[1:], c
    assert "0.700" in README and "27 of 64" in README and "27 of 64" in results


@pytest.mark.parametrize("name", ["RESULTS.md", "LIMITATIONS.md", "SPEC_V2.md"])
def test_docs_start_with_a_three_line_summary(name):
    lines = (ROOT / "docs" / name).read_text().splitlines()
    assert lines[0].startswith("# ") and lines[1] == ""
    assert all(ln.startswith("> ") for ln in lines[2:5]) and lines[2].startswith("> **Summary for readers.**") and lines[5] == ""


def test_every_relative_markdown_link_resolves():
    broken, n = [], 0
    for f in tracked_md():
        text = re.sub(r"```.*?```", "", (ROOT / f).read_text(), flags=re.S)
        text = re.sub(r"`[^`\n]*`", "", text)
        for m in LINK.finditer(text):
            u = m.group(1)
            if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", u) or u.startswith("#"):
                continue
            n += 1
            if not (ROOT / f).parent.joinpath(u.split("#")[0].split("?")[0]).resolve().exists():
                broken.append((f, u))
    assert n >= 8 and not broken, broken
