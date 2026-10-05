"""The pinned dependency list must cover everything pyproject.toml asks for."""
import re
import tomllib
from pathlib import Path


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_names(specs):
    return {_norm(re.split(r"[<>=!~\[; ]", s, maxsplit=1)[0]) for s in specs}


def _lock():
    pins = {}
    for line in Path("requirements-lock.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([^\s;]+)", line)
            assert m, f"not an exact pin: {line!r}"
            pins[_norm(m.group(1))] = m.group(2)
    return pins


def test_every_direct_dependency_is_pinned_exactly():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    wanted = _requirement_names(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        wanted |= _requirement_names(extra)
    missing = wanted - set(_lock())
    assert not missing, f"not in requirements-lock.txt: {sorted(missing)}"


def test_lock_has_no_editable_or_local_entries():
    pins = [l for l in Path("requirements-lock.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert pins and not any("vine-risk" in l or l.startswith("-e") or "file://" in l or "@ " in l for l in pins)


def test_requires_python_matches_the_pinned_libraries():
    # numpy 2.5 / pandas 3 (pinned) need Python >= 3.11
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    assert project["requires-python"] == ">=3.11"
