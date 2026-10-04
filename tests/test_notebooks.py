"""The demonstration notebooks must be executed, error-free, portable, and use names that exist."""
import ast
import importlib
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS = sorted(Path("notebooks").glob("*.ipynb"))
EXPECTED = ["01_data_exploration", "02_static_vine", "03_rolling_vine", "04_dependence_changes",
            "05_portfolio_risk", "06_live_monitor"]


def test_all_notebooks_present():
    assert [p.stem for p in NOTEBOOKS] == EXPECTED


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_is_valid_executed_and_clean(path):
    nb = nbformat.read(path, 4)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert code and nb.cells[0].cell_type == "markdown"  # starts with an explanation
    assert all(c.execution_count is not None for c in code), "run the notebook top to bottom before committing"
    for c in code:
        assert not [o for o in c.outputs if o.output_type == "error"]
    text = path.read_text()
    assert "/Users/" not in text and "/private/" not in text  # no local paths in the saved outputs
    assert "os.chdir(ROOT)" in code[0].source  # works whether started from the root or from notebooks/


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_imports_resolve(path):
    """Catches API drift: every `from vine_risk.x import y` in a notebook still exists."""
    nb = nbformat.read(path, 4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = "\n".join(l for l in cell.source.splitlines() if not l.lstrip().startswith(("%", "!")))
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("vine_risk"):
                mod = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(mod, alias.name), f"{node.module}.{alias.name} (used in {path.name})"
