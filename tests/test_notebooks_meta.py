"""Committed notebooks must be executed, error-free and honestly labelled."""
from __future__ import annotations

import json

import pytest

from src import REPO_ROOT, RESULTS_ROOT

NB = REPO_ROOT / "notebooks"
NAMES = ["01_sum_to_one_eda.ipynb", "02_view_results.ipynb"]


@pytest.mark.parametrize("name", NAMES)
def test_notebook_executed_and_labelled(name):
    path = NB / name
    if not path.exists():
        pytest.skip("notebook not built")
    nb = json.loads(path.read_text())
    meta = nb["metadata"].get("bayes")
    assert meta and meta["data_kinds"], "notebook metadata must record its data kinds"
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(c.get("execution_count") for c in code), "every code cell must have been executed"
    outputs = json.dumps([c.get("outputs", []) for c in code])
    assert '"output_type": "error"' not in outputs
    if "synthetic" in meta["data_kinds"]:
        assert "SYNTHETIC DATA" in outputs
    else:
        assert "DATA: REAL" in outputs and "SYNTHETIC DATA" not in outputs


def test_readme_block_matches_metrics():
    readme = (REPO_ROOT / "README.md").read_text()
    if "<!-- RESULTS:START -->" not in readme or not (RESULTS_ROOT / "metrics.json").exists():
        pytest.skip("results not generated")
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "notebooks"))
    from readme_results import render

    block = readme.split("<!-- RESULTS:START -->")[1].split("<!-- RESULTS:END -->")[0].strip()
    assert block == render(RESULTS_ROOT).strip(), "README results are stale: run scripts/build_notebooks.py --update-readme"
