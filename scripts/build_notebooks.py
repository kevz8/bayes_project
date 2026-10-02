"""Build, execute and stamp the research notebooks (reproducibly, from code).

    python scripts/build_notebooks.py                 # all three, data mode auto (real data preferred)
    python scripts/build_notebooks.py --only 03       # one notebook
    python scripts/build_notebooks.py --update-readme # also refresh the README results block

Each notebook's cells live in ``scripts/notebooks/nbXX.py`` as ``CELLS = [("md"|"code", source), ...]``.
After execution the notebook's metadata records the data kind, code version and build time,
and the build fails if any cell errored or the provenance banner is missing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import RESULTS_ROOT, code_version  # noqa: E402

NB_DIR = ROOT / "notebooks"
SPEC_DIR = Path(__file__).resolve().parent / "notebooks"
NOTEBOOKS = {
    "01": "01_sum_to_one_eda.ipynb",
    "02": "02_cross_market_ols.ipynb",
    "03": "03_view_results.ipynb",
}
BANNER_RE = re.compile(r"DATA: REAL|SYNTHETIC DATA")


def load_cells(key: str) -> list[tuple[str, str]]:
    spec = importlib.util.spec_from_file_location(f"nb{key}", SPEC_DIR / f"nb{key}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.CELLS


def build(key: str, *, execute: bool = True, timeout: int = 3600) -> Path:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, src in load_cells(key):
        src = src.strip("\n")
        nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src))
    path = NB_DIR / NOTEBOOKS[key]
    if execute:
        t0 = time.time()
        NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}}).execute()
        errors = [c for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
        if errors:
            raise RuntimeError(f"{NOTEBOOKS[key]}: a cell raised an error")
        text = json.dumps([c.get("outputs", []) for c in nb.cells if c.cell_type == "code"])
        if not BANNER_RE.search(text):
            raise RuntimeError(f"{NOTEBOOKS[key]}: no data-provenance banner in the outputs")
        info_path = RESULTS_ROOT / f"dataset_info_{key}.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        nb.metadata["bayes"] = {"data_kinds": info.get("data_kinds"), "baskets": info.get("baskets"),
                                "code_version": code_version(),
                                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "build_seconds": round(time.time() - t0, 1)}
        for c in nb.cells:  # strip volatile per-cell timing metadata
            c.metadata.pop("execution", None)
    nbformat.write(nb, path)
    return path


def update_readme(readme: Path = ROOT / "README.md") -> None:
    sys.path.insert(0, str(SPEC_DIR))
    from readme_results import render  # noqa: E402

    text = readme.read_text(encoding="utf-8")
    block = render(RESULTS_ROOT)
    new = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
                 f"<!-- RESULTS:START -->\n{block}\n<!-- RESULTS:END -->", text, flags=re.S)
    readme.write_text(new, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="01,02,03")
    ap.add_argument("--data-mode", default=None, help="sets BAYES_DATA_MODE for the kernels")
    ap.add_argument("--no-execute", action="store_true")
    ap.add_argument("--update-readme", action="store_true")
    args = ap.parse_args(argv)
    if args.data_mode:
        os.environ["BAYES_DATA_MODE"] = args.data_mode
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    for key in args.only.split(","):
        t0 = time.time()
        p = build(key.strip(), execute=not args.no_execute)
        print(f"{p.relative_to(ROOT)} built in {time.time() - t0:.0f}s")
    if args.update_readme:
        update_readme()
        print("README results block updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
