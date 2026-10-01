"""Polymarket statistical-arbitrage research platform.

Package layout (see README.md for the full pipeline):

* ``events`` / ``orderbook`` / ``clob_client`` - wire format, L2 books, live feeds (CS side)
* ``arb_engine`` - streaming basket sums, rolling z-scores, signal state machine
* ``execution_sim`` / ``metrics`` - friction-aware execution and performance statistics
* ``stats_tools`` / ``plotting`` - research helpers used by the notebooks
"""
from __future__ import annotations

import subprocess
from pathlib import Path

__version__ = "0.1.0"

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "markets.json"
DATA_ROOT = REPO_ROOT / "data"
HIST_ROOT = DATA_ROOT / "historical_books"
PRICES_ROOT = DATA_ROOT / "prices_history"
SYNTH_ROOT = DATA_ROOT / "synthetic"
RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_ROOT = RESULTS_ROOT / "figures"


def code_version() -> str:
    """Short git SHA of the working tree (``"unknown"`` outside a git checkout)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - provenance only, never fatal
        return "unknown"
