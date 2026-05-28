from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
CONFIGS = ROOT / "configs"
STATE = ROOT / "state"
WORKSPACES = ROOT / "kernel_workspaces"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs"
QUEUE = STATE / "queue.json"
RUNS = STATE / "runs.json"
LEADERBOARD = STATE / "leaderboard.csv"
