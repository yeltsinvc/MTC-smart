from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import load_class_map, load_openai_fallback, load_tracking_postprocess, load_typology_rules, resolve_rare_classes, settings
from .experiments import load_queue, load_runs, save_runs, update_queue_item
from .io import file_lock, utc_stamp, write_json
from .kaggle_worker_template import render_notebook, render_worker
from .paths import OUTPUTS, RUNS, WORKSPACES


def slugify(value: str, max_len: int = 56) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return re.sub(r"-+", "-", value)[:max_len].strip("-") or "experiment"


def run_cmd(args: list[str], cwd: Path | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(args, cwd=str(cwd) if cwd else None, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check and proc.returncode:
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{proc.stdout}")
    return proc


def build_workspace(exp: dict[str, Any]) -> Path:
    cfg = settings()
    exp = dict(exp)
    if cfg.dataset_slug:
        exp.setdefault("dataset_slug", cfg.dataset_slug)
    # Inject the rare-class set so the worker can compute rare_map50_95. Without
    # this the worker's EXP.get("rare_classes", []) is always empty and the whole
    # rare-class tracking/ranking loop is dead.
    if "rare_classes" not in exp:
        rare = resolve_rare_classes()
        if rare:
            exp["rare_classes"] = rare
    exp_id = slugify(str(exp["id"]))
    ws = WORKSPACES / exp_id
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True, exist_ok=True)
    script = render_worker(exp, load_class_map(), load_typology_rules(), load_tracking_postprocess(), load_openai_fallback())
    title = f"DVKA {exp_id}"
    notebook_name = f"{exp_id}.ipynb"
    (ws / "experiment.json").write_text(json.dumps(exp, indent=2) + "\n", encoding="utf-8")
    (ws / "train_kernel.py").write_text(script, encoding="utf-8")
    (ws / notebook_name).write_text(json.dumps(render_notebook(script, title), indent=2), encoding="utf-8")
    metadata = {
        "id": f"{cfg.owner}/{slugify(title)}",
        "title": title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [cfg.dataset_slug] if cfg.dataset_slug else [],
        "competition_sources": [cfg.competition_slug] if cfg.competition_slug else [],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": cfg.accelerator,
    }
    (ws / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return ws


def build(exp_id: str | None = None, limit: int = 1) -> list[str]:
    queue = load_queue()
    candidates = [item for item in queue if item.get("status") == "queued"]
    if exp_id:
        candidates = [item for item in queue if str(item.get("id")) == exp_id]
    built = []
    for exp in candidates[:limit]:
        built.append(str(build_workspace(exp)))
    return built


def launch(limit: int = 1, timeout: int = 7200) -> list[dict[str, Any]]:
    queue = load_queue()
    launched: list[dict[str, Any]] = []
    for exp in [item for item in queue if item.get("status") == "queued"][:limit]:
        ws = build_workspace(exp)
        proc = run_cmd(["kaggle", "kernels", "push", "-p", str(ws)], cwd=ws)
        kernel_id = json.loads((ws / "kernel-metadata.json").read_text(encoding="utf-8"))["id"]
        status = "submitted" if proc.returncode == 0 else "error"
        update_queue_item(str(exp["id"]), status=status, kernel_id=kernel_id, workspace=str(ws), launch_output=proc.stdout)
        run = {"experiment_id": exp["id"], "kernel_id": kernel_id, "status": status, "workspace": str(ws), "submitted_at": utc_stamp()}
        with file_lock(RUNS):
            runs = load_runs()
            runs.append(run)
            save_runs(runs)
        launched.append(run)
    return launched


def refresh() -> list[dict[str, Any]]:
    refreshed: list[dict[str, Any]] = []
    with file_lock(RUNS):
        runs = load_runs()
        for run in runs:
            if run.get("status") not in {"submitted", "running", "queued_remote"}:
                continue
            kernel_id = run.get("kernel_id")
            if not kernel_id:
                # Without a kernel_id we cannot query Kaggle. Surface the run as
                # errored instead of silently leaving it stuck in "submitted".
                run.update({
                    "status": "error",
                    "last_status_output": "missing kernel_id",
                    "last_status_at": utc_stamp(),
                })
                update_queue_item(str(run.get("experiment_id")), status="error")
                refreshed.append(run)
                continue
            proc = run_cmd(["kaggle", "kernels", "status", str(kernel_id)])
            text = proc.stdout
            low = text.lower()
            if "complete" in low:
                status = "complete"
            elif "error" in low or "failed" in low:
                status = "error"
            elif "running" in low:
                status = "running"
            else:
                status = "submitted"
            run.update({"status": status, "last_status_output": text, "last_status_at": utc_stamp()})
            update_queue_item(str(run["experiment_id"]), status=status, kernel_id=kernel_id)
            refreshed.append(run)
        save_runs(runs)
    return refreshed


def pull(output_root: Path = OUTPUTS, skip_existing: bool = True) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    pulled: list[Path] = []
    for run in load_runs():
        if run.get("status") != "complete":
            continue
        dest = output_root / slugify(str(run["experiment_id"]))
        if skip_existing and (dest / "artifacts" / "metrics.json").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        proc = run_cmd(["kaggle", "kernels", "output", str(run["kernel_id"]), "-p", str(dest)])
        (dest / "pull.log").write_text(proc.stdout, encoding="utf-8")
        pulled.append(dest)
    return pulled
