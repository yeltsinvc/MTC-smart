from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .io import read_json, utc_stamp, write_json
from .paths import CONFIGS, ROOT


def load_kaggle_inference_config(path: Path | None = None) -> dict[str, Any]:
    return read_json(path or CONFIGS / "kaggle_inference.json", {})


def slugify(value: str, max_len: int = 48) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    value = re.sub(r"-+", "-", value)
    return (value or "job")[:max_len].strip("-")


def run_cmd(args: list[str], cwd: Path | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(args, cwd=str(cwd) if cwd else None, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check and proc.returncode:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(args)}\n{proc.stdout}")
    return proc


def submit_remote_video_job(video_path: Path, job_dir: Path, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_kaggle_inference_config()
    owner = str(cfg.get("owner") or "")
    if not owner:
        raise ValueError("configs/kaggle_inference.json must define owner")
    job_id = job_dir.name
    dataset_slug = f"{owner}/{slugify('dvka-video-' + job_id, 44)}"
    kernel_slug = slugify("dvka-infer-" + job_id, 44)
    kernel_id = f"{owner}/{kernel_slug}"

    dataset_ws = job_dir / "kaggle_dataset"
    kernel_ws = job_dir / "kaggle_kernel"
    dataset_ws.mkdir(parents=True, exist_ok=True)
    kernel_ws.mkdir(parents=True, exist_ok=True)

    packaged_video = dataset_ws / f"input{video_path.suffix.lower()}"
    shutil.copy2(video_path, packaged_video)

    local_model = ROOT / str(cfg.get("local_model_path", "models/best.pt"))
    include_model = bool(cfg.get("include_local_model_in_job_dataset", True))
    model_sources = []
    if cfg.get("model_dataset_slug"):
        model_sources.append(str(cfg["model_dataset_slug"]))
    elif include_model:
        if not local_model.exists():
            raise FileNotFoundError(f"Local model not found: {local_model}")
        shutil.copy2(local_model, dataset_ws / str(cfg.get("model_filename", "best.pt")))

    metadata = {
        "title": f"DVKA Video {job_id}",
        "id": dataset_slug,
        "licenses": [{"name": str(cfg.get("dataset_license", "unknown"))}],
    }
    (dataset_ws / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    dataset_proc = run_cmd(["kaggle", "datasets", "create", "-p", str(dataset_ws), "--dir-mode", "zip"], cwd=dataset_ws)
    if dataset_proc.returncode != 0 and "already exists" in dataset_proc.stdout.lower():
        dataset_proc = run_cmd(["kaggle", "datasets", "version", "-p", str(dataset_ws), "-m", f"update {job_id}", "--dir-mode", "zip"], cwd=dataset_ws)
    if dataset_proc.returncode != 0:
        raise RuntimeError(dataset_proc.stdout)

    script = render_remote_kernel(cfg, dataset_slug)
    notebook_name = "infer.ipynb"
    (kernel_ws / notebook_name).write_text(json.dumps(render_notebook(script), indent=2), encoding="utf-8")
    (kernel_ws / "infer_kernel.py").write_text(script, encoding="utf-8")
    kernel_metadata = {
        "id": kernel_id,
        "title": f"DVKA Infer {job_id}",
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true" if cfg.get("enable_internet", True) else "false",
        "dataset_sources": [dataset_slug] + model_sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": str(cfg.get("accelerator", "NvidiaTeslaT4")),
    }
    (kernel_ws / "kernel-metadata.json").write_text(json.dumps(kernel_metadata, indent=2) + "\n", encoding="utf-8")
    kernel_proc = run_cmd(["kaggle", "kernels", "push", "-p", str(kernel_ws)], cwd=kernel_ws)
    if kernel_proc.returncode != 0:
        raise RuntimeError(kernel_proc.stdout)

    state = {
        "job_id": job_id,
        "status": "submitted",
        "created_at": utc_stamp(),
        "video_path": str(video_path),
        "job_dir": str(job_dir),
        "dataset_slug": dataset_slug,
        "kernel_id": kernel_id,
        "dataset_output": dataset_proc.stdout,
        "kernel_output": kernel_proc.stdout,
    }
    write_json(job_dir / "remote_job.json", state)
    return state


def refresh_remote_job(job_dir: Path) -> dict[str, Any]:
    state_path = job_dir / "remote_job.json"
    state = read_json(state_path, {})
    if not state:
        raise FileNotFoundError(state_path)
    proc = run_cmd(["kaggle", "kernels", "status", str(state["kernel_id"])])
    low = proc.stdout.lower()
    if "complete" in low:
        status = "complete"
    elif "error" in low or "failed" in low:
        status = "error"
    elif "running" in low:
        status = "running"
    else:
        status = "submitted"
    state.update({"status": status, "last_status_at": utc_stamp(), "last_status_output": proc.stdout})
    write_json(state_path, state)
    return state


def pull_remote_job(job_dir: Path) -> dict[str, Any]:
    state = refresh_remote_job(job_dir)
    output_dir = job_dir / "kaggle_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if state["status"] != "complete":
        return {**state, "pulled": False, "output_dir": str(output_dir)}
    proc = run_cmd(["kaggle", "kernels", "output", str(state["kernel_id"]), "-p", str(output_dir)])
    state.update({"pulled": proc.returncode == 0, "pull_output": proc.stdout, "output_dir": str(output_dir)})
    write_json(job_dir / "remote_job.json", state)
    return state


def render_remote_kernel(cfg: dict[str, Any], dataset_slug: str) -> str:
    return f'''
from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "ultralytics>=8.3.0", "opencv-python"])

from ultralytics import YOLO

KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
ART = WORK / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {{".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}}
DATASET_ROOT = KAGGLE_INPUT / "{dataset_slug.split('/')[-1]}"
videos = [p for p in DATASET_ROOT.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
if not videos:
    raise FileNotFoundError("No video file found in job dataset")
video = videos[0]

model_candidates = list(DATASET_ROOT.rglob("*.pt")) + list(KAGGLE_INPUT.rglob("{cfg.get('model_filename', 'best.pt')}"))
if not model_candidates:
    raise FileNotFoundError("No .pt model found. Set model_dataset_slug or include_local_model_in_job_dataset=true")
model_path = model_candidates[0]

model = YOLO(str(model_path))
rows = []
for frame_idx, result in enumerate(model.track(
    source=str(video),
    imgsz={int(cfg.get("imgsz", 960))},
    conf={float(cfg.get("conf", 0.05))},
    iou={float(cfg.get("iou", 0.5))},
    tracker="{cfg.get('tracker', 'bytetrack.yaml')}",
    persist=True,
    stream=True,
    verbose=False,
)):
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        continue
    image_h, image_w = getattr(result, "orig_shape", [0, 0])
    for box in boxes:
        raw_id = getattr(box, "id", None)
        track_id = int(raw_id.item()) if raw_id is not None else -1
        cls_id = int(box.cls.item())
        xyxy = [float(v) for v in box.xyxy[0].tolist()]
        width = max(0.0, xyxy[2] - xyxy[0])
        height = max(0.0, xyxy[3] - xyxy[1])
        norm_area = (width * height) / max(1.0, float(image_w * image_h))
        aspect = (width / max(1.0, image_w)) / max(1e-9, (height / max(1.0, image_h)))
        rows.append({{
            "video": video.name,
            "frame": frame_idx,
            "track_id": track_id if track_id >= 0 else "",
            "class_id": cls_id,
            "class": result.names.get(cls_id, str(cls_id)),
            "conf": float(box.conf.item()),
            "xyxy": json.dumps(xyxy),
            "image_width": int(image_w),
            "image_height": int(image_h),
            "bbox_normalized_area": norm_area,
            "bbox_normalized_aspect": aspect,
        }})

def write_csv(path, data):
    if not data:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({{k for row in data for k in row}})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)

write_csv(ART / "tracking_frame_detections.csv", rows)

by_track = defaultdict(list)
for row in rows:
    if row["track_id"] != "":
        by_track[(row["video"], row["track_id"])].append(row)

summary = []
votes = []
for (video_name, tid), track_rows in by_track.items():
    counter = Counter(row["class"] for row in track_rows)
    conf_sum = defaultdict(float)
    for row in track_rows:
        conf_sum[row["class"]] += float(row["conf"])
    scored = sorted(counter, key=lambda cls: (counter[cls], conf_sum[cls]), reverse=True)
    final = scored[0] if scored else ""
    total = sum(counter.values()) or 1
    for cls in scored:
        votes.append({{"video": video_name, "track_id": tid, "class": cls, "frequency": counter[cls], "score": counter[cls] / total}})
    summary.append({{
        "video": video_name,
        "track_id": tid,
        "frames": len(track_rows),
        "final_class": final,
        "final_score": counter[final] / total if final else 0,
        "stable": (counter[final] / total) >= 0.6 if final else False,
        "mean_bbox_normalized_area": sum(float(r["bbox_normalized_area"]) for r in track_rows) / len(track_rows),
    }})

write_csv(ART / "tracking_track_summary.csv", summary)
write_csv(ART / "tracking_class_votes.csv", votes)
(ART / "summary.json").write_text(json.dumps({{
    "video": video.name,
    "model_path": str(model_path),
    "frame_detections": len(rows),
    "tracks": len(summary),
}}, indent=2), encoding="utf-8")
'''


def render_notebook(script: str) -> dict[str, Any]:
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# DVKA remote video inference\\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in script.splitlines()]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
