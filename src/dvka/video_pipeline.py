from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_class_map, load_openai_fallback, load_tracking_postprocess, load_typology_rules
from .io import read_json
from .openai_fallback import build_fallback_candidates, resolve_candidates, write_openai_fallback_outputs
from .tracking_postprocess import consensus_tracks, write_tracking_outputs
from .typology_agent import enrich_detections, load_calibration


@dataclass
class VideoPipelineConfig:
    model_path: Path
    imgsz: int = 960
    conf: float = 0.05
    iou: float = 0.5
    tracker: str = "bytetrack.yaml"
    device: str = "auto"
    save_annotated_video: bool = False
    enable_openai_fallback: bool = False


def load_server_config(path: Path) -> dict[str, Any]:
    return read_json(path, {})


def run_video_pipeline(video_path: Path, output_dir: Path, cfg: VideoPipelineConfig) -> dict[str, Any]:
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(cfg.model_path))
    frame_rows = _track_video(model, video_path, cfg, output_dir)
    rules = load_typology_rules()
    calibration = load_calibration()
    enriched = enrich_detections(frame_rows, rules, calibration)
    consensus = consensus_tracks(enriched, load_tracking_postprocess())
    write_tracking_outputs(consensus, output_dir, load_tracking_postprocess())

    openai_result = {"enabled": False, "candidates": 0, "decisions": 0}
    if cfg.enable_openai_fallback:
        openai_cfg = dict(load_openai_fallback())
        openai_cfg["enabled"] = True
        candidates = build_fallback_candidates(consensus["frame_rows"], consensus["track_summary"], consensus["vote_rows"], openai_cfg)
        decisions = resolve_candidates(candidates, openai_cfg)
        write_openai_fallback_outputs(candidates, decisions, output_dir, openai_cfg)
        openai_result = {"enabled": True, "candidates": len(candidates), "decisions": len(decisions)}

    summary = {
        "video": str(video_path),
        "model_path": str(cfg.model_path),
        "frames_with_detections": len({row.get("frame") for row in consensus["frame_rows"]}),
        "frame_detections": len(consensus["frame_rows"]),
        "tracks": len(consensus["track_summary"]),
        "openai_fallback": openai_result,
        "artifacts": {
            "frame_detections_csv": str(output_dir / "tracking_frame_detections.csv"),
            "track_summary_csv": str(output_dir / "tracking_track_summary.csv"),
            "track_class_votes_csv": str(output_dir / "tracking_class_votes.csv"),
            "summary_json": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _track_video(model: Any, video_path: Path, cfg: VideoPipelineConfig, output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    kwargs = {
        "source": str(video_path),
        "imgsz": cfg.imgsz,
        "conf": cfg.conf,
        "iou": cfg.iou,
        "tracker": cfg.tracker,
        "persist": True,
        "stream": True,
        "verbose": False,
        "save": cfg.save_annotated_video,
        "project": str(output_dir),
        "name": "annotated",
        "exist_ok": True,
    }
    if cfg.device != "auto":
        kwargs["device"] = cfg.device
    for frame_idx, result in enumerate(model.track(**kwargs)):
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        image_h, image_w = getattr(result, "orig_shape", [0, 0])
        for box in boxes:
            raw_id = getattr(box, "id", None)
            track_id = int(raw_id.item()) if raw_id is not None else -1
            cls_id = int(box.cls.item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            rows.append(
                {
                    "video": video_path.name,
                    "frame": frame_idx,
                    "track_id": track_id if track_id >= 0 else "",
                    "class_id": cls_id,
                    "class": result.names.get(cls_id, str(cls_id)),
                    "conf": float(box.conf.item()),
                    "xyxy": xyxy,
                    "image_width": int(image_w),
                    "image_height": int(image_h),
                }
            )
    _write_csv(output_dir / "raw_frame_detections.csv", rows)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat_rows = []
    for row in rows:
        flat_rows.append({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    fields = sorted({key for row in flat_rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)
