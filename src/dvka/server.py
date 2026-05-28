from __future__ import annotations

import asyncio
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .kaggle_remote import pull_remote_job, refresh_remote_job, submit_remote_video_job, load_kaggle_inference_config
from .paths import ROOT
from .paths import CONFIGS
from .video_pipeline import VideoPipelineConfig, load_server_config, run_video_pipeline


_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _validate_job_id(job_id: str) -> str:
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id")
    return job_id


def create_app() -> FastAPI:
    app = FastAPI(title="Drone Vehicle Detection API", version="0.1.0")
    server_cfg = load_server_config(CONFIGS / "server.json")
    jobs_dir = ROOT / str(server_cfg.get("jobs_dir", "server_runs"))
    jobs_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/health")
    def health() -> dict[str, Any]:
        model_path = ROOT / str(server_cfg.get("model_path", "models/best.pt"))
        return {"ok": True, "model_exists": model_path.exists(), "model_path": str(model_path)}

    @app.post("/v1/videos/analyze")
    async def analyze_video(video: UploadFile = File(...)) -> dict[str, Any]:
        if not video.filename:
            raise HTTPException(status_code=400, detail="Missing filename")
        suffix = Path(video.filename).suffix.lower()
        if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}:
            raise HTTPException(status_code=400, detail="Unsupported video extension")

        job_id = uuid.uuid4().hex
        job_dir = jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        input_path = job_dir / f"input{suffix}"
        max_bytes = int(server_cfg.get("max_upload_mb", 1024)) * 1024 * 1024
        written = 0
        with input_path.open("wb") as handle:
            while chunk := await video.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(status_code=413, detail="Video too large")
                handle.write(chunk)

        model_path = ROOT / str(server_cfg.get("model_path", "models/best.pt"))
        if not model_path.exists():
            raise HTTPException(status_code=500, detail=f"Model not found: {model_path}")
        cfg = VideoPipelineConfig(
            model_path=model_path,
            imgsz=int(server_cfg.get("imgsz", 960)),
            conf=float(server_cfg.get("conf", 0.05)),
            iou=float(server_cfg.get("iou", 0.5)),
            tracker=str(server_cfg.get("tracker", "bytetrack.yaml")),
            device=str(server_cfg.get("device", "auto")),
            save_annotated_video=bool(server_cfg.get("save_annotated_video", False)),
            enable_openai_fallback=bool(server_cfg.get("enable_openai_fallback", False)),
        )
        # run_video_pipeline is CPU/GPU-bound and synchronous; off-load it so we
        # don't block the uvicorn event loop (no other request, not even /health,
        # would be served during inference).
        summary = await asyncio.to_thread(run_video_pipeline, input_path, job_dir / "artifacts", cfg)
        summary["job_id"] = job_id
        summary["download_base"] = f"/v1/jobs/{job_id}/artifacts"
        return summary

    @app.post("/v1/videos/analyze-kaggle")
    async def analyze_video_on_kaggle(video: UploadFile = File(...)) -> dict[str, Any]:
        """Receive video locally, run inference remotely on Kaggle GPU."""
        if not video.filename:
            raise HTTPException(status_code=400, detail="Missing filename")
        suffix = Path(video.filename).suffix.lower()
        if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}:
            raise HTTPException(status_code=400, detail="Unsupported video extension")

        remote_cfg = load_kaggle_inference_config()
        job_id = uuid.uuid4().hex
        job_dir = jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        input_path = job_dir / f"input{suffix}"
        max_bytes = int(server_cfg.get("max_upload_mb", 1024)) * 1024 * 1024
        written = 0
        with input_path.open("wb") as handle:
            while chunk := await video.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(status_code=413, detail="Video too large")
                handle.write(chunk)
        try:
            state = submit_remote_video_job(input_path, job_dir, remote_cfg)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            **state,
            "status_url": f"/v1/kaggle-jobs/{job_id}",
            "pull_url": f"/v1/kaggle-jobs/{job_id}/pull",
        }

    @app.get("/v1/kaggle-jobs/{job_id}")
    def kaggle_job_status(job_id: str) -> dict[str, Any]:
        _validate_job_id(job_id)
        job_dir = jobs_dir / job_id
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        try:
            return refresh_remote_job(job_dir)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/v1/kaggle-jobs/{job_id}/pull")
    def kaggle_job_pull(job_id: str) -> dict[str, Any]:
        _validate_job_id(job_id)
        job_dir = jobs_dir / job_id
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        try:
            state = pull_remote_job(job_dir)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        state["download_base"] = f"/v1/kaggle-jobs/{job_id}/artifacts"
        return state

    @app.get("/v1/kaggle-jobs/{job_id}/artifacts/{name}")
    def get_kaggle_artifact(job_id: str, name: str) -> FileResponse:
        _validate_job_id(job_id)
        allowed = {
            "summary.json",
            "tracking_frame_detections.csv",
            "tracking_track_summary.csv",
            "tracking_class_votes.csv",
        }
        if name not in allowed:
            raise HTTPException(status_code=404, detail="Unknown artifact")
        path = jobs_dir / job_id / "kaggle_output" / "artifacts" / name
        if not path.exists():
            # Kaggle sometimes flattens output contents.
            path = jobs_dir / job_id / "kaggle_output" / name
        if not path.exists():
            raise HTTPException(status_code=404, detail="Artifact not found. Pull the Kaggle job first.")
        return FileResponse(path)

    @app.get("/v1/jobs/{job_id}/artifacts/{name}")
    def get_artifact(job_id: str, name: str) -> FileResponse:
        _validate_job_id(job_id)
        allowed = {
            "summary.json",
            "raw_frame_detections.csv",
            "tracking_frame_detections.csv",
            "tracking_track_summary.csv",
            "tracking_class_votes.csv",
            "openai_fallback_candidates.csv",
            "openai_fallback_decisions.csv",
        }
        if name not in allowed:
            raise HTTPException(status_code=404, detail="Unknown artifact")
        path = jobs_dir / job_id / "artifacts" / name
        if not path.exists():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(path)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("dvka.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
