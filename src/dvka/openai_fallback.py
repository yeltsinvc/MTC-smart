from __future__ import annotations

import base64
import csv
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .io import read_json
from .paths import CONFIGS


def load_openai_fallback_config(path: Path | None = None) -> dict[str, Any]:
    return read_json(path or CONFIGS / "openai_fallback.json", {})


def build_fallback_candidates(
    frame_rows: list[dict[str, Any]],
    track_rows: list[dict[str, Any]],
    vote_rows: list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = cfg or load_openai_fallback_config()
    only = cfg.get("only_when") or {}
    max_items = int(cfg.get("max_items_per_run", 50))
    votes_by_track: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for vote in vote_rows:
        key = (str(vote.get("video", "")), str(vote.get("track_id", "")))
        votes_by_track.setdefault(key, []).append(vote)

    frames_by_track: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in frame_rows:
        key = (str(row.get("video", "")), str(row.get("track_id", "")))
        frames_by_track.setdefault(key, []).append(row)

    candidates: list[dict[str, Any]] = []
    for track in track_rows:
        key = (str(track.get("video", "")), str(track.get("track_id", "")))
        votes = sorted(votes_by_track.get(key, []), key=lambda item: _float(item.get("score")), reverse=True)
        if len(votes) < 2:
            continue
        top1, top2 = votes[0], votes[1]
        margin = _float(top1.get("score")) - _float(top2.get("score"))
        unstable_hit = bool(only.get("track_stable_is_false", True)) and str(track.get("stable")).lower() in {"false", "0", ""}
        low_score_hit = _float(track.get("final_score")) <= float(only.get("max_track_final_score", 0.62))
        low_margin_hit = margin <= float(only.get("max_vote_margin", 0.08))
        if not (unstable_hit or low_score_hit or low_margin_hit):
            continue
        representative = _representative_frame(frames_by_track.get(key, []))
        if only.get("max_detection_conf") is not None and _float(representative.get("conf")) > float(only["max_detection_conf"]) and not low_margin_hit:
            continue
        candidates.append(
            {
                "video": key[0],
                "track_id": key[1],
                "candidate_a": top1.get("class"),
                "candidate_b": top2.get("class"),
                "candidate_a_score": top1.get("score"),
                "candidate_b_score": top2.get("score"),
                "vote_margin": round(margin, 6),
                "track_final_class": track.get("final_class"),
                "track_final_score": track.get("final_score"),
                "frame": representative.get("frame", ""),
                "crop_path": representative.get("crop_path", ""),
                "area_ratio_to_car": representative.get("area_ratio_to_car", ""),
                "bbox_normalized_area": representative.get("bbox_normalized_area", ""),
                "bbox_normalized_aspect": representative.get("bbox_normalized_aspect", ""),
                "reason": "unstable_or_low_margin_track",
            }
        )
        if len(candidates) >= max_items:
            break
    return candidates


def resolve_candidates(candidates: list[dict[str, Any]], cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = cfg or load_openai_fallback_config()
    if not cfg.get("enabled", False):
        return [_decision(candidate, "skipped_disabled", "uncertain", 0.0, "OpenAI fallback disabled") for candidate in candidates]
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return [_decision(candidate, "skipped_no_api_key", "uncertain", 0.0, "OPENAI_API_KEY not set") for candidate in candidates]
    decisions = []
    for candidate in candidates[: int(cfg.get("max_items_per_run", 50))]:
        crop_path = Path(str(candidate.get("crop_path") or ""))
        if not crop_path.exists():
            decisions.append(_decision(candidate, "skipped_no_crop", "uncertain", 0.0, "Missing crop image"))
            continue
        try:
            decisions.append(_call_openai(candidate, crop_path, cfg, api_key))
        except Exception as exc:
            decisions.append(_decision(candidate, "error", "uncertain", 0.0, _sanitize_error(exc, api_key)))
    return decisions


def _sanitize_error(exc: BaseException, api_key: str) -> str:
    """Strip the API key and any Authorization header from an error message
    before persisting it to disk."""
    message = f"{type(exc).__name__}: {exc}"
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    # Defensive: scrub any "Bearer ..." token that slipped through (different key, etc.).
    import re as _re
    message = _re.sub(r"Bearer\s+[A-Za-z0-9._\-]+", "Bearer [REDACTED]", message)
    return message[:500]


def write_openai_fallback_outputs(candidates: list[dict[str, Any]], decisions: list[dict[str, Any]], output_dir: Path, cfg: dict[str, Any] | None = None) -> None:
    cfg = cfg or load_openai_fallback_config()
    outputs = cfg.get("outputs") or {}
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / outputs.get("candidates", "openai_fallback_candidates.csv"), candidates)
    _write_csv(output_dir / outputs.get("decisions", "openai_fallback_decisions.csv"), decisions)


def _call_openai(candidate: dict[str, Any], crop_path: Path, cfg: dict[str, Any], api_key: str) -> dict[str, Any]:
    image_b64 = base64.b64encode(crop_path.read_bytes()).decode("ascii")
    mime = "image/png" if crop_path.suffix.lower() == ".png" else "image/jpeg"
    candidate_a = str(candidate.get("candidate_a"))
    candidate_b = str(candidate.get("candidate_b"))
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "chosen_class": {"type": "string", "enum": [candidate_a, candidate_b, "uncertain"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["chosen_class", "confidence", "reason"],
    }
    payload = {
        "model": cfg.get("model", "gpt-5-mini"),
        "reasoning": {"effort": cfg.get("reasoning_effort", "minimal")},
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are resolving an ambiguous vehicle crop from a drone video. "
                            f"Choose only between '{candidate_a}' and '{candidate_b}', or 'uncertain'. "
                            "Use visible vehicle shape, relative size, and proportions. Do not invent another class. "
                            f"Geometry context: area_ratio_to_car={candidate.get('area_ratio_to_car')}, "
                            f"normalized_area={candidate.get('bbox_normalized_area')}, "
                            f"normalized_aspect={candidate.get('bbox_normalized_aspect')}."
                        ),
                    },
                    {"type": "input_image", "image_url": f"data:{mime};base64,{image_b64}", "detail": "low"},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "vehicle_tiebreak",
                "schema": schema,
                "strict": True,
            }
        },
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = _extract_output_text(body)
    parsed = json.loads(text)
    min_conf = float((cfg.get("decision_policy") or {}).get("min_model_confidence", 0.55))
    chosen = parsed.get("chosen_class", "uncertain")
    confidence = float(parsed.get("confidence", 0.0))
    if chosen not in {candidate_a, candidate_b, "uncertain"}:
        chosen = "uncertain"
    if confidence < min_conf:
        chosen = "uncertain"
    return _decision(candidate, "resolved", chosen, confidence, parsed.get("reason", ""))


def _extract_output_text(body: dict[str, Any]) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    for item in body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return str(content.get("text", ""))
    raise ValueError("OpenAI response did not contain output_text")


def _representative_frame(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return sorted(rows, key=lambda row: (_float(row.get("postprocessed_conf", row.get("conf"))), _float(row.get("typology_geometry_score"))), reverse=True)[0]


def _decision(candidate: dict[str, Any], status: str, chosen: str, confidence: float, reason: str) -> dict[str, Any]:
    out = dict(candidate)
    out.update({"openai_status": status, "openai_chosen_class": chosen, "openai_confidence": round(confidence, 6), "openai_reason": reason})
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
