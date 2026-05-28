from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import read_json
from .paths import CONFIGS


def load_tracking_config(path: Path | None = None) -> dict[str, Any]:
    return read_json(path or CONFIGS / "tracking_postprocess.json", {})


def consensus_tracks(rows: list[dict[str, Any]], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_tracking_config()
    class_cfg = cfg.get("class_consensus") or {}
    min_len = int(class_cfg.get("min_track_length", 3))
    confidence_w = float(class_cfg.get("confidence_weight", 0.55))
    geometry_w = float(class_cfg.get("geometry_weight", 0.30))
    frequency_w = float(class_cfg.get("frequency_weight", 0.15))
    margin = float(class_cfg.get("switch_margin", 0.08))

    # Detections without a track_id (the tracker could not associate them) cannot
    # participate in the consensus vote, but dropping them silently loses recall.
    # We keep them as pass-through rows in `corrected_rows` so the downstream CSV
    # still sees every detection; only the per-track aggregates exclude them.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    untracked: list[dict[str, Any]] = []
    for row in rows:
        track_id = str(row.get("track_id", ""))
        if not track_id:
            untracked.append(row)
            continue
        grouped[(str(row.get("video", "")), track_id)].append(row)

    summaries: list[dict[str, Any]] = []
    vote_rows: list[dict[str, Any]] = []
    corrected_rows: list[dict[str, Any]] = []
    for (video, track_id), track_rows in grouped.items():
        votes: dict[str, dict[str, float]] = defaultdict(lambda: {"conf": 0.0, "geom": 0.0, "freq": 0.0})
        class_counter: Counter[str] = Counter()
        for row in track_rows:
            cls = str(row.get("postprocessed_class") or row.get("class") or "")
            if not cls:
                continue
            conf = _float(row.get("postprocessed_conf", row.get("conf", 0.0)))
            geom = _float(row.get("typology_geometry_score", 0.0))
            votes[cls]["conf"] += conf
            votes[cls]["geom"] += geom
            votes[cls]["freq"] += 1.0
            class_counter[cls] += 1

        scored = []
        total_freq = max(1.0, sum(v["freq"] for v in votes.values()))
        total_conf = max(1e-9, sum(v["conf"] for v in votes.values()))
        total_geom = max(1e-9, sum(v["geom"] for v in votes.values()))
        for cls, vals in votes.items():
            score = (
                confidence_w * (vals["conf"] / total_conf)
                + geometry_w * (vals["geom"] / total_geom)
                + frequency_w * (vals["freq"] / total_freq)
            )
            scored.append((cls, score, vals))
            vote_rows.append(
                {
                    "video": video,
                    "track_id": track_id,
                    "class": cls,
                    "score": round(score, 6),
                    "conf_vote": round(vals["conf"], 6),
                    "geometry_vote": round(vals["geom"], 6),
                    "frequency": int(vals["freq"]),
                }
            )
        scored.sort(key=lambda item: item[1], reverse=True)
        final_class = scored[0][0] if scored else ""
        final_score = scored[0][1] if scored else 0.0
        runner_up = scored[1][1] if len(scored) > 1 else 0.0
        stable = len(track_rows) >= min_len and (final_score - runner_up) >= margin
        if not stable and class_counter:
            final_class = class_counter.most_common(1)[0][0]

        summaries.append(
            {
                "video": video,
                "track_id": track_id,
                "frames": len(track_rows),
                "final_class": final_class,
                "final_score": round(final_score, 6),
                "runner_up_score": round(runner_up, 6),
                "stable": stable,
                "raw_classes": "|".join(f"{cls}:{count}" for cls, count in class_counter.most_common()),
                "mean_area_ratio_to_car": round(_mean(_float(r.get("area_ratio_to_car", 0.0)) for r in track_rows), 6),
                "mean_bbox_normalized_area": round(_mean(_float(r.get("bbox_normalized_area", 0.0)) for r in track_rows), 10),
            }
        )

        for row in track_rows:
            corrected = dict(row)
            corrected["track_final_class"] = final_class
            corrected["track_final_score"] = round(final_score, 6)
            corrected["track_class_stable"] = stable
            corrected_rows.append(corrected)

    for row in untracked:
        corrected = dict(row)
        corrected["track_final_class"] = str(row.get("postprocessed_class") or row.get("class") or "")
        corrected["track_final_score"] = 0.0
        corrected["track_class_stable"] = False
        corrected_rows.append(corrected)

    return {"frame_rows": corrected_rows, "track_summary": summaries, "vote_rows": vote_rows}


def write_tracking_outputs(result: dict[str, Any], output_dir: Path, cfg: dict[str, Any] | None = None) -> None:
    cfg = cfg or load_tracking_config()
    outputs = cfg.get("outputs") or {}
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / outputs.get("frame_detections", "tracking_frame_detections.csv"), result.get("frame_rows", []))
    _write_csv(output_dir / outputs.get("track_summary", "tracking_track_summary.csv"), result.get("track_summary", []))
    _write_csv(output_dir / outputs.get("track_class_votes", "tracking_class_votes.csv"), result.get("vote_rows", []))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _mean(values: Any) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
