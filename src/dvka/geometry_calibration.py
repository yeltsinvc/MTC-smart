from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from .config import load_class_map
from .io import write_json
from .paths import STATE


def calibrate_from_dataset(dataset_root: Path, out_path: Path | None = None) -> dict[str, Any]:
    """Learn bbox geometry priors from labeled boxes.

    The calibration is intentionally lightweight: normalized area and normalized
    aspect ratio are available from YOLO labels and Roboflow CSV annotations.
    """
    class_map = load_class_map()
    samples: dict[str, list[dict[str, float]]] = defaultdict(list)
    samples.update(_read_roboflow_csv(dataset_root, class_map))
    yolo_samples = _read_yolo_labels(dataset_root, class_map)
    for cls, rows in yolo_samples.items():
        samples[cls].extend(rows)

    priors: dict[str, Any] = {}
    for cls, rows in sorted(samples.items()):
        areas = sorted(row["normalized_area"] for row in rows if row["normalized_area"] > 0)
        aspects = sorted(row["normalized_aspect"] for row in rows if row["normalized_aspect"] > 0)
        if not areas or not aspects:
            continue
        priors[cls] = {
            "count": len(rows),
            "area_p05": _quantile(areas, 0.05),
            "area_p10": _quantile(areas, 0.10),
            "area_p50": _quantile(areas, 0.50),
            "area_p90": _quantile(areas, 0.90),
            "area_p95": _quantile(areas, 0.95),
            "aspect_p10": _quantile(aspects, 0.10),
            "aspect_p50": _quantile(aspects, 0.50),
            "aspect_p90": _quantile(aspects, 0.90),
        }

    car_area = priors.get("car", {}).get("area_p50")
    for cls, row in priors.items():
        if car_area:
            row["area_ratio_to_car_p50"] = row["area_p50"] / car_area

    payload = {
        "dataset_root": str(dataset_root.resolve()),
        "classes": list(priors),
        "reference_class": "car" if "car" in priors else (next(iter(priors), "")),
        "priors": priors,
        "notes": "Generated from labeled bounding boxes. Calibrate again whenever labels or splits change.",
    }
    write_json(out_path or STATE / "geometry_calibration.json", payload)
    return payload


def score_against_priors(
    normalized_area: float,
    normalized_aspect: float,
    base_class: str,
    priors_payload: dict[str, Any],
    allowed_classes: list[str] | None = None,
) -> tuple[str | None, float, str]:
    priors = priors_payload.get("priors") or {}
    if not priors:
        return None, 0.0, "no_calibration_priors"
    allowed = allowed_classes or list(priors)
    candidates: list[tuple[str, float, str]] = []
    for cls in allowed:
        prior = priors.get(cls)
        if not prior:
            continue
        area_score = _range_score(normalized_area, prior["area_p10"], prior["area_p50"], prior["area_p90"])
        aspect_score = _range_score(normalized_aspect, prior["aspect_p10"], prior["aspect_p50"], prior["aspect_p90"])
        base_bonus = 0.12 if cls == base_class else 0.0
        score = min(1.0, 0.62 * area_score + 0.26 * aspect_score + base_bonus)
        candidates.append((cls, score, f"calibrated_area={normalized_area:.6g},aspect={normalized_aspect:.3f}"))
    if not candidates:
        return None, 0.0, "no_matching_calibration_class"
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0]


def _read_roboflow_csv(dataset_root: Path, class_map: dict[str, Any]) -> dict[str, list[dict[str, float]]]:
    aliases = {str(k).lower(): str(v) for k, v in (class_map.get("aliases") or {}).items()}
    out: dict[str, list[dict[str, float]]] = defaultdict(list)
    required = {"width", "height", "class", "xmin", "ymin", "xmax", "ymax"}
    for csv_path in dataset_root.rglob("_annotations.csv"):
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not required <= set(reader.fieldnames or []):
                continue
            for row in reader:
                cls = _canonical(str(row.get("class", "")), aliases)
                try:
                    image_w = float(row["width"])
                    image_h = float(row["height"])
                    box_w = max(0.0, float(row["xmax"]) - float(row["xmin"]))
                    box_h = max(0.0, float(row["ymax"]) - float(row["ymin"]))
                except ValueError:
                    continue
                if image_w <= 0 or image_h <= 0 or box_w <= 0 or box_h <= 0:
                    continue
                out[cls].append(
                    {
                        "normalized_area": (box_w * box_h) / (image_w * image_h),
                        "normalized_aspect": (box_w / image_w) / (box_h / image_h),
                    }
                )
    return out


def _read_yolo_labels(dataset_root: Path, class_map: dict[str, Any]) -> dict[str, list[dict[str, float]]]:
    names = _read_yaml_names(dataset_root) or list(class_map.get("classes") or [])
    out: dict[str, list[dict[str, float]]] = defaultdict(list)
    for label_path in dataset_root.rglob("*.txt"):
        if "labels" not in {part.lower() for part in label_path.parts}:
            continue
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                norm_w = float(parts[3])
                norm_h = float(parts[4])
            except ValueError:
                continue
            if cls_id < 0 or cls_id >= len(names) or norm_w <= 0 or norm_h <= 0:
                continue
            out[names[cls_id]].append(
                {
                    "normalized_area": norm_w * norm_h,
                    "normalized_aspect": norm_w / norm_h,
                }
            )
    return out


def _read_yaml_names(dataset_root: Path) -> list[str]:
    for yaml_path in list(dataset_root.rglob("data.yaml")) + list(dataset_root.rglob("*.yaml")):
        text = yaml_path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("names:") and "[" in stripped and "]" in stripped:
                raw = stripped.split(":", 1)[1].strip().strip("[]")
                return [item.strip(" '\"") for item in raw.split(",") if item.strip()]
    return []


def _canonical(name: str, aliases: dict[str, str]) -> str:
    clean = " ".join(name.strip().replace("_", " ").lower().split())
    return aliases.get(clean, name.strip())


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return float(values[idx])


def _range_score(value: float, low: float, center: float, high: float) -> float:
    if value <= 0 or low <= 0 or high <= 0:
        return 0.0
    if low <= value <= high:
        denom = max(center - low, high - center, 1e-9)
        return max(0.25, 1.0 - abs(value - center) / denom * 0.45)
    if value < low:
        return max(0.0, 0.25 * value / low)
    return max(0.0, 0.25 * high / value)
