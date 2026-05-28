from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import read_json
from .paths import CONFIGS
from .geometry_calibration import score_against_priors


@dataclass(frozen=True)
class Detection:
    image: str
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    image_width: float
    image_height: float


def load_rules(path: Path | None = None) -> dict[str, Any]:
    return read_json(path or CONFIGS / "typology_rules.json", {})


def load_calibration(path: Path | None = None) -> dict[str, Any]:
    from .paths import STATE

    return read_json(path or STATE / "geometry_calibration.json", {})


def enrich_detections(
    detections: list[dict[str, Any]],
    rules: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Add geometry features and optional typology override.

    This is intentionally deterministic. It should be calibrated on validation
    folds before being trusted for final submissions.
    """
    rules = rules or load_rules()
    calibration = calibration or load_calibration()
    if not rules.get("enabled", True):
        return detections

    grouped: dict[str, list[dict[str, Any]]] = {}
    for det in detections:
        grouped.setdefault(str(det.get("image", "")), []).append(det)

    enriched: list[dict[str, Any]] = []
    for image_name, image_detections in grouped.items():
        ref_area = _reference_area(image_detections, rules)
        for det in image_detections:
            item = dict(det)
            features = _features(item, ref_area)
            item.update(features)
            typology, geometry_score, reason = _classify_typology(item, rules, calibration)
            item["base_class"] = item.get("class")
            item["typology_class"] = typology
            item["typology_geometry_score"] = round(geometry_score, 6)
            item["typology_reason"] = reason
            item["postprocessed_class"] = typology or item.get("class")
            item["postprocessed_conf"] = _blend_confidence(float(item.get("conf", 0.0)), geometry_score, rules)
            enriched.append(item)
    return enriched


def _reference_area(detections: list[dict[str, Any]], rules: dict[str, Any]) -> float:
    ref_cfg = rules.get("reference") or {}
    ref_class = str(ref_cfg.get("class") or "car")
    min_conf = float(ref_cfg.get("min_conf") or 0.0)
    fallback = float(ref_cfg.get("fallback_normalized_area") or 0.001)
    areas = []
    for det in detections:
        if det.get("class") != ref_class or float(det.get("conf", 0.0)) < min_conf:
            continue
        area = _normalized_area(det)
        if area > 0:
            areas.append(area)
    if not areas:
        return fallback
    areas.sort()
    return areas[len(areas) // 2]


def _normalized_area(det: dict[str, Any]) -> float:
    xyxy = det.get("xyxy") or det.get("box")
    if not xyxy or len(xyxy) < 4:
        return 0.0
    width = max(0.0, float(xyxy[2]) - float(xyxy[0]))
    height = max(0.0, float(xyxy[3]) - float(xyxy[1]))
    image_width = float(det.get("image_width") or det.get("width") or 1.0)
    image_height = float(det.get("image_height") or det.get("height") or 1.0)
    if image_width <= 0 or image_height <= 0:
        return 0.0
    return (width * height) / (image_width * image_height)


def _features(det: dict[str, Any], ref_area: float) -> dict[str, float]:
    xyxy = det.get("xyxy") or det.get("box") or [0, 0, 0, 0]
    width = max(0.0, float(xyxy[2]) - float(xyxy[0]))
    height = max(0.0, float(xyxy[3]) - float(xyxy[1]))
    area = _normalized_area(det)
    image_width = float(det.get("image_width") or det.get("width") or 1.0)
    image_height = float(det.get("image_height") or det.get("height") or 1.0)
    normalized_aspect = (width / image_width) / (height / image_height) if height > 0 and image_width > 0 and image_height > 0 else 0.0
    return {
        "bbox_width_px": round(width, 6),
        "bbox_height_px": round(height, 6),
        "bbox_aspect_ratio": round(width / height, 6) if height > 0 else 0.0,
        "bbox_normalized_area": round(area, 10),
        "bbox_normalized_aspect": round(normalized_aspect, 6),
        "reference_car_area": round(ref_area, 10),
        "area_ratio_to_car": round(area / ref_area, 6) if ref_area > 0 else 0.0,
    }


def _classify_typology(
    det: dict[str, Any],
    rules: dict[str, Any],
    calibration: dict[str, Any] | None = None,
) -> tuple[str | None, float, str]:
    base = str(det.get("class") or "")
    if calibration and calibration.get("priors"):
        allowed = [str(rule.get("target")) for rule in rules.get("rules", []) if base in set(rule.get("allowed_base_classes") or [base])]
        calibrated = score_against_priors(
            float(det.get("bbox_normalized_area") or 0.0),
            float(det.get("bbox_normalized_aspect") or 0.0),
            base,
            calibration,
            allowed_classes=allowed or None,
        )
        if calibrated[1] >= 0.45:
            return calibrated
    ratio = float(det.get("area_ratio_to_car") or 0.0)
    aspect = float(det.get("bbox_aspect_ratio") or 0.0)
    candidates: list[tuple[str, float, str]] = []
    for rule in rules.get("rules", []):
        allowed = set(rule.get("allowed_base_classes") or [])
        if allowed and base not in allowed:
            continue
        if "min_area_ratio_to_car" in rule and ratio < float(rule["min_area_ratio_to_car"]):
            continue
        if "max_area_ratio_to_car" in rule and ratio > float(rule["max_area_ratio_to_car"]):
            continue
        if "min_aspect_ratio" in rule and aspect < float(rule["min_aspect_ratio"]):
            continue
        if "max_aspect_ratio" in rule and aspect > float(rule["max_aspect_ratio"]):
            continue
        score = _geometry_score(rule, ratio, aspect)
        candidates.append((str(rule["target"]), score, f"matched_area_ratio={ratio:.3f},aspect={aspect:.3f}"))
    if not candidates:
        return None, 0.0, "no_geometry_rule_match"
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0]


def _geometry_score(rule: dict[str, Any], ratio: float, aspect: float) -> float:
    score = 0.5
    min_ratio = rule.get("min_area_ratio_to_car")
    max_ratio = rule.get("max_area_ratio_to_car")
    if min_ratio is not None and max_ratio is not None:
        center = (float(min_ratio) + float(max_ratio)) / 2.0
        span = max(1e-6, float(max_ratio) - float(min_ratio))
        score += max(0.0, 0.35 * (1.0 - abs(ratio - center) / span))
    else:
        score += 0.2
    min_aspect = rule.get("min_aspect_ratio")
    max_aspect = rule.get("max_aspect_ratio")
    if min_aspect is not None and aspect >= float(min_aspect):
        score += 0.1
    if max_aspect is not None and aspect <= float(max_aspect):
        score += 0.1
    return min(1.0, score)


def _blend_confidence(detector_conf: float, geometry_score: float, rules: dict[str, Any]) -> float:
    cfg = rules.get("confidence_blend") or {}
    detector_weight = float(cfg.get("detector_weight", 0.75))
    geometry_weight = float(cfg.get("geometry_weight", 0.25))
    min_output = float(cfg.get("min_output_conf", 0.001))
    if geometry_score <= 0:
        return max(min_output, detector_conf)
    return max(min_output, detector_conf * detector_weight + geometry_score * geometry_weight)
