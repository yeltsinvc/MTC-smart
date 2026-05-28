from __future__ import annotations

"""Measure whether the geometric typology layer helps or hurts the competition metric.

The typology / tracking layers *re-label* detections. If the official metric is a
standard per-detection mAP, re-labeling can just as easily flip a correct class to a
wrong one. This module evaluates, on a labeled set, the mAP@0.5 of the raw detector
classes (`base_class`) against the post-processed classes (`postprocessed_class`),
plus a class-flip breakdown, and returns a verdict so the layer is only shipped when
it actually improves the objective.

Everything here is dependency-free and unit-testable: boxes are normalized [0,1] xyxy.
"""

import json
from pathlib import Path
from typing import Any

Box = list[float]


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _average_precision(recalls: list[float], precisions: list[float]) -> float:
    """All-point (VOC2010+) area under the precision-recall curve."""
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def compute_map50(
    predictions: list[dict[str, Any]],
    ground_truth: dict[str, list[dict[str, Any]]],
    classes: list[str],
    label_field: str,
    iou_thr: float = 0.5,
) -> dict[str, Any]:
    """mAP@0.5 using `label_field` ("base_class" or "postprocessed_class") as the
    predicted class. predictions: [{image, conf, box(norm xyxy), <label_field>}].
    ground_truth: {image: [{class, box(norm xyxy)}]}.
    """
    gt_counts: dict[str, int] = {c: 0 for c in classes}
    for boxes in ground_truth.values():
        for g in boxes:
            cls = str(g.get("class"))
            if cls in gt_counts:
                gt_counts[cls] += 1

    ap_per_class: dict[str, float] = {}
    for cls in classes:
        cls_preds = [p for p in predictions if str(p.get(label_field)) == cls]
        cls_preds.sort(key=lambda p: float(p.get("conf", 0.0)), reverse=True)
        n_gt = gt_counts.get(cls, 0)
        if n_gt == 0:
            # No ground truth for this class: AP is 0 if there are false positives,
            # otherwise undefined -> excluded from the mean by convention (skip).
            ap_per_class[cls] = 0.0 if cls_preds else float("nan")
            continue
        matched: dict[str, set[int]] = {}
        tp: list[int] = []
        fp: list[int] = []
        for p in cls_preds:
            image = str(p.get("image"))
            gts = [(i, g) for i, g in enumerate(ground_truth.get(image, [])) if str(g.get("class")) == cls]
            best_iou, best_idx = iou_thr, -1
            for i, g in gts:
                iou = _iou(p.get("box", [0, 0, 0, 0]), g.get("box", [0, 0, 0, 0]))
                if iou >= best_iou and i not in matched.get(image, set()):
                    best_iou, best_idx = iou, i
            if best_idx >= 0:
                matched.setdefault(image, set()).add(best_idx)
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
        # cumulative precision / recall
        cum_tp = cum_fp = 0
        recalls: list[float] = []
        precisions: list[float] = []
        for t, f in zip(tp, fp):
            cum_tp += t
            cum_fp += f
            recalls.append(cum_tp / n_gt)
            precisions.append(cum_tp / (cum_tp + cum_fp))
        ap_per_class[cls] = _average_precision(recalls, precisions) if recalls else 0.0

    valid = [v for v in ap_per_class.values() if v == v]  # drop NaN
    mean_ap = sum(valid) / len(valid) if valid else 0.0
    return {"map50": mean_ap, "ap_per_class": ap_per_class, "gt_counts": gt_counts}


def _flip_impact(
    predictions: list[dict[str, Any]],
    ground_truth: dict[str, list[dict[str, Any]]],
    iou_thr: float = 0.5,
) -> dict[str, int]:
    """For predictions matched to a GT box, count how the typology relabeling changed
    correctness vs the raw class: helped (wrong->right), hurt (right->wrong), or no-op."""
    counts = {"matched": 0, "helped": 0, "hurt": 0, "unchanged_correct": 0, "unchanged_wrong": 0}
    used: dict[str, set[int]] = {}
    ordered = sorted(predictions, key=lambda p: float(p.get("conf", 0.0)), reverse=True)
    for p in ordered:
        image = str(p.get("image"))
        gts = ground_truth.get(image, [])
        best_iou, best_idx = iou_thr, -1
        for i, g in enumerate(gts):
            if i in used.get(image, set()):
                continue
            iou = _iou(p.get("box", [0, 0, 0, 0]), g.get("box", [0, 0, 0, 0]))
            if iou >= best_iou:
                best_iou, best_idx = iou, i
        if best_idx < 0:
            continue
        used.setdefault(image, set()).add(best_idx)
        truth = str(gts[best_idx].get("class"))
        base = str(p.get("base_class"))
        post = str(p.get("postprocessed_class"))
        counts["matched"] += 1
        base_ok = base == truth
        post_ok = post == truth
        if base_ok and post_ok:
            counts["unchanged_correct"] += 1
        elif not base_ok and not post_ok:
            counts["unchanged_wrong"] += 1
        elif post_ok and not base_ok:
            counts["helped"] += 1
        elif base_ok and not post_ok:
            counts["hurt"] += 1
    return counts


def evaluate_typology(
    predictions: list[dict[str, Any]],
    ground_truth: dict[str, list[dict[str, Any]]],
    classes: list[str],
    iou_thr: float = 0.5,
) -> dict[str, Any]:
    base = compute_map50(predictions, ground_truth, classes, "base_class", iou_thr)
    post = compute_map50(predictions, ground_truth, classes, "postprocessed_class", iou_thr)
    flips = _flip_impact(predictions, ground_truth, iou_thr)
    delta = post["map50"] - base["map50"]
    if delta > 1e-6:
        verdict = "USE_TYPOLOGY: post-processing improves mAP50"
    elif delta < -1e-6:
        verdict = "DROP_TYPOLOGY: post-processing hurts mAP50; submit the raw detector classes"
    else:
        verdict = "NEUTRAL: no measurable effect on mAP50"
    return {
        "base_map50": round(base["map50"], 6),
        "post_map50": round(post["map50"], 6),
        "delta_map50": round(delta, 6),
        "verdict": verdict,
        "flip_impact": flips,
        "base_ap_per_class": {k: round(v, 6) for k, v in base["ap_per_class"].items()},
        "post_ap_per_class": {k: round(v, 6) for k, v in post["ap_per_class"].items()},
    }


# --- loaders -----------------------------------------------------------------

def normalize_predictions(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accept the worker's predictions.json rows (xyxy in pixels + image_width/height)
    and emit normalized-box rows keyed by image stem."""
    out: list[dict[str, Any]] = []
    for r in raw:
        xyxy = r.get("xyxy") or r.get("bbox_xyxy") or [0, 0, 0, 0]
        w = float(r.get("image_width") or 0) or 1.0
        h = float(r.get("image_height") or 0) or 1.0
        # If the box already looks normalized (<=1), keep it; else divide by image size.
        if max(xyxy) <= 1.0:
            box = [float(v) for v in xyxy]
        else:
            box = [float(xyxy[0]) / w, float(xyxy[1]) / h, float(xyxy[2]) / w, float(xyxy[3]) / h]
        out.append(
            {
                "image": Path(str(r.get("image", ""))).stem,
                "conf": float(r.get("conf", r.get("postprocessed_conf", 0.0)) or 0.0),
                "box": box,
                "base_class": str(r.get("base_class") or r.get("class") or ""),
                "postprocessed_class": str(r.get("postprocessed_class") or r.get("class") or ""),
            }
        )
    return out


def load_yolo_ground_truth(labels_dir: Path, names: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Read YOLO label files (class cx cy w h, normalized) into {image_stem: [...]}."""
    gt: dict[str, list[dict[str, Any]]] = {}
    for txt in labels_dir.rglob("*.txt"):
        boxes: list[dict[str, Any]] = []
        for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            if cid < 0 or cid >= len(names):
                continue
            boxes.append({"class": names[cid], "box": [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]})
        gt[txt.stem] = boxes
    return gt


def evaluate_from_files(
    predictions_path: Path,
    ground_truth: dict[str, list[dict[str, Any]]],
    classes: list[str],
    iou_thr: float = 0.5,
) -> dict[str, Any]:
    raw = json.loads(predictions_path.read_text(encoding="utf-8"))
    rows = raw.get("predictions", raw) if isinstance(raw, dict) else raw
    preds = normalize_predictions(rows)
    return evaluate_typology(preds, ground_truth, classes, iou_thr)
