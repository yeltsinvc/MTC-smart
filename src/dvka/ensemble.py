from __future__ import annotations

"""Weighted Boxes Fusion (WBF) + optional Test-Time Augmentation (TTA).

For a detection competition the single biggest lever beyond a good model is usually
fusing several models and averaging over augmented views. This module provides:

  * weighted_boxes_fusion(): a dependency-free, faithful implementation of WBF
    (Solovyev et al., 2019) so it can be unit-tested offline without torch.
  * fuse_models_on_images(): runs each trained model (optionally with TTA) over a
    folder of images and fuses the per-image detections. ultralytics is imported
    lazily so importing this module never requires a GPU stack.
  * resolve_top_k_weights(): picks the best.pt of the top-k leaderboard configs.

WBF differs from NMS: instead of discarding overlapping boxes it averages them,
weighting by confidence, and rescales the fused confidence by how many models
agreed. Boxes that every model predicts keep their score; boxes only one model
saw are strongly down-weighted.
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


def _clip01(box: Box) -> Box:
    x1, y1, x2, y2 = box
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return [min(1.0, max(0.0, c)) for c in (x1, y1, x2, y2)]


def weighted_boxes_fusion(
    boxes_list: list[list[Box]],
    scores_list: list[list[float]],
    labels_list: list[list[Any]],
    weights: list[float] | None = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    conf_type: str = "avg",
) -> tuple[list[Box], list[float], list[Any]]:
    """Fuse boxes from several models. Boxes are normalized [0,1] xyxy.

    Returns fused (boxes, scores, labels) sorted by descending score.
    """
    n_models = len(boxes_list)
    if weights is None:
        weights = [1.0] * n_models
    if len(weights) != n_models:
        raise ValueError("weights length must match number of models")
    wsum = float(sum(weights)) or 1.0

    by_label: dict[Any, list[dict[str, Any]]] = {}
    for m in range(n_models):
        w = float(weights[m])
        for box, score, label in zip(boxes_list[m], scores_list[m], labels_list[m]):
            if score < skip_box_thr:
                continue
            by_label.setdefault(label, []).append(
                {"box": _clip01(list(box)), "score": float(score) * w}
            )

    out_boxes: list[Box] = []
    out_scores: list[float] = []
    out_labels: list[Any] = []
    for label, items in by_label.items():
        items.sort(key=lambda e: e["score"], reverse=True)
        clusters: list[dict[str, Any]] = []
        for e in items:
            best_ci, best_iou = -1, iou_thr
            for ci, c in enumerate(clusters):
                iou = _iou(e["box"], c["fused"])
                if iou > best_iou:
                    best_iou, best_ci = iou, ci
            if best_ci == -1:
                clusters.append({"boxes": [e["box"]], "scores": [e["score"]], "fused": list(e["box"])})
            else:
                c = clusters[best_ci]
                c["boxes"].append(e["box"])
                c["scores"].append(e["score"])
                sw = sum(c["scores"])
                fused = [0.0, 0.0, 0.0, 0.0]
                for bb, ss in zip(c["boxes"], c["scores"]):
                    for k in range(4):
                        fused[k] += ss * bb[k]
                c["fused"] = [f / sw for f in fused] if sw > 0 else list(c["boxes"][0])
        for c in clusters:
            count = len(c["scores"])
            conf = max(c["scores"]) if conf_type == "max" else sum(c["scores"]) / count
            # Rescale by model agreement: a box seen by many models keeps its score,
            # a box seen by one of N models is down-weighted toward score/N.
            conf = conf * min(n_models, count) / wsum
            out_boxes.append(c["fused"])
            out_scores.append(conf)
            out_labels.append(label)

    order = sorted(range(len(out_scores)), key=lambda i: -out_scores[i])
    return [out_boxes[i] for i in order], [out_scores[i] for i in order], [out_labels[i] for i in order]


def resolve_top_k_weights(top_k: int, output_root: Path) -> list[Path]:
    """Best.pt of the top-k leaderboard configurations (by mean selection_score)."""
    from .review_agent import summarize

    summary = summarize(output_root)
    paths: list[Path] = []
    for group in summary["groups"][:top_k]:
        run = group.get("best_run") or {}
        artifact_dir = run.get("path")
        if not artifact_dir:
            continue
        weight = Path(artifact_dir) / "best.pt"
        if weight.exists():
            paths.append(weight)
    return paths


def fuse_models_on_images(
    model_paths: list[Path],
    images_dir: Path,
    output_path: Path,
    imgsz: int = 960,
    conf: float = 0.001,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    tta: bool = True,
    weights: list[float] | None = None,
) -> dict[str, Any]:
    """Run each model (optionally with TTA) over images_dir and fuse per image.

    ultralytics/torch are imported here so the rest of the module stays importable
    on machines without a GPU stack.
    """
    from ultralytics import YOLO  # noqa: PLC0415 - lazy on purpose

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in exts)
    if not images:
        raise FileNotFoundError(f"No images found under {images_dir}")
    if not model_paths:
        raise ValueError("No model weights provided to ensemble")

    # Per image: accumulate each model's normalized detections.
    per_image_boxes: dict[str, list[list[Box]]] = {p.name: [] for p in images}
    per_image_scores: dict[str, list[list[float]]] = {p.name: [] for p in images}
    per_image_labels: dict[str, list[list[int]]] = {p.name: [] for p in images}
    names: dict[int, str] = {}

    for model_path in model_paths:
        model = YOLO(str(model_path))
        results = model.predict(
            source=[str(p) for p in images],
            imgsz=imgsz,
            conf=conf,
            iou=0.7,
            augment=tta,
            save=False,
            verbose=False,
        )
        for result in results:
            name = Path(result.path).name
            if not names:
                names = {int(k): str(v) for k, v in result.names.items()}
            h, w = getattr(result, "orig_shape", (1, 1))
            boxes_obj = getattr(result, "boxes", None)
            img_boxes: list[Box] = []
            img_scores: list[float] = []
            img_labels: list[int] = []
            if boxes_obj is not None and w and h:
                for box in boxes_obj:
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                    img_boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
                    img_scores.append(float(box.conf.item()))
                    img_labels.append(int(box.cls.item()))
            per_image_boxes.setdefault(name, []).append(img_boxes)
            per_image_scores.setdefault(name, []).append(img_scores)
            per_image_labels.setdefault(name, []).append(img_labels)

    fused: dict[str, list[dict[str, Any]]] = {}
    for name in per_image_boxes:
        boxes, scores, labels = weighted_boxes_fusion(
            per_image_boxes[name],
            per_image_scores[name],
            per_image_labels[name],
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
            conf_type="avg",
        )
        fused[name] = [
            {
                "bbox_xyxy_normalized": [round(c, 6) for c in box],
                "score": round(score, 6),
                "class_id": int(label),
                "class": names.get(int(label), str(label)),
            }
            for box, score, label in zip(boxes, scores, labels)
        ]

    payload = {
        "models": [str(p) for p in model_paths],
        "tta": tta,
        "iou_thr": iou_thr,
        "imgsz": imgsz,
        "images": len(images),
        "detections": fused,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"images": len(images), "models": len(model_paths), "output": str(output_path)}
