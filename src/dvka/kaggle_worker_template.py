from __future__ import annotations

import json
from textwrap import dedent
from typing import Any


def render_worker(
    exp: dict[str, Any],
    class_map: dict[str, Any],
    typology_rules: dict[str, Any] | None = None,
    tracking_config: dict[str, Any] | None = None,
    openai_fallback: dict[str, Any] | None = None,
) -> str:
    # Serialize as base64-wrapped JSON so arbitrary string content (quotes, triple
    # quotes, backslashes, newlines) cannot break the raw-string template or inject
    # Python tokens. The worker decodes back into a dict at startup.
    def encode(payload: Any) -> str:
        raw = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        import base64
        return base64.b64encode(raw).decode("ascii")

    return (
        WORKER.replace("__EXPERIMENT_JSON_B64__", encode(exp))
        .replace("__CLASS_MAP_JSON_B64__", encode(class_map))
        .replace("__TYPOLOGY_RULES_JSON_B64__", encode(typology_rules or {}))
        .replace("__TRACKING_CONFIG_JSON_B64__", encode(tracking_config or {}))
        .replace("__OPENAI_FALLBACK_JSON_B64__", encode(openai_fallback or {}))
    )


WORKER = r'''
from __future__ import annotations

import csv
import base64
import glob
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

def _load_config(token: str) -> dict[str, Any]:
    return json.loads(base64.b64decode(token).decode("utf-8"))


EXP = _load_config("__EXPERIMENT_JSON_B64__")
CLASS_MAP = _load_config("__CLASS_MAP_JSON_B64__")
TYPOLOGY_RULES = _load_config("__TYPOLOGY_RULES_JSON_B64__")
TRACKING_CONFIG = _load_config("__TRACKING_CONFIG_JSON_B64__")
OPENAI_FALLBACK = _load_config("__OPENAI_FALLBACK_JSON_B64__")
KAGGLE_INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
KAGGLE_WORKING = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
ARTIFACTS = KAGGLE_WORKING / "artifacts"
DATASET_WORK = KAGGLE_WORKING / "dataset_yolo"
CROP_DIR = ARTIFACTS / "openai_crops"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
GEOMETRY_PRIORS: dict[str, Any] = {}


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def install() -> None:
    sh([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "ultralytics>=8.3.0", "pyyaml"])


def find_dataset_root() -> Path:
    explicit = os.environ.get("KAGGLE_DATASET_DIR", "").strip()
    if explicit and Path(explicit).exists():
        return Path(explicit)
    slug = str(EXP.get("dataset_slug") or "").strip()
    if slug:
        owner, _, name = slug.partition("/")
        candidates = [KAGGLE_INPUT / name, KAGGLE_INPUT / slug, KAGGLE_INPUT / "datasets" / owner / name]
        for path in candidates:
            if path.exists():
                return path
    children = [p for p in KAGGLE_INPUT.iterdir() if p.is_dir()] if KAGGLE_INPUT.exists() else []
    if len(children) == 1:
        return children[0]
    raise FileNotFoundError(f"Cannot resolve dataset root under {KAGGLE_INPUT}. Set dataset_slug in configs/project.json.")


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def canonical_class(name: str) -> str:
    aliases = {str(k).lower(): str(v) for k, v in (CLASS_MAP.get("aliases") or {}).items()}
    clean = " ".join(name.strip().replace("_", " ").lower().split())
    return aliases.get(clean, name.strip())


def image_files(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def video_files(root: Path) -> list[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def normalize_existing_yaml(dataset_root: Path, yaml_path: Path) -> Path:
    cfg = load_yaml(yaml_path)
    cfg["path"] = str(Path(cfg.get("path") or yaml_path.parent).resolve())
    if "val" not in cfg and "valid" in cfg:
        cfg["val"] = cfg["valid"]
    if "names" not in cfg:
        cfg["names"] = CLASS_MAP.get("classes", [])
    out = KAGGLE_WORKING / "data.yaml"
    write_yaml(out, cfg)
    return out


def yolo_from_csv(dataset_root: Path) -> Path | None:
    csv_paths = sorted(dataset_root.rglob("_annotations.csv"))
    if not csv_paths:
        return None
    classes = list(CLASS_MAP.get("classes") or [])
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    for split_dir in ("train", "val", "test"):
        (DATASET_WORK / "images" / split_dir).mkdir(parents=True, exist_ok=True)
        (DATASET_WORK / "labels" / split_dir).mkdir(parents=True, exist_ok=True)
    required = {"filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"}
    for csv_path in csv_paths:
        split = csv_path.parent.name.lower()
        split = "val" if split in {"valid", "validation"} else split
        split = split if split in {"train", "val", "test"} else "train"
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not required <= set(reader.fieldnames or []):
                continue
            labels: dict[str, list[str]] = {}
            for row in reader:
                cls = canonical_class(str(row.get("class", "")))
                if cls not in class_to_id:
                    continue
                width = float(row["width"]); height = float(row["height"])
                xmin = max(0.0, min(width, float(row["xmin"])))
                ymin = max(0.0, min(height, float(row["ymin"])))
                xmax = max(0.0, min(width, float(row["xmax"])))
                ymax = max(0.0, min(height, float(row["ymax"])))
                if width <= 0 or height <= 0 or xmax <= xmin or ymax <= ymin:
                    continue
                x = ((xmin + xmax) / 2.0) / width
                y = ((ymin + ymax) / 2.0) / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                labels.setdefault(row["filename"], []).append(f"{class_to_id[cls]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            for image_name, lines in labels.items():
                src = csv_path.parent / image_name
                if not src.exists():
                    matches = list(csv_path.parent.rglob(image_name))
                    src = matches[0] if matches else src
                if not src.exists():
                    continue
                shutil.copy2(src, DATASET_WORK / "images" / split / src.name)
                (DATASET_WORK / "labels" / split / f"{src.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    data = {"path": str(DATASET_WORK), "train": "images/train", "val": "images/val", "test": "images/test", "names": classes}
    out = KAGGLE_WORKING / "data.yaml"
    write_yaml(out, data)
    return out


def infer_yolo_dataset(dataset_root: Path) -> Path:
    yaml_candidates = sorted(dataset_root.rglob("data.yaml")) + sorted(dataset_root.rglob("*.yaml"))
    if yaml_candidates:
        return normalize_existing_yaml(dataset_root, yaml_candidates[0])
    converted = yolo_from_csv(dataset_root)
    if converted:
        return converted
    raise FileNotFoundError("Dataset must provide data.yaml or Roboflow _annotations.csv files.")


def build_model(model_name: str):
    from ultralytics import YOLO
    try:
        if model_name.lower().startswith("rtdetr"):
            from ultralytics import RTDETR
            return RTDETR(model_name)
    except Exception as exc:
        print("RTDETR import failed, falling back to YOLO:", exc)
    return YOLO(model_name)


def metric_dict(metrics: Any, names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    box = getattr(metrics, "box", None)
    aliases = {"map": "map50_95", "map50": "map50", "mp": "precision", "mr": "recall"}
    if box is not None:
        for attr, key in aliases.items():
            value = getattr(box, attr, None)
            if value is not None:
                out[key] = float(value)
        maps = getattr(box, "maps", None)
        if maps is not None:
            per_class = {}
            for idx, value in enumerate(list(maps)):
                label = names[idx] if idx < len(names) else str(idx)
                per_class[label] = float(value)
            out["per_class_map50_95"] = per_class
            rare = [name for name in EXP.get("rare_classes", []) if name in per_class]
            if rare:
                out["rare_map50_95"] = sum(per_class[name] for name in rare) / len(rare)
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            try:
                out[str(key)] = float(value)
            except Exception:
                pass
    out["gpu_count"] = __import__("torch").cuda.device_count()
    return out


def sample_predictions(model: Any, data_yaml: Path) -> list[dict[str, Any]]:
    cfg = load_yaml(data_yaml)
    root = Path(cfg.get("path", data_yaml.parent))
    source = cfg.get("test") or cfg.get("val") or cfg.get("train")
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = root / source_path
    samples = image_files(source_path)[:8]
    if not samples:
        return []
    results = model.predict(source=[str(p) for p in samples], imgsz=int(EXP["imgsz"]), conf=0.001, save=False, verbose=False)
    rows = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes[:100]:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            rows.append({
                "image": Path(result.path).name,
                "class_id": cls_id,
                "class": result.names.get(cls_id, str(cls_id)),
                "conf": conf,
                "xyxy": xyxy,
                "image_width": int(getattr(result, "orig_shape", [0, 0])[1]),
                "image_height": int(getattr(result, "orig_shape", [0, 0])[0]),
            })
    return enrich_typology(rows)


def track_videos(model: Any, dataset_root: Path) -> dict[str, Any]:
    if not TRACKING_CONFIG.get("enabled", True):
        return {"status": "disabled", "videos": 0}
    videos = video_files(dataset_root)[: int(TRACKING_CONFIG.get("max_videos", 3))]
    if not videos:
        return {"status": "no_videos_found", "videos": 0}
    frame_rows: list[dict[str, Any]] = []
    tracker = TRACKING_CONFIG.get("tracker", "bytetrack.yaml")
    for video in videos:
        kwargs = {
            "source": str(video),
            "imgsz": int(EXP["imgsz"]),
            "conf": float(TRACKING_CONFIG.get("conf", 0.05)),
            "iou": float(TRACKING_CONFIG.get("iou", 0.5)),
            "tracker": tracker,
            "persist": True,
            "stream": True,
            "verbose": False,
        }
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
                conf = float(box.conf.item())
                crop_path = save_openai_candidate_crop(result, xyxy, video.name, frame_idx, track_id, conf)
                frame_rows.append(
                    {
                        "video": video.name,
                        "frame": frame_idx,
                        "track_id": track_id if track_id >= 0 else "",
                        "class_id": cls_id,
                        "class": result.names.get(cls_id, str(cls_id)),
                        "conf": conf,
                        "xyxy": xyxy,
                        "image_width": int(image_w),
                        "image_height": int(image_h),
                        "crop_path": crop_path,
                    }
                )
    enriched = enrich_typology(frame_rows)
    consensus = consensus_tracks(enriched)
    write_tracking_csvs(consensus)
    openai_summary = run_openai_fallback(consensus)
    return {
        "status": "ok",
        "videos": len(videos),
        "frame_detections": len(consensus["frame_rows"]),
        "tracks": len(consensus["track_summary"]),
        "tracker": tracker,
        "openai_fallback": openai_summary,
    }


def enrich_typology(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not TYPOLOGY_RULES.get("enabled", True):
        return detections
    grouped: dict[str, list[dict[str, Any]]] = {}
    for det in detections:
        # Tracking rows expose "video" (per-video reference area, since flight altitude
        # is stable inside a video); sample predictions expose "image" (per-image).
        # Without this distinction every video would share one reference area.
        key = str(det.get("video") or det.get("image", ""))
        grouped.setdefault(key, []).append(det)
    enriched: list[dict[str, Any]] = []
    for _, image_dets in grouped.items():
        ref_area = reference_car_area(image_dets)
        for det in image_dets:
            item = dict(det)
            item.update(geometry_features(item, ref_area))
            typology, score, reason = classify_by_calibration(item)
            if not typology:
                typology, score, reason = classify_by_geometry(item)
            item["base_class"] = item.get("class")
            item["typology_class"] = typology
            item["postprocessed_class"] = typology or item.get("class")
            item["typology_geometry_score"] = round(score, 6)
            item["typology_reason"] = reason
            item["postprocessed_conf"] = blend_conf(float(item.get("conf", 0.0)), score)
            enriched.append(item)
    return enriched


def reference_car_area(detections: list[dict[str, Any]]) -> float:
    cfg = TYPOLOGY_RULES.get("reference") or {}
    ref_class = str(cfg.get("class") or "car")
    min_conf = float(cfg.get("min_conf") or 0.0)
    fallback = float(cfg.get("fallback_normalized_area") or 0.001)
    areas = [normalized_area(det) for det in detections if det.get("class") == ref_class and float(det.get("conf", 0.0)) >= min_conf]
    areas = sorted(area for area in areas if area > 0)
    return areas[len(areas) // 2] if areas else fallback


def normalized_area(det: dict[str, Any]) -> float:
    xyxy = det.get("xyxy") or [0, 0, 0, 0]
    width = max(0.0, float(xyxy[2]) - float(xyxy[0]))
    height = max(0.0, float(xyxy[3]) - float(xyxy[1]))
    image_width = float(det.get("image_width") or 1.0)
    image_height = float(det.get("image_height") or 1.0)
    return (width * height) / (image_width * image_height) if image_width > 0 and image_height > 0 else 0.0


def geometry_features(det: dict[str, Any], ref_area: float) -> dict[str, float]:
    xyxy = det.get("xyxy") or [0, 0, 0, 0]
    width = max(0.0, float(xyxy[2]) - float(xyxy[0]))
    height = max(0.0, float(xyxy[3]) - float(xyxy[1]))
    area = normalized_area(det)
    image_width = float(det.get("image_width") or 1.0)
    image_height = float(det.get("image_height") or 1.0)
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


def _resolve_label_dir(image_path: Path, root: Path, split_key: str) -> Path:
    # YOLO convention: sibling "labels" dir with matching split subfolder.
    # Walk up from the images path replacing the deepest "images" segment with
    # "labels" so we cope with imgs/, dataset/images/train, or absolute paths.
    parts = list(image_path.parts)
    for idx in range(len(parts) - 1, -1, -1):
        seg = parts[idx]
        if seg.lower() in {"images", "imgs", "img"}:
            candidate = Path(*parts[:idx], "labels", *parts[idx + 1:])
            if candidate.exists():
                return candidate
    # Fallback: sibling labels/<split> directory under the dataset root.
    fallback = root / "labels" / ("val" if split_key == "val" else split_key)
    if fallback.exists():
        return fallback
    # Last resort: search anywhere under root for a labels/<split> tree.
    for cand in root.rglob("labels"):
        sub = cand / ("val" if split_key == "val" else split_key)
        if sub.is_dir():
            return sub
    return fallback


def calibrate_geometry(data_yaml: Path) -> dict[str, Any]:
    cfg = load_yaml(data_yaml)
    root = Path(cfg.get("path", data_yaml.parent))
    names = list(cfg.get("names") or CLASS_MAP.get("classes") or [])
    samples: dict[str, list[dict[str, float]]] = {}
    for split_key in ("train", "val"):
        image_dir = cfg.get(split_key)
        if not image_dir:
            continue
        image_path = Path(image_dir)
        if not image_path.is_absolute():
            image_path = root / image_path
        label_path = _resolve_label_dir(image_path, root, split_key)
        for txt in label_path.rglob("*.txt") if label_path.exists() else []:
            for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
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
                cls = names[cls_id]
                samples.setdefault(cls, []).append({"normalized_area": norm_w * norm_h, "normalized_aspect": norm_w / norm_h})
    priors = {}
    for cls, rows in sorted(samples.items()):
        areas = sorted(row["normalized_area"] for row in rows if row["normalized_area"] > 0)
        aspects = sorted(row["normalized_aspect"] for row in rows if row["normalized_aspect"] > 0)
        if not areas or not aspects:
            continue
        priors[cls] = {
            "count": len(rows),
            "area_p10": quantile(areas, 0.10),
            "area_p50": quantile(areas, 0.50),
            "area_p90": quantile(areas, 0.90),
            "aspect_p10": quantile(aspects, 0.10),
            "aspect_p50": quantile(aspects, 0.50),
            "aspect_p90": quantile(aspects, 0.90),
        }
    car_area = priors.get("car", {}).get("area_p50")
    for row in priors.values():
        if car_area:
            row["area_ratio_to_car_p50"] = row["area_p50"] / car_area
    payload = {"reference_class": "car" if "car" in priors else (next(iter(priors), "")), "priors": priors}
    (ARTIFACTS / "geometry_calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return float(values[idx])


def classify_by_calibration(det: dict[str, Any]) -> tuple[str | None, float, str]:
    priors = GEOMETRY_PRIORS.get("priors") or {}
    if not priors:
        return None, 0.0, "no_calibration_priors"
    base = str(det.get("class") or "")
    allowed = []
    for rule in TYPOLOGY_RULES.get("rules", []):
        if base in set(rule.get("allowed_base_classes") or [base]):
            allowed.append(str(rule.get("target")))
    allowed = allowed or list(priors)
    candidates: list[tuple[str, float, str]] = []
    area = float(det.get("bbox_normalized_area") or 0.0)
    aspect = float(det.get("bbox_normalized_aspect") or 0.0)
    for cls in allowed:
        prior = priors.get(cls)
        if not prior:
            continue
        area_score = range_score(area, prior["area_p10"], prior["area_p50"], prior["area_p90"])
        aspect_score = range_score(aspect, prior["aspect_p10"], prior["aspect_p50"], prior["aspect_p90"])
        base_bonus = 0.12 if cls == base else 0.0
        score = min(1.0, 0.62 * area_score + 0.26 * aspect_score + base_bonus)
        candidates.append((cls, score, f"calibrated_area={area:.6g},aspect={aspect:.3f}"))
    if not candidates:
        return None, 0.0, "no_matching_calibration_class"
    candidates.sort(key=lambda item: item[1], reverse=True)
    if candidates[0][1] < 0.45:
        return None, candidates[0][1], "weak_calibration_match"
    return candidates[0]


def range_score(value: float, low: float, center: float, high: float) -> float:
    if value <= 0 or low <= 0 or high <= 0:
        return 0.0
    if low <= value <= high:
        denom = max(center - low, high - center, 1e-9)
        return max(0.25, 1.0 - abs(value - center) / denom * 0.45)
    if value < low:
        return max(0.0, 0.25 * value / low)
    return max(0.0, 0.25 * high / value)


def classify_by_geometry(det: dict[str, Any]) -> tuple[str | None, float, str]:
    base = str(det.get("class") or "")
    ratio = float(det.get("area_ratio_to_car") or 0.0)
    aspect = float(det.get("bbox_aspect_ratio") or 0.0)
    candidates: list[tuple[str, float, str]] = []
    for rule in TYPOLOGY_RULES.get("rules", []):
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
        score = geometry_score(rule, ratio, aspect)
        candidates.append((str(rule["target"]), score, f"matched_area_ratio={ratio:.3f},aspect={aspect:.3f}"))
    if not candidates:
        return None, 0.0, "no_geometry_rule_match"
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0]


def geometry_score(rule: dict[str, Any], ratio: float, aspect: float) -> float:
    score = 0.5
    min_ratio = rule.get("min_area_ratio_to_car")
    max_ratio = rule.get("max_area_ratio_to_car")
    if min_ratio is not None and max_ratio is not None:
        center = (float(min_ratio) + float(max_ratio)) / 2.0
        span = max(1e-6, float(max_ratio) - float(min_ratio))
        score += max(0.0, 0.35 * (1.0 - abs(ratio - center) / span))
    else:
        score += 0.2
    if rule.get("min_aspect_ratio") is not None and aspect >= float(rule["min_aspect_ratio"]):
        score += 0.1
    if rule.get("max_aspect_ratio") is not None and aspect <= float(rule["max_aspect_ratio"]):
        score += 0.1
    return min(1.0, score)


def blend_conf(detector_conf: float, geometry_score_value: float) -> float:
    cfg = TYPOLOGY_RULES.get("confidence_blend") or {}
    if geometry_score_value <= 0:
        return detector_conf
    return max(
        float(cfg.get("min_output_conf", 0.001)),
        detector_conf * float(cfg.get("detector_weight", 0.75)) + geometry_score_value * float(cfg.get("geometry_weight", 0.25)),
    )


def consensus_tracks(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    cfg = TRACKING_CONFIG.get("class_consensus") or {}
    min_len = int(cfg.get("min_track_length", 3))
    confidence_w = float(cfg.get("confidence_weight", 0.55))
    geometry_w = float(cfg.get("geometry_weight", 0.30))
    frequency_w = float(cfg.get("frequency_weight", 0.15))
    margin = float(cfg.get("switch_margin", 0.08))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    untracked: list[dict[str, Any]] = []
    for row in rows:
        tid = str(row.get("track_id", ""))
        if not tid:
            # Preserve detections the tracker could not associate so we do not
            # silently drop them from the artefacts CSV. They skip the vote.
            untracked.append(row)
            continue
        grouped.setdefault((str(row.get("video", "")), tid), []).append(row)
    frame_rows: list[dict[str, Any]] = []
    track_summary: list[dict[str, Any]] = []
    vote_rows: list[dict[str, Any]] = []
    for (video, tid), track_rows in grouped.items():
        votes: dict[str, dict[str, float]] = {}
        for row in track_rows:
            cls = str(row.get("postprocessed_class") or row.get("class") or "")
            votes.setdefault(cls, {"conf": 0.0, "geom": 0.0, "freq": 0.0})
            votes[cls]["conf"] += float(row.get("postprocessed_conf", row.get("conf", 0.0)) or 0.0)
            votes[cls]["geom"] += float(row.get("typology_geometry_score", 0.0) or 0.0)
            votes[cls]["freq"] += 1.0
        total_conf = max(1e-9, sum(v["conf"] for v in votes.values()))
        total_geom = max(1e-9, sum(v["geom"] for v in votes.values()))
        total_freq = max(1.0, sum(v["freq"] for v in votes.values()))
        scored = []
        for cls, vals in votes.items():
            score = confidence_w * vals["conf"] / total_conf + geometry_w * vals["geom"] / total_geom + frequency_w * vals["freq"] / total_freq
            scored.append((cls, score, vals))
            vote_rows.append({"video": video, "track_id": tid, "class": cls, "score": round(score, 6), "frequency": int(vals["freq"])})
        scored.sort(key=lambda item: item[1], reverse=True)
        final_class = scored[0][0] if scored else ""
        final_score = scored[0][1] if scored else 0.0
        runner_up = scored[1][1] if len(scored) > 1 else 0.0
        stable = len(track_rows) >= min_len and (final_score - runner_up) >= margin
        track_summary.append({
            "video": video,
            "track_id": tid,
            "frames": len(track_rows),
            "final_class": final_class,
            "final_score": round(final_score, 6),
            "runner_up_score": round(runner_up, 6),
            "stable": stable,
            "mean_area_ratio_to_car": round(sum(float(r.get("area_ratio_to_car", 0.0) or 0.0) for r in track_rows) / max(1, len(track_rows)), 6),
        })
        for row in track_rows:
            item = dict(row)
            item["track_final_class"] = final_class
            item["track_final_score"] = round(final_score, 6)
            item["track_class_stable"] = stable
            frame_rows.append(item)
    for row in untracked:
        item = dict(row)
        item["track_final_class"] = str(row.get("postprocessed_class") or row.get("class") or "")
        item["track_final_score"] = 0.0
        item["track_class_stable"] = False
        frame_rows.append(item)
    return {"frame_rows": frame_rows, "track_summary": track_summary, "vote_rows": vote_rows}


def write_tracking_csvs(consensus: dict[str, list[dict[str, Any]]]) -> None:
    outputs = TRACKING_CONFIG.get("outputs") or {}
    write_csv(ARTIFACTS / outputs.get("frame_detections", "tracking_frame_detections.csv"), consensus["frame_rows"])
    write_csv(ARTIFACTS / outputs.get("track_summary", "tracking_track_summary.csv"), consensus["track_summary"])
    write_csv(ARTIFACTS / outputs.get("track_class_votes", "tracking_class_votes.csv"), consensus["vote_rows"])


def save_openai_candidate_crop(result: Any, xyxy: list[float], video_name: str, frame_idx: int, track_id: int, conf: float) -> str:
    if not OPENAI_FALLBACK.get("enabled", False):
        return ""
    threshold = float((OPENAI_FALLBACK.get("only_when") or {}).get("max_detection_conf", 0.38))
    if conf > threshold:
        return ""
    try:
        import cv2
        image = getattr(result, "orig_img", None)
        if image is None:
            return ""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        pad = 8
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return ""
        CROP_DIR.mkdir(parents=True, exist_ok=True)
        safe_video = "".join(ch if ch.isalnum() else "_" for ch in video_name)[:40]
        path = CROP_DIR / f"{safe_video}_f{frame_idx:06d}_t{track_id}.jpg"
        cv2.imwrite(str(path), image[y1:y2, x1:x2])
        return str(path)
    except Exception as exc:
        print("crop_save_failed:", exc)
        return ""


def run_openai_fallback(consensus: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    candidates = build_openai_candidates(consensus)
    outputs = OPENAI_FALLBACK.get("outputs") or {}
    write_csv(ARTIFACTS / outputs.get("candidates", "openai_fallback_candidates.csv"), candidates)
    decisions = resolve_openai_candidates(candidates)
    write_csv(ARTIFACTS / outputs.get("decisions", "openai_fallback_decisions.csv"), decisions)
    return {"enabled": bool(OPENAI_FALLBACK.get("enabled")), "candidates": len(candidates), "decisions": len(decisions)}


def build_openai_candidates(consensus: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    cfg = OPENAI_FALLBACK
    only = cfg.get("only_when") or {}
    max_items = int(cfg.get("max_items_per_run", 50))
    votes_by_track: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for vote in consensus["vote_rows"]:
        key = (str(vote.get("video", "")), str(vote.get("track_id", "")))
        votes_by_track.setdefault(key, []).append(vote)
    frames_by_track: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in consensus["frame_rows"]:
        key = (str(row.get("video", "")), str(row.get("track_id", "")))
        frames_by_track.setdefault(key, []).append(row)
    candidates = []
    for track in consensus["track_summary"]:
        key = (str(track.get("video", "")), str(track.get("track_id", "")))
        votes = sorted(votes_by_track.get(key, []), key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        if len(votes) < 2:
            continue
        margin = float(votes[0].get("score", 0.0) or 0.0) - float(votes[1].get("score", 0.0) or 0.0)
        unstable = str(track.get("stable")).lower() in {"false", "0", ""}
        low_score = float(track.get("final_score", 0.0) or 0.0) <= float(only.get("max_track_final_score", 0.62))
        low_margin = margin <= float(only.get("max_vote_margin", 0.08))
        if not (unstable or low_score or low_margin):
            continue
        rep = representative_crop_frame(frames_by_track.get(key, []))
        candidates.append({
            "video": key[0],
            "track_id": key[1],
            "candidate_a": votes[0].get("class"),
            "candidate_b": votes[1].get("class"),
            "candidate_a_score": votes[0].get("score"),
            "candidate_b_score": votes[1].get("score"),
            "vote_margin": round(margin, 6),
            "track_final_class": track.get("final_class"),
            "track_final_score": track.get("final_score"),
            "frame": rep.get("frame", ""),
            "crop_path": rep.get("crop_path", ""),
            "area_ratio_to_car": rep.get("area_ratio_to_car", ""),
            "bbox_normalized_area": rep.get("bbox_normalized_area", ""),
            "bbox_normalized_aspect": rep.get("bbox_normalized_aspect", ""),
        })
        if len(candidates) >= max_items:
            break
    return candidates


def representative_crop_frame(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in rows if row.get("crop_path")]
    if not rows:
        return {}
    return sorted(rows, key=lambda row: (float(row.get("postprocessed_conf", row.get("conf", 0.0)) or 0.0), float(row.get("typology_geometry_score", 0.0) or 0.0)), reverse=True)[0]


def resolve_openai_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not OPENAI_FALLBACK.get("enabled", False):
        return [openai_decision(row, "skipped_disabled", "uncertain", 0.0, "OpenAI fallback disabled") for row in candidates]
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return [openai_decision(row, "skipped_no_api_key", "uncertain", 0.0, "OPENAI_API_KEY not set") for row in candidates]
    decisions = []
    for row in candidates:
        crop_path = Path(str(row.get("crop_path") or ""))
        if not crop_path.exists():
            decisions.append(openai_decision(row, "skipped_no_crop", "uncertain", 0.0, "Missing crop image"))
            continue
        try:
            decisions.append(call_openai_tiebreak(row, crop_path, api_key))
        except Exception as exc:
            decisions.append(openai_decision(row, "error", "uncertain", 0.0, str(exc)))
    return decisions


def call_openai_tiebreak(row: dict[str, Any], crop_path: Path, api_key: str) -> dict[str, Any]:
    image_b64 = base64.b64encode(crop_path.read_bytes()).decode("ascii")
    candidate_a = str(row.get("candidate_a"))
    candidate_b = str(row.get("candidate_b"))
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "chosen_class": {"type": "string", "enum": [candidate_a, candidate_b, "uncertain"]},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["chosen_class", "confidence", "reason"],
    }
    payload = {
        "model": OPENAI_FALLBACK.get("model", "gpt-5-mini"),
        "reasoning": {"effort": OPENAI_FALLBACK.get("reasoning_effort", "minimal")},
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": f"Drone vehicle crop. Decide only between {candidate_a} and {candidate_b}, or uncertain. Do not invent another class. area_ratio_to_car={row.get('area_ratio_to_car')}, normalized_area={row.get('bbox_normalized_area')}, normalized_aspect={row.get('bbox_normalized_aspect')}."},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}", "detail": "low"},
            ],
        }],
        "text": {"format": {"type": "json_schema", "name": "vehicle_tiebreak", "schema": schema, "strict": True}},
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=45) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = extract_openai_text(body)
    parsed = json.loads(text)
    min_conf = float((OPENAI_FALLBACK.get("decision_policy") or {}).get("min_model_confidence", 0.55))
    chosen = parsed.get("chosen_class", "uncertain")
    conf = float(parsed.get("confidence", 0.0) or 0.0)
    if chosen not in {candidate_a, candidate_b, "uncertain"}:
        chosen = "uncertain"
    if conf < min_conf:
        chosen = "uncertain"
    return openai_decision(row, "resolved", chosen, conf, parsed.get("reason", ""))


def extract_openai_text(body: dict[str, Any]) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    for item in body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return str(content.get("text", ""))
    raise ValueError("OpenAI response did not contain output_text")


def openai_decision(row: dict[str, Any], status: str, chosen: str, confidence: float, reason: str) -> dict[str, Any]:
    out = dict(row)
    out.update({"openai_status": status, "openai_chosen_class": chosen, "openai_confidence": round(confidence, 6), "openai_reason": reason})
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat_rows = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            flat[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
        flat_rows.append(flat)
    fields = sorted({key for row in flat_rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    global GEOMETRY_PRIORS
    install()
    import torch

    dataset_root = find_dataset_root()
    data_yaml = infer_yolo_dataset(dataset_root)
    GEOMETRY_PRIORS = calibrate_geometry(data_yaml)
    cfg = load_yaml(data_yaml)
    names = list(cfg.get("names") or CLASS_MAP.get("classes") or [])
    model = build_model(str(EXP["model"]))
    device = "0,1" if torch.cuda.device_count() >= 2 else (0 if torch.cuda.is_available() else "cpu")
    train_args = {
        "data": str(data_yaml),
        "imgsz": int(EXP["imgsz"]),
        "epochs": int(EXP["epochs"]),
        "batch": int(EXP["batch"]),
        "optimizer": EXP.get("optimizer", "AdamW"),
        "lr0": float(EXP.get("lr0", 0.001)),
        "patience": int(EXP.get("patience", 20)),
        "seed": int(EXP.get("seed", 0)),
        "device": device,
        "project": str(KAGGLE_WORKING / "runs"),
        "name": str(EXP["id"]),
        "exist_ok": True,
        "verbose": True,
    }
    train_args.update(EXP.get("augment") or {})
    train_args.update(EXP.get("train_args") or {})
    print("TRAIN_ARGS", json.dumps(train_args, indent=2, default=str))
    model.train(**train_args)
    metrics = model.val(data=str(data_yaml), imgsz=int(EXP["imgsz"]), batch=int(EXP["batch"]), device=device, split="val")
    md = metric_dict(metrics, names)
    preds = sample_predictions(model, data_yaml)
    tracking = track_videos(model, dataset_root)
    md["tracking_status"] = tracking.get("status")
    md["tracking_videos"] = tracking.get("videos", 0)
    md["tracking_tracks"] = tracking.get("tracks", 0)
    run_dir = Path(train_args["project"]) / str(EXP["id"])
    for name in ("best.pt", "last.pt", "results.csv", "args.yaml"):
        src = run_dir / "weights" / name if name.endswith(".pt") else run_dir / name
        if src.exists():
            shutil.copy2(src, ARTIFACTS / name)
    (ARTIFACTS / "experiment.json").write_text(json.dumps(EXP, indent=2), encoding="utf-8")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(md, indent=2), encoding="utf-8")
    (ARTIFACTS / "predictions.json").write_text(json.dumps(preds, indent=2), encoding="utf-8")
    (ARTIFACTS / "tracking_summary.json").write_text(json.dumps(tracking, indent=2), encoding="utf-8")
    (ARTIFACTS / "summary.json").write_text(json.dumps({"dataset_root": str(dataset_root), "data_yaml": str(data_yaml), "names": names}, indent=2), encoding="utf-8")
    print("FINAL_METRICS", json.dumps(md, sort_keys=True))


main()
'''


def render_notebook(script: str, title: str) -> dict[str, Any]:
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n", "\nAutogenerated Kaggle training notebook.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in script.splitlines()]},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
