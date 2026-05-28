from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import load_class_map
from .io import write_json
from .paths import STATE

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTS = {".txt", ".json", ".csv"}


def canonical_class(name: str, class_map: dict[str, Any]) -> str:
    aliases = {str(k).lower(): str(v) for k, v in (class_map.get("aliases") or {}).items()}
    clean = " ".join(name.strip().replace("_", " ").lower().split())
    return aliases.get(clean, name.strip())


def audit_dataset(dataset_root: Path, out_path: Path | None = None) -> dict[str, Any]:
    class_map = load_class_map()
    target_classes = list(class_map.get("classes") or [])
    image_files = sorted(p for p in dataset_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    label_files = sorted(p for p in dataset_root.rglob("*") if p.suffix.lower() in LABEL_EXTS)
    class_counts: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    issues: list[dict[str, Any]] = []

    yaml_names = _read_yaml_names(dataset_root)
    csv_counts, csv_issues = _audit_roboflow_csv(dataset_root, class_map)
    issues.extend(csv_issues)
    for split, counts in csv_counts.items():
        split_counts[split].update(counts)
        class_counts.update(counts)

    yolo_counts, yolo_issues = _audit_yolo_labels(dataset_root, yaml_names, target_classes)
    issues.extend(yolo_issues)
    if yolo_counts:
        for split, counts in yolo_counts.items():
            split_counts[split].update(counts)
            class_counts.update(counts)

    missing_classes = [name for name in target_classes if class_counts.get(name, 0) == 0]
    nonzero = [count for count in class_counts.values() if count > 0]
    median = sorted(nonzero)[len(nonzero) // 2] if nonzero else 0
    rare = [name for name, count in class_counts.items() if count > 0 and median and count <= max(3, median * 0.35)]
    summary = {
        "dataset_root": str(dataset_root.resolve()),
        "images": len(image_files),
        "label_files": len(label_files),
        "target_classes": target_classes,
        "class_counts": dict(sorted(class_counts.items())),
        "split_class_counts": {split: dict(sorted(counts.items())) for split, counts in split_counts.items()},
        "missing_target_classes": missing_classes,
        "rare_classes": sorted(rare),
        "issues": issues[:500],
        "issue_count": len(issues),
    }
    resolved = out_path or STATE / "dataset_audit.json"
    write_json(resolved, summary)
    return summary


def _read_yaml_names(dataset_root: Path) -> list[str]:
    for yaml_path in list(dataset_root.rglob("data.yaml")) + list(dataset_root.rglob("*.yaml")):
        text = yaml_path.read_text(encoding="utf-8", errors="ignore")
        names = []
        in_names = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("names:"):
                raw = stripped.split(":", 1)[1].strip()
                if raw.startswith("[") and raw.endswith("]"):
                    return [part.strip(" '\"") for part in raw.strip("[]").split(",") if part.strip()]
                in_names = True
                continue
            if in_names:
                if not stripped.startswith("-"):
                    break
                names.append(stripped[1:].strip(" '\""))
        if names:
            return names
    return []


def _split_name(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "valid" in parts or "val" in parts:
        return "val"
    if "test" in parts:
        return "test"
    return "train"


def _audit_roboflow_csv(dataset_root: Path, class_map: dict[str, Any]) -> tuple[dict[str, Counter[str]], list[dict[str, Any]]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    issues: list[dict[str, Any]] = []
    required = {"filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"}
    for csv_path in dataset_root.rglob("_annotations.csv"):
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if not required <= fields:
                issues.append({"file": str(csv_path), "issue": "csv_missing_required_columns"})
                continue
            split = _split_name(csv_path)
            for row_idx, row in enumerate(reader, start=2):
                cls = canonical_class(str(row.get("class", "")), class_map)
                try:
                    width = float(row.get("width", "0"))
                    height = float(row.get("height", "0"))
                    xmin = float(row.get("xmin", "nan"))
                    ymin = float(row.get("ymin", "nan"))
                    xmax = float(row.get("xmax", "nan"))
                    ymax = float(row.get("ymax", "nan"))
                except ValueError:
                    issues.append({"file": str(csv_path), "row": row_idx, "issue": "non_numeric_box"})
                    continue
                if width <= 0 or height <= 0 or xmax <= xmin or ymax <= ymin:
                    issues.append({"file": str(csv_path), "row": row_idx, "issue": "invalid_box"})
                    continue
                counts[split][cls] += 1
    return counts, issues


def _audit_yolo_labels(
    dataset_root: Path,
    yaml_names: list[str],
    target_classes: list[str],
) -> tuple[dict[str, Counter[str]], list[dict[str, Any]]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    issues: list[dict[str, Any]] = []
    names = yaml_names or target_classes
    txt_files = [p for p in dataset_root.rglob("*.txt") if "labels" in {part.lower() for part in p.parts}]
    for label_path in txt_files:
        split = _split_name(label_path)
        for line_idx, line in enumerate(label_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            parts = line.split()
            if len(parts) < 5:
                issues.append({"file": str(label_path), "line": line_idx, "issue": "short_yolo_label"})
                continue
            try:
                cls_id = int(float(parts[0]))
                coords = [float(v) for v in parts[1:5]]
            except ValueError:
                issues.append({"file": str(label_path), "line": line_idx, "issue": "non_numeric_yolo_label"})
                continue
            if cls_id < 0 or cls_id >= len(names):
                issues.append({"file": str(label_path), "line": line_idx, "issue": "class_id_out_of_range"})
                continue
            if any(v < 0 or v > 1 for v in coords) or coords[2] <= 0 or coords[3] <= 0:
                issues.append({"file": str(label_path), "line": line_idx, "issue": "box_outside_normalized_range"})
            counts[split][names[cls_id]] += 1
    return counts, issues
