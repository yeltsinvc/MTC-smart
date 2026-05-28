from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import read_json
from .paths import CONFIGS, STATE


@dataclass(frozen=True)
class Settings:
    owner: str
    dataset_slug: str
    competition_slug: str
    accelerator: str
    max_active_kernels: int
    metric: str
    target_classes: list[str]
    minority_ap_weight: float
    rare_classes: list[str]


def load_project(path: Path | None = None) -> dict[str, Any]:
    return read_json(path or CONFIGS / "project.json", {})


def load_search_space(path: Path | None = None) -> dict[str, Any]:
    return read_json(path or CONFIGS / "search_space.json", {})


def load_class_map(path: Path | None = None) -> dict[str, Any]:
    return read_json(path or CONFIGS / "class_map.json", {})


def load_typology_rules(path: Path | None = None) -> dict[str, Any]:
    return read_json(path or CONFIGS / "typology_rules.json", {})


def load_tracking_postprocess(path: Path | None = None) -> dict[str, Any]:
    return read_json(path or CONFIGS / "tracking_postprocess.json", {})


def load_openai_fallback(path: Path | None = None) -> dict[str, Any]:
    return read_json(path or CONFIGS / "openai_fallback.json", {})


def load_dataset_audit(path: Path | None = None) -> dict[str, Any]:
    return read_json(path or STATE / "dataset_audit.json", {})


def resolve_rare_classes() -> list[str]:
    """Single source of truth for the rare-class set used to compute rare_map50_95.

    Prefers the data-driven list produced by DataAgent (state/dataset_audit.json);
    falls back to an explicit project.json override so the closed loop still works
    before any audit has run. Returns [] only if neither is configured.
    """
    audit = load_dataset_audit()
    rare = audit.get("rare_classes") if isinstance(audit, dict) else None
    if rare:
        return [str(name) for name in rare]
    project = load_project()
    return [str(name) for name in (project.get("rare_classes") or [])]


def settings() -> Settings:
    cfg = load_project()
    return Settings(
        owner=str(cfg.get("owner") or ""),
        dataset_slug=str(cfg.get("dataset_slug") or ""),
        competition_slug=str(cfg.get("competition_slug") or ""),
        accelerator=str(cfg.get("accelerator") or "NvidiaTeslaT4"),
        max_active_kernels=int(cfg.get("max_active_kernels") or 2),
        metric=str(cfg.get("metric") or "map50_95"),
        target_classes=list(cfg.get("target_classes") or []),
        minority_ap_weight=float(cfg.get("minority_ap_weight") or 0.0),
        rare_classes=list(cfg.get("rare_classes") or []),
    )
