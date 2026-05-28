from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from .config import load_search_space, settings
from .io import read_json, utc_stamp, write_json
from .paths import LEADERBOARD, OUTPUTS, STATE

REVIEW_HISTORY = STATE / "review_history.json"


METRIC_ALIASES = {
    "map50_95": ["map50_95", "map", "metrics/mAP50-95(B)", "metrics/map50-95(B)", "metrics/map50-95"],
    "map50": ["map50", "metrics/mAP50(B)", "metrics/map50(B)", "metrics/map50"],
    "precision": ["precision", "mp", "metrics/precision(B)", "metrics/precision"],
    "recall": ["recall", "mr", "metrics/recall(B)", "metrics/recall"],
    "rare_map50_95": ["rare_map50_95"],
}


def metric(metrics: dict[str, Any], key: str) -> float | None:
    aliases = {alias.lower().replace("_", "").replace("/", "").replace("-", "") for alias in METRIC_ALIASES[key]}
    for raw_key, value in metrics.items():
        normalized = str(raw_key).lower().replace("_", "").replace("/", "").replace("-", "")
        if normalized in aliases:
            try:
                return float(value)
            except Exception:
                return None
    return None


def selection_score(map50_95: float | None, rare_map50_95: float | None, weight: float) -> float | None:
    """Ranking objective: overall mAP plus a bonus for the rare classes that decide
    this competition. A run with no measured mAP is unrankable (None). When rare AP
    is missing we fall back to the overall mAP with no bonus, so such runs are never
    rewarded for the absence of the rare signal."""
    if map50_95 is None:
        return None
    return float(map50_95) + (weight * float(rare_map50_95) if rare_map50_95 is not None else 0.0)


def discover_runs(output_root: Path = OUTPUTS) -> list[dict[str, Any]]:
    weight = settings().minority_ap_weight
    runs: list[dict[str, Any]] = []
    for metrics_path in output_root.rglob("metrics.json"):
        artifact_dir = metrics_path.parent
        exp = read_json(artifact_dir / "experiment.json", {})
        metrics = read_json(metrics_path, {})
        per_class = metrics.get("per_class_map50_95") if isinstance(metrics, dict) else {}
        map50_95 = metric(metrics, "map50_95")
        rare = metric(metrics, "rare_map50_95")
        run = {
            "experiment_id": str(exp.get("id") or artifact_dir.parent.name),
            "model": str(exp.get("model") or ""),
            "family": str(exp.get("family") or ""),
            "imgsz": exp.get("imgsz"),
            "epochs": exp.get("epochs"),
            "batch": exp.get("batch"),
            "seed": exp.get("seed"),
            "augment_name": exp.get("augment_name") or exp.get("augment"),
            "map50_95": map50_95,
            "map50": metric(metrics, "map50"),
            "precision": metric(metrics, "precision"),
            "recall": metric(metrics, "recall"),
            "rare_map50_95": rare,
            "selection_score": selection_score(map50_95, rare, weight),
            "per_class_map50_95": per_class if isinstance(per_class, dict) else {},
            "path": str(artifact_dir),
        }
        runs.append(run)
    runs.sort(key=lambda x: (x.get("selection_score") is None, -(x.get("selection_score") or -1.0)))
    return runs


def config_key(run: dict[str, Any]) -> str:
    """Identity of a network configuration, independent of the random seed, so that
    seed replicas of the same config land in the same group for variance estimation."""
    return "|".join(
        str(run.get(field, ""))
        for field in ("model", "imgsz", "epochs", "batch", "augment_name")
    )


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None
    mean = statistics.fmean(clean)
    std = statistics.stdev(clean) if len(clean) >= 2 else 0.0
    return mean, std


def aggregate_groups(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse per-run rows into one entry per configuration, reporting the mean and
    standard deviation of each metric across seeds. Single-seed groups report std=0
    with n_seeds=1, which the stopping logic treats as 'sigma unknown'."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        if run.get("selection_score") is None:
            continue
        buckets.setdefault(config_key(run), []).append(run)
    groups: list[dict[str, Any]] = []
    for key, members in buckets.items():
        members_sorted = sorted(members, key=lambda r: -(r.get("selection_score") or -1.0))
        rep = members_sorted[0]
        sel_mean, sel_std = _mean_std([r.get("selection_score") for r in members])
        map_mean, map_std = _mean_std([r.get("map50_95") for r in members])
        rare_mean, rare_std = _mean_std([r.get("rare_map50_95") for r in members])
        groups.append(
            {
                "config_key": key,
                "model": rep.get("model"),
                "imgsz": rep.get("imgsz"),
                "epochs": rep.get("epochs"),
                "batch": rep.get("batch"),
                "augment_name": rep.get("augment_name"),
                "n_seeds": len(members),
                "seeds": [r.get("seed") for r in members if r.get("seed") is not None],
                "selection_score_mean": sel_mean,
                "selection_score_std": sel_std,
                "map50_95_mean": map_mean,
                "map50_95_std": map_std,
                "rare_map50_95_mean": rare_mean,
                "rare_map50_95_std": rare_std,
                "experiment_ids": [r.get("experiment_id") for r in members_sorted],
                "best_run": rep,
            }
        )
    groups.sort(key=lambda g: (g.get("selection_score_mean") is None, -(g.get("selection_score_mean") or -1.0)))
    return groups


def _group_as_best(group: dict[str, Any]) -> dict[str, Any]:
    """Project a config group into the legacy 'best' dict shape consumed by the
    EvolutionAgent and the recommendation, using the across-seed means."""
    rep = group.get("best_run") or {}
    return {
        "experiment_id": group["experiment_ids"][0] if group.get("experiment_ids") else rep.get("experiment_id"),
        "config_key": group.get("config_key"),
        "model": group.get("model"),
        "imgsz": group.get("imgsz"),
        "epochs": group.get("epochs"),
        "batch": group.get("batch"),
        "augment_name": group.get("augment_name"),
        "n_seeds": group.get("n_seeds"),
        "selection_score": group.get("selection_score_mean"),
        "selection_score_std": group.get("selection_score_std"),
        "map50_95": group.get("map50_95_mean"),
        "map50_95_std": group.get("map50_95_std"),
        "rare_map50_95": group.get("rare_map50_95_mean"),
        "rare_map50_95_std": group.get("rare_map50_95_std"),
        "per_class_map50_95": rep.get("per_class_map50_95", {}),
        "path": rep.get("path"),
    }


def summarize(output_root: Path = OUTPUTS) -> dict[str, Any]:
    """Single source of truth used by both review() and the EvolutionAgent."""
    runs = discover_runs(output_root)
    groups = aggregate_groups(runs)
    return {"runs": runs, "groups": groups}


def write_leaderboard(output_root: Path = OUTPUTS) -> list[dict[str, Any]]:
    rows = discover_runs(output_root)
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    fields = ["experiment_id", "model", "imgsz", "epochs", "batch", "augment_name", "selection_score", "map50_95", "map50", "precision", "recall", "rare_map50_95", "path"]
    with LEADERBOARD.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    write_json(STATE / "leaderboard.json", rows)
    return rows


def _stopping_config() -> dict[str, Any]:
    cfg = (load_search_space().get("stopping") or {})
    return {
        "patience_reviews": int(cfg.get("patience_reviews", 3)),
        "min_seeds_for_sigma": int(cfg.get("min_seeds_for_sigma", 2)),
        "sigma_multiplier": float(cfg.get("sigma_multiplier", 1.0)),
    }


def _append_history(entry: dict[str, Any]) -> list[dict[str, Any]]:
    history = read_json(REVIEW_HISTORY, [])
    if not isinstance(history, list):
        history = []
    history.append(entry)
    write_json(REVIEW_HISTORY, history)
    return history


def evaluate_objective(best_group: dict[str, Any] | None, history: list[dict[str, Any]]) -> dict[str, Any]:
    """Decide whether the search has converged: the objective is 'reached' once the
    best score has not improved by more than sigma over the patience window AND we
    actually have a sigma (>= min_seeds_for_sigma seeds on the leader). Without that
    variance estimate the leader's score is just a noisy point estimate, so we keep
    going rather than declare victory on noise."""
    cfg = _stopping_config()
    if not best_group:
        return {"reached": False, "reason": "no_completed_runs", **cfg}
    score = best_group.get("selection_score_mean")
    std = best_group.get("selection_score_std")
    n_seeds = int(best_group.get("n_seeds") or 0)
    have_sigma = n_seeds >= cfg["min_seeds_for_sigma"] and std is not None
    threshold = (std or 0.0) * cfg["sigma_multiplier"]

    # Best score over the patience window *before* this review (history excludes the
    # entry we are about to append).
    window = [h for h in history if h.get("best_selection_score") is not None][-cfg["patience_reviews"]:]
    prev_best = max((float(h["best_selection_score"]) for h in window), default=None)
    improvement = None if (score is None or prev_best is None) else float(score) - prev_best

    if not have_sigma:
        reason = f"need >= {cfg['min_seeds_for_sigma']} seeds on the leader to estimate sigma (have {n_seeds}); run `dvka confirm-leader`."
        reached = False
    elif prev_best is None or improvement is None:
        reason = f"not enough review history yet (need {cfg['patience_reviews']} prior reviews)."
        reached = False
    elif improvement <= threshold:
        reason = f"improvement {improvement:+.4f} over last {len(window)} reviews <= sigma*{cfg['sigma_multiplier']}={threshold:.4f}: converged."
        reached = True
    else:
        reason = f"still improving: {improvement:+.4f} > sigma threshold {threshold:.4f}."
        reached = False
    return {
        "reached": reached,
        "reason": reason,
        "best_selection_score": score,
        "sigma": std,
        "n_seeds": n_seeds,
        "improvement_over_window": improvement,
        "threshold": threshold,
        **cfg,
    }


def review(output_root: Path = OUTPUTS) -> dict[str, Any]:
    rows = write_leaderboard(output_root)
    groups = aggregate_groups(rows)
    best_group = groups[0] if groups else None
    best = _group_as_best(best_group) if best_group else None

    weak_classes: list[str] = []
    if best:
        per_class = best.get("per_class_map50_95") or {}
        if isinstance(per_class, dict) and per_class:
            weak_classes = [name for name, value in sorted(per_class.items(), key=lambda item: item[1])[:3]]

    history = read_json(REVIEW_HISTORY, [])
    if not isinstance(history, list):
        history = []
    objective = evaluate_objective(best_group, history)

    report = {
        "completed_runs": len(rows),
        "configs": len(groups),
        "best": best,
        "groups": groups,
        "weak_classes": weak_classes,
        "objective": objective,
        "recommendation": _recommendation(best, weak_classes, objective),
    }
    write_json(STATE / "review.json", report)

    # Record this review in the history *after* evaluating, so the objective compares
    # against prior reviews only.
    if best_group is not None:
        _append_history(
            {
                "at": utc_stamp(),
                "best_config_key": best_group.get("config_key"),
                "best_selection_score": best_group.get("selection_score_mean"),
                "best_sigma": best_group.get("selection_score_std"),
                "n_seeds": best_group.get("n_seeds"),
                "completed_runs": len(rows),
                "objective_reached": objective.get("reached"),
            }
        )
    return report


def _recommendation(best: dict[str, Any] | None, weak_classes: list[str], objective: dict[str, Any] | None = None) -> str:
    if not best:
        return "Run the seed pilots first."
    parts = [
        f"Current leader is {best.get('experiment_id')} "
        f"({best.get('n_seeds')} seed(s)) with selection_score="
        f"{best.get('selection_score')} +/- {best.get('selection_score_std')} "
        f"(mAP50-95={best.get('map50_95')}, rare_mAP50-95={best.get('rare_map50_95')})."
    ]
    if objective and objective.get("reached"):
        parts.append("OBJECTIVE REACHED: " + str(objective.get("reason")) + " Build the WBF+TTA ensemble of the top configs for submission.")
        return " ".join(parts)
    if objective and objective.get("reason"):
        parts.append("Stopping check: " + str(objective.get("reason")))
    if best.get("rare_map50_95") is None:
        parts.append("Warning: rare_mAP50-95 is missing; check that rare_classes are injected into experiments and present in per-class AP.")
    if weak_classes:
        parts.append("Promote higher resolution or class-balanced augmentation for: " + ", ".join(weak_classes) + ".")
    if int(best.get("imgsz") or 0) < 1280:
        parts.append("Next branch should test the same model at 1280 before changing architecture.")
    return " ".join(parts)
