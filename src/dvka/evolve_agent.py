from __future__ import annotations

import copy
import secrets
from datetime import datetime, timezone
from typing import Any

from pathlib import Path

from .config import load_search_space
from .experiments import enqueue_many
from .io import read_json, utc_stamp, write_json
from .paths import OUTPUTS, STATE
from .review_agent import review


SEED_POOL = [17, 23, 31, 41, 101, 202, 303, 404, 505, 606, 707, 808, 909]


def confirm_leader(count: int | None = None, output_root: Path = OUTPUTS) -> dict[str, Any]:
    """Enqueue seed replicas of the current best configuration so the ReviewAgent can
    estimate sigma. Without >= 2 seeds on the leader the stopping criterion can never
    fire, because a single run is just a noisy point estimate."""
    rev = review(output_root)
    groups = rev.get("groups") or []
    if not groups:
        return {"added": [], "reason": "no_completed_runs"}
    leader = groups[0]
    path = (leader.get("best_run") or {}).get("path")
    base = read_json(Path(path) / "experiment.json", {}) if path else {}
    if not base:
        return {"added": [], "reason": "leader_experiment_config_not_found", "config_key": leader.get("config_key")}

    stopping = load_search_space().get("stopping") or {}
    target = int(stopping.get("min_seeds_for_sigma", 2)) + 1  # one above the minimum for a robust sigma
    have = int(leader.get("n_seeds") or 1)
    needed = count if count is not None else max(0, target - have)
    if needed <= 0:
        return {"added": [], "reason": f"leader already has {have} seeds (>= target {target})", "config_key": leader.get("config_key")}

    used = {int(s) for s in (leader.get("seeds") or []) if s is not None}
    fresh = [s for s in SEED_POOL if s not in used][:needed]
    stamp = _evo_stamp()
    replicas: list[dict[str, Any]] = []
    for idx, seed in enumerate(fresh, start=1):
        item = copy.deepcopy(base)
        item.pop("status", None)
        item.pop("created_at", None)
        item.pop("updated_at", None)
        item.pop("kernel_id", None)
        item.pop("workspace", None)
        item.pop("launch_output", None)
        item.pop("rare_classes", None)  # re-injected at build time from current audit
        item["seed"] = seed
        item["source"] = "ConfirmLeader"
        item["priority"] = 1  # confirm the leader before exploring further
        item["budget"] = "confirm"
        item["id"] = f"SEED_{stamp}_{idx:02d}_{str(item.get('model','')).replace('.', '_')}_{item.get('imgsz')}_s{seed}"
        item["notes"] = f"seed replica of leader {leader.get('config_key')} to estimate sigma"
        replicas.append(item)
    added = enqueue_many(replicas)
    payload = {"added": added, "config_key": leader.get("config_key"), "seeds": fresh, "had_seeds": have}
    write_json(STATE / "confirm_leader.json", payload)
    return payload


def _evo_stamp() -> str:
    # utc_stamp() has second resolution, so two evolve() calls in the same second
    # produce identical IDs and enqueue_many silently drops the duplicates.
    # Include microseconds + a short random suffix so IDs are unique.
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"{now}{secrets.token_hex(2)}"


def _scale_batch(model: str, imgsz: int) -> int:
    scale = "m"
    stem = model.rsplit(".", 1)[0].lower()
    for suffix in ("n", "s", "m", "l", "x"):
        if stem.endswith(suffix):
            scale = suffix
    table = {
        "n": {640: 32, 960: 16, 1280: 8, 1536: 4},
        "s": {640: 24, 960: 12, 1280: 6, 1536: 3},
        "m": {640: 16, 960: 8, 1280: 4, 1536: 2},
        "l": {640: 8, 960: 4, 1280: 2, 1536: 1},
        "x": {640: 6, 960: 2, 1280: 1, 1536: 1},
    }
    return table.get(scale, table["m"]).get(imgsz, 2)


def evolve(max_new: int | None = None) -> dict[str, Any]:
    rev = review(OUTPUTS)
    best = rev.get("best")
    if not best:
        return {"added": [], "reason": "no_completed_runs"}
    # Refuse to evolve from a leader without a measured score. Otherwise the
    # whole population would inherit from a run we never validated.
    if best.get("map50_95") is None:
        return {"added": [], "reason": "best_run_missing_map50_95", "best": best}
    groups = rev.get("groups") or []
    space = load_search_space()
    evo = space.get("evolution", {})
    presets = space.get("augment_presets", {})
    max_new = int(max_new or evo.get("max_new_per_review") or 6)
    epochs_factor = float(evo.get("promote_epochs_factor", 1.5))
    near_tie_delta = float(evo.get("near_tie_delta", 0.0))
    candidates: list[dict[str, Any]] = []
    base_model = str(best.get("model"))
    base_imgsz = int(best.get("imgsz") or 960)

    # Near-tie leaders: every configuration whose mean score is within near_tie_delta
    # of the best mean. Branching architecture probes from all of them (not just the
    # single best) adds light exploration and avoids locking onto a greedy local
    # optimum when the top configs are statistically indistinguishable.
    best_score = best.get("selection_score")
    leaders: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    for group in groups:
        score = group.get("selection_score_mean")
        if score is None:
            continue
        if best_score is not None and (best_score - float(score)) > near_tie_delta:
            break  # groups are sorted descending; no further near-ties exist
        model = str(group.get("model"))
        if model not in seen_models:
            seen_models.add(model)
            leaders.append(group)
    if not leaders:
        leaders = [best]

    # 1. Resolution promotion from the best leader.
    next_sizes = [size for size in evo.get("promote_imgsz", [960, 1280]) if int(size) > base_imgsz]
    if next_sizes:
        size = int(next_sizes[0])
        candidates.append(_proposal(best, base_model, size, presets, "small_objects", "promote_resolution", epochs_factor))
    # 2. Epoch extension from the best leader.
    candidates.append(_proposal(best, base_model, base_imgsz, presets, "small_objects", "extend_epochs", epochs_factor))
    # 3. Architecture probes from each near-tie leader, at that leader's resolution.
    probed: set[tuple[str, int]] = set()
    for leader in leaders:
        leader_model = str(leader.get("model"))
        leader_imgsz = int(leader.get("imgsz") or base_imgsz)
        for model in evo.get("candidate_models", []):
            model = str(model)
            if model == leader_model or (model, leader_imgsz) in probed:
                continue
            probed.add((model, leader_imgsz))
            aug = "low_aug" if model.startswith("rtdetr") else "small_objects"
            candidates.append(_proposal(leader, model, leader_imgsz, presets, aug, "architecture_probe", epochs_factor))
    stamp = _evo_stamp()
    for idx, item in enumerate(candidates[:max_new], start=1):
        item["id"] = f"EVO_{stamp}_{idx:02d}_{item['model'].replace('.', '_')}_{item['imgsz']}"
    added = enqueue_many(candidates[:max_new])
    payload = {"added": added, "proposals": candidates[:max_new], "review": rev}
    write_json(STATE / "evolution.json", payload)
    return payload


def _proposal(
    best: dict[str, Any],
    model: str,
    imgsz: int,
    presets: dict[str, Any],
    augment_name: str,
    reason: str,
    epochs_factor: float = 1.5,
) -> dict[str, Any]:
    base_epochs = int(best.get("epochs") or 40)
    epochs = max(base_epochs + 15, int(base_epochs * epochs_factor)) if reason == "extend_epochs" else max(45, base_epochs)
    parent_id = best.get("experiment_id") or (best.get("experiment_ids") or [None])[0]
    parent_score = best.get("map50_95") if best.get("map50_95") is not None else best.get("map50_95_mean")
    family = "ultralytics"
    return {
        "id": "EVO_PENDING",
        "family": family,
        "model": model,
        "imgsz": imgsz,
        "epochs": epochs,
        "batch": _scale_batch(model, imgsz),
        "optimizer": "AdamW",
        "lr0": 0.0002 if model.startswith("rtdetr") else 0.001,
        "patience": 16,
        "seed": 101,
        "augment_name": augment_name,
        "augment": copy.deepcopy(presets.get(augment_name, {})),
        "priority": 2 if reason != "architecture_probe" else 3,
        "budget": "evolved",
        "source": "EvolutionAgent",
        "notes": f"{reason}: generated from leader {parent_id} with mAP50-95={parent_score}",
    }
