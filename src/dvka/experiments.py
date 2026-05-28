from __future__ import annotations

import copy
from collections import Counter
from typing import Any

from .config import load_search_space
from .io import file_lock, read_json, utc_stamp, write_json
from .paths import QUEUE, RUNS, STATE, WORKSPACES

ACTIVE = {"queued_remote", "submitted", "running"}


def validate_experiment(exp: dict[str, Any]) -> None:
    required = {"id", "family", "model", "imgsz", "epochs", "batch", "optimizer", "augment", "priority", "budget"}
    missing = required - set(exp)
    if missing:
        raise ValueError(f"{exp.get('id', '<unknown>')} missing fields: {sorted(missing)}")
    if int(exp["imgsz"]) < 320:
        raise ValueError(f"{exp['id']} has too-small imgsz")
    if int(exp["epochs"]) <= 0:
        raise ValueError(f"{exp['id']} has invalid epochs")


def seed_queue(force: bool = False) -> list[dict[str, Any]]:
    STATE.mkdir(parents=True, exist_ok=True)
    WORKSPACES.mkdir(parents=True, exist_ok=True)
    if QUEUE.exists() and not force:
        return read_json(QUEUE, [])
    space = load_search_space()
    presets = space.get("augment_presets", {})
    queue: list[dict[str, Any]] = []
    for raw in space.get("seed_experiments", []):
        item = copy.deepcopy(raw)
        augment_name = item.get("augment")
        item["augment_name"] = augment_name
        item["augment"] = copy.deepcopy(presets.get(str(augment_name), {}))
        item["status"] = "queued"
        item["created_at"] = utc_stamp()
        validate_experiment(item)
        queue.append(item)
    queue.sort(key=lambda x: (int(x.get("priority", 100)), str(x.get("id"))))
    write_json(QUEUE, queue)
    if not RUNS.exists():
        write_json(RUNS, [])
    return queue


def load_queue() -> list[dict[str, Any]]:
    if not QUEUE.exists():
        return seed_queue(False)
    return read_json(QUEUE, [])


def save_queue(queue: list[dict[str, Any]]) -> None:
    write_json(QUEUE, queue)


def load_runs() -> list[dict[str, Any]]:
    return read_json(RUNS, [])


def save_runs(runs: list[dict[str, Any]]) -> None:
    write_json(RUNS, runs)


def status_counts() -> dict[str, int]:
    return dict(Counter(str(item.get("status", "unknown")) for item in load_queue()))


def active_count() -> int:
    return sum(1 for item in load_queue() if item.get("status") in ACTIVE)


def next_queued(limit: int) -> list[dict[str, Any]]:
    return [item for item in load_queue() if item.get("status") == "queued"][:limit]


def update_queue_item(exp_id: str, **updates: Any) -> dict[str, Any] | None:
    with file_lock(QUEUE):
        queue = load_queue()
        updated = None
        for item in queue:
            if str(item.get("id")) == exp_id:
                item.update(updates)
                item["updated_at"] = utc_stamp()
                updated = item
                break
        save_queue(queue)
    return updated


def enqueue_many(experiments: list[dict[str, Any]]) -> list[str]:
    with file_lock(QUEUE):
        queue = load_queue()
        existing = {str(item.get("id")) for item in queue}
        added: list[str] = []
        for exp in experiments:
            if str(exp.get("id")) in existing:
                continue
            item = copy.deepcopy(exp)
            item.setdefault("status", "queued")
            item.setdefault("created_at", utc_stamp())
            validate_experiment(item)
            queue.append(item)
            existing.add(str(item["id"]))
            added.append(str(item["id"]))
        queue.sort(key=lambda x: (int(x.get("priority", 100)), str(x.get("id"))))
        save_queue(queue)
    return added
