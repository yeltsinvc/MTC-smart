from __future__ import annotations

"""DirectorAgent: the policy that decides *what to do next*.

The other agents (Review, Evolution, ConfirmLeader, Ensemble) are passive: they each
do one job when invoked. `dvka cycle` chains them but always evolves, even when the
methodology says we should first measure sigma or that we are already done.

The Director closes that loop. Given the current review, it picks the single next
action that the methodology prescribes:

    no completed runs        -> run the seed pilots
    objective reached        -> stop searching, build the WBF+TTA ensemble
    leader has < N seeds      -> confirm-leader (replicate seeds to estimate sigma)
    otherwise                -> evolve (propose new candidates)

`direct()` performs one autonomous step (refresh -> pull -> review -> decide -> act ->
launch within capacity). It never blocks on Kaggle: call it on a cadence (cron/loop)
and it advances the search by exactly one safe step each time.
"""

from pathlib import Path
from typing import Any

from .config import settings
from .evolve_agent import confirm_leader, evolve
from .experiments import active_count, next_queued
from .io import write_json
from .kaggle_agent import launch, pull, refresh
from .paths import OUTPUTS, STATE
from .review_agent import review


def decide_next_action(review_report: dict[str, Any]) -> dict[str, Any]:
    """Pure decision function: maps a review report to the next action + reason.

    Returns {action, reason} where action is one of:
    run_seeds | confirm_leader | evolve | ensemble | wait.
    """
    completed = int(review_report.get("completed_runs") or 0)
    objective = review_report.get("objective") or {}
    best = review_report.get("best") or {}

    if completed == 0:
        return {"action": "run_seeds", "reason": "no completed runs yet; launch the seed pilots."}
    if objective.get("reached"):
        return {"action": "ensemble", "reason": objective.get("reason") or "objective reached; build the final ensemble."}

    n_seeds = int(best.get("n_seeds") or 0)
    min_seeds = int(objective.get("min_seeds_for_sigma") or 2)
    if n_seeds < min_seeds:
        return {
            "action": "confirm_leader",
            "reason": f"leader has {n_seeds} seed(s); need {min_seeds} to estimate sigma before stopping.",
        }
    return {"action": "evolve", "reason": objective.get("reason") or "still improving; propose new candidates."}


def direct(
    output_root: Path = OUTPUTS,
    launch_limit: int = 2,
    max_new: int | None = None,
    timeout: int = 7200,
    ensemble_top_k: int = 3,
    act: bool = True,
) -> dict[str, Any]:
    """Run one autonomous Director step.

    With act=False the Director only reports the decision (dry run) without enqueuing
    or launching anything.
    """
    cfg = settings()
    refreshed = refresh() if act else []
    pulled = [str(p) for p in (pull(output_root, skip_existing=True) if act else [])]
    rev = review(output_root)
    decision = decide_next_action(rev)
    action = decision["action"]

    performed: dict[str, Any] = {"action": action, "reason": decision["reason"]}

    if act:
        if action == "confirm_leader":
            performed["confirm_leader"] = confirm_leader(output_root=output_root)
        elif action == "evolve":
            performed["evolved"] = evolve(max_new=max_new).get("added", [])
        elif action == "ensemble":
            # The Director cannot run inference on its own (it needs a holdout images
            # folder and a GPU), so it resolves the top-k weights and hands back the
            # exact command to run. This is the 'proposal' for the final submission.
            from .ensemble import resolve_top_k_weights

            weights = [str(p) for p in resolve_top_k_weights(ensemble_top_k, output_root)]
            performed["ensemble_plan"] = {
                "top_k_weights": weights,
                "next_command": (
                    "dvka eval-typology --predictions <preds.json> --gt-yolo <labels/val>  "
                    "&&  dvka ensemble --top-k "
                    f"{ensemble_top_k} --images <holdout_images> --output outputs/ensemble_predictions.json"
                ),
                "note": "Validate typology on a holdout before submitting; then fuse the top-k models with WBF+TTA.",
            }

    # Launch queued work within capacity, except when we have converged.
    launched: list[dict[str, Any]] = []
    if act and action != "ensemble":
        capacity = max(0, cfg.max_active_kernels - active_count())
        pending = len(next_queued(launch_limit))
        if capacity > 0 and launch_limit > 0 and pending > 0:
            launched = launch(limit=min(launch_limit, capacity), timeout=timeout)

    report = {
        "decision": decision,
        "performed": performed,
        "refreshed": len(refreshed),
        "pulled": pulled,
        "completed_runs": rev.get("completed_runs"),
        "best": rev.get("best"),
        "objective": rev.get("objective"),
        "launched": launched,
        "active": active_count(),
        "recommendation": rev.get("recommendation"),
    }
    write_json(STATE / "director.json", report)
    return report
