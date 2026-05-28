from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import __version__
from .config import settings
from .data_agent import audit_dataset
from .evolve_agent import confirm_leader, evolve
from .experiments import active_count, load_queue, seed_queue, status_counts
from .geometry_calibration import calibrate_from_dataset
from .kaggle_agent import build, launch, pull, refresh
from .openai_fallback import build_fallback_candidates, load_openai_fallback_config, resolve_candidates, write_openai_fallback_outputs
from .paths import LEADERBOARD, OUTPUTS, QUEUE, STATE
from .review_agent import review, write_leaderboard
from .typology_agent import enrich_detections, load_rules
from .tracking_postprocess import consensus_tracks, load_tracking_config, write_tracking_outputs


def print_json(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def cmd_init(args: argparse.Namespace) -> None:
    queue = seed_queue(force=args.force)
    print(f"Initialized queue with {len(queue)} experiments at {QUEUE}")


def cmd_status(_: argparse.Namespace) -> None:
    cfg = settings()
    print_json(
        {
            "version": __version__,
            "dataset_slug": cfg.dataset_slug,
            "competition_slug": cfg.competition_slug,
            "accelerator": cfg.accelerator,
            "queue": status_counts(),
            "active": active_count(),
            "queue_file": str(QUEUE),
            "leaderboard": str(LEADERBOARD),
        }
    )


def cmd_queue(_: argparse.Namespace) -> None:
    rows = load_queue()
    print_json(rows)


def cmd_audit(args: argparse.Namespace) -> None:
    summary = audit_dataset(Path(args.dataset), Path(args.output) if args.output else STATE / "dataset_audit.json")
    print_json(summary)


def cmd_calibrate_geometry(args: argparse.Namespace) -> None:
    payload = calibrate_from_dataset(Path(args.dataset), Path(args.output) if args.output else STATE / "geometry_calibration.json")
    print_json(payload)


def cmd_build(args: argparse.Namespace) -> None:
    print_json({"built": build(exp_id=args.exp_id, limit=args.limit)})


def cmd_launch(args: argparse.Namespace) -> None:
    print_json({"launched": launch(limit=args.limit, timeout=args.timeout)})


def cmd_refresh(_: argparse.Namespace) -> None:
    print_json({"refreshed": refresh(), "queue": status_counts()})


def cmd_pull(args: argparse.Namespace) -> None:
    print_json({"pulled": [str(p) for p in pull(Path(args.output_root), skip_existing=args.skip_existing)]})


def cmd_leaderboard(args: argparse.Namespace) -> None:
    rows = write_leaderboard(Path(args.output_root))
    print_json({"rows": len(rows), "leaderboard": str(LEADERBOARD), "top": rows[:5]})


def cmd_review(args: argparse.Namespace) -> None:
    print_json(review(Path(args.output_root)))


def cmd_evolve(args: argparse.Namespace) -> None:
    print_json(evolve(max_new=args.max_new))


def cmd_confirm_leader(args: argparse.Namespace) -> None:
    print_json(confirm_leader(count=args.count, output_root=Path(args.output_root)))


def cmd_ensemble(args: argparse.Namespace) -> None:
    from .ensemble import fuse_models_on_images, resolve_top_k_weights

    if args.models:
        model_paths = [Path(m) for m in args.models]
    else:
        model_paths = resolve_top_k_weights(args.top_k, Path(args.output_root))
    if not model_paths:
        print_json({"error": "no model weights resolved; pass --models or pull top-k outputs with best.pt"})
        return
    result = fuse_models_on_images(
        model_paths=model_paths,
        images_dir=Path(args.images),
        output_path=Path(args.output),
        imgsz=args.imgsz,
        conf=args.conf,
        iou_thr=args.iou,
        tta=not args.no_tta,
    )
    print_json(result)


def cmd_typology(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    detections = payload.get("detections", payload) if isinstance(payload, dict) else payload
    enriched = enrich_detections(detections, load_rules())
    if args.output:
        Path(args.output).write_text(json.dumps(enriched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_json({"detections": len(enriched), "output": args.output, "sample": enriched[:5]})


def cmd_track_consensus(args: argparse.Namespace) -> None:
    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = consensus_tracks(rows, load_tracking_config())
    out_dir = Path(args.output_dir)
    write_tracking_outputs(result, out_dir, load_tracking_config())
    print_json(
        {
            "input_rows": len(rows),
            "frame_rows": len(result.get("frame_rows", [])),
            "tracks": len(result.get("track_summary", [])),
            "output_dir": str(out_dir),
        }
    )


def cmd_openai_fallback(args: argparse.Namespace) -> None:
    cfg = load_openai_fallback_config()
    with Path(args.frames).open(newline="", encoding="utf-8") as handle:
        frame_rows = list(csv.DictReader(handle))
    with Path(args.tracks).open(newline="", encoding="utf-8") as handle:
        track_rows = list(csv.DictReader(handle))
    with Path(args.votes).open(newline="", encoding="utf-8") as handle:
        vote_rows = list(csv.DictReader(handle))
    candidates = build_fallback_candidates(frame_rows, track_rows, vote_rows, cfg)
    decisions = resolve_candidates(candidates, cfg)
    write_openai_fallback_outputs(candidates, decisions, Path(args.output_dir), cfg)
    print_json(
        {
            "enabled": bool(cfg.get("enabled")),
            "candidates": len(candidates),
            "decisions": len(decisions),
            "output_dir": args.output_dir,
        }
    )


def cmd_eval_typology(args: argparse.Namespace) -> None:
    from .config import load_class_map
    from .typology_eval import evaluate_from_files, load_yolo_ground_truth

    classes = list(load_class_map().get("classes") or [])
    if args.gt_yolo:
        ground_truth = load_yolo_ground_truth(Path(args.gt_yolo), classes)
    else:
        gt_raw = json.loads(Path(args.gt_json).read_text(encoding="utf-8"))
        ground_truth = gt_raw.get("ground_truth", gt_raw) if isinstance(gt_raw, dict) else {}
    report = evaluate_from_files(Path(args.predictions), ground_truth, classes, iou_thr=args.iou)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_json(report)


def cmd_auto(args: argparse.Namespace) -> None:
    from .director import direct

    print_json(
        direct(
            output_root=Path(args.output_root),
            launch_limit=args.launch_limit,
            max_new=args.max_new,
            timeout=args.timeout,
            ensemble_top_k=args.ensemble_top_k,
            act=not args.dry_run,
        )
    )


def cmd_cycle(args: argparse.Namespace) -> None:
    cfg = settings()
    refreshed = refresh()
    pulled = pull(Path(args.output_root), skip_existing=True)
    rows = write_leaderboard(Path(args.output_root))
    rev = review(Path(args.output_root))
    evo = evolve(max_new=args.max_new)
    capacity = max(0, cfg.max_active_kernels - active_count())
    launched = []
    if capacity > 0 and args.launch_limit > 0:
        launched = launch(limit=min(args.launch_limit, capacity), timeout=args.timeout)
    print_json(
        {
            "refreshed": len(refreshed),
            "pulled": [str(p) for p in pulled],
            "leaderboard_rows": len(rows),
            "best": rev.get("best"),
            "objective": rev.get("objective"),
            "recommendation": rev.get("recommendation"),
            "evolved": evo.get("added", []),
            "active": active_count(),
            "launched": launched,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dvka", description="Drone Vehicle Kaggle Agents")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="initialize or reset the experiment queue")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_status = sub.add_parser("status", help="show local queue and Kaggle config state")
    p_status.set_defaults(func=cmd_status)

    p_queue = sub.add_parser("queue", help="print the full local queue")
    p_queue.set_defaults(func=cmd_queue)

    p_audit = sub.add_parser("audit-data", help="audit a local dataset before uploading/using it in Kaggle")
    p_audit.add_argument("dataset")
    p_audit.add_argument("--output")
    p_audit.set_defaults(func=cmd_audit)

    p_calibrate = sub.add_parser("calibrate-geometry", help="learn class area/aspect priors from labeled bounding boxes")
    p_calibrate.add_argument("dataset")
    p_calibrate.add_argument("--output")
    p_calibrate.set_defaults(func=cmd_calibrate_geometry)

    p_build = sub.add_parser("build", help="generate Kaggle notebook workspace(s) without launching")
    p_build.add_argument("--exp-id")
    p_build.add_argument("--limit", type=int, default=1)
    p_build.set_defaults(func=cmd_build)

    p_launch = sub.add_parser("launch", help="build and push queued Kaggle notebook(s)")
    p_launch.add_argument("--limit", type=int, default=1)
    p_launch.add_argument("--timeout", type=int, default=7200)
    p_launch.set_defaults(func=cmd_launch)

    p_refresh = sub.add_parser("refresh", help="refresh active Kaggle kernel statuses")
    p_refresh.set_defaults(func=cmd_refresh)

    p_pull = sub.add_parser("pull", help="download completed Kaggle kernel outputs")
    p_pull.add_argument("--output-root", default=str(OUTPUTS))
    p_pull.add_argument("--skip-existing", action="store_true")
    p_pull.set_defaults(func=cmd_pull)

    p_lb = sub.add_parser("leaderboard", help="rebuild leaderboard from pulled outputs")
    p_lb.add_argument("--output-root", default=str(OUTPUTS))
    p_lb.set_defaults(func=cmd_leaderboard)

    p_review = sub.add_parser("review", help="analyze completed runs and weak classes")
    p_review.add_argument("--output-root", default=str(OUTPUTS))
    p_review.set_defaults(func=cmd_review)

    p_evolve = sub.add_parser("evolve", help="enqueue new experiments from the current leaderboard")
    p_evolve.add_argument("--max-new", type=int)
    p_evolve.set_defaults(func=cmd_evolve)

    p_confirm = sub.add_parser("confirm-leader", help="enqueue seed replicas of the current best config to estimate sigma")
    p_confirm.add_argument("--count", type=int, default=None, help="number of extra seeds (default: reach min_seeds_for_sigma+1)")
    p_confirm.add_argument("--output-root", default=str(OUTPUTS))
    p_confirm.set_defaults(func=cmd_confirm_leader)

    p_ensemble = sub.add_parser("ensemble", help="fuse top-k models with WBF (+TTA) over an images folder for submission")
    p_ensemble.add_argument("--images", required=True, help="folder of images to run inference on")
    p_ensemble.add_argument("--output", default="outputs/ensemble_predictions.json")
    p_ensemble.add_argument("--models", nargs="*", help="explicit best.pt paths; if omitted, uses --top-k from the leaderboard")
    p_ensemble.add_argument("--top-k", type=int, default=3)
    p_ensemble.add_argument("--output-root", default=str(OUTPUTS))
    p_ensemble.add_argument("--imgsz", type=int, default=960)
    p_ensemble.add_argument("--conf", type=float, default=0.001)
    p_ensemble.add_argument("--iou", type=float, default=0.55)
    p_ensemble.add_argument("--no-tta", action="store_true", help="disable test-time augmentation")
    p_ensemble.set_defaults(func=cmd_ensemble)

    p_typology = sub.add_parser("typology", help="apply area/aspect typology rules to a detections JSON")
    p_typology.add_argument("input")
    p_typology.add_argument("--output")
    p_typology.set_defaults(func=cmd_typology)

    p_track_consensus = sub.add_parser("track-consensus", help="stabilize class labels across a tracking detections CSV")
    p_track_consensus.add_argument("input")
    p_track_consensus.add_argument("--output-dir", default="outputs/track_consensus")
    p_track_consensus.set_defaults(func=cmd_track_consensus)

    p_openai = sub.add_parser("openai-fallback", help="resolve only ambiguous low-confidence tracks with OpenAI as last resort")
    p_openai.add_argument("--frames", required=True)
    p_openai.add_argument("--tracks", required=True)
    p_openai.add_argument("--votes", required=True)
    p_openai.add_argument("--output-dir", default="outputs/openai_fallback")
    p_openai.set_defaults(func=cmd_openai_fallback)

    p_eval_typ = sub.add_parser("eval-typology", help="measure raw vs post-processed mAP50 to decide if typology helps the metric")
    p_eval_typ.add_argument("--predictions", required=True, help="predictions JSON with base_class/postprocessed_class")
    p_eval_typ.add_argument("--gt-yolo", help="YOLO labels dir (class cx cy w h)")
    p_eval_typ.add_argument("--gt-json", help="ground-truth JSON {image_stem: [{class, box(norm xyxy)}]}")
    p_eval_typ.add_argument("--iou", type=float, default=0.5)
    p_eval_typ.add_argument("--output")
    p_eval_typ.set_defaults(func=cmd_eval_typology)

    p_cycle = sub.add_parser("cycle", help="refresh, pull, leaderboard, review, evolve, and launch within capacity")
    p_cycle.add_argument("--output-root", default=str(OUTPUTS))
    p_cycle.add_argument("--launch-limit", type=int, default=1)
    p_cycle.add_argument("--max-new", type=int)
    p_cycle.add_argument("--timeout", type=int, default=7200)
    p_cycle.set_defaults(func=cmd_cycle)

    p_auto = sub.add_parser("auto", help="DirectorAgent: decide and perform the next methodology step (seeds/confirm/evolve/ensemble)")
    p_auto.add_argument("--output-root", default=str(OUTPUTS))
    p_auto.add_argument("--launch-limit", type=int, default=2)
    p_auto.add_argument("--max-new", type=int)
    p_auto.add_argument("--timeout", type=int, default=7200)
    p_auto.add_argument("--ensemble-top-k", type=int, default=3)
    p_auto.add_argument("--dry-run", action="store_true", help="only report the decision; do not enqueue or launch")
    p_auto.set_defaults(func=cmd_auto)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
