"""Simulacion: cuanto penaliza la metrica oficial detectar vehiculos REALES que
NO estan etiquetados (el caso de los vehiculos fuera de la ROI).

Modelo del experimento (sobre una muestra de frames reales de train.csv):
  * GT_eval = las cajas etiquetadas del frame (verdad-terreno).
  * Predicciones = TODAS las cajas GT (como TP, con score realista variado)
                   + K cajas EXTRA (vehiculos reales no etiquetados -> FP),
                     con tamano/clase muestreados de la distribucion real,
                     colocadas sin solapar el GT.
  * Se mide Macro AP-rIoU@[0.50:0.80] variando:
      (A) el SCORE de los FP, con K fijo (= 30% del GT).
      (B) la CANTIDAD de FP (0%..150% del GT), con score de FP alto (0.9).

Hipotesis: (A) FP con score < score de los TP casi no penaliza; FP con score alto
si. (B) cuantos mas FP confiados, mas cae el AP.

Salidas: eval/out/fp_score_sweep.png, eval/out/fp_count_sweep.png, eval/out/fp_sim.json

Uso:
    python eval/fp_simulation.py --train ../data/train.csv --frames 200
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric import macro_ap_riou, rotated_iou, ID2NAME  # noqa: E402

W, H = 1920, 1080
random.seed(42)


def load_frames(train_csv: Path, n_frames: int):
    frames = {}
    size_pool = {c: [] for c in ID2NAME}
    with open(train_csv, newline="", encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for line in r:
            if len(line) < 2:
                continue
            fid, tgt = line[0].strip(), line[1].strip()
            if not tgt or tgt.lower() == "none":
                continue
            boxes = []
            for d in tgt.split(";"):
                p = d.split()
                if len(p) < 6:
                    continue
                cid = int(float(p[0])); cx, cy, w, h, ang = map(float, p[1:6])
                if w > 0 and h > 0:
                    boxes.append((cid, cx, cy, w, h, ang))
                    size_pool[cid].append((w, h))
            if len(boxes) >= 3:   # frames con varias cajas (mas informativos)
                frames[fid] = boxes
            if len(frames) >= n_frames:
                # seguir leyendo un poco para enriquecer size_pool no hace falta
                break
    return frames, size_pool


def place_extra(boxes, size_pool, k):
    """Genera k cajas extra (vehiculos reales no etiquetados) sin solapar el GT."""
    extra = []
    classes = [c for c in size_pool if size_pool[c]]
    attempts = 0
    while len(extra) < k and attempts < k * 30:
        attempts += 1
        cid = random.choice(classes)
        w, h = random.choice(size_pool[cid])
        cx = random.uniform(0.05 * W, 0.95 * W)
        cy = random.uniform(0.05 * H, 0.95 * H)
        ang = random.uniform(0, 360)
        cand = (cid, cx, cy, w, h, ang)
        # no debe solapar ninguna GT (si solapa, seria un TP, no un FP)
        ok = True
        for g in boxes:
            if g[0] == cid and rotated_iou(cand[1:], g[1:]) > 0.1:
                ok = False; break
        if ok:
            extra.append(cand)
    return extra


def build_predictions(frames, fp_frac, fp_score, size_pool, tp_score_range=(0.5, 0.99)):
    """GT como TP (score variado) + FP extra (score = fp_score)."""
    preds = {}
    extras_by_frame = {}
    for fid, boxes in frames.items():
        plist = []
        for (cid, cx, cy, w, h, ang) in boxes:
            s = random.uniform(*tp_score_range)
            plist.append((s, cid, cx, cy, w, h, ang))
        k = int(round(len(boxes) * fp_frac))
        extra = extras_by_frame.get(fid)
        if extra is None:
            extra = place_extra(boxes, size_pool, max(k, 0)) if k > 0 else []
            extras_by_frame[fid] = extra
        for (cid, cx, cy, w, h, ang) in extra[:k]:
            plist.append((fp_score, cid, cx, cy, w, h, ang))
        preds[fid] = plist
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/train.csv")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--outdir", default="eval/out")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    print(f"Cargando {args.frames} frames de {args.train} ...")
    frames, size_pool = load_frames(Path(args.train), args.frames)
    n_boxes = sum(len(v) for v in frames.values())
    print(f"  frames={len(frames)}  cajas_gt={n_boxes}")

    # sanity: GT perfecto
    perfect = {f: [(1.0, *b) for b in boxes] for f, boxes in frames.items()}
    base = macro_ap_riou(frames, perfect)["macro_ap"]
    print(f"  sanity GT-perfecto macro_ap = {base:.4f} (debe ser ~1.0)")

    results = {"baseline_perfect": base, "n_frames": len(frames), "n_boxes": n_boxes}

    # ---- Experimento A: variar el SCORE de los FP (K fijo = 30% del GT) ----
    print("\n=== A) Efecto del SCORE de los FP (cantidad fija = 30% del GT) ===")
    fp_scores = [None, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    a_scores = []; a_aps = []
    # referencia sin FP
    pred0 = build_predictions(frames, 0.0, 0.0, size_pool)
    ap0 = macro_ap_riou(frames, pred0)["macro_ap"]
    print(f"  sin FP:            macro_ap = {ap0:.4f}")
    a_scores.append("sin FP"); a_aps.append(ap0)
    for s in fp_scores:
        if s is None:
            continue
        pred = build_predictions(frames, 0.30, s, size_pool)
        m = macro_ap_riou(frames, pred)["macro_ap"]
        print(f"  FP score={s:<4}:       macro_ap = {m:.4f}  (caida {(ap0-m):+.4f})")
        a_scores.append(f"{s}"); a_aps.append(m)
    results["score_sweep"] = {"labels": a_scores, "macro_ap": a_aps, "fp_frac": 0.30}

    # ---- Experimento B: variar la CANTIDAD de FP (score alto = 0.9) ----
    print("\n=== B) Efecto de la CANTIDAD de FP confiados (score=0.9) ===")
    fracs = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
    b_aps = []
    for fr in fracs:
        pred = build_predictions(frames, fr, 0.9, size_pool)
        m = macro_ap_riou(frames, pred)["macro_ap"]
        print(f"  FP = {int(fr*100):>3}% del GT: macro_ap = {m:.4f}  (caida {(ap0-m):+.4f})")
        b_aps.append(m)
    results["count_sweep"] = {"fp_frac": fracs, "macro_ap": b_aps, "fp_score": 0.9}

    (out / "fp_sim.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- figuras ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(a_aps)), a_aps, "o-", color="#d6336c")
    ax.set_xticks(range(len(a_scores))); ax.set_xticklabels(a_scores)
    ax.set_xlabel("score asignado a los FP (vehiculos reales no etiquetados)")
    ax.set_ylabel("Macro AP-rIoU@[0.50:0.80]")
    ax.set_title("A) Penalizacion segun el SCORE de los FP\n(cantidad fija = 30% del GT; TP score 0.5-0.99)")
    ax.axhline(ap0, ls="--", color="gray", lw=1, label=f"sin FP = {ap0:.3f}")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "fp_score_sweep.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([f * 100 for f in fracs], b_aps, "s-", color="#1c7ed6")
    ax.set_xlabel("FP anadidos como % del numero de cajas etiquetadas")
    ax.set_ylabel("Macro AP-rIoU@[0.50:0.80]")
    ax.set_title("B) Penalizacion segun la CANTIDAD de FP confiados (score=0.9)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fp_count_sweep.png", dpi=130); plt.close(fig)

    print(f"\nGuardado en {out}: fp_sim.json, fp_score_sweep.png, fp_count_sweep.png")


if __name__ == "__main__":
    main()
