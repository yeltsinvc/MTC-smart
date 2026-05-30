"""Implementacion local de la metrica oficial: Macro AP-rIoU@[0.50:0.80].

Reproduce la evaluacion del SMART CHALLENGE 2026 para poder medir modelos sin
gastar submissions de Kaggle. Sin dependencias externas (rotated-IoU implementado
a mano con recorte de poligonos convexos de Sutherland-Hodgman).

Definicion (segun la pestana Evaluation del concurso):
  * rotated IoU (rIoU) entre cajas OBB prediccion vs real.
  * Para cada clase (1..9) y cada umbral t in {0.50,0.55,...,0.80}:
      - ordenar predicciones de esa clase por score desc;
      - match codicioso: una prediccion matchea un GT del MISMO frame y MISMA clase
        con rIoU >= t que no haya sido ya asignado; si matchea -> TP, si no -> FP;
      - AP = area bajo la curva precision-recall (interpolacion VOC all-points).
  * Score final = promedio NO ponderado de AP sobre las 9 clases y los 7 umbrales.

Formatos (ver concurso):
  GT  por caja:          category_id cx cy width height angle_deg
  Pred por caja: score   category_id cx cy width height angle_deg

API principal:
    macro_ap_riou(gt_by_frame, pred_by_frame) -> dict con score y desgloses
donde gt_by_frame[frame] = [(cid, cx, cy, w, h, ang), ...]
      pred_by_frame[frame] = [(score, cid, cx, cy, w, h, ang), ...]
"""
from __future__ import annotations

import math
from typing import Iterable

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
CLASS_IDS = list(range(1, 10))
ID2NAME = {1: "auto", 2: "combi", 3: "microbus", 4: "minibus", 5: "omnibus",
           6: "articulado", 7: "camion", 8: "mototaxi", 9: "motocicleta"}


# ----------------------------- geometria rotada -----------------------------
def rect_corners(cx: float, cy: float, w: float, h: float, angle_deg: float):
    """4 esquinas de la caja OBB. Convencion ccw_math validada (theta directo)."""
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    dx, dy = w / 2.0, h / 2.0
    pts = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    return [(cx + px * c - py * s, cy + px * s + py * c) for px, py in pts]


def _signed_area(poly):
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a / 2.0


def _poly_area(poly):
    return abs(_signed_area(poly))


def _ccw(poly):
    """Devuelve el poligono en orden antihorario (signed area > 0)."""
    return poly if _signed_area(poly) >= 0 else poly[::-1]


def _clip(subject, clipper):
    """Sutherland-Hodgman: recorta el poligono 'subject' contra 'clipper' (convexo, CCW)."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12

    def isect(p1, p2, a, b):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = a; x4, y4 = b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-12:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    output = list(subject)
    n = len(clipper)
    for i in range(n):
        a = clipper[i]; b = clipper[(i + 1) % n]
        inp = output; output = []
        if not inp:
            break
        prev = inp[-1]
        for cur in inp:
            if inside(cur, a, b):
                if not inside(prev, a, b):
                    output.append(isect(prev, cur, a, b))
                output.append(cur)
            elif inside(prev, a, b):
                output.append(isect(prev, cur, a, b))
            prev = cur
    return output


def rotated_iou(box1, box2) -> float:
    """rIoU entre dos cajas (cx,cy,w,h,angle_deg)."""
    p1 = _ccw(rect_corners(*box1))
    p2 = _ccw(rect_corners(*box2))
    inter_poly = _clip(p1, p2)
    if len(inter_poly) < 3:
        return 0.0
    inter = _poly_area(inter_poly)
    a1 = _poly_area(p1); a2 = _poly_area(p2)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# ----------------------------- AP por clase/umbral -----------------------------
def _voc_ap(rec, prec) -> float:
    """AP por interpolacion VOC all-points (area bajo la envolvente decreciente)."""
    mrec = [0.0] + list(rec) + [1.0]
    mpre = [0.0] + list(prec) + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def _ap_for_class_threshold(preds, gts, iou_cache, thr) -> float | None:
    """preds: lista (score, frame, idx). gts: dict frame->num_gt. iou_cache[(frame,pidx,gidx)]=iou.
    Devuelve AP, o None si no hay GT de esa clase."""
    npos = sum(gts.values())
    if npos == 0:
        return None
    preds_sorted = sorted(preds, key=lambda x: -x[0])
    assigned: set[tuple] = set()
    tp = [0] * len(preds_sorted)
    fp = [0] * len(preds_sorted)
    for i, (score, frame, pidx) in enumerate(preds_sorted):
        best_iou = 0.0; best_g = -1
        for gidx in range(gts_frame_count(gts, frame)):
            if (frame, gidx) in assigned:
                continue
            iou = iou_cache.get((frame, pidx, gidx), 0.0)
            if iou >= thr and iou > best_iou:
                best_iou = iou; best_g = gidx
        if best_g >= 0:
            tp[i] = 1; assigned.add((frame, best_g))
        else:
            fp[i] = 1
    # acumulados
    tp_c = 0; fp_c = 0; rec = []; prec = []
    for i in range(len(preds_sorted)):
        tp_c += tp[i]; fp_c += fp[i]
        rec.append(tp_c / npos)
        prec.append(tp_c / (tp_c + fp_c))
    return _voc_ap(rec, prec)


def gts_frame_count(gts_frame_counts: dict, frame) -> int:
    return gts_frame_counts.get(frame, 0)


def macro_ap_riou(gt_by_frame: dict, pred_by_frame: dict, verbose: bool = False) -> dict:
    """Calcula Macro AP-rIoU@[0.50:0.80]. Devuelve score global y desgloses por
    clase y por umbral."""
    # indexar GT y preds por clase
    gt_idx: dict[int, dict] = {c: {} for c in CLASS_IDS}      # cid -> {frame: [box,...]}
    for frame, boxes in gt_by_frame.items():
        for (cid, cx, cy, w, h, ang) in boxes:
            gt_idx.setdefault(cid, {}).setdefault(frame, []).append((cx, cy, w, h, ang))

    pred_idx: dict[int, dict] = {c: {} for c in CLASS_IDS}    # cid -> {frame: [(score,box),...]}
    for frame, boxes in pred_by_frame.items():
        for (score, cid, cx, cy, w, h, ang) in boxes:
            pred_idx.setdefault(cid, {}).setdefault(frame, []).append((score, (cx, cy, w, h, ang)))

    per_class = {}
    per_cell = {}  # (clase, umbral) -> AP
    for cid in CLASS_IDS:
        gframes = gt_idx.get(cid, {})
        gt_counts = {f: len(v) for f, v in gframes.items()}
        pframes = pred_idx.get(cid, {})
        # precomputar IoU pred-gt por frame (compartido entre umbrales)
        iou_cache: dict[tuple, float] = {}
        flat_preds = []
        for frame, plist in pframes.items():
            glist = gframes.get(frame, [])
            for pidx, (score, pbox) in enumerate(plist):
                flat_preds.append((score, frame, pidx))
                for gidx, gbox in enumerate(glist):
                    iou_cache[(frame, pidx, gidx)] = rotated_iou(pbox, gbox)
        aps = []
        for thr in THRESHOLDS:
            ap = _ap_for_class_threshold(flat_preds, gt_counts, iou_cache, thr)
            if ap is not None:
                per_cell[(cid, thr)] = ap
                aps.append(ap)
        if aps:
            per_class[cid] = sum(aps) / len(aps)
        if verbose and aps:
            print(f"  {ID2NAME[cid]:<12} AP={per_class[cid]:.4f}  (n_gt={sum(gt_counts.values())})")

    classes_present = [c for c in CLASS_IDS if c in per_class]
    macro = sum(per_class[c] for c in classes_present) / len(classes_present) if classes_present else 0.0
    return {
        "macro_ap": macro,
        "per_class_ap": {ID2NAME[c]: per_class.get(c) for c in CLASS_IDS},
        "per_threshold_ap": {
            t: (sum(per_cell[(c, t)] for c in classes_present if (c, t) in per_cell) /
                max(1, sum(1 for c in classes_present if (c, t) in per_cell)))
            for t in THRESHOLDS
        },
        "classes_evaluated": [ID2NAME[c] for c in classes_present],
    }


# ----------------------------- auto-test -----------------------------
def _selftest() -> None:
    print("=== auto-test rotated_iou ===")
    # 1) cajas identicas -> 1.0
    b = (100, 100, 40, 20, 0.0)
    assert abs(rotated_iou(b, b) - 1.0) < 1e-6, rotated_iou(b, b)
    print(f"  identicas: {rotated_iou(b,b):.4f} (esperado 1.0) OK")
    # 2) axis-aligned con solape conocido: dos 10x10, desplazadas 5 en x
    a = (0, 0, 10, 10, 0.0); c = (5, 0, 10, 10, 0.0)
    # interseccion 5x10=50, union 100+100-50=150 -> 1/3
    iou = rotated_iou(a, c)
    assert abs(iou - 1/3) < 1e-6, iou
    print(f"  solape 50%: {iou:.4f} (esperado 0.3333) OK")
    # 3) sin solape -> 0
    d = (100, 100, 10, 10, 0.0)
    assert rotated_iou(a, d) == 0.0
    print(f"  disjuntas: {rotated_iou(a,d):.4f} (esperado 0.0) OK")
    # 4) rotacion 90 grados de cuadrado -> identica (1.0)
    sq = (0, 0, 20, 20, 0.0); sqr = (0, 0, 20, 20, 90.0)
    assert abs(rotated_iou(sq, sqr) - 1.0) < 1e-6, rotated_iou(sq, sqr)
    print(f"  cuadrado rot90: {rotated_iou(sq,sqr):.4f} (esperado 1.0) OK")
    # 5) rectangulo rotado 45: IoU < 1
    r0 = (0, 0, 40, 10, 0.0); r45 = (0, 0, 40, 10, 45.0)
    iou45 = rotated_iou(r0, r45)
    assert 0.1 < iou45 < 0.5, iou45
    print(f"  rect rot45: {iou45:.4f} (esperado ~0.2-0.4) OK")

    print("\n=== auto-test AP (GT perfecto -> macro_ap = 1.0) ===")
    gt = {
        "f1": [(1, 100, 100, 40, 20, 0.0), (9, 200, 200, 20, 15, 10.0)],
        "f2": [(1, 50, 60, 40, 20, 0.0)],
    }
    pred = {f: [(1.0, *box) for box in boxes] for f, boxes in gt.items()}
    res = macro_ap_riou(gt, pred)
    print(f"  macro_ap={res['macro_ap']:.4f} (esperado 1.0), clases={res['classes_evaluated']}")
    assert abs(res["macro_ap"] - 1.0) < 1e-6, res["macro_ap"]
    print("\nTODOS LOS TESTS PASARON.")


if __name__ == "__main__":
    _selftest()
