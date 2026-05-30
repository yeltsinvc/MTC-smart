"""Etapa 07 - Define la ROI de anotacion por video y la verifica visualmente.

Para cada video:
  * junta las 4 esquinas de TODAS las cajas OBB de TODOS sus frames (convencion ccw_math);
  * calcula el casco convexo (convex hull) de esas esquinas;
  * lo dilata un margen (escala desde el centroide + margen absoluto) para no cortar
    vehiculos en el borde de la ROI;
  * guarda el poligono ROI y su bbox (para recorte rectangular).

Salidas:
  * obb/roi_polygons.json  -> {video: {polygon:[[x,y]...], bbox:[x0,y0,x1,y1], n_boxes, n_frames, area_frac}}
  * obb/roi_verify/<video>.jpg -> overlay (cajas + poligono ROI + bbox) para inspeccion

Uso:
    python obb/build_roi.py --train ../data/train.csv --zip ../data/train.zip \
        --margin-scale 1.08 --margin-px 25 --n_verify 12
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    from scipy.spatial import ConvexHull
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

W, H = 1920, 1080
ID2NAME = {1: "auto", 2: "combi", 3: "microbus", 4: "minibus", 5: "omnibus",
           6: "articulado", 7: "camion", 8: "mototaxi", 9: "motocicleta"}
COLORS = {1: (230, 25, 75), 2: (60, 180, 75), 3: (255, 225, 25), 4: (67, 99, 216),
          5: (245, 130, 49), 6: (145, 30, 180), 7: (66, 212, 244), 8: (240, 50, 230),
          9: (191, 239, 69)}


def corners(cx, cy, w, h, angle_deg):
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    dx, dy = w / 2.0, h / 2.0
    pts = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    return [(cx + px * c - py * s, cy + px * s + py * c) for px, py in pts]


def parse_by_video(train_csv: Path):
    """video -> {'frames': set, 'boxes': [(cid,cx,cy,w,h,ang)], 'corners': [(x,y)]}"""
    vids = {}
    with open(train_csv, newline="", encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for line in r:
            if len(line) < 2:
                continue
            fid, tgt = line[0].strip(), line[1].strip()
            vid = fid.rsplit("_", 1)[0]
            d = vids.setdefault(vid, {"frames": set(), "boxes": [], "corners": []})
            d["frames"].add(fid)
            if not tgt or tgt.lower() == "none":
                continue
            for det in tgt.split(";"):
                p = det.split()
                if len(p) < 6:
                    continue
                cid = int(float(p[0])); cx, cy, w, h, ang = map(float, p[1:6])
                if w <= 0 or h <= 0:
                    continue
                d["boxes"].append((cid, cx, cy, w, h, ang))
                d["corners"].extend(corners(cx, cy, w, h, ang))
    return vids


def convex_hull(points: np.ndarray) -> np.ndarray:
    if HAVE_SCIPY and len(points) >= 3:
        try:
            h = ConvexHull(points)
            return points[h.vertices]
        except Exception:
            pass
    # fallback: bbox
    x0, y0 = points.min(0); x1, y1 = points.max(0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def dilate(poly: np.ndarray, scale: float, margin_px: float) -> np.ndarray:
    """Escala el poligono desde su centroide y luego empuja cada vertice margin_px hacia afuera."""
    cen = poly.mean(0)
    out = cen + (poly - cen) * scale
    # empuje radial adicional
    vecs = out - cen
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    out = out + vecs / norms * margin_px
    out[:, 0] = np.clip(out[:, 0], 0, W)
    out[:, 1] = np.clip(out[:, 1], 0, H)
    return out


def poly_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/train.csv")
    ap.add_argument("--zip", default="../data/train.zip")
    ap.add_argument("--margin-scale", type=float, default=1.08)
    ap.add_argument("--margin-px", type=float, default=25.0)
    ap.add_argument("--n_verify", type=int, default=12)
    ap.add_argument("--outdir", default="obb")
    args = ap.parse_args()

    out = Path(args.outdir); (out / "roi_verify").mkdir(parents=True, exist_ok=True)
    print(f"scipy ConvexHull: {HAVE_SCIPY}")
    print("Parseando train.csv por video ...")
    vids = parse_by_video(Path(args.train))
    print(f"  videos: {len(vids)}")

    roi = {}
    areas = []
    for vid, d in vids.items():
        if not d["corners"]:
            # video sin cajas: ROI vacia (se excluira del entrenamiento)
            roi[vid] = {"polygon": [], "bbox": None, "n_boxes": 0,
                        "n_frames": len(d["frames"]), "area_frac": 0.0}
            continue
        pts = np.array(d["corners"])
        hull = convex_hull(pts)
        dil = dilate(hull, args.margin_scale, args.margin_px)
        x0, y0 = dil.min(0); x1, y1 = dil.max(0)
        area_frac = poly_area(dil) / (W * H)
        areas.append(area_frac)
        roi[vid] = {
            "polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in dil],
            "bbox": [int(x0), int(y0), int(math.ceil(x1)), int(math.ceil(y1))],
            "n_boxes": len(d["boxes"]),
            "n_frames": len(d["frames"]),
            "area_frac": round(area_frac, 4),
        }

    (out / "roi_polygons.json").write_text(json.dumps(roi, ensure_ascii=False), encoding="utf-8")
    areas = np.array(areas)
    print("\n=== Cobertura de la ROI dilatada (poligono) ===")
    print(f"  mediana {np.median(areas):.2f} | p10 {np.percentile(areas,10):.2f} | p90 {np.percentile(areas,90):.2f}")
    print(f"  videos con ROI < 60% del cuadro: {(areas<0.6).mean()*100:.0f}%")
    n_empty = sum(1 for v in roi.values() if v['n_boxes'] == 0)
    print(f"  videos sin cajas (ROI vacia, se excluiran): {n_empty}")

    # verificacion visual: mezcla de ROIs pequenas, medianas y grandes
    items = sorted([(v, roi[v]["area_frac"]) for v in roi if roi[v]["n_boxes"] > 0], key=lambda x: x[1])
    pick = []
    if items:
        idxs = np.linspace(0, len(items) - 1, args.n_verify).astype(int)
        pick = [items[i][0] for i in idxs]
    with zipfile.ZipFile(args.zip) as z:
        name_map = {}
        for n in z.namelist():
            if n.lower().endswith(".jpg"):
                stem = Path(n).stem; v = stem.rsplit("_", 1)[0]
                name_map.setdefault(v, n)
        for vid in pick:
            if vid not in name_map:
                continue
            with z.open(name_map[vid]) as fh:
                im = Image.open(io.BytesIO(fh.read())).convert("RGB")
            dr = ImageDraw.Draw(im, "RGBA")
            poly = [tuple(p) for p in roi[vid]["polygon"]]
            # sombrear lo de FUERA de la ROI (oscurecer) para ver que se descarta
            if len(poly) >= 3:
                dr.polygon(poly, outline=(57, 255, 20), width=4)
            bb = roi[vid]["bbox"]
            if bb:
                dr.rectangle(bb, outline=(255, 20, 147), width=2)
            for (cid, cx, cy, w, h, ang) in vids[vid]["boxes"]:
                cs = corners(cx, cy, w, h, ang)
                dr.line(cs + [cs[0]], fill=COLORS.get(cid, (255, 255, 255)), width=2)
            dr.text((6, 6), f"{vid}  ROI={roi[vid]['area_frac']:.2f}  cajas={roi[vid]['n_boxes']}",
                    fill=(255, 255, 255))
            im.save(out / "roi_verify" / f"{vid}.jpg", quality=85)
    print(f"\nGuardado: obb/roi_polygons.json y {len(pick)} overlays en obb/roi_verify/")
    print("Verifica: el poligono verde debe contener TODOS los vehiculos de interes;")
    print("los vehiculos FUERA del verde son los que se descartan (no deberian ser de la calzada principal).")


if __name__ == "__main__":
    main()
