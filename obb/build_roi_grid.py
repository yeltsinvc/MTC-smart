"""Etapa 07b - ROI por REJILLA DE OCUPACION (concava), sustituye al casco convexo.

Motivo: las anotaciones forman una CRUZ (dos avenidas). El casco convexo de una cruz
rellena las esquinas (manzanas con autos estacionados NO etiquetados) -> re-contamina.
La rejilla de ocupacion sigue la forma real de la zona anotada.

Metodo por video:
  * rejilla de celdas de cell_px (def 64) sobre 1920x1080;
  * marca una celda si alguna caja OBB (su bbox) la toca, acumulado en TODOS los frames;
  * dilatacion morfologica de 'dilate' celdas (margen para no cortar vehiculos del borde);
  * la mascara resultante (booleana por celda) ES la ROI.

Salidas:
  * obb/roi_grid.npz   -> cell_px + una mascara booleana por video (clave=video)
  * obb/roi_grid_meta.json -> {video: {cells_on, area_frac, n_boxes, n_frames}}
  * obb/roi_verify_grid/<video>.jpg -> overlay (mascara sombreada + cajas) para comparar

Uso:
    python obb/build_roi_grid.py --train ../data/train.csv --zip ../data/train.zip \
        --cell-px 64 --dilate 1 --n_verify 12
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
    vids = {}
    with open(train_csv, newline="", encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for line in r:
            if len(line) < 2:
                continue
            fid, tgt = line[0].strip(), line[1].strip()
            vid = fid.rsplit("_", 1)[0]
            d = vids.setdefault(vid, {"frames": set(), "boxes": []})
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
    return vids


def dilate_mask(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask
    out = mask.copy()
    for _ in range(k):
        m = out.copy()
        m[:-1, :] |= out[1:, :]
        m[1:, :] |= out[:-1, :]
        m[:, :-1] |= out[:, 1:]
        m[:, 1:] |= out[:, :-1]
        out = m
    return out


def build_grid(boxes, cell_px, dilate):
    gw = int(math.ceil(W / cell_px)); gh = int(math.ceil(H / cell_px))
    grid = np.zeros((gh, gw), dtype=bool)
    for (cid, cx, cy, w, h, ang) in boxes:
        cs = corners(cx, cy, w, h, ang)
        xs = [p[0] for p in cs]; ys = [p[1] for p in cs]
        c0 = max(0, int(min(xs) // cell_px)); c1 = min(gw - 1, int(max(xs) // cell_px))
        r0 = max(0, int(min(ys) // cell_px)); r1 = min(gh - 1, int(max(ys) // cell_px))
        grid[r0:r1 + 1, c0:c1 + 1] = True
    return dilate_mask(grid, dilate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/train.csv")
    ap.add_argument("--zip", default="../data/train.zip")
    ap.add_argument("--cell-px", type=int, default=64)
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--n_verify", type=int, default=12)
    ap.add_argument("--outdir", default="obb")
    args = ap.parse_args()

    out = Path(args.outdir); (out / "roi_verify_grid").mkdir(parents=True, exist_ok=True)
    print("Parseando train.csv por video ...")
    vids = parse_by_video(Path(args.train))
    print(f"  videos: {len(vids)}  cell_px={args.cell_px} dilate={args.dilate}")

    masks = {}; meta = {}
    areas = []
    for vid, d in vids.items():
        if not d["boxes"]:
            gw = int(math.ceil(W / args.cell_px)); gh = int(math.ceil(H / args.cell_px))
            masks[vid] = np.zeros((gh, gw), dtype=bool)
            meta[vid] = {"cells_on": 0, "area_frac": 0.0, "n_boxes": 0, "n_frames": len(d["frames"])}
            continue
        grid = build_grid(d["boxes"], args.cell_px, args.dilate)
        af = float(grid.mean())
        areas.append(af)
        masks[vid] = grid
        meta[vid] = {"cells_on": int(grid.sum()), "area_frac": round(af, 4),
                     "n_boxes": len(d["boxes"]), "n_frames": len(d["frames"])}

    np.savez_compressed(out / "roi_grid.npz", cell_px=args.cell_px, **{k: v for k, v in masks.items()})
    (out / "roi_grid_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    areas = np.array(areas)
    print("\n=== Cobertura ROI por rejilla (fraccion de celdas activas) ===")
    print(f"  mediana {np.median(areas):.2f} | p10 {np.percentile(areas,10):.2f} | p90 {np.percentile(areas,90):.2f}")
    print(f"  videos con ROI < 40% del cuadro: {(areas<0.4).mean()*100:.0f}%")

    # overlays comparables (mismos rangos de tamano)
    items = sorted([(v, meta[v]["area_frac"]) for v in meta if meta[v]["n_boxes"] > 0], key=lambda x: x[1])
    pick = [items[i][0] for i in np.linspace(0, len(items) - 1, args.n_verify).astype(int)] if items else []
    cell = args.cell_px
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
            ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
            dr = ImageDraw.Draw(ov)
            grid = masks[vid]
            # sombrear celdas FUERA de la ROI (rojo translucido) = lo que se descarta
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    if not grid[r, c]:
                        dr.rectangle([c * cell, r * cell, (c + 1) * cell, (r + 1) * cell],
                                     fill=(255, 0, 0, 70))
            im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
            dr2 = ImageDraw.Draw(im)
            for (cid, cx, cy, w, h, ang) in vids[vid]["boxes"]:
                cs = corners(cx, cy, w, h, ang)
                dr2.line(cs + [cs[0]], fill=COLORS.get(cid, (255, 255, 255)), width=2)
            dr2.text((6, 6), f"{vid}  ROI={meta[vid]['area_frac']:.2f}  cajas={meta[vid]['n_boxes']}  (rojo=descartado)",
                     fill=(255, 255, 255))
            im.save(out / "roi_verify_grid" / f"{vid}.jpg", quality=85)
    print(f"\nGuardado: obb/roi_grid.npz, roi_grid_meta.json y {len(pick)} overlays en obb/roi_verify_grid/")


if __name__ == "__main__":
    main()
