"""Etapa 08 - Conversor train.csv -> dataset YOLO-OBB, recortado+ennegrecido a la ROI.

Para cada frame:
  * carga la imagen desde train.zip;
  * recorta al bbox de la ROI del video (roi_grid.npz);
  * ennegrece las celdas de la rejilla que quedan FUERA de la ROI dentro de ese recorte
    (negativos limpios: elimina manzanas/esquinas con vehiculos no etiquetados);
  * conserva solo las cajas cuyo CENTRO cae dentro de la ROI;
  * escribe la etiqueta en formato Ultralytics OBB (DOTA): clase x1 y1 x2 y2 x3 y3 x4 y4
    con coordenadas normalizadas [0,1] respecto al recorte (clase 0-based = category_id-1).

Split por VIDEO (no por frame) para evitar fuga: val_frac de los videos van a val.

Salidas (en --out, por defecto dataset_obb/):
  images/train/*.jpg, images/val/*.jpg
  labels/train/*.txt, labels/val/*.txt
  data.yaml
  convert_report.json

Modos:
  --limit N        procesa solo N videos (prueba)
  --verify-dir D   ademas dibuja las etiquetas YOLO-OBB de vuelta sobre el recorte (control)

Uso (prueba):
  python obb/convert_to_yolo_obb.py --limit 3 --verify-dir obb/convert_check
Uso (completo):
  python obb/convert_to_yolo_obb.py
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
NAMES_0BASED = [ID2NAME[i] for i in range(1, 10)]  # indice YOLO = id-1


def corners(cx, cy, w, h, angle_deg):
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    dx, dy = w / 2.0, h / 2.0
    pts = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    return [(cx + px * c - py * s, cy + px * s + py * c) for px, py in pts]


def parse_train(train_csv: Path):
    frames = {}
    order = []
    with open(train_csv, newline="", encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for line in r:
            if len(line) < 2:
                continue
            fid, tgt = line[0].strip(), line[1].strip()
            boxes = []
            if tgt and tgt.lower() != "none":
                for det in tgt.split(";"):
                    p = det.split()
                    if len(p) < 6:
                        continue
                    cid = int(float(p[0])); cx, cy, w, h, ang = map(float, p[1:6])
                    if w > 0 and h > 0:
                        boxes.append((cid, cx, cy, w, h, ang))
            frames[fid] = boxes
            order.append(fid)
    return frames, order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/train.csv")
    ap.add_argument("--zip", default="../data/train.zip")
    ap.add_argument("--roi", default="obb/roi_grid.npz")
    ap.add_argument("--out", default="dataset_obb")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=0, help="0 = todos los videos")
    ap.add_argument("--no-blackout", action="store_true", help="no ennegrecer fuera de la ROI")
    ap.add_argument("--verify-dir", default="")
    args = ap.parse_args()

    out = Path(args.out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    verify = Path(args.verify_dir) if args.verify_dir else None
    if verify:
        verify.mkdir(parents=True, exist_ok=True)

    roi_npz = np.load(args.roi, allow_pickle=True)
    cell_px = int(roi_npz["cell_px"])
    roi_masks = {k: roi_npz[k] for k in roi_npz.files if k != "cell_px"}

    print("Parseando train.csv ...")
    frames, _ = parse_train(Path(args.train))

    # videos validos: con ROI no vacia
    videos = sorted({fid.rsplit("_", 1)[0] for fid in frames})
    videos = [v for v in videos if v in roi_masks and roi_masks[v].sum() > 0]
    if args.limit:
        videos = videos[: args.limit]
    # split por video (determinista: hash del nombre)
    val_videos = set(v for v in videos if (hash(v) % 1000) / 1000.0 < args.val_frac)
    print(f"  videos a procesar: {len(videos)}  (val: {len(val_videos)})  cell_px={cell_px}")

    report = {"videos": len(videos), "val_videos": len(val_videos),
              "frames_written": 0, "boxes_kept": 0, "boxes_dropped_outside_roi": 0,
              "class_counts": {n: 0 for n in NAMES_0BASED}}

    name_index = None
    with zipfile.ZipFile(args.zip) as z:
        if name_index is None:
            name_index = {Path(n).stem: n for n in z.namelist() if n.lower().endswith(".jpg")}
        for vi, vid in enumerate(videos):
            mask = roi_masks[vid]  # (gh, gw) bool
            gh, gw = mask.shape
            ys, xs = np.where(mask)
            x0 = int(xs.min() * cell_px); x1 = int(min(W, (xs.max() + 1) * cell_px))
            y0 = int(ys.min() * cell_px); y1 = int(min(H, (ys.max() + 1) * cell_px))
            split = "val" if vid in val_videos else "train"
            # frames del video
            vframes = [fid for fid in frames if fid.rsplit("_", 1)[0] == vid]
            for fid in vframes:
                if fid not in name_index:
                    continue
                with z.open(name_index[fid]) as fh:
                    img = Image.open(io.BytesIO(fh.read())).convert("RGB")
                arr = np.asarray(img)
                if not args.no_blackout:
                    # ennegrecer celdas fuera de la ROI (mascara expandida a pixeles, vectorizado)
                    full_mask = np.repeat(np.repeat(mask, cell_px, axis=0), cell_px, axis=1)[:H, :W]
                    arr = np.where(full_mask[:, :, None], arr, 0)
                crop = arr[y0:y1, x0:x1]
                ch, cw = crop.shape[:2]
                # labels
                lines = []
                for (cid, cx, cy, w, h, ang) in frames[fid]:
                    # centro dentro de la ROI?
                    gc = int(min(gw - 1, max(0, cx // cell_px))); gr = int(min(gh - 1, max(0, cy // cell_px)))
                    if not mask[gr, gc]:
                        report["boxes_dropped_outside_roi"] += 1
                        continue
                    cs = corners(cx, cy, w, h, ang)
                    norm = []
                    for (px, py) in cs:
                        nx = (px - x0) / cw
                        ny = (py - y0) / ch
                        norm.extend([nx, ny])
                    # descartar si queda completamente fuera del recorte
                    xs_n = norm[0::2]; ys_n = norm[1::2]
                    if max(xs_n) < 0 or min(xs_n) > 1 or max(ys_n) < 0 or min(ys_n) > 1:
                        report["boxes_dropped_outside_roi"] += 1
                        continue
                    norm = [min(1.0, max(0.0, v)) for v in norm]
                    cls0 = cid - 1
                    lines.append(f"{cls0} " + " ".join(f"{v:.6f}" for v in norm))
                    report["boxes_kept"] += 1
                    report["class_counts"][NAMES_0BASED[cls0]] += 1
                # guardar imagen + label
                Image.fromarray(crop).save(out / "images" / split / f"{fid}.jpg", quality=88)
                (out / "labels" / split / f"{fid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                report["frames_written"] += 1

                # verificacion: dibujar etiquetas YOLO-OBB de vuelta
                if verify and len(lines) > 0 and report["frames_written"] % 7 == 1:
                    vim = Image.fromarray(crop).convert("RGB")
                    dr = ImageDraw.Draw(vim)
                    for ln in lines:
                        parts = ln.split()
                        coords = list(map(float, parts[1:]))
                        pts = [(coords[i] * cw, coords[i + 1] * ch) for i in range(0, 8, 2)]
                        dr.line(pts + [pts[0]], fill=(57, 255, 20), width=2)
                    vim.save(verify / f"{fid}.jpg", quality=85)
            if (vi + 1) % 50 == 0 or args.limit:
                print(f"  [{vi+1}/{len(videos)}] {vid} split={split} crop=({cw}x{ch}) frames_ok={report['frames_written']}")

    # data.yaml
    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: NAMES_0BASED[i] for i in range(len(NAMES_0BASED))},
    }
    import yaml
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (out / "convert_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Reporte ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nGuardado dataset YOLO-OBB en {out}")
    if verify:
        print(f"Overlays de verificacion en {verify}")


if __name__ == "__main__":
    main()
