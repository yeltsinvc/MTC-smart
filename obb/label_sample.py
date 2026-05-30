"""Dibuja una muestra de N frames con TODAS sus cajas OBB (convencion ccw_math
validada) para inspeccion visual de las etiquetas antes de convertir todo.

- Colorea por clase, escribe la etiqueta de cada caja y una leyenda.
- Prioriza frames que contengan clases raras (articulado, omnibus, microbus,
  mototaxi, combi) para poder revisarlas, ademas de frames variados.

Uso:
    python obb/label_sample.py --train ../data/train.csv --zip ../data/train.zip --n 50
"""
from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ID2NAME = {1: "auto", 2: "combi", 3: "microbus", 4: "minibus", 5: "omnibus",
           6: "articulado", 7: "camion", 8: "mototaxi", 9: "motocicleta"}
COLORS = {1: (230, 25, 75), 2: (60, 180, 75), 3: (255, 225, 25), 4: (67, 99, 216),
          5: (245, 130, 49), 6: (145, 30, 180), 7: (66, 212, 244), 8: (240, 50, 230),
          9: (191, 239, 69)}
RARE = {"articulado", "omnibus", "microbus", "mototaxi", "combi"}
SIGN = +1.0  # ccw_math (validado en obb/validate_angle_zoom.py)


def corners(cx, cy, w, h, angle_deg):
    th = np.deg2rad(angle_deg) * SIGN
    c, s = np.cos(th), np.sin(th)
    dx, dy = w / 2.0, h / 2.0
    pts = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    R = np.array([[c, -s], [s, c]])
    rot = pts @ R.T
    rot[:, 0] += cx; rot[:, 1] += cy
    return [tuple(p) for p in rot]


def parse_all(train_csv: Path):
    frames = {}
    with open(train_csv, newline="", encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for line in r:
            if len(line) < 2:
                continue
            fid, tgt = line[0].strip(), line[1].strip()
            if not tgt or tgt.lower() == "none":
                continue
            dets = []
            for d in tgt.split(";"):
                p = d.split()
                if len(p) < 6:
                    continue
                cid = int(float(p[0])); cx, cy, w, h, ang = map(float, p[1:6])
                if w > 0 and h > 0:
                    dets.append((cid, cx, cy, w, h, ang))
            if dets:
                frames[fid] = dets
    return frames


def select_sample(frames: dict, n: int):
    """Elige n frames: primero uno por video que muestre cada clase rara, luego
    completa con frames variados de distintos videos."""
    by_video = {}
    for fid in frames:
        vid = fid.rsplit("_", 1)[0]
        by_video.setdefault(vid, []).append(fid)

    chosen = []
    chosen_set = set()

    # 1) cubrir clases raras: buscar frames con la clase rara mas grande/visible
    rare_ids = [cid for cid, nm in ID2NAME.items() if nm in RARE]
    for cid in rare_ids:
        best = None; best_area = 0
        for fid, dets in frames.items():
            for (c, cx, cy, w, h, ang) in dets:
                if c == cid and w * h > best_area:
                    best_area = w * h; best = fid
        if best and best not in chosen_set:
            chosen.append(best); chosen_set.add(best)

    # 2) completar con frames de videos distintos (diversidad de escenas)
    vids = sorted(by_video)
    i = 0
    while len(chosen) < n and i < len(vids) * 5:
        vid = vids[i % len(vids)]
        # frame con mas cajas de ese video
        cand = max(by_video[vid], key=lambda f: len(frames[f]))
        if cand not in chosen_set:
            chosen.append(cand); chosen_set.add(cand)
        i += 1
    return chosen[:n]


def draw_frame(base: Image.Image, dets, font):
    im = base.copy()
    dr = ImageDraw.Draw(im)
    counts = {}
    for (cid, cx, cy, w, h, ang) in dets:
        col = COLORS.get(cid, (255, 255, 255))
        pts = corners(cx, cy, w, h, ang)
        dr.line(pts + [pts[0]], fill=col, width=2)
        dr.text((min(p[0] for p in pts), min(p[1] for p in pts) - 11),
                ID2NAME.get(cid, str(cid)), fill=col, font=font)
        counts[ID2NAME.get(cid, str(cid))] = counts.get(ID2NAME.get(cid, str(cid)), 0) + 1
    # leyenda arriba-izquierda
    y = 6
    dr.rectangle([0, 0, 230, 14 + 13 * len(counts)], fill=(0, 0, 0))
    dr.text((6, y), f"{len(dets)} cajas", fill=(255, 255, 255), font=font); y += 13
    for nm, c in sorted(counts.items(), key=lambda x: -x[1]):
        cid = next(k for k, v in ID2NAME.items() if v == nm)
        dr.text((6, y), f"{nm}: {c}", fill=COLORS[cid], font=font); y += 13
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/train.csv")
    ap.add_argument("--zip", default="../data/train.zip")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--outdir", default="obb/sample_labels")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    print("Parseando train.csv ...")
    frames = parse_all(Path(args.train))
    sample = select_sample(frames, args.n)
    print(f"Frames seleccionados: {len(sample)}")

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    want = set(sample)
    saved = 0
    with zipfile.ZipFile(args.zip) as z:
        name_map = {Path(n).stem: n for n in z.namelist() if n.lower().endswith(".jpg")}
        # ordenar la muestra y guardar
        for idx, fid in enumerate(sample):
            if fid not in name_map:
                print(f"  [{idx}] {fid}: imagen no hallada"); continue
            with z.open(name_map[fid]) as fh:
                base = Image.open(io.BytesIO(fh.read())).convert("RGB")
            im = draw_frame(base, frames[fid], font)
            im.save(out / f"{idx:02d}_{fid}.jpg", quality=85)
            saved += 1
    print(f"Guardadas {saved} imagenes etiquetadas en {out}")

    # contar clases presentes en la muestra
    from collections import Counter
    cnt = Counter()
    for fid in sample:
        for (cid, *_rest) in frames[fid]:
            cnt[ID2NAME[cid]] += 1
    print("Cajas por clase en la muestra:")
    for nm in ID2NAME.values():
        print(f"  {nm:<12} {cnt.get(nm,0)}")


if __name__ == "__main__":
    main()
