"""Zoom comparativo de convenciones de angulo: recorta cada vehiculo rotado y
dibuja la caja con ambas convenciones, lado a lado, ampliado.

Uso:
    python obb/validate_angle_zoom.py --train ../data/train.csv --zip ../data/train.zip
"""
from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ID2NAME = {1: "auto", 2: "combi", 3: "microbus", 4: "minibus", 5: "omnibus",
           6: "articulado", 7: "camion", 8: "mototaxi", 9: "motocicleta"}
CONVENTIONS = {"ccw_math": +1.0, "cw_image": -1.0}


def corners(cx, cy, w, h, angle_deg, sign):
    th = np.deg2rad(angle_deg) * sign
    c, s = np.cos(th), np.sin(th)
    dx, dy = w / 2.0, h / 2.0
    pts = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    R = np.array([[c, -s], [s, c]])
    rot = pts @ R.T
    rot[:, 0] += cx; rot[:, 1] += cy
    return rot


def first_rotated_boxes(train_csv: Path, k: int):
    """Devuelve hasta k cajas (frame_id, cid, cx, cy, w, h, ang) bien rotadas y grandes."""
    found = []
    with open(train_csv, newline="", encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for line in r:
            if len(line) < 2:
                continue
            fid, tgt = line[0].strip(), line[1].strip()
            if not tgt or tgt.lower() == "none":
                continue
            for d in tgt.split(";"):
                p = d.split()
                if len(p) < 6:
                    continue
                cid = int(float(p[0])); cx, cy, w, h, ang = map(float, p[1:6])
                # caja grande (visible) y claramente rotada
                if w * h > 4000 and 20 < (abs(ang) % 180) < 160:
                    found.append((fid, cid, cx, cy, w, h, ang))
                    if len(found) >= k:
                        return found
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/train.csv")
    ap.add_argument("--zip", default="../data/train.zip")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--outdir", default="obb/angle_check")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    boxes = first_rotated_boxes(Path(args.train), args.k)
    print(f"Cajas rotadas elegidas: {[(b[0], ID2NAME[b[1]], round(b[6],1)) for b in boxes]}")

    with zipfile.ZipFile(args.zip) as z:
        names = {Path(n).stem: n for n in z.namelist() if n.lower().endswith(".jpg")}
        for i, (fid, cid, cx, cy, w, h, ang) in enumerate(boxes):
            if fid not in names:
                continue
            with z.open(names[fid]) as fh:
                base = Image.open(io.BytesIO(fh.read())).convert("RGB")
            half = int(max(w, h) * 1.6)
            box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
            panels = []
            for conv, sign in CONVENTIONS.items():
                im = base.copy()
                dr = ImageDraw.Draw(im)
                pts = corners(cx, cy, w, h, ang, sign)
                dr.line([tuple(p) for p in pts] + [tuple(pts[0])], fill="#39FF14", width=2)
                # marcar el primer vertice (esquina sup-izq local) para ver orientacion
                dr.ellipse([pts[0][0]-3, pts[0][1]-3, pts[0][0]+3, pts[0][1]+3], fill="#FF1493")
                crop = im.crop(box).resize((360, 360), Image.NEAREST)
                d2 = ImageDraw.Draw(crop)
                d2.text((6, 6), conv, fill="#39FF14")
                panels.append(crop)
            combo = Image.new("RGB", (360 * len(panels) + 10 * (len(panels)-1), 360), "#000000")
            x = 0
            for pnl in panels:
                combo.paste(pnl, (x, 0)); x += 370
            tag = f"{ID2NAME[cid]}_{abs(int(ang))}"
            combo.save(out / f"zoom_{i}_{tag}.jpg", quality=90)
            print(f"  guardado zoom_{i}_{tag}.jpg  ({ID2NAME[cid]}, ang={ang:.1f})")
    print(f"\nCompara en {out}/zoom_*.jpg: izq=ccw_math, der=cw_image. Gana la que alinea la caja con el vehiculo.")


if __name__ == "__main__":
    main()
