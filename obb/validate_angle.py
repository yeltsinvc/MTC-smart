"""Valida la convencion del angulo de las cajas OBB del concurso.

El concurso da cada caja como:  category_id cx cy width height angle_deg
Pero NO documenta el signo/sentido del angulo ni si y crece hacia abajo.
Convertir mal => entrenar con basura. Aqui dibujamos cajas reales sobre imagenes
reales con varias convenciones y guardamos overlays para inspeccion visual.

Genera obb/angle_check/<frame>__<conv>.jpg para cada convencion candidata.

Uso:
    python obb/validate_angle.py --train ../data/train.csv --zip ../data/train.zip --n 6
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
COLORS = {1: "#e6194B", 2: "#3cb44b", 3: "#ffe119", 4: "#4363d8", 5: "#f58231",
          6: "#911eb4", 7: "#42d4f4", 8: "#f032e6", 9: "#bfef45"}

# Convenciones candidatas: (nombre, signo_aplicado_a_theta, descripcion)
# theta_rad = deg2rad(angle_deg) * sign
CONVENTIONS = {
    "ccw_math": +1.0,   # antihorario en eje matematico
    "cw_image": -1.0,   # horario (comun cuando y crece hacia abajo)
}


def corners(cx, cy, w, h, angle_deg, sign):
    th = np.deg2rad(angle_deg) * sign
    c, s = np.cos(th), np.sin(th)
    dx, dy = w / 2.0, h / 2.0
    pts = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    R = np.array([[c, -s], [s, c]])
    rot = pts @ R.T
    rot[:, 0] += cx
    rot[:, 1] += cy
    return [tuple(p) for p in rot]


def parse_frames(train_csv: Path, want: int):
    """Devuelve {frame_id: [(cid,cx,cy,w,h,ang), ...]} para frames con varias cajas rotadas."""
    out = {}
    with open(train_csv, newline="", encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for line in r:
            if len(line) < 2:
                continue
            fid, tgt = line[0].strip(), line[1].strip()
            if not tgt or tgt.lower() == "none":
                continue
            dets = []
            rotated = 0
            for d in tgt.split(";"):
                p = d.split()
                if len(p) < 6:
                    continue
                cid = int(float(p[0])); vals = list(map(float, p[1:6]))
                dets.append((cid, *vals))
                if abs(vals[4]) > 5:
                    rotated += 1
            # preferir frames con varias cajas rotadas (mas informativos)
            if rotated >= 3:
                out[fid] = dets
            if len(out) >= want:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/train.csv")
    ap.add_argument("--zip", default="../data/train.zip")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--outdir", default="obb/angle_check")
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    frames = parse_frames(Path(args.train), args.n)
    print(f"Frames seleccionados: {list(frames)}")

    with zipfile.ZipFile(args.zip) as z:
        names = {Path(n).stem: n for n in z.namelist() if n.lower().endswith(".jpg")}
        for fid, dets in frames.items():
            if fid not in names:
                print(f"  {fid}: imagen no encontrada en zip"); continue
            with z.open(names[fid]) as fh:
                base = Image.open(io.BytesIO(fh.read())).convert("RGB")
            for conv, sign in CONVENTIONS.items():
                im = base.copy()
                dr = ImageDraw.Draw(im)
                for cid, cx, cy, w, h, ang in dets:
                    pts = corners(cx, cy, w, h, ang, sign)
                    dr.line(pts + [pts[0]], fill=COLORS.get(cid, "#ffffff"), width=3)
                    dr.text((pts[0][0], pts[0][1] - 12), ID2NAME.get(cid, str(cid)),
                            fill=COLORS.get(cid, "#ffffff"))
                p = out / f"{fid}__{conv}.jpg"
                im.save(p, quality=85)
            print(f"  {fid}: {len(dets)} cajas -> overlays {list(CONVENTIONS)}")
    print(f"\nRevisa visualmente {out}: la convencion correcta es la que ENCIERRA bien los vehiculos.")


if __name__ == "__main__":
    main()
