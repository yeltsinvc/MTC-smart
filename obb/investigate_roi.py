"""Investiga si cada video tiene una ROI estable donde se concentran las anotaciones.

Para cada video:
  * acumula los centros (cx,cy) de TODAS las cajas de TODOS sus frames;
  * mide la envolvente (bbox + casco convexo) y que fraccion de la imagen cubre;
  * estima 'densidad de borde': si la ROI fuera toda la imagen, las cajas tocarian
    los bordes; si es central, no.

Salidas:
  * obb/roi_out/roi_stats.csv  (una fila por video)
  * obb/roi_out/heatmap_global.png  (densidad de centros sobre todos los videos)
  * obb/roi_out/roi_<video>.png  (heatmap + envolvente para una muestra de videos)
  * resumen agregado por consola

Uso:
    python obb/investigate_roi.py --train ../data/train.csv --zip ../data/train.zip
"""
from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

W, H = 1920, 1080  # resolucion confirmada


def parse_by_video(train_csv: Path):
    vids = {}
    n_frames = {}
    with open(train_csv, newline="", encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for line in r:
            if len(line) < 2:
                continue
            fid, tgt = line[0].strip(), line[1].strip()
            vid = fid.rsplit("_", 1)[0]
            n_frames[vid] = n_frames.get(vid, 0) + 1
            if not tgt or tgt.lower() == "none":
                continue
            for d in tgt.split(";"):
                p = d.split()
                if len(p) < 6:
                    continue
                cx, cy, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                vids.setdefault(vid, []).append((cx, cy, w, h))
    return vids, n_frames


def convex_hull_area(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(pts).volume)  # 'volume' en 2D = area
    except Exception:
        # fallback: area del bbox
        return float((pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/train.csv")
    ap.add_argument("--zip", default="../data/train.zip")
    ap.add_argument("--outdir", default="obb/roi_out")
    ap.add_argument("--n_overlays", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    print("Parseando train.csv por video ...")
    vids, n_frames = parse_by_video(args.train)
    print(f"  videos con cajas: {len(vids)}")

    rows = []
    global_centers = []
    for vid, boxes in vids.items():
        arr = np.array(boxes)  # cx,cy,w,h
        centers = arr[:, :2]
        global_centers.append(centers)
        # envolvente bbox de centros
        x0, y0 = centers[:, 0].min(), centers[:, 1].min()
        x1, y1 = centers[:, 0].max(), centers[:, 1].max()
        bbox_w, bbox_h = x1 - x0, y1 - y0
        bbox_frac = (bbox_w * bbox_h) / (W * H)
        hull_frac = convex_hull_area(centers) / (W * H)
        # cobertura: percentil 5-95 para robustez a outliers
        px0, px1 = np.percentile(centers[:, 0], [2.5, 97.5])
        py0, py1 = np.percentile(centers[:, 1], [2.5, 97.5])
        core_frac = ((px1 - px0) * (py1 - py0)) / (W * H)
        rows.append({
            "video": vid,
            "n_boxes": len(boxes),
            "n_frames": n_frames.get(vid, 0),
            "boxes_per_frame": round(len(boxes) / max(1, n_frames.get(vid, 0)), 1),
            "bbox_frac": round(bbox_frac, 3),
            "hull_frac": round(hull_frac, 3),
            "core95_frac": round(core_frac, 3),
            "cx_min": int(x0), "cx_max": int(x1), "cy_min": int(y0), "cy_max": int(y1),
        })

    # CSV
    rows.sort(key=lambda r: r["hull_frac"])
    fields = list(rows[0].keys())
    with (out / "roi_stats.csv").open("w", newline="", encoding="utf-8") as fh:
        wri = csv.DictWriter(fh, fieldnames=fields); wri.writeheader(); wri.writerows(rows)

    # resumen agregado
    hull = np.array([r["hull_frac"] for r in rows])
    core = np.array([r["core95_frac"] for r in rows])
    bpf = np.array([r["boxes_per_frame"] for r in rows])
    print("\n=== Cobertura de la ROI (fraccion de la imagen) ===")
    print(f"  casco convexo de centros  -> mediana {np.median(hull):.2f} | p10 {np.percentile(hull,10):.2f} | p90 {np.percentile(hull,90):.2f}")
    print(f"  nucleo 95% (robusto)      -> mediana {np.median(core):.2f} | p10 {np.percentile(core,10):.2f} | p90 {np.percentile(core,90):.2f}")
    print(f"  cajas por frame           -> mediana {np.median(bpf):.1f} | min {bpf.min():.1f} | max {bpf.max():.1f}")
    frac_small = float((hull < 0.5).mean())
    print(f"  videos cuya ROI cubre <50% de la imagen: {frac_small*100:.0f}%")
    frac_big = float((hull > 0.85).mean())
    print(f"  videos cuya ROI cubre >85% de la imagen: {frac_big*100:.0f}%")

    # heatmap global
    allc = np.vstack(global_centers)
    plt.figure(figsize=(9, 5))
    plt.hist2d(allc[:, 0], allc[:, 1], bins=[96, 54], range=[[0, W], [0, H]], cmap="inferno")
    plt.gca().invert_yaxis()
    plt.title(f"Densidad de centros de cajas (todos los videos, n={len(allc):,})")
    plt.xlabel("x"); plt.ylabel("y"); plt.colorbar(label="cajas")
    plt.tight_layout(); plt.savefig(out / "heatmap_global.png", dpi=130); plt.close()

    # overlays para una muestra de videos (los de ROI mas pequena + algunos grandes)
    sample = rows[: args.n_overlays // 2] + rows[-(args.n_overlays - args.n_overlays // 2):]
    with zipfile.ZipFile(args.zip) as z:
        name_map = {}
        for n in z.namelist():
            if n.lower().endswith(".jpg"):
                stem = Path(n).stem
                v = stem.rsplit("_", 1)[0]
                name_map.setdefault(v, n)  # primer frame del video
        for r in sample:
            vid = r["video"]
            if vid not in name_map:
                continue
            with z.open(name_map[vid]) as fh:
                img = Image.open(io.BytesIO(fh.read())).convert("RGB")
            centers = np.array([(b[0], b[1]) for b in vids[vid]])
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.imshow(img)
            ax.scatter(centers[:, 0], centers[:, 1], s=4, c="#39FF14", alpha=0.5)
            # rectangulo envolvente
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((r["cx_min"], r["cy_min"]), r["cx_max"]-r["cx_min"],
                                   r["cy_max"]-r["cy_min"], fill=False, edgecolor="#FF1493", lw=2))
            ax.set_title(f"{vid}  hull={r['hull_frac']:.2f}  cajas/frame={r['boxes_per_frame']}")
            ax.axis("off")
            fig.tight_layout(); fig.savefig(out / f"roi_{vid}.png", dpi=120); plt.close(fig)

    print(f"\nGuardado en {out}: roi_stats.csv, heatmap_global.png, roi_<video>.png ({len(sample)} overlays)")


if __name__ == "__main__":
    main()
