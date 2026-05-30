"""Construye la tabla de features geometricas a partir de train.csv.

Cada deteccion (caja OBB) del concurso es:  category_id cx cy width height angle_deg
Salida: geo_model/dataset.pkl con una fila por caja y columnas de features.

Uso:
    python geo_model/build_dataset.py --train ../data/train.csv --out geo_model/dataset.pkl
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

ID2NAME = {1: "auto", 2: "combi", 3: "microbus", 4: "minibus", 5: "omnibus",
           6: "articulado", 7: "camion", 8: "mototaxi", 9: "motocicleta"}


def parse_train(train_csv: Path) -> pd.DataFrame:
    rows = []
    with open(train_csv, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)  # cabecera Id,Target
        for line in r:
            if len(line) < 2:
                continue
            frame_id, target = line[0].strip(), line[1].strip()
            if not target or target.lower() == "none":
                continue
            # video_id = todo menos el sufijo _<frame> (formato v_<video>_<frame>)
            video_id = frame_id.rsplit("_", 1)[0]
            for det in target.split(";"):
                p = det.split()
                if len(p) < 6:
                    continue
                try:
                    cid = int(float(p[0])); cx = float(p[1]); cy = float(p[2])
                    w = float(p[3]); h = float(p[4]); ang = float(p[5])
                except ValueError:
                    continue
                if w <= 0 or h <= 0:
                    continue
                rows.append((video_id, frame_id, cid, cx, cy, w, h, ang))
    df = pd.DataFrame(rows, columns=["video_id", "frame_id", "cid", "cx", "cy", "w", "h", "angle_deg"])
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["area"] = df["w"] * df["h"]
    df["long_side"] = df[["w", "h"]].max(axis=1)
    df["short_side"] = df[["w", "h"]].min(axis=1)
    df["aspect"] = df["long_side"] / df["short_side"]          # elongacion >=1 (robusta a orientacion)
    df["log_area"] = np.log(df["area"])
    df["log_long"] = np.log(df["long_side"])
    df["log_short"] = np.log(df["short_side"])
    # angulo OBB: periodo 180 -> usar 2*theta para continuidad
    th = np.deg2rad(df["angle_deg"].to_numpy() * 2.0)
    df["ang_sin2"] = np.sin(th)
    df["ang_cos2"] = np.cos(th)

    # area_ratio_to_auto: relativo a la mediana de area de 'auto' EN EL MISMO FRAME.
    # Fallbacks: mediana de auto por video -> mediana global de auto.
    auto = df[df["cid"] == 1]
    med_global = float(auto["area"].median()) if len(auto) else float(df["area"].median())
    med_frame = auto.groupby("frame_id")["area"].median()
    med_video = auto.groupby("video_id")["area"].median()
    ref_frame = df["frame_id"].map(med_frame)
    ref_video = df["video_id"].map(med_video)
    ref = ref_frame.fillna(ref_video).fillna(med_global)
    df["ref_auto_area"] = ref
    df["area_ratio_to_auto"] = df["area"] / ref
    df["log_area_ratio"] = np.log(df["area_ratio_to_auto"])
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/train.csv")
    ap.add_argument("--out", default="geo_model/dataset.pkl")
    args = ap.parse_args()

    train_csv = Path(args.train)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Parseando {train_csv} ...")
    df = parse_train(train_csv)
    print(f"  cajas: {len(df):,} | videos: {df['video_id'].nunique():,} | frames: {df['frame_id'].nunique():,}")
    df = add_features(df)
    df["clase"] = df["cid"].map(ID2NAME)

    print("Distribucion por clase:")
    vc = df["clase"].value_counts().reindex(list(ID2NAME.values()))
    for name, n in vc.items():
        print(f"  {name:<12} {int(n):>8,}")

    df.to_pickle(out)
    print(f"Guardado -> {out}  ({len(df.columns)} columnas)")


if __name__ == "__main__":
    main()
