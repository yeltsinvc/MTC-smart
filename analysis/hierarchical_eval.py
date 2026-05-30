"""Etapa 09 - Valida la arquitectura JERARQUICA propuesta:
   detectar super-grupos y desambiguar la clase fina con RELACIONES (post-proceso).

Pregunta que responde, con datos reales (train.csv) y SIN entrenar YOLO:
   "Dado el super-grupo correcto, cuanto recupera la clase fina un clasificador
    basado en RELACIONES (geometria + contexto del frame)?"

Compara, todo con macro-F1 (como el concurso) y CV por video (StratifiedGroupKFold):
  (1) Fino directo        : clasificar las 9 clases de una vez (relaciones).
  (2) Jerarquico ORACLE   : asumir super-grupo correcto -> un clasificador fino por
                            grupo (relaciones) -> reensamblar a 9 clases. Es la COTA
                            SUPERIOR de "que aporta la 2a etapa de relaciones".
  (3) Solo super-grupos   : macro-F1 a nivel grupo (referencia de lo facil que es).

Si (2) >> (1): la jerarquia ayuda y las relaciones bastan para la 2a etapa.
Si (2) ~ (1): la 2a etapa necesita APARIENCIA (crops), no solo relaciones.

Features de RELACIONES anadidas (ademas de geometria de etapa 02):
  * n_veh_frame         : nro de vehiculos en el frame
  * area_rank_pct       : percentil del area de la caja dentro del frame
  * area_rel_medfr      : area / mediana de areas del frame
  * nn_dist_norm        : distancia al vecino mas cercano (normalizada por sqrt(area))
  * nn_area_ratio       : area / area del vecino mas cercano
  * pos_x, pos_y        : centro normalizado en el frame (1920x1080)

Uso:
    python analysis/hierarchical_eval.py --data geo_model/dataset.pkl --outdir analysis/out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

W, H = 1920, 1080
ID2NAME = {1: "auto", 2: "combi", 3: "microbus", 4: "minibus", 5: "omnibus",
           6: "articulado", 7: "camion", 8: "mototaxi", 9: "motocicleta"}
CLASSES = list(ID2NAME.values())

# Super-grupos semanticos (4) - de analysis/REPORT.md
GROUPS = {
    "dos_ruedas": ["motocicleta", "mototaxi"],
    "livianos": ["auto", "combi", "minibus"],
    "medianos": ["microbus", "camion"],
    "grandes": ["omnibus", "articulado"],
}
NAME2GROUP = {c: g for g, cs in GROUPS.items() for c in cs}

BASE_FEATS = ["log_area_ratio", "aspect", "ang_sin2", "ang_cos2"]
REL_FEATS = ["n_veh_frame", "area_rank_pct", "area_rel_medfr", "nn_dist_norm",
             "nn_area_ratio", "pos_x", "pos_y"]
ALL_FEATS = BASE_FEATS + REL_FEATS


def add_relation_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    df["pos_x"] = df["cx"] / W
    df["pos_y"] = df["cy"] / H
    # por frame
    g = df.groupby("frame_id")
    df["n_veh_frame"] = g["area"].transform("size")
    df["area_rank_pct"] = g["area"].rank(pct=True)
    df["area_rel_medfr"] = df["area"] / g["area"].transform("median")
    # vecino mas cercano (por frame), vectorizado por grupo con indices posicionales
    n = len(df)
    nn_dist = np.full(n, 50.0)
    nn_aratio = np.ones(n)
    cx = df["cx"].to_numpy(); cy = df["cy"].to_numpy(); area = df["area"].to_numpy()
    # grupos como listas de posiciones (rapido: groupby sobre indices ya 0..n-1)
    for _, pos in df.groupby("frame_id").indices.items():
        if len(pos) < 2:
            continue
        P = np.stack([cx[pos], cy[pos]], axis=1)
        D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(D, np.inf)
        nn = D.argmin(1)
        d = D[np.arange(len(pos)), nn]
        nn_dist[pos] = np.clip(d / np.maximum(1.0, np.sqrt(area[pos])), 0, 50)
        nn_aratio[pos] = np.clip(area[pos] / np.maximum(1.0, area[pos][nn]), 0, 20)
    df["nn_dist_norm"] = nn_dist
    df["nn_area_ratio"] = nn_aratio
    return df


def make_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=300, max_leaf_nodes=31,
        min_samples_leaf=150, l2_regularization=1.0, random_state=42)


def cv_fine_direct(df, feats, n_splits=5):
    X = df[feats].to_numpy(float); y = df["cid"].to_numpy(); grp = df["video_id"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    yt, yp = [], []
    for tr, te in sgkf.split(X, y, grp):
        m = make_model(); m.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        yt.append(y[te]); yp.append(m.predict(X[te]))
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    return yt, yp


def cv_hierarchical_oracle(df, feats, n_splits=5):
    """Asume grupo verdadero. Entrena un clasificador fino por grupo (con sus miembros)
    y predice dentro del grupo verdadero de cada caja de test."""
    X = df[feats].to_numpy(float); y = df["cid"].to_numpy(); grp = df["video_id"].to_numpy()
    gid = df["clase"].map(NAME2GROUP).to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    yt, yp = [], []
    for tr, te in sgkf.split(X, y, grp):
        pred = np.empty(len(te), dtype=int)
        for gname, members in GROUPS.items():
            if len(members) == 1:
                # grupo de 1 clase: prediccion trivial
                only_cid = [k for k, v in ID2NAME.items() if v == members[0]][0]
                sel_te = np.where(gid[te] == gname)[0]
                pred[sel_te] = only_cid
                continue
            tr_mask = (gid[tr] == gname)
            te_mask = (gid[te] == gname)
            if tr_mask.sum() == 0 or te_mask.sum() == 0:
                continue
            m = make_model()
            m.fit(X[tr][tr_mask], y[tr][tr_mask],
                  sample_weight=compute_sample_weight("balanced", y[tr][tr_mask]))
            pred[np.where(te_mask)[0]] = m.predict(X[te][te_mask])
        yt.append(y[te]); yp.append(pred)
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    return yt, yp


def cv_group_level(df, feats, n_splits=5):
    """macro-F1 a nivel super-grupo (lo facil)."""
    X = df[feats].to_numpy(float)
    gid = df["clase"].map(NAME2GROUP)
    gcode = gid.astype("category").cat.codes.to_numpy()
    y = df["cid"].to_numpy(); grp = df["video_id"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    yt, yp = [], []
    for tr, te in sgkf.split(X, y, grp):
        m = make_model(); m.fit(X[tr], gcode[tr], sample_weight=compute_sample_weight("balanced", gcode[tr]))
        yt.append(gcode[te]); yp.append(m.predict(X[te]))
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    return yt, yp


def report_fine(yt, yp, tag):
    f1 = f1_score(yt, yp, average="macro", labels=list(ID2NAME))
    per = {ID2NAME[c]: round(f1_score(yt == c, yp == c, average="binary", zero_division=0), 3)
           for c in ID2NAME}
    print(f"\n[{tag}] macro-F1 (9 clases) = {f1:.3f}")
    for c in CLASSES:
        print(f"    {c:<12} F1={per[c]:.3f}")
    return f1, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="geo_model/dataset.pkl")
    ap.add_argument("--outdir", default="analysis/out")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    print("Cargando dataset y construyendo features de relaciones ...")
    df = pd.read_pickle(args.data)
    df = add_relation_features(df)
    print(f"  cajas={len(df):,} videos={df['video_id'].nunique()} feats={len(ALL_FEATS)}")

    results = {"groups": GROUPS, "features": ALL_FEATS}

    # referencia geometria sola (etapa 02) vs +relaciones, fino directo
    yt, yp = cv_fine_direct(df, BASE_FEATS)
    f1_geo, _ = report_fine(yt, yp, "Fino directo - solo geometria")
    yt2, yp2 = cv_fine_direct(df, ALL_FEATS)
    f1_rel, per_rel = report_fine(yt2, yp2, "Fino directo - geometria + relaciones")

    # jerarquico oracle (grupo verdadero) con todas las features
    yt3, yp3 = cv_hierarchical_oracle(df, ALL_FEATS)
    f1_hier, per_hier = report_fine(yt3, yp3, "Jerarquico ORACLE (grupo correcto) + relaciones")

    # nivel grupo
    gyt, gyp = cv_group_level(df, ALL_FEATS)
    gnames = sorted(GROUPS)
    f1_grp = f1_score(gyt, gyp, average="macro")
    print(f"\n[Solo super-grupos] macro-F1 (4 grupos) = {f1_grp:.3f}")

    results["macro_f1"] = {
        "fino_directo_geo": round(f1_geo, 3),
        "fino_directo_geo_rel": round(f1_rel, 3),
        "jerarquico_oracle_rel": round(f1_hier, 3),
        "solo_super_grupos": round(f1_grp, 3),
    }
    results["per_class_fino_rel"] = per_rel
    results["per_class_jerarquico_oracle"] = per_hier
    (out / "hierarchical_eval.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================= RESUMEN =================")
    print(f"  Fino directo (solo geometria)            : {f1_geo:.3f}")
    print(f"  Fino directo (geometria + relaciones)    : {f1_rel:.3f}  (+{f1_rel-f1_geo:.3f})")
    print(f"  Jerarquico ORACLE (grupo OK + relaciones): {f1_hier:.3f}  (cota superior 2a etapa)")
    print(f"  Solo super-grupos (4)                    : {f1_grp:.3f}  (lo facil)")
    print("\nLectura:")
    gain = f1_hier - f1_rel
    if gain > 0.05:
        print(f"  El grupo correcto SUBE el fino +{gain:.3f}: la jerarquia ayuda, pero el techo")
        print("  de la 2a etapa SOLO con relaciones es ese valor. Lo que falte hasta ~1.0")
        print("  debe aportarlo la APARIENCIA (clasificador de crops) en la 2a etapa.")
    else:
        print(f"  El grupo correcto casi no cambia el fino (+{gain:.3f}): las relaciones NO")
        print("  bastan para la 2a etapa; hace falta APARIENCIA (crops) si o si.")
    print(f"\nGuardado: {out}/hierarchical_eval.json")


if __name__ == "__main__":
    main()
