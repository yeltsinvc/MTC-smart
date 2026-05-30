"""Analisis de separabilidad por tipologia -> conviene agrupar clases para entrenar?

Responde, con datos de train.csv, a:
  1. Que tan desbalanceadas estan las clases (soporte e imbalance ratio).
  2. Que tan separables son geometricamente (solapamiento de distribuciones).
  3. Que clases se confunden entre si (matriz de confusion en validacion por video).
  4. Que agrupamientos (super-clases) emergen de forma natural (clustering jerarquico).
  5. Co-ocurrencia: que clases aparecen juntas en el mismo frame.

Genera plots + un JSON con grupos propuestos en analysis/out/.

Uso:
    python analysis/typology_grouping.py --data geo_model/dataset.pkl --outdir analysis/out
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

ID2NAME = {1: "auto", 2: "combi", 3: "microbus", 4: "minibus", 5: "omnibus",
           6: "articulado", 7: "camion", 8: "mototaxi", 9: "motocicleta"}
CLASSES = list(ID2NAME.values())
GEO_FEATS = ["log_area_ratio", "aspect", "ang_sin2", "ang_cos2"]


# ----------------------------- 1. Soporte / desbalance -----------------------------
def support_table(df: pd.DataFrame) -> pd.DataFrame:
    n = df["clase"].value_counts().reindex(CLASSES)
    tab = pd.DataFrame({"n": n})
    tab["pct"] = (tab["n"] / tab["n"].sum() * 100).round(2)
    tab["imbalance_vs_auto"] = (tab["n"].max() / tab["n"]).round(1)
    return tab


# ------------------------ 2. Solapamiento geometrico (Bhattacharyya) ---------------
def class_gaussians(df: pd.DataFrame):
    """Media y covarianza por clase en el espacio [log_area_ratio, log_aspect]."""
    X = df[["log_area_ratio"]].copy()
    X["log_aspect"] = np.log(df["aspect"].to_numpy())
    stats = {}
    for cid, name in ID2NAME.items():
        sub = X[df["cid"].to_numpy() == cid].to_numpy()
        if len(sub) < 5:
            continue
        mu = sub.mean(axis=0)
        cov = np.cov(sub, rowvar=False) + np.eye(2) * 1e-6
        stats[name] = (mu, cov)
    return stats


def bhattacharyya(mu1, cov1, mu2, cov2):
    cov = (cov1 + cov2) / 2.0
    dmu = np.asarray(mu1 - mu2).ravel()
    inv = np.linalg.inv(cov)
    term1 = 0.125 * float(dmu @ inv @ dmu)
    d1, d2, d = np.linalg.det(cov1), np.linalg.det(cov2), np.linalg.det(cov)
    term2 = 0.5 * np.log(max(d, 1e-12) / np.sqrt(max(d1 * d2, 1e-24)))
    db = term1 + term2
    bc = float(np.exp(-db))   # coef. de Bhattacharyya in [0,1]: 1 = identicas, 0 = disjuntas
    return db, bc


def overlap_matrices(stats):
    names = [n for n in CLASSES if n in stats]
    k = len(names)
    BC = np.eye(k)
    DB = np.zeros((k, k))
    for i, j in combinations(range(k), 2):
        db, bc = bhattacharyya(*stats[names[i]], *stats[names[j]])
        BC[i, j] = BC[j, i] = bc
        DB[i, j] = DB[j, i] = db
    return names, BC, DB


# ------------------------ 3. Confusion en validacion por video ---------------------
def heldout_confusion(df: pd.DataFrame):
    X = df[GEO_FEATS].to_numpy(float)
    y = df["cid"].to_numpy()
    g = df["video_id"].to_numpy()
    tr, te = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, g))
    m = HistGradientBoostingClassifier(learning_rate=0.08, max_iter=300, min_samples_leaf=200,
                                       l2_regularization=1.0, random_state=42)
    m.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
    p = m.predict(X[te])
    cm = confusion_matrix(y[te], p, labels=list(ID2NAME))
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    return cmn, y[te], p


def group_collapse_eval(y_true, y_pred, groups: list[list[str]]) -> dict:
    """Compara macro-F1 a nivel fino (9 clases) vs colapsando a super-grupos.

    Si el macro-F1 sube MUCHO al agrupar, el error es sobre todo 'dentro del grupo'
    (clases parecidas) -> agrupar/ jerarquizar ayuda. Si casi no sube, el problema
    no es el agrupamiento sino la deteccion misma.
    """
    name2group = {}
    for gi, g in enumerate(groups):
        for name in g:
            name2group[name] = gi
    # mapear cid -> indice de grupo
    cid2group = {cid: name2group[ID2NAME[cid]] for cid in ID2NAME}
    yt_g = np.array([cid2group[c] for c in y_true])
    yp_g = np.array([cid2group[c] for c in y_pred])
    fine = f1_score(y_true, y_pred, average="macro", labels=list(ID2NAME))
    coarse = f1_score(yt_g, yp_g, average="macro", labels=sorted(set(cid2group.values())))
    return {"macro_f1_fino_9": round(float(fine), 3),
            "macro_f1_grupos": round(float(coarse), 3),
            "grupos": groups}


# ----------------------------- 4. Co-ocurrencia en frames --------------------------
def cooccurrence(df: pd.DataFrame):
    k = len(CLASSES); idx = {c: i for i, c in enumerate(CLASSES)}
    co = np.zeros((k, k))
    for _, grp in df.groupby("frame_id")["clase"]:
        present = set(grp)
        for a in present:
            for b in present:
                co[idx[a], idx[b]] += 1
    diag = np.diag(co).copy()
    # P(b presente | a presente)
    cond = co / diag.reshape(-1, 1).clip(min=1)
    return cond


# ----------------------------- Plots -----------------------------
def heatmap(M, labels, title, path, fmt="{:.2f}", cmap="magma"):
    fig, ax = plt.subplots(figsize=(8, 6.8))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = M[i, j]
            if v > 0.005:
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=7)
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def violin_area(df: pd.DataFrame, path: Path):
    order = sorted(CLASSES, key=lambda c: df.loc[df["clase"] == c, "area_ratio_to_auto"].median())
    data = [np.log10(df.loc[df["clase"] == c, "area_ratio_to_auto"].to_numpy().clip(1e-3)) for c in order]
    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot(data, showmedians=True, widths=0.9)
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("log10(area_ratio_to_auto)")
    ax.axhline(0, color="gray", ls="--", lw=0.8, label="tamano = auto")
    ax.set_title("Distribucion de tamano relativo al auto, por clase")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def dendro(DB, names, path: Path):
    Z = linkage(squareform(DB, checks=False), method="average")
    fig, ax = plt.subplots(figsize=(8, 5))
    dendrogram(Z, labels=names, ax=ax, color_threshold=0.7 * DB.max())
    ax.set_title("Clustering jerarquico por distancia geometrica (Bhattacharyya)")
    ax.set_ylabel("distancia"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return Z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="geo_model/dataset.pkl")
    ap.add_argument("--outdir", default="analysis/out")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(args.data)
    report = {}

    # 1. soporte
    tab = support_table(df)
    report["support"] = tab.reset_index(names="clase").to_dict(orient="records")
    print("=== 1. Soporte / desbalance ===")
    print(tab.to_string())

    # 2. solapamiento geometrico
    stats = class_gaussians(df)
    names, BC, DB = overlap_matrices(stats)
    heatmap(BC, names, "Solapamiento geometrico (Bhattacharyya, 1=identicas)", out / "overlap_geo.png")
    violin_area(df, out / "area_violin.png")
    Z = dendro(DB, names, out / "dendro_geo.png")
    # vecino mas solapado por clase
    nn = {}
    for i, a in enumerate(names):
        j = int(np.argsort(BC[i])[::-1][1])  # el mayor distinto de si mismo
        nn[a] = {"mas_solapado_con": names[j], "BC": round(float(BC[i, j]), 3)}
    report["overlap_nearest"] = nn
    print("\n=== 2. Clase mas solapada geometricamente (BC alto = dificil de separar) ===")
    for a, d in nn.items():
        print(f"  {a:<12} -> {d['mas_solapado_con']:<12} BC={d['BC']}")

    # grupos por corte del dendrograma a varias alturas
    groupings = {}
    for k in (3, 4, 5, 6):
        labels = fcluster(Z, t=k, criterion="maxclust")
        grp = {}
        for name, lab in zip(names, labels):
            grp.setdefault(int(lab), []).append(name)
        groupings[f"k={k}"] = list(grp.values())
    report["geo_groupings"] = groupings
    print("\n=== 4. Agrupamientos geometricos naturales (dendrograma) ===")
    for k, grps in groupings.items():
        print(f"  {k}: " + " | ".join("{" + ", ".join(g) + "}" for g in grps))

    # 3. confusion
    cmn, yt, yp = heldout_confusion(df)
    heatmap(cmn, CLASSES, "Confusion en validacion por video (modelo geo)", out / "confusion_holdout.png")
    # confusiones fuera de la diagonal mas fuertes
    conf_pairs = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and cmn[i, j] >= 0.10:
                conf_pairs.append({"real": CLASSES[i], "predicho": CLASSES[j], "tasa": round(float(cmn[i, j]), 3)})
    conf_pairs.sort(key=lambda d: -d["tasa"])
    report["top_confusions"] = conf_pairs
    print("\n=== 3. Confusiones mas fuertes (real -> predicho, tasa>=0.10) ===")
    for d in conf_pairs[:20]:
        print(f"  {d['real']:<12} -> {d['predicho']:<12} {d['tasa']:.2f}")

    # 3b. cuanto ayuda agrupar: macro-F1 fino vs colapsado a cada agrupamiento
    print("\n=== 3b. Ganancia de macro-F1 al colapsar en super-grupos (modelo solo-geometria) ===")
    collapse = {}
    for k, grps in groupings.items():
        ev = group_collapse_eval(yt, yp, grps)
        collapse[k] = ev
        print(f"  {k}: fino(9)={ev['macro_f1_fino_9']}  ->  grupos({len(grps)})={ev['macro_f1_grupos']}")
    report["group_collapse_eval"] = collapse

    # 5. co-ocurrencia
    cond = cooccurrence(df)
    heatmap(cond, CLASSES, "Co-ocurrencia P(col presente | fila presente)", out / "cooccurrence.png")

    (out / "grouping_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {out}: overlap_geo.png, area_violin.png, dendro_geo.png, confusion_holdout.png, cooccurrence.png, grouping_report.json")


if __name__ == "__main__":
    main()
