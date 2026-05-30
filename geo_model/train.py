"""Entrena un clasificador de tipologia vehicular SOLO con geometria.

Idea: dada la geometria de una caja OBB (sobre todo su tamano relativo al 'auto'
del mismo frame), predecir la clase del concurso (1..9).

Rigor metodologico:
  * Split por video (StratifiedGroupKFold): los ~50 frames de un clip son casi
    identicos; un split aleatorio filtraria informacion y inflaria la metrica.
  * Metrica MACRO (F1 por clase) como en el concurso: 'auto' (481k cajas) no
    puede tapar a 'articulado' (250 cajas).
  * sample_weight balanceado para el fuerte desbalance de clases.

Entrena y compara dos modelos:
  * ratio   -> 1 sola feature: area_ratio_to_auto  (lo que pediste: "te doy el ratio")
  * geo     -> ratio + aspecto + angulo (todas invariantes a escala) [RECOMENDADO]

Uso:
    python geo_model/train.py --data geo_model/dataset.pkl --outdir geo_model/artifacts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

ID2NAME = {1: "auto", 2: "combi", 3: "microbus", 4: "minibus", 5: "omnibus",
           6: "articulado", 7: "camion", 8: "mototaxi", 9: "motocicleta"}
CLASSES = list(ID2NAME.values())

FEATURE_SETS = {
    # Lo minimo: solo el ratio de tamano respecto al auto del frame.
    "ratio": ["log_area_ratio"],
    # Recomendado: todas invariantes a la escala de la camara (portables entre videos).
    "geo": ["log_area_ratio", "aspect", "ang_sin2", "ang_cos2"],
}


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )


def cv_evaluate(df: pd.DataFrame, feats: list[str], n_splits: int = 5) -> dict:
    """StratifiedGroupKFold por video: macro-F1 y balanced-acc por fold."""
    X = df[feats].to_numpy(dtype=float)
    y = df["cid"].to_numpy()
    groups = df["video_id"].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    macro_f1, bal_acc = [], []
    # Acumular predicciones out-of-fold para una matriz de confusion global honesta.
    oof_true, oof_pred = [], []
    for tr, te in sgkf.split(X, y, groups):
        m = make_model()
        sw = compute_sample_weight("balanced", y[tr])
        m.fit(X[tr], y[tr], sample_weight=sw)
        p = m.predict(X[te])
        macro_f1.append(f1_score(y[te], p, average="macro", labels=list(ID2NAME)))
        bal_acc.append(balanced_accuracy_score(y[te], p))
        oof_true.append(y[te]); oof_pred.append(p)
    return {
        "macro_f1_mean": float(np.mean(macro_f1)),
        "macro_f1_std": float(np.std(macro_f1)),
        "balanced_acc_mean": float(np.mean(bal_acc)),
        "balanced_acc_std": float(np.std(bal_acc)),
        "folds": n_splits,
        "oof_true": np.concatenate(oof_true),
        "oof_pred": np.concatenate(oof_pred),
    }


def per_class_report(y_true, y_pred) -> dict:
    rep = classification_report(
        y_true, y_pred, labels=list(ID2NAME), target_names=CLASSES,
        output_dict=True, zero_division=0,
    )
    return rep


def save_confusion(y_true, y_pred, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(ID2NAME))
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(9)); ax.set_yticks(range(9))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Prediccion"); ax.set_ylabel("Real")
    ax.set_title(title)
    for i in range(9):
        for j in range(9):
            v = cmn[i, j]
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=7)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def ratio_decision_bins(df: pd.DataFrame, model, feats: list[str]) -> list[dict]:
    """Para el modelo 'ratio', barre el ratio y devuelve los tramos donde cambia
    la clase predicha -> tabla legible: 'si ratio en [a,b) -> clase X'."""
    grid = np.geomspace(0.03, 15.0, 600)
    Xg = np.log(grid).reshape(-1, 1)
    pred = model.predict(Xg)
    bins = []
    start = 0
    for i in range(1, len(grid) + 1):
        if i == len(grid) or pred[i] != pred[start]:
            bins.append({"ratio_min": round(float(grid[start]), 3),
                         "ratio_max": round(float(grid[i - 1]), 3),
                         "clase": ID2NAME[int(pred[start])]})
            start = i
    return bins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="geo_model/dataset.pkl")
    ap.add_argument("--outdir", default="geo_model/artifacts")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(args.data)
    print(f"Dataset: {len(df):,} cajas | {df['video_id'].nunique()} videos | {df['frame_id'].nunique():,} frames")

    summary = {"n_boxes": int(len(df)), "n_videos": int(df["video_id"].nunique()), "models": {}}

    for name, feats in FEATURE_SETS.items():
        print(f"\n===== Modelo '{name}'  features={feats} =====")
        res = cv_evaluate(df, feats)
        print(f"  macro-F1 (CV por video): {res['macro_f1_mean']:.3f} +/- {res['macro_f1_std']:.3f}")
        print(f"  balanced-acc:            {res['balanced_acc_mean']:.3f} +/- {res['balanced_acc_std']:.3f}")

        rep = per_class_report(res["oof_true"], res["oof_pred"])
        print("  F1 por clase (out-of-fold):")
        for c in CLASSES:
            print(f"    {c:<12} F1={rep[c]['f1-score']:.3f}  P={rep[c]['precision']:.3f}  R={rep[c]['recall']:.3f}  n={int(rep[c]['support'])}")

        save_confusion(res["oof_true"], res["oof_pred"], outdir / f"confusion_{name}.png",
                       f"Matriz de confusion (norm.) - modelo '{name}'  macroF1={res['macro_f1_mean']:.3f}")

        # Reentrenar en TODO el dataset y guardar el modelo final.
        X = df[feats].to_numpy(dtype=float); y = df["cid"].to_numpy()
        final = make_model()
        final.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        bundle = {"model": final, "features": feats, "id2name": ID2NAME, "classes": CLASSES}
        joblib.dump(bundle, outdir / f"model_{name}.joblib")

        entry = {
            "features": feats,
            "macro_f1_cv": [res["macro_f1_mean"], res["macro_f1_std"]],
            "balanced_acc_cv": [res["balanced_acc_mean"], res["balanced_acc_std"]],
            "per_class_f1": {c: round(rep[c]["f1-score"], 3) for c in CLASSES},
            "per_class_precision": {c: round(rep[c]["precision"], 3) for c in CLASSES},
            "per_class_recall": {c: round(rep[c]["recall"], 3) for c in CLASSES},
            "support": {c: int(rep[c]["support"]) for c in CLASSES},
        }
        if name == "ratio":
            entry["ratio_decision_table"] = ratio_decision_bins(df, final, feats)
            print("  Tabla de decision por ratio:")
            for b in entry["ratio_decision_table"]:
                print(f"    ratio [{b['ratio_min']:>6}, {b['ratio_max']:>6}) -> {b['clase']}")
        summary["models"][name] = entry

    (outdir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado: {outdir}/model_ratio.joblib, model_geo.joblib, metrics.json, confusion_*.png")


if __name__ == "__main__":
    main()
