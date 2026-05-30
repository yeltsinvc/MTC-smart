"""Predice la tipologia vehicular del concurso a partir de geometria.

Carga el modelo entrenado (geo_model/train.py) y expone:
  * funcion importable predict_typology(...)
  * CLI rapido

Ejemplos CLI:
    # Modelo 'ratio': solo el tamano relativo al auto del frame
    python geo_model/predict.py --ratio 3.1
    python geo_model/predict.py --ratio 0.18 --topk 3

    # Modelo 'geo': ratio + dimensiones de la caja (calcula aspecto y angulo)
    python geo_model/predict.py --model geo --ratio 3.1 --w 220 --h 70 --angle 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

ART = Path(__file__).resolve().parent / "artifacts"


def load_bundle(model: str = "geo"):
    path = ART / f"model_{model}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Entrena primero con geo_model/train.py")
    return joblib.load(path)


def _features_from_inputs(feats: list[str], ratio: float, w: float | None,
                          h: float | None, angle: float | None) -> np.ndarray:
    vals = {}
    vals["log_area_ratio"] = np.log(max(ratio, 1e-9))
    if "aspect" in feats:
        if w is None or h is None:
            raise ValueError("El modelo 'geo' necesita --w y --h para el aspecto.")
        long_s, short_s = max(w, h), min(w, h)
        vals["aspect"] = long_s / max(short_s, 1e-9)
    if "ang_sin2" in feats:
        a = np.deg2rad((angle or 0.0) * 2.0)
        vals["ang_sin2"] = np.sin(a); vals["ang_cos2"] = np.cos(a)
    return np.array([[vals[f] for f in feats]], dtype=float)


def predict_typology(ratio: float, model: str = "geo", w: float | None = None,
                     h: float | None = None, angle: float | None = None,
                     topk: int = 1, bundle=None):
    """Devuelve [(clase, probabilidad), ...] ordenado por probabilidad."""
    bundle = bundle or load_bundle(model)
    X = _features_from_inputs(bundle["features"], ratio, w, h, angle)
    clf = bundle["model"]
    proba = clf.predict_proba(X)[0]
    order = np.argsort(proba)[::-1]
    id2name = bundle["id2name"]
    out = [(id2name[int(clf.classes_[i])], float(proba[i])) for i in order[:topk]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ratio", "geo"], default="geo")
    ap.add_argument("--ratio", type=float, required=True, help="area_ratio_to_auto (tamano relativo al auto del frame)")
    ap.add_argument("--w", type=float, default=None)
    ap.add_argument("--h", type=float, default=None)
    ap.add_argument("--angle", type=float, default=None)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    res = predict_typology(args.ratio, model=args.model, w=args.w, h=args.h,
                           angle=args.angle, topk=args.topk)
    print(f"Modelo '{args.model}'  ratio={args.ratio}"
          + (f" w={args.w} h={args.h} angle={args.angle}" if args.model == "geo" else ""))
    for i, (name, p) in enumerate(res, 1):
        print(f"  {i}. {name:<12} {p*100:5.1f}%")


if __name__ == "__main__":
    main()
