"""Notebook de entrenamiento YOLO-OBB para Kaggle (GPU T4) - SMART CHALLENGE 2026.

Pensado para pegarse en una celda de un notebook de Kaggle con GPU activada y el
dataset YOLO-OBB adjunto como Kaggle Dataset.

Que hace:
  1. Localiza el dataset OBB adjunto (busca data.yaml bajo /kaggle/input).
  2. Entrena un modelo Ultralytics OBB (yolo11s-obb / yolo11m-obb).
  3. Valida y guarda metricas por clase (proxy del macro AP del concurso).
  4. Copia best.pt y resultados a /kaggle/working para descargarlos.

Notas de diseno (ver analysis/REPORT.md y pipeline/PIPELINE_LOG.md):
  * 9 clases OBB; metrica oficial macro -> las clases raras importan tanto como 'auto'.
  * imgsz alto (1280) porque las motos miden ~20 px tras el recorte a la ROI.
  * 'cls' gain subido y mosaic moderado para no romper objetos pequenos.
  * El desbalance se ataca con augmentation; el oversampling por clase se puede
    activar duplicando frames de clases raras (ver build_oversampled_list, opcional).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

# ------------------------- configuracion -------------------------
MODEL = os.environ.get("OBB_MODEL", "yolo11s-obb.pt")  # o yolo11m-obb.pt
IMGSZ = int(os.environ.get("OBB_IMGSZ", "1280"))
EPOCHS = int(os.environ.get("OBB_EPOCHS", "60"))
BATCH = int(os.environ.get("OBB_BATCH", "8"))
PATIENCE = int(os.environ.get("OBB_PATIENCE", "15"))
SEED = int(os.environ.get("OBB_SEED", "17"))

KAGGLE_INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
ARTIFACTS = WORKING / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

RARE = ["combi", "microbus", "minibus", "omnibus", "articulado", "mototaxi"]


def find_data_yaml() -> Path:
    cands = sorted(KAGGLE_INPUT.rglob("data.yaml"))
    if not cands:
        raise FileNotFoundError("No se encontro data.yaml bajo /kaggle/input. Adjunta el dataset OBB.")
    print("data.yaml encontrado:", cands[0])
    return cands[0]


def install():
    os.system("pip install -q 'ultralytics>=8.3.0'")


def main():
    install()
    import torch
    from ultralytics import YOLO

    data_yaml = find_data_yaml()
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"device={device} cuda={torch.cuda.is_available()} model={MODEL} imgsz={IMGSZ} epochs={EPOCHS}")

    model = YOLO(MODEL)
    train_args = dict(
        data=str(data_yaml),
        task="obb",
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        patience=PATIENCE,
        seed=SEED,
        device=device,
        optimizer="AdamW",
        lr0=0.001,
        # augmentation orientada a objetos pequenos y robustez aerea
        hsv_h=0.012, hsv_s=0.5, hsv_v=0.4,
        degrees=180.0,        # rotacion completa: las tomas aereas no tienen "arriba" canonico
        translate=0.1, scale=0.5, shear=0.0,
        fliplr=0.5, flipud=0.5,
        mosaic=0.8, close_mosaic=10,
        cls=0.7,              # peso de clasificacion algo mayor (clases dificiles)
        project=str(WORKING / "runs"),
        name="obb_train",
        exist_ok=True,
        verbose=True,
    )
    print("TRAIN_ARGS", json.dumps(train_args, indent=2, default=str))
    model.train(**train_args)

    # validacion final
    metrics = model.val(data=str(data_yaml), task="obb", imgsz=IMGSZ, split="val", device=device)
    out = {}
    box = getattr(metrics, "box", None)
    if box is not None:
        out["map50_95"] = float(getattr(box, "map", 0.0))
        out["map50"] = float(getattr(box, "map50", 0.0))
        maps = getattr(box, "maps", None)
        names = list(getattr(model, "names", {}).values()) if hasattr(model, "names") else []
        if maps is not None and names:
            per_class = {names[i]: float(v) for i, v in enumerate(list(maps)) if i < len(names)}
            out["per_class_map50_95"] = per_class
            rare = [per_class[c] for c in RARE if c in per_class]
            if rare:
                out["rare_map50_95"] = sum(rare) / len(rare)
    print("VAL_METRICS", json.dumps(out, indent=2))
    (ARTIFACTS / "val_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # copiar pesos y curvas
    run_dir = WORKING / "runs" / "obb_train"
    for name in ["weights/best.pt", "weights/last.pt", "results.csv", "args.yaml",
                 "confusion_matrix.png", "results.png"]:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, ARTIFACTS / Path(name).name)
    print("Artefactos en", ARTIFACTS, ":", [p.name for p in ARTIFACTS.iterdir()])


if __name__ == "__main__":
    main()
