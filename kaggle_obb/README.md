# Entrenamiento YOLO-OBB en Kaggle (GPU T4)

Guía para entrenar el detector OBB del SMART CHALLENGE 2026 con la GPU gratuita de Kaggle.

## 0. Prerrequisito: dataset convertido

En local, generar el dataset YOLO-OBB (recortado+ennegrecido a la ROI):

```powershell
cd C:\Users\yelts\Downloads\kagle\MTC-smart
python obb/convert_to_yolo_obb.py --out dataset_obb
```

Esto crea `dataset_obb/` con:
```
dataset_obb/
├── images/{train,val}/*.jpg
├── labels/{train,val}/*.txt   # formato Ultralytics OBB (8 coords normalizadas)
└── data.yaml
```

## 1. Subir el dataset a Kaggle

Opción A — CLI (recomendado):
```powershell
cd C:\Users\yelts\Downloads\kagle\MTC-smart\dataset_obb
kaggle datasets init -p .
# editar dataset-metadata.json: poner "title" y "id": "yeltsinvalero/mtc-obb-dataset"
kaggle datasets create -p . --dir-mode zip
```

Opción B — interfaz web: New Dataset → arrastrar la carpeta `dataset_obb`.

## 2. Crear el notebook en Kaggle

1. Kaggle → **Create → New Notebook**.
2. **Settings → Accelerator → GPU T4 x2** (o P100).
3. **Add Input →** tu dataset `mtc-obb-dataset`.
4. En una celda, pegar el contenido de `train_obb_kaggle.py` (o `%load`).
5. Run All.

Variables de entorno opcionales (celda previa):
```python
import os
os.environ["OBB_MODEL"] = "yolo11s-obb.pt"   # o yolo11m-obb.pt (mas fuerte, mas lento)
os.environ["OBB_IMGSZ"] = "1280"
os.environ["OBB_EPOCHS"] = "60"
os.environ["OBB_BATCH"] = "8"
```

## 3. Descargar resultados

Al terminar, en `/kaggle/working/artifacts/`:
- `best.pt` — pesos del modelo (para inferencia y ensamble).
- `val_metrics.json` — mAP50-95 global y **por clase** (proxy del macro AP del concurso).
- `results.png`, `confusion_matrix.png` — curvas para el artículo.

## Decisiones de diseño (resumen; detalle en `../pipeline/PIPELINE_LOG.md`)

| Decisión | Motivo |
|---|---|
| Modelo `-obb` (no normal) | El concurso usa cajas orientadas (rotated IoU). |
| `imgsz=1280` | Tras recortar a la ROI, las motos miden ~20 px; alta resolución es clave. |
| `degrees=180`, `flipud=0.5` | Tomas aéreas sin orientación canónica; rotación completa ayuda. |
| 9 clases (sin fusionar) | La métrica es macro sobre 9 clases; fusionar daría AP=0 en las clases unidas. |
| ROI (recorte+ennegrecido) | Evita enseñar "vehículo no etiquetado = fondo" (falsos negativos). |

## Pendiente (ver `../pipeline/PENDIENTES.md`)
- **P1:** confirmar con organizadores si el test usa la misma ROI (afecta el filtrado en inferencia).
- Oversampling de clases raras (`articulado` 250 cajas): mejora futura si el mAP de raras es bajo.
