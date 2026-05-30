# Clasificador de tipología vehicular por geometría

Modelo que, dada la **geometría** de una caja OBB (sobre todo su tamaño relativo al `auto`
del mismo frame), predice la **tipología** del concurso SMART CHALLENGE 2026 (clases 1–9).

> **TL;DR honesto:** el ratio de tamaño **sí** separa bien los extremos
> (`motocicleta`, `auto`) pero **no** las clases medianas (`combi`, `minibus`, `microbus`,
> `omnibus`, `camion`), porque sus tamaños se solapan casi por completo. La geometría sola
> alcanza un **macro-F1 ≈ 0.23** (validación por video). Sirve como *prior* / desempate
> geométrico, **no** como clasificador final — para eso hace falta la apariencia (el detector
> visual YOLO/RT-DETR del pipeline principal).

## Archivos

| Archivo | Qué hace |
|---|---|
| `build_dataset.py` | Parsea `data/train.csv` → tabla de features (`dataset.pkl`). |
| `train.py` | Entrena y evalúa con CV por video; guarda modelos + métricas + matrices de confusión. |
| `predict.py` | Predice la tipología dado un ratio (y opcionalmente w/h/ángulo). Importable y CLI. |
| `artifacts/` | `model_ratio.joblib`, `model_geo.joblib`, `metrics.json`, `confusion_*.png`. |

## Uso

```powershell
cd C:\Users\yelts\Downloads\kagle\MTC-smart
python geo_model/build_dataset.py --train ../data/train.csv --out geo_model/dataset.pkl
python geo_model/train.py --data geo_model/dataset.pkl --outdir geo_model/artifacts

# Predecir solo con el ratio:
python geo_model/predict.py --model ratio --ratio 3.1 --topk 3
# Predecir con geometría completa (mejor):
python geo_model/predict.py --model geo --ratio 4.0 --w 300 --h 70 --angle 10
```

Desde Python:

```python
from geo_model.predict import predict_typology
predict_typology(ratio=0.18, model="ratio", topk=3)
# -> [('motocicleta', 0.908), ('mototaxi', 0.084), ('auto', 0.004)]
```

## Qué es el "ratio"

`area_ratio_to_auto = área_de_la_caja / mediana_del_área_de_los_autos_del_mismo_frame`.

Se normaliza **por frame** (con fallback a por-vídeo y luego global) para que sea invariante a
la altura/zoom de cada cámara: un `auto` mide ~1.0 en cualquier intersección.

## Modelos

- **`ratio`** — 1 sola feature: `log(area_ratio_to_auto)`. Es lo que pediste: *"te doy el ratio
  y te digo la tipología"*. Es un árbol (HistGradientBoosting) → equivale a una **tabla de
  cortes por ratio** (ver abajo).
- **`geo`** — `log_area_ratio` + `aspect` (elongación lado_largo/lado_corto, ≥1) +
  ángulo OBB codificado como `sin(2θ), cos(2θ)`. Todas invariantes a escala. **Recomendado.**

### Tabla de decisión del modelo `ratio`

| ratio (área vs auto) | tipología predicha |
|---|---|
| 0.03 – 0.28 | **motocicleta** |
| 0.28 – 0.58 | **mototaxi** |
| 0.58 – 1.44 | **auto** |
| 1.44 – 2.18 | **minibus** |
| 2.18 – 3.41 | **microbus** |
| 3.41 – 15.0 | **camion** |

> Nota: `combi`, `omnibus` y `articulado` **nunca** ganan por ratio solo (siempre quedan
> dominadas por otra clase de tamaño parecido). Por eso el modelo `geo` y, sobre todo, el
> detector visual son necesarios para esas clases.

## Rigor de la evaluación

- **Split por vídeo** con `StratifiedGroupKFold` (5 folds). Los ~50 frames de un clip son casi
  idénticos; un split aleatorio filtraría información y **inflaría** la métrica. Todos los frames
  de un vídeo caen en el mismo fold.
- **Métrica macro** (igual que el concurso): cada clase pesa lo mismo. `auto` (481 731 cajas)
  no puede tapar a `articulado` (250 cajas).
- `sample_weight="balanced"` para el fuerte desbalance.
- Métricas y matrices de confusión *out-of-fold* en `artifacts/`.

## Resultados (CV por vídeo, out-of-fold)

| Modelo | macro-F1 | balanced-acc |
|---|---|---|
| `ratio` | 0.205 ± 0.013 | 0.296 ± 0.022 |
| `geo`   | **0.227 ± 0.017** | **0.302 ± 0.017** |

**F1 por clase (modelo `geo`):**

| clase | F1 | precisión | recall | soporte |
|---|---|---|---|---|
| auto | **0.823** | 0.949 | 0.726 | 481 731 |
| motocicleta | **0.769** | 0.846 | 0.705 | 47 568 |
| mototaxi | 0.234 | 0.215 | 0.255 | 5 539 |
| camion | 0.234 | 0.198 | 0.285 | 32 668 |
| minibus | 0.094 | 0.073 | 0.131 | 18 941 |
| omnibus | 0.072 | 0.041 | 0.293 | 2 283 |
| microbus | 0.057 | 0.035 | 0.152 | 2 802 |
| articulado | 0.045 | 0.024 | 0.516 | 250 |
| combi | 0.038 | 0.023 | 0.116 | 10 152 |

### Lectura

- **Funciona bien** en los extremos: `auto` (F1 0.82) y `motocicleta` (F1 0.77).
- **Falla** en las clases medianas: en tamaño, `combi`/`minibus`/`microbus`/`omnibus`/`camion`
  ocupan casi el mismo rango (área ~1.1–3.1× auto) y se confunden entre sí — son visualmente
  distintas pero geométricamente casi idénticas vistas desde arriba.
- `articulado` tiene recall alto (0.52) pero precisión ínfima (0.02): la geometría la "sospecha"
  por ser grande y alargada, pero hay muchos falsos positivos (camiones, ómnibus largos).

## Recomendación de uso en el pipeline

Usar este modelo como **prior geométrico** que se *mezcla* con la confianza del detector
visual (es justo lo que ya hace `configs/typology_rules.json` con `confidence_blend`), **no**
como clasificador autónomo. Para las clases medianas, la decisión debe venir del detector
y del consenso por tracking (`TrackingPostprocessAgent`).
