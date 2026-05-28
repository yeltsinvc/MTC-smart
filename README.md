# Drone Vehicle Kaggle Agents

Pipeline desde cero para una competencia Kaggle de detección y clasificación de vehículos desde drone, optimizado para `mAP50-95`.

Clases objetivo:

```text
bicycle, motorcycle, mototaxi, car, microbus, bus, articulated_bus, truck
```

## Idea

El proyecto separa el trabajo en agentes ejecutables:

- `DataAgent`: audita dataset local, clases, labels, cajas y desbalance; produce la lista de clases raras.
- `ExperimentAgent`: mantiene la cola reproducible en `state/queue.json`.
- `KaggleWorkerAgent`: genera notebooks Kaggle por experimento y los lanza; inyecta `rare_classes` para que el worker calcule `rare_map50_95`.
- `ReviewAgent`: baja outputs, reconstruye leaderboard, agrupa runs por configuración (media ± sigma entre seeds), clasifica por `selection_score` y evalúa el criterio de parada.
- `EvolutionAgent`: propone nuevos fine-tunings desde los líderes en cuasi-empate, y replica el líder con más seeds (`confirm-leader`) para estimar sigma.
- `TypologyAgent`: usa área relativa al auto y aspecto de caja para corregir o inferir tipologías raras.
- `TrackingPostprocessAgent`: estabiliza la clase final por vehículo usando consenso de trayectoria.
- `OpenAIFallbackAgent`: último recurso para desempatar tracks ambiguos entre dos clases candidatas.
- `EnsembleAgent`: fusiona los mejores modelos con WBF + TTA para la submission final.
- `DirectorAgent`: decide y ejecuta el siguiente paso de la metodología (correr seeds, confirmar el líder, evolucionar o ensamblar). Es el único agente que decide; los demás solo ejecutan su tarea.

## Setup

```powershell
cd "C:\Users\yeltsin.valero\Downloads\Nouveau dossier (40)\drone-vehicle-kaggle-agents"
python -m pip install -e .
dvka init
dvka status
```

Edita `configs/project.json` antes de lanzar Kaggle:

```json
{
  "owner": "tu_usuario_kaggle",
  "dataset_slug": "owner/dataset-name",
  "competition_slug": "",
  "accelerator": "NvidiaTeslaT4"
}
```

## Flujo Normal

Auditar un dataset local, si ya lo tienes descargado:

```powershell
dvka audit-data "C:\path\to\dataset"
dvka calibrate-geometry "C:\path\to\dataset"
```

Generar un notebook Kaggle sin lanzarlo:

```powershell
dvka build --limit 1
```

Lanzar experimentos:

```powershell
dvka launch --limit 2
```

Actualizar resultados y evolucionar:

```powershell
dvka cycle --launch-limit 2
```

Ver el ranking:

```powershell
dvka leaderboard
Get-Content state\leaderboard.csv
```

### Modo Autónomo (DirectorAgent)

En vez de decidir a mano entre confirmar, evolucionar o parar, deja que el Director elija el siguiente paso correcto según el estado:

```powershell
dvka auto                 # ejecuta un paso autónomo (refresh, pull, review, decidir, actuar, lanzar)
dvka auto --dry-run       # solo muestra la decisión, sin tocar Kaggle ni la cola
```

El Director aplica esta política:

```text
sin runs completados      -> correr los seeds
objetivo alcanzado        -> parar y armar el ensamble WBF+TTA
líder con < N seeds       -> confirm-leader (replicar seeds para estimar sigma)
en otro caso              -> evolve (proponer nuevos candidatos)
```

Como los kernels de Kaggle son asíncronos, llama `dvka auto` de forma periódica (cron, tarea programada): cada llamada avanza la búsqueda un paso seguro. La decisión queda en `state/director.json`.

Confirmar el líder con más seeds a mano (necesario para estimar sigma y poder parar):

```powershell
dvka confirm-leader
```

Cuando el ciclo reporta `objective.reached = true`, armar el ensamble final:

```powershell
dvka eval-typology --predictions outputs\run\artifacts\predictions.json --gt-yolo "C:\path\to\dataset\labels\val"
dvka ensemble --top-k 3 --images "C:\path\to\holdout" --output outputs\ensemble_predictions.json
```

## Arquitecturas Iniciales

La búsqueda arranca con una **comparación justa de arquitecturas a resolución constante** (`imgsz=960`, mismas épocas/patience), para que el líder inicial refleje la calidad del modelo y no un sesgo de resolución:

- `yolo11s.pt` a 960
- `yolo11m.pt` a 960
- `yolov8x.pt` a 960
- `rtdetr-l.pt` a 960 (augmentation ligera y `lr0` menor, estándar para DETR)

Después, `EvolutionAgent` promueve:

- mismo modelo a mayor resolución (la escala de resolución se explora aquí, no en los seeds);
- más épocas según `promote_epochs_factor` si el líder aún mejora;
- comparadores de arquitectura desde **todos los líderes en cuasi-empate** (dentro de `near_tie_delta`), no solo el mejor único;
- augmentations para objetos pequeños y clases raras.

## Métrica y Clasificación

El objetivo de ranking es `selection_score = mAP50-95 + minority_ap_weight * rare_mAP50-95`. Premia las clases minoritarias (`mototaxi`, `microbus`, `articulated_bus`) que deciden esta competencia, en vez de optimizar solo el mAP global. El sistema guarda además:

- `mAP50`, `precision`, `recall`
- `per_class_map50_95`
- `rare_map50_95` (calculado sobre `rare_classes`, inyectadas desde el audit o `project.json`)

Los runs se **agrupan por configuración** y se reportan como media ± desviación entre seeds. El líder es la mejor configuración en promedio, no un run suelto con suerte.

Configurables en `configs/project.json` (`minority_ap_weight`, `rare_classes`) y `configs/search_space.json` (`near_tie_delta`, `promote_epochs_factor`).

## Criterio de Parada (¿cuándo se alcanzó el objetivo?)

El objetivo se considera **alcanzado** solo cuando se cumplen ambas:

1. el líder tiene al menos `min_seeds_for_sigma` seeds, de modo que sigma (la variabilidad entre seeds) es real, y
2. la mejora del mejor `selection_score` durante las últimas `patience_reviews` revisiones es menor o igual a `sigma_multiplier * sigma` (la mejora ya está por debajo del ruido).

Si todavía no hay sigma, `dvka cycle` recomienda correr `dvka confirm-leader`. La decisión y su justificación quedan en `state/review.json` (campo `objective`) y el historial en `state/review_history.json`. Configurable en el bloque `stopping` de `configs/search_space.json`.

## Dataset Soportado

El worker Kaggle acepta:

- dataset YOLO con `data.yaml`;
- dataset Roboflow con `_annotations.csv`.

El `class_map` está en `configs/class_map.json`. Ahí se normalizan alias como `moto`, `auto`, `van`, `articulated bus`, `tuk tuk`.

## Tipologías Por Área

Cuando faltan datos para clases como `mototaxi`, `microbus` o `articulated_bus`, el pipeline usa una segunda capa geométrica calibrada desde los labels.

Archivo:

```text
configs/typology_rules.json
```

El modelo geométrico se aprende automáticamente desde los bounding boxes etiquetados:

```powershell
dvka calibrate-geometry "C:\path\to\dataset"
```

Eso genera:

```text
state/geometry_calibration.json
```

La regla central calibrada es:

```text
area_ratio_to_car = bbox_normalized_area / median_car_normalized_area
```

Ejemplo de lectura:

- `car`: cerca de `1.0`
- `microbus`: más grande que auto, menor que bus
- `bus`: varias veces el área del auto
- `articulated_bus`: muy grande y alargado
- `mototaxi`: mayor que moto, menor o similar a auto

El worker Kaggle guarda estos campos en `artifacts/predictions.json`:

```text
bbox_normalized_area
area_ratio_to_car
bbox_aspect_ratio
base_class
typology_class
postprocessed_class
postprocessed_conf
```

Y también guarda los priors aprendidos en:

```text
artifacts/geometry_calibration.json
```

Para probar reglas localmente con un JSON de detecciones:

```powershell
dvka typology detections.json --output detections_typology.json
```

Esta capa se debe validar. Si el benchmark oficial evalúa solo detección multiclase, compara el modelo puro (`base_class`) contra `postprocessed_class` antes de usarlo en la submission final:

```powershell
dvka eval-typology `
  --predictions outputs\run\artifacts\predictions.json `
  --gt-yolo "C:\path\to\dataset\labels\val"
```

El comando calcula `mAP50` con las clases crudas y con las post-procesadas, cuenta cuántas etiquetas la tipología corrigió (`helped`) o rompió (`hurt`), y devuelve un veredicto `USE_TYPOLOGY` o `DROP_TYPOLOGY`. Solo activa la capa en la submission si mejora el `mAP50`.

## Compatibilización Por Tracking

En video, el mismo vehículo puede cambiar de clase frame a frame. El pipeline corrige eso con consenso por `track_id`.

Config:

```text
configs/tracking_postprocess.json
```

Si el dataset Kaggle contiene videos, el worker ejecuta `model.track(...)` y exporta:

```text
artifacts/tracking_frame_detections.csv
artifacts/tracking_track_summary.csv
artifacts/tracking_class_votes.csv
artifacts/tracking_summary.json
```

La clase final por vehículo queda en:

```text
track_final_class
track_final_score
track_class_stable
```

El voto combina:

```text
confianza del detector + score geométrico calibrado + frecuencia temporal
```

Para aplicar consenso localmente a un CSV ya generado:

```powershell
dvka track-consensus outputs\run\artifacts\tracking_frame_detections.csv --output-dir outputs\run\track_fixed
```

## Fallback OpenAI Para Casos Dudosos

Como último recurso, el pipeline puede enviar solo el crop del vehículo a OpenAI para decidir entre las dos clases candidatas más probables.

Config:

```text
configs/openai_fallback.json
```

Está desactivado por defecto:

```json
"enabled": false
```

Para usarlo:

```powershell
$env:OPENAI_API_KEY="..."
```

Luego cambia `enabled` a `true`.

El fallback solo se aplica si el track es inestable, tiene bajo score o margen bajo entre las dos clases. No inventa clases nuevas: responde `candidate_a`, `candidate_b` o `uncertain`.

Outputs:

```text
artifacts/openai_fallback_candidates.csv
artifacts/openai_fallback_decisions.csv
```

Comando local:

```powershell
dvka openai-fallback `
  --frames outputs\run\artifacts\tracking_frame_detections.csv `
  --tracks outputs\run\artifacts\tracking_track_summary.csv `
  --votes outputs\run\artifacts\tracking_class_votes.csv `
  --output-dir outputs\run\openai_fallback
```

Importante: revisa las reglas de la competencia antes de usar APIs externas en Kaggle. Si no está permitido, deja este fallback apagado y usa solo los CSV de candidatos para análisis fuera del submit.

## Ensamble WBF + TTA

El mayor salto de `mAP` para ganar suele venir de fusionar varios modelos, no de un único modelo. `dvka ensemble` toma los `best.pt` de las mejores configuraciones del leaderboard, corre inferencia (con TTA por defecto) y fusiona las cajas con Weighted Boxes Fusion:

```powershell
dvka ensemble --top-k 3 --images "C:\path\to\images" --output outputs\ensemble_predictions.json
```

O con pesos explícitos y sin TTA:

```powershell
dvka ensemble `
  --models outputs\run_a\artifacts\best.pt outputs\run_b\artifacts\best.pt `
  --images "C:\path\to\images" `
  --no-tta
```

WBF no descarta las cajas solapadas como NMS: promedia las coordenadas ponderando por confianza y reescala la confianza final según cuántos modelos coincidieron. Una caja que todos los modelos predicen conserva su score; una que solo vio un modelo se penaliza hacia `score / N`.

## Carpetas

```text
configs/             configs editables
src/dvka/            agentes y CLI
state/               cola, runs, leaderboard, reviews, historial y objetivo
kernel_workspaces/   notebooks Kaggle generados
outputs/             outputs descargados de Kaggle
docs/                notas operativas
```

## API Para Jurado

Para la versión final donde el jurado envía un video a tu servidor:

[docs/SERVER_API.md](docs/SERVER_API.md)

Comando:

```powershell
dvka-server
```

Endpoint:

```text
POST /v1/videos/analyze
```

Si el servidor no tiene GPU, usa:

```text
POST /v1/videos/analyze-kaggle
```

Ese modo recibe el video, lanza un notebook Kaggle con GPU y luego permite bajar los artefactos desde el servidor.
