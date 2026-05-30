
## [2026-05-30 01:40:00] 00_obtener_datos — Obtencion del dataset del concurso

**Entradas:** `kaggle_slug`=mtc-smart-challenge-ia-para-la-movilidad-del-peru
**Salidas:** `train_csv`=data/train.csv, `train_zip`=data/train.zip (43 GB), `test_zip`=data/test.zip (18.6 GB), `sample_submission`=data/sample_submission.csv
**Métricas:**
- `train_frames`: 54263
- `test_frames`: 23579
- `img_resolution`: 1920x1080
**Hallazgos:**
- Kaggle no entrega los datos: entrega un HTML que apunta a una carpeta de Google Drive.
- Datos reales descargados desde Drive (train.zip imagenes, test.zip imagenes, train.csv labels).
- No hay labels del test (ocultos en el servidor de Kaggle, como es habitual).
**Decisión:** Trabajar con train.csv como verdad-terreno y test.zip como conjunto a predecir.

## [2026-05-30 01:40:00] 01_actualizar_clases — Alineacion del repo a las 9 clases oficiales

**Entradas:** `fuente`=pestana Overview del concurso
**Salidas:** `class_map`=configs/class_map.json, `project`=configs/project.json, `typology`=configs/typology_rules.json
**Métricas:**
- `n_clases`: 9
**Hallazgos:**
- Clases oficiales (id->nombre): 1 auto, 2 combi, 3 microbus, 4 minibus, 5 omnibus, 6 articulado, 7 camion, 8 mototaxi, 9 motocicleta.
- El repo clonado usaba otras clases (bicycle/car/bus/truck...). Se reescribieron los 3 configs.
- Metrica oficial: Macro AP-rIoU@[0.50:0.80] (rotated IoU, promedio no ponderado por clase).
**Decisión:** indice YOLO = category_id - 1. Configs como unica fuente de verdad de clases.

## [2026-05-30 01:40:00] 02_clasificador_geometrico — Modelo de tipologia por geometria (prior)

**Entradas:** `data`=geo_model/dataset.pkl (601934 cajas)
**Salidas:** `modelos`=geo_model/artifacts/model_{ratio,geo}.joblib, `metricas`=geo_model/artifacts/metrics.json
**Métricas:**
- `macro_f1_ratio`: 0.345
- `macro_f1_geo`: 0.612
- `balanced_acc_geo`: 0.758
- `validacion`: StratifiedGroupKFold por video (5 folds)
**Hallazgos:**
- Solo-geometria separa bien los extremos (auto F1 0.94, motocicleta 0.96) y mal los medianos.
- El ratio de tamano por si solo (modelo 'ratio') da macro-F1 0.345; anadir aspecto+angulo sube a 0.612.
**Decisión:** Usar la geometria como PRIOR/desempate, no como clasificador final (la apariencia decide).

## [2026-05-30 01:40:00] 03_analisis_agrupamiento — Analisis de separabilidad y agrupamiento de tipologias

**Entradas:** `data`=geo_model/dataset.pkl
**Salidas:** `reporte`=analysis/REPORT.md, `figuras`=analysis/out/*.png, `json`=analysis/out/grouping_report.json
**Métricas:**
- `imbalance_auto_vs_articulado`: 1927:1
- `macro_f1_fino_9`: 0.603
- `macro_f1_3grupos`: 0.849
- `articulado_n`: 250
**Hallazgos:**
- Desbalance extremo: auto 80% del dataset; articulado solo 250 cajas (0.04%).
- Parejas casi inseparables por geometria (Bhattacharyya~0.95): auto-minibus, microbus-camion, combi-minibus.
- Colapsar a 3 super-grupos sube macro-F1 0.60->0.85: el error grave es INTRA-grupo y de apariencia.
**Decisión:** NO fusionar la salida (metrica exige 9 clases). Si usar grupos como etiqueta jerarquica + 2a etapa visual + oversampling de raras.

## [2026-05-30 01:40:00] 04_validar_angulo_obb — Validacion de la convencion del angulo OBB

**Entradas:** `train`=data/train.csv, `train_zip`=data/train.zip
**Salidas:** `overlays`=obb/angle_check/zoom_*.jpg
**Métricas:**
- `convencion_correcta`: ccw_math (theta directo, antihorario)
**Hallazgos:**
- Se dibujaron cajas reales con 2 convenciones (ccw_math vs cw_image) sobre vehiculos rotados.
- ccw_math encierra perfectamente los vehiculos; cw_image los deja torcidos.
**Decisión:** El conversor a YOLO-OBB usara theta_rad = deg2rad(angle_deg), sin invertir el signo.

## [2026-05-30 01:40:00] 05_investigar_roi — Investigacion del etiquetado parcial / ROI por video

**Entradas:** `train`=data/train.csv, `train_zip`=data/train.zip
**Salidas:** `stats`=obb/roi_out/roi_stats.csv, `heatmap`=obb/roi_out/heatmap_global.png, `overlays`=obb/roi_out/roi_v_*.png
**Métricas:**
- `hull_frac_mediana`: 0.31
- `core95_frac_mediana`: 0.43
- `videos_roi_lt_50pct`: 0.83
- `videos_roi_gt_85pct`: 0.0
- `cajas_por_frame_mediana`: 8.4
**Hallazgos:**
- Las anotaciones siguen los CORREDORES viales de la interseccion (patron de cruz en el heatmap global), no toda la imagen.
- 83% de los videos tienen su zona anotada cubriendo <50% de la imagen; NINGUN video cubre >85%.
- Vehiculos estacionados/fuera de la calzada de interes quedan SIN etiquetar -> etiquetado parcial.
- Cobertura muy variable por video (de ~0.1 a 51.6 cajas/frame): algunos clips casi no tienen trafico anotado.
**Decisión:** Pendiente: manejar el etiquetado parcial (enmascarar a la ROI por video) antes de convertir a YOLO-OBB para no inyectar falsos negativos.

## [2026-05-30 02:06:09] 06_metrica_y_simulacion_FP — Metrica oficial local + simulacion de penalizacion por etiquetado parcial

**Entradas:** `train`=data/train.csv (200 frames)
**Salidas:** `metrica`=eval/metric.py, `simulacion`=eval/fp_simulation.py, `figuras`=eval/out/fp_score_sweep.png, fp_count_sweep.png, `correo`=pipeline/correo_organizadores.md
**Métricas:**
- `self_test`: PASS (rIoU y AP)
- `ap_sin_FP`: 1.0
- `ap_FP_score0.5`: 1.0
- `ap_FP_score0.7`: 0.816
- `ap_FP_score0.9`: 0.602
- `ap_FP_score0.99`: 0.525
- `ap_FP_10pct_score0.9`: 0.786
- `ap_FP_50pct_score0.9`: 0.545
**Hallazgos:**
- Implementada y validada la metrica oficial Macro AP-rIoU@[0.50:0.80] (rotated IoU propio).
- FP con score <=0.5 NO penalizan; FP con score alto si: detectar vehiculos reales no etiquetados con alta confianza hunde el AP.
- Un detector MEJOR puede puntuar PEOR si el test tiene etiquetado parcial (vehiculos reales = FP).
**Decisión:** Redactado correo a organizadores pidiendo aclarar si el test usa ROI y si hay mascara de evaluacion. En inferencia, filtrar/penalizar predicciones fuera de la ROI estimada.

## [2026-05-30 02:42:52] 07_definir_roi — Definicion de la ROI de anotacion por video (rejilla de ocupacion)

**Entradas:** `train`=data/train.csv, `train_zip`=data/train.zip
**Salidas:** `convex_hull`=obb/roi_polygons.json (descartado), `grid_masks`=obb/roi_grid.npz, `grid_meta`=obb/roi_grid_meta.json, `overlays`=obb/roi_verify/ y obb/roi_verify_grid/
**Métricas:**
- `metodo_elegido`: rejilla de ocupacion (cell_px=64, dilate=1)
- `roi_area_mediana_grid`: 0.28
- `roi_area_mediana_hull`: 0.39
- `videos_roi_lt40pct_grid`: 0.88
**Hallazgos:**
- El casco convexo RELLENA las esquinas (forma de cruz->diamante) e incluye manzanas con autos no etiquetados: descartado.
- La rejilla de ocupacion sigue la forma real de la zona anotada (calzadas en cruz/T) sin rellenar esquinas.
- Verificacion visual OK: la ROI cubre las calzadas con trafico anotado y excluye azoteas/manzanas.
- Residual: dentro de la ROI aun quedan algunos autos estacionados en los bordes sin etiquetar (contaminacion menor, pendiente P2).
**Decisión:** Usar la ROI por rejilla (roi_grid.npz) para la conversion a YOLO-OBB: descartar/enmascarar lo de fuera de la ROI en entrenamiento.

## [2026-05-30 02:51:35] 08_conversor_yolo_obb — Conversor train.csv -> dataset YOLO-OBB (recorte + ennegrecido a la ROI)

**Entradas:** `train`=data/train.csv, `train_zip`=data/train.zip, `roi`=obb/roi_grid.npz
**Salidas:** `script`=obb/convert_to_yolo_obb.py, `dataset`=dataset_obb/ (pendiente run completo)
**Métricas:**
- `prueba_videos`: 3
- `frames_ok`: 150
- `cajas_kept`: 566
- `cajas_dropped`: 0
- `formato`: Ultralytics OBB (DOTA, 8 coords normalizadas, clase 0-based)
- `split`: por video (val_frac=0.15, hash determinista)
**Hallazgos:**
- Verificacion visual OK: cajas OBB re-dibujadas alinean con los vehiculos (convencion ccw_math confirmada tras recorte+normalizacion).
- Ennegrecido por celdas sigue la forma de cruz: las esquinas de manzana con autos no etiquetados quedan en negro.
- cajas_dropped=0 es esperado: la ROI se construye desde las cajas, todo centro cae en celda activa; el valor de la mascara es limpiar el FONDO, no descartar cajas.
**Decisión:** Conversor validado. Pendiente: lanzar conversion COMPLETA (54263 frames) y elegir modelo YOLO-OBB para entrenar.

## [2026-05-30 03:34:40] 10b_subida_dataset_train — Subida de train.zip+train.csv como Kaggle Dataset (train primero)

**Entradas:** `local`=kaggle_dataset/ train.zip 40.3GB + train.csv + sample_submission.csv
**Salidas:** `dataset`=yeltsinvalero/mtc-smart-challenge-2026-data
**Métricas:**
- `estrategia`: train primero, test despues
- `train_zip_GB`: 40.3
**Hallazgos:**
- gdown del Drive del concurso falla por cuota en zips grandes.
- Solucion: subir como Kaggle Dataset privado y adjuntar al notebook.
- test.zip (17.3GB) en _test_hold, se sube tras train.
**Decisión:** Subiendo train en background (bjmw6xshy). Al terminar: re-push del kernel y Run para entrenar. Luego subir test para inferencia.
