"""Rellena la bitacora con las etapas ya ejecutadas (00-04) con sus metricas reales.
Ejecutar una sola vez para inicializar pipeline/PIPELINE_LOG.md y run_log.jsonl.
"""
from _log import log_stage

log_stage(
    stage="00_obtener_datos",
    title="Obtencion del dataset del concurso",
    inputs={"kaggle_slug": "mtc-smart-challenge-ia-para-la-movilidad-del-peru"},
    outputs={"train_csv": "data/train.csv", "train_zip": "data/train.zip (43 GB)",
             "test_zip": "data/test.zip (18.6 GB)", "sample_submission": "data/sample_submission.csv"},
    metrics={"train_frames": 54263, "test_frames": 23579, "img_resolution": "1920x1080"},
    findings=[
        "Kaggle no entrega los datos: entrega un HTML que apunta a una carpeta de Google Drive.",
        "Datos reales descargados desde Drive (train.zip imagenes, test.zip imagenes, train.csv labels).",
        "No hay labels del test (ocultos en el servidor de Kaggle, como es habitual).",
    ],
    decision="Trabajar con train.csv como verdad-terreno y test.zip como conjunto a predecir.",
)

log_stage(
    stage="01_actualizar_clases",
    title="Alineacion del repo a las 9 clases oficiales",
    inputs={"fuente": "pestana Overview del concurso"},
    outputs={"class_map": "configs/class_map.json", "project": "configs/project.json",
             "typology": "configs/typology_rules.json"},
    metrics={"n_clases": 9},
    findings=[
        "Clases oficiales (id->nombre): 1 auto, 2 combi, 3 microbus, 4 minibus, 5 omnibus, 6 articulado, 7 camion, 8 mototaxi, 9 motocicleta.",
        "El repo clonado usaba otras clases (bicycle/car/bus/truck...). Se reescribieron los 3 configs.",
        "Metrica oficial: Macro AP-rIoU@[0.50:0.80] (rotated IoU, promedio no ponderado por clase).",
    ],
    decision="indice YOLO = category_id - 1. Configs como unica fuente de verdad de clases.",
)

log_stage(
    stage="02_clasificador_geometrico",
    title="Modelo de tipologia por geometria (prior)",
    inputs={"data": "geo_model/dataset.pkl (601934 cajas)"},
    outputs={"modelos": "geo_model/artifacts/model_{ratio,geo}.joblib",
             "metricas": "geo_model/artifacts/metrics.json"},
    metrics={"macro_f1_ratio": 0.345, "macro_f1_geo": 0.612, "balanced_acc_geo": 0.758,
             "validacion": "StratifiedGroupKFold por video (5 folds)"},
    findings=[
        "Solo-geometria separa bien los extremos (auto F1 0.94, motocicleta 0.96) y mal los medianos.",
        "El ratio de tamano por si solo (modelo 'ratio') da macro-F1 0.345; anadir aspecto+angulo sube a 0.612.",
    ],
    decision="Usar la geometria como PRIOR/desempate, no como clasificador final (la apariencia decide).",
)

log_stage(
    stage="03_analisis_agrupamiento",
    title="Analisis de separabilidad y agrupamiento de tipologias",
    inputs={"data": "geo_model/dataset.pkl"},
    outputs={"reporte": "analysis/REPORT.md", "figuras": "analysis/out/*.png",
             "json": "analysis/out/grouping_report.json"},
    metrics={"imbalance_auto_vs_articulado": "1927:1", "macro_f1_fino_9": 0.603,
             "macro_f1_3grupos": 0.849, "articulado_n": 250},
    findings=[
        "Desbalance extremo: auto 80% del dataset; articulado solo 250 cajas (0.04%).",
        "Parejas casi inseparables por geometria (Bhattacharyya~0.95): auto-minibus, microbus-camion, combi-minibus.",
        "Colapsar a 3 super-grupos sube macro-F1 0.60->0.85: el error grave es INTRA-grupo y de apariencia.",
    ],
    decision="NO fusionar la salida (metrica exige 9 clases). Si usar grupos como etiqueta jerarquica + 2a etapa visual + oversampling de raras.",
)

log_stage(
    stage="04_validar_angulo_obb",
    title="Validacion de la convencion del angulo OBB",
    inputs={"train": "data/train.csv", "train_zip": "data/train.zip"},
    outputs={"overlays": "obb/angle_check/zoom_*.jpg"},
    metrics={"convencion_correcta": "ccw_math (theta directo, antihorario)"},
    findings=[
        "Se dibujaron cajas reales con 2 convenciones (ccw_math vs cw_image) sobre vehiculos rotados.",
        "ccw_math encierra perfectamente los vehiculos; cw_image los deja torcidos.",
    ],
    decision="El conversor a YOLO-OBB usara theta_rad = deg2rad(angle_deg), sin invertir el signo.",
)

log_stage(
    stage="05_investigar_roi",
    title="Investigacion del etiquetado parcial / ROI por video",
    inputs={"train": "data/train.csv", "train_zip": "data/train.zip"},
    outputs={"stats": "obb/roi_out/roi_stats.csv", "heatmap": "obb/roi_out/heatmap_global.png",
             "overlays": "obb/roi_out/roi_v_*.png"},
    metrics={"hull_frac_mediana": 0.31, "core95_frac_mediana": 0.43,
             "videos_roi_lt_50pct": 0.83, "videos_roi_gt_85pct": 0.0,
             "cajas_por_frame_mediana": 8.4},
    findings=[
        "Las anotaciones siguen los CORREDORES viales de la interseccion (patron de cruz en el heatmap global), no toda la imagen.",
        "83% de los videos tienen su zona anotada cubriendo <50% de la imagen; NINGUN video cubre >85%.",
        "Vehiculos estacionados/fuera de la calzada de interes quedan SIN etiquetar -> etiquetado parcial.",
        "Cobertura muy variable por video (de ~0.1 a 51.6 cajas/frame): algunos clips casi no tienen trafico anotado.",
    ],
    decision="Pendiente: manejar el etiquetado parcial (enmascarar a la ROI por video) antes de convertir a YOLO-OBB para no inyectar falsos negativos.",
)

print("\nBitacora inicializada. Revisa pipeline/PIPELINE_LOG.md")
