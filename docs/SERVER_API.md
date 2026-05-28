# API Para El Jurado

Esta parte es para la versión que se sube/demuestra al jurado: ellos envían un video a tu servidor y reciben resultados procesados.

## Flujo

```text
POST /v1/videos/analyze
  -> guarda video
  -> carga models/best.pt
  -> YOLO/RT-DETR tracking
  -> calibración geométrica
  -> consenso por track
  -> fallback OpenAI opcional
  -> devuelve job_id + resumen + artefactos
```

## Preparar Modelo

Coloca el mejor peso final en:

```text
models/best.pt
```

O cambia:

```text
configs/server.json
```

```json
{
  "model_path": "models/best.pt"
}
```

## Ejecutar Local

```powershell
cd "C:\Users\yeltsin.valero\Downloads\Nouveau dossier (40)\drone-vehicle-kaggle-agents"
python -m pip install -r requirements-server.txt
python -m pip install -e .
dvka-server
```

Servidor:

```text
http://localhost:8000
```

Healthcheck:

```powershell
curl http://localhost:8000/health
```

Enviar video:

```powershell
curl -X POST "http://localhost:8000/v1/videos/analyze" `
  -F "video=@C:\path\to\video.mp4"
```

Respuesta típica:

```json
{
  "job_id": "...",
  "frame_detections": 1234,
  "tracks": 87,
  "artifacts": {
    "frame_detections_csv": "...",
    "track_summary_csv": "...",
    "track_class_votes_csv": "...",
    "summary_json": "..."
  },
  "download_base": "/v1/jobs/{job_id}/artifacts"
}
```

Descargar artefacto:

```powershell
curl -O "http://localhost:8000/v1/jobs/{job_id}/artifacts/tracking_track_summary.csv"
```

## Docker

```powershell
docker build -f Dockerfile.server -t drone-vehicle-api .
docker run --rm -p 8000:8000 `
  -v ${PWD}\models:/app/models `
  -v ${PWD}\server_runs:/app/server_runs `
  drone-vehicle-api
```

Con GPU, usa una imagen CUDA/PyTorch si el servidor lo necesita. Este Dockerfile base está pensado para CPU o para adaptarlo a tu entorno.

## OpenAI Fallback En Servidor

Por defecto está apagado:

```json
"enable_openai_fallback": false
```

Para activarlo:

1. En `configs/server.json`, pon:

```json
"enable_openai_fallback": true
```

2. Define:

```powershell
$env:OPENAI_API_KEY="..."
```

El fallback solo se aplica a tracks ambiguos. No manda el video completo, solo crops de vehículos dudosos.

## Modo Servidor Ligero + Kaggle GPU

Si tu servidor no tiene GPU, usa este endpoint:

```text
POST /v1/videos/analyze-kaggle
```

Notebook cliente listo para demo:

```text
notebooks/api_remote_kaggle_demo.ipynb
```

El notebook Kaggle que corre dentro de Kaggle se genera por cada video en:

```text
server_runs/{job_id}/kaggle_kernel/infer.ipynb
```

Flujo:

```text
servidor recibe video
-> crea dataset privado temporal en Kaggle con el video
-> incluye models/best.pt o usa un dataset Kaggle con el modelo
-> genera notebook de inferencia
-> kaggle kernels push
-> devuelve job_id/kernel_id
-> cliente consulta estado
-> servidor baja outputs cuando termina
```

Config:

```text
configs/kaggle_inference.json
```

Modo simple, subiendo el modelo junto con cada video:

```json
{
  "include_local_model_in_job_dataset": true,
  "local_model_path": "models/best.pt",
  "model_dataset_slug": ""
}
```

Modo recomendado si el modelo es grande: súbelo una vez como dataset privado en Kaggle y referencia ese dataset:

```json
{
  "include_local_model_in_job_dataset": false,
  "model_dataset_slug": "tu_usuario/tu-modelo-best-pt",
  "model_filename": "best.pt"
}
```

Enviar video:

```powershell
curl -X POST "http://localhost:8000/v1/videos/analyze-kaggle" `
  -F "video=@C:\path\to\video.mp4"
```

Respuesta:

```json
{
  "job_id": "...",
  "dataset_slug": "user/dvka-video-...",
  "kernel_id": "user/dvka-infer-...",
  "status": "submitted",
  "status_url": "/v1/kaggle-jobs/{job_id}",
  "pull_url": "/v1/kaggle-jobs/{job_id}/pull"
}
```

Consultar:

```powershell
curl "http://localhost:8000/v1/kaggle-jobs/{job_id}"
```

Bajar outputs cuando esté `complete`:

```powershell
curl -X POST "http://localhost:8000/v1/kaggle-jobs/{job_id}/pull"
```

Descargar resultado:

```powershell
curl -O "http://localhost:8000/v1/kaggle-jobs/{job_id}/artifacts/tracking_track_summary.csv"
```

Requisitos:

- Kaggle CLI configurado en el servidor;
- `KAGGLE_USERNAME` y `KAGGLE_KEY`, o `~/.kaggle/kaggle.json`;
- cuenta Kaggle con GPU habilitada;
- modelo `best.pt` local o subido como dataset privado.
