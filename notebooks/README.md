# Notebooks

## `api_remote_kaggle_demo.ipynb`

Notebook cliente para la demo del jurado:

1. envía un video a tu servidor;
2. el servidor lanza Kaggle;
3. consulta estado;
4. hace pull de outputs;
5. descarga CSV/JSON de resultados.

El notebook Kaggle real de inferencia se genera dinámicamente en:

```text
server_runs/{job_id}/kaggle_kernel/infer.ipynb
```

después de llamar:

```text
POST /v1/videos/analyze-kaggle
```
