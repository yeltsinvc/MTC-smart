"""Utilidad de bitacora para el pipeline OBB del MTC SMART Challenge.

Cada etapa (script) llama a log_stage(...) y queda registrada de forma estructurada
en dos sitios, pensados para luego escribir un articulo/documentacion:

  * pipeline/run_log.jsonl  -> una linea JSON por evento (maquina-legible, reproducible)
  * pipeline/PIPELINE_LOG.md -> bitacora humana en Markdown (append-only)

Uso tipico dentro de un script de etapa:

    from pipeline._log import log_stage
    log_stage(
        stage="02_investigate_roi",
        title="Investigacion de la ROI de anotacion",
        inputs={"train": "data/train.csv"},
        outputs={"stats": "obb/roi_out/roi_stats.csv"},
        metrics={"hull_frac_median": 0.31, "videos_roi_lt50pct": 0.83},
        findings=["Las anotaciones se concentran en los corredores de la interseccion."],
        decision="Entrenar enmascarando a la ROI por video.",
    )

No usa Date.now del entorno-modelo; toma la hora del sistema operativo real donde
corre Python (esto SI esta permitido en scripts normales de usuario).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
JSONL = ROOT / "run_log.jsonl"
MD = ROOT / "PIPELINE_LOG.md"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_stage(
    stage: str,
    title: str,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    findings: list[str] | None = None,
    decision: str | None = None,
    notes: str | None = None,
) -> None:
    ts = _now()
    record = {
        "ts": ts,
        "stage": stage,
        "title": title,
        "inputs": inputs or {},
        "outputs": outputs or {},
        "metrics": metrics or {},
        "findings": findings or [],
        "decision": decision or "",
        "notes": notes or "",
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    with JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # markdown legible
    lines = [f"\n## [{ts}] {stage} — {title}\n"]
    if inputs:
        lines.append("**Entradas:** " + ", ".join(f"`{k}`={v}" for k, v in inputs.items()))
    if outputs:
        lines.append("**Salidas:** " + ", ".join(f"`{k}`={v}" for k, v in outputs.items()))
    if metrics:
        lines.append("**Métricas:**")
        for k, v in metrics.items():
            lines.append(f"- `{k}`: {v}")
    if findings:
        lines.append("**Hallazgos:**")
        for fnd in findings:
            lines.append(f"- {fnd}")
    if decision:
        lines.append(f"**Decisión:** {decision}")
    if notes:
        lines.append(f"**Notas:** {notes}")
    with MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[log] etapa '{stage}' registrada en {JSONL.name} y {MD.name}")


if __name__ == "__main__":
    log_stage(stage="00_smoke_test", title="Prueba de la utilidad de log",
              notes="Si ves esto en PIPELINE_LOG.md, el logger funciona.")
