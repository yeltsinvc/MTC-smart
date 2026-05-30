# Pendientes / decisiones abiertas

## P1 — Confirmar criterio de etiquetado del TEST (bloqueante para inferencia)
**Estado:** ABIERTO — esperando respuesta de organizadores.
**Qué:** ¿El conjunto de prueba (public/private) usa la misma ROI por intersección que el train?
¿Existe una máscara de evaluación oficial por clip?
**Por qué importa:** si el test está etiquetado parcialmente, detectar vehículos reales fuera de
la ROI cuenta como falso positivo y hunde el AP (ver `eval/fp_simulation.py`). La estrategia de
inferencia (filtrar predicciones a la ROI) depende de esta respuesta.
**Acción:** correo redactado en `correo_organizadores.md` (enviar por foro de Kaggle).

## Supuesto de trabajo asumido (mientras P1 no se aclare)
- **Entrenamos SOLO dentro de la ROI de cada vídeo** (recorte/máscara), para no inyectar
  falsos negativos (vehículos visibles no etiquetados aprendidos como fondo).
- En inferencia, queda pendiente decidir si filtrar las predicciones a la ROI estimada del test
  (depende de P1).

## P2 — Verificar completitud del etiquetado DENTRO de la ROI
**Estado:** en curso (etapa 07).
**Qué:** confirmar visualmente que dentro de la ROI no quedan vehículos sin etiquetar.
Si quedan, la contaminación de negativos se reduce pero no desaparece.
