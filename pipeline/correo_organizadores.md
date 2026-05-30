# Borrador de correo a los organizadores (SMART CHALLENGE 2026)

**Para:** organizadores MTC SC26 / Artificio (vía foro de discusión de Kaggle o correo de contacto)
**Asunto:** Consulta sobre etiquetado parcial y su efecto en la métrica (posibles falsos positivos por vehículos no anotados)

---

Estimado equipo organizador del SMART CHALLENGE 2026:

Antes que nada, gracias por la organización del concurso y por poner a disposición un dataset
tan valioso para la movilidad urbana del Perú.

Escribo para plantear una consulta técnica sobre la interacción entre el **etiquetado del
dataset** y la **métrica oficial** (Macro AP-rIoU\@[0.50:0.80]), por si pudieran aclararla.

**Observación 1 — El etiquetado parece ser parcial (por zona de interés).**
Al analizar `train.csv` notamos que, en muchos clips, no todos los vehículos visibles están
anotados: las anotaciones se concentran en los corredores de la intersección, mientras que
vehículos claramente visibles fuera de esa zona (p. ej. estacionados o en calles aledañas)
no tienen caja. De forma agregada, en una porción importante de los vídeos la región anotada
cubre menos de la mitad del cuadro. Entendemos que esto es razonable para un objetivo de
conteo en intersecciones, pero tiene una consecuencia sobre la evaluación (Observación 2).

**Observación 2 — La métrica penaliza detectar vehículos reales no etiquetados.**
Como la métrica AP cuenta como falso positivo (FP) toda predicción que no hace match con una
caja del ground-truth, un detector que funcione *bien* y encuentre vehículos reales que no
fueron etiquetados será penalizado por ello. El efecto es especialmente fuerte porque esas
detecciones suelen tener **alta confianza** (son vehículos nítidos), y en AP los FP de score
alto son los que más bajan el puntaje. En una simulación con la métrica oficial observamos que
agregar falsos positivos de alta confianza equivalentes a solo el 10–30% del ground-truth puede
reducir el score de ~1.0 a ~0.6–0.8, aun cuando esas detecciones correspondan a vehículos reales.

En la práctica, esto significa que **un modelo mejor (que detecta más vehículos reales) podría
obtener un puntaje peor** si el conjunto de prueba está etiquetado parcialmente, lo que
introduciría ruido en el ranking del leaderboard.

**Consultas concretas:**
1. ¿El conjunto de prueba (público y privado) está etiquetado con el **mismo criterio de zona**
   que el de entrenamiento? Es decir, ¿solo se anotan los vehículos dentro de una región de
   interés por intersección?
2. En caso afirmativo, ¿existe una **máscara o región de evaluación** oficial por clip que
   delimite la zona donde se evalúan las predicciones, de modo que las detecciones fuera de
   esa zona no se cuenten como falsos positivos?
3. Si no existe tal máscara, ¿podrían considerar publicarla, o aclarar la recomendación para
   que los participantes filtren sus predicciones a la zona anotada y no sean penalizados por
   detectar vehículos reales fuera de ella?

Creemos que aclarar este punto ayudaría a que el ranking refleje mejor la calidad real de los
modelos y a igualar las condiciones para todos los participantes.

Quedamos atentos a su respuesta. Muchas gracias por su tiempo.

Atentamente,
[Tu nombre / equipo]
[Usuario de Kaggle]

---

## Notas de respaldo (no incluir en el correo; evidencia interna)

- ROI: 83% de los vídeos con región anotada < 50% del cuadro; 0% > 85%
  (ver `obb/roi_out/heatmap_global.png` y `roi_stats.csv`).
- Simulación de penalización por FP (métrica oficial reimplementada y validada):
  - FP con score ≤ 0.5: caída 0.000 (no penaliza).
  - FP con score 0.7 / 0.9 / 0.99: Macro AP 0.82 / 0.60 / 0.53.
  - FP confiados (score 0.9) = 10% del GT → AP 1.00→0.79; = 50% → 0.55.
  - (ver `eval/fp_simulation.py`, `eval/out/fp_score_sweep.png`, `fp_count_sweep.png`).
