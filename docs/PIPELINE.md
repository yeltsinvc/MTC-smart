# Pipeline Competitivo

## Bucle

1. `DataAgent` valida labels y desbalance.
2. `ExperimentAgent` ordena cola por prioridad.
3. `KaggleWorkerAgent` genera un notebook por experimento.
4. Kaggle entrena en T4 y exporta `artifacts/metrics.json`.
5. `ReviewAgent` arma `state/leaderboard.csv`.
6. `EvolutionAgent` encola variantes del líder.
7. `TypologyAgent` aplica calibración geométrica aprendida desde labels y reglas de respaldo.
8. `TrackingPostprocessAgent` estabiliza la clase final por vehículo usando todos sus frames.
9. `OpenAIFallbackAgent` desempata solo tracks ambiguos, si la competencia permite API externa.

## Reglas de Promoción

- Si el líder está en `960`, probar `1280`.
- Si una clase tiene bajo AP, mantener modelo y subir resolución antes de cambiar arquitectura.
- Si `RT-DETR` gana precisión pero es caro, conservarlo como candidato final, no como único camino.
- Si `mAP50` sube pero `mAP50-95` no, mejorar localización: resolución, labels limpios, menos augment destructivo.
- Si clases raras fallan, usar `small_objects`, copy-paste ligero y más datos de esa clase.

## Cascada o Áreas

Para el concurso, primero conviene tener un detector único multiclase como baseline. Después se prueba una cascada geométrica:

- detector general `vehicle` para localizar todo;
- detector/clasificador base para clases conocidas;
- regla de área relativa al auto para `microbus`, `bus`, `articulated_bus`, `mototaxi`;
- clasificador/crop model si hay suficientes crops;
- ensemble de detectores si la submission permite fusionar predicciones.

Esa cascada debe compararse contra el detector único, porque puede mejorar clasificación fina pero empeorar mAP si reasigna clases de forma agresiva.

## Regla De Área

Primero se aprende desde los labels:

```powershell
dvka calibrate-geometry "C:\path\to\dataset"
```

Esto calcula por clase:

- percentiles de área normalizada: `area_p10`, `area_p50`, `area_p90`;
- percentiles de aspecto normalizado: `aspect_p10`, `aspect_p50`, `aspect_p90`;
- ratio mediano contra el auto: `area_ratio_to_car_p50`.

Luego, para cada imagen o escena:

```text
car_area_ref = mediana del área normalizada de autos confiables
area_ratio_to_car = área_normalizada_objeto / car_area_ref
```

Luego:

- `motorcycle`: ratio bajo.
- `mototaxi`: mayor que moto, menor o similar a auto.
- `car`: ratio alrededor de 1.
- `microbus`: ratio entre auto y bus.
- `bus`: ratio alto.
- `articulated_bus`: ratio alto y aspecto alargado.
- `truck`: ratio alto, pero se conserva si el detector ya lo predice.

El clasificador geométrico compara cada detección contra las distribuciones aprendidas. Las reglas en `configs/typology_rules.json` funcionan como restricciones y fallback, no como única fuente de verdad.

## Consenso Temporal

En video, un vehículo puede alternar entre clases:

```text
car -> microbus -> car -> bus -> microbus
```

Por eso el pipeline usa tracking:

```text
detecciones por frame -> track_id -> votos por clase -> track_final_class
```

El voto final pondera:

- confianza del detector;
- score geométrico calibrado;
- frecuencia de la clase dentro del track.

Esto produce una clase compatible para toda la trayectoria sin cambiar la caja frame a frame.

## Fallback OpenAI

Se usa solo al final:

```text
detector -> tipología geométrica -> consenso por tracking -> OpenAI solo si sigue ambiguo
```

Criterios típicos:

- `track_class_stable = false`;
- `track_final_score` bajo;
- margen bajo entre las dos clases más votadas;
- crop disponible;
- `OPENAI_API_KEY` presente;
- `configs/openai_fallback.json` con `enabled=true`.

La consulta envía solo el crop del vehículo y pide una respuesta estructurada:

```text
chosen_class: candidate_a | candidate_b | uncertain
confidence: 0..1
reason: texto corto
```

Según la documentación oficial de OpenAI, la Responses API acepta entrada de imagen y permite `json_schema` para salidas estructuradas. El pipeline usa esa forma para evitar respuestas libres.

No debe usarse si las reglas de Kaggle prohíben llamadas externas durante inferencia o generación de submission.

## Comando Automático

```powershell
dvka cycle --launch-limit 2
```

Ese comando refresca Kaggle, baja outputs, reconstruye leaderboard, genera nuevas propuestas y lanza si hay capacidad.
