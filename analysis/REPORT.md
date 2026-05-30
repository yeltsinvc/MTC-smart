# Análisis de tipologías: ¿conviene agrupar clases para el entrenamiento?

> Análisis sobre `data/train.csv` (601 934 cajas OBB, 1 047 vídeos, 50 868 frames).
> Reproducible con `python analysis/typology_grouping.py`. Figuras en `analysis/out/`.

## TL;DR — la decisión

1. **No colapsar la salida final.** La métrica del concurso es **Macro AP-rIoU sobre las 9
   clases**: cada clase pesa 1/9. Si entrenas/entregas super-clases, las fusionadas sacan
   **AP = 0** y hunden el score. La submission **debe** distinguir las 9.
2. **Sí usar los grupos como estructura interna de entrenamiento.** Colapsar a 3 super-grupos
   sube el macro-F1 (solo-geometría) de **0.60 → 0.85**: el error grave es **intra-grupo** y de
   **apariencia**, no de escala → cabeza jerárquica + 2ª etapa visual + aumento de raras, **no**
   fusionar clases.
3. **El cuello de botella real es el desbalance.** `articulado` = **250 cajas** (0.04 %) vs
   481 731 de `auto` → **1927:1**. Sin muestreo balanceado y copy-paste, las raras son invisibles
   para una métrica macro.

---

## 1. Desbalance de clases (decisivo: la métrica es macro)

| clase | cajas | % | desbalance vs auto |
|---|---:|---:|---:|
| auto | 481 731 | 80.03 % | 1× |
| motocicleta | 47 568 | 7.90 % | 10× |
| camion | 32 668 | 5.43 % | 15× |
| minibus | 18 941 | 3.15 % | 25× |
| combi | 10 152 | 1.69 % | 48× |
| mototaxi | 5 539 | 0.92 % | 87× |
| microbus | 2 802 | 0.47 % | 172× |
| omnibus | 2 283 | 0.38 % | 211× |
| **articulado** | **250** | **0.04 %** | **1927×** |

Cada clase aporta 1/9 al score → las **clases raras deciden el ranking**; `auto` ya está resuelto.

---

## 2. Separabilidad geométrica (Bhattacharyya, 1 = inseparables)

Pareja más solapada de cada clase (espacio tamaño-relativo × elongación), de `out/overlap_geo.png`:

| clase | más solapada con | BC |
|---|---|---:|
| **auto** | **minibus** | **0.975** |
| **minibus** | **auto** | **0.975** |
| **microbus** | **camion** | **0.945** |
| **camion** | **microbus** | **0.945** |
| **combi** | **minibus** | **0.944** |
| omnibus | microbus | 0.665 |
| mototaxi | auto | 0.586 |
| articulado | omnibus | 0.447 |
| motocicleta | mototaxi | 0.184 |

Tres parejas son **casi inseparables por geometría** (BC ≈ 0.95): `auto↔minibus`,
`microbus↔camion`, `combi↔minibus`. Solo `motocicleta` está bien aislada (BC máx 0.18).
El violín de tamaño (`out/area_violin.png`) lo confirma: los medianos se solapan casi por completo.

---

## 3. Confusiones reales (modelo solo-geometría, validación por vídeo)

Confusiones más fuertes (real → predicho), de `out/confusion_holdout.png`:

```
microbus → camion      0.33        omnibus  → camion      0.19
minibus  → auto        0.32        mototaxi → auto        0.12
omnibus  → articulado  0.20        articulado → microbus  0.12
                                   combi    → auto        0.10
```

Diagonal (recall) del mismo modelo: `motocicleta` 0.99, `auto` 0.90, `articulado` 0.88\*,
`mototaxi` 0.85, `combi` 0.77, `camion` 0.75, `omnibus` 0.60, `minibus` 0.56, `microbus` 0.53.

> \* El recall alto de `articulado` (0.88) engaña: con `sample_weight` balanceado la geometría la
> "sospecha" por ser grande, pero su **precisión es ínfima** (muchísimos falsos positivos de
> ómnibus/camión). En F1 sigue siendo la peor (≈0.19). Es un problema de **precisión**, no de recall.

Casi todas las confusiones graves caen **dentro de un mismo grupo de tamaño**; entre grupos
lejanos (p. ej. `motocicleta`↔bus) no hay confusión.

---

## 4. Super-grupos naturales (clustering jerárquico, `out/dendro_geo.png`)

Cortes del dendrograma (distancia de Bhattacharyya, enlace promedio):

| k | grupos (data-driven) |
|---|---|
| **3** | **{auto, combi, microbus, minibus, camion, mototaxi}** · **{omnibus, articulado}** · **{motocicleta}** |
| 4 | {auto, combi, microbus, minibus, camion} · {omnibus, articulado} · {mototaxi} · {motocicleta} |
| 5 | {auto, combi, microbus, minibus, camion} · {omnibus} · {articulado} · {mototaxi} · {motocicleta} |
| 6 | {auto, combi, minibus} · {microbus, camion} · {omnibus} · {articulado} · {mototaxi} · {motocicleta} |

El dendrograma separa con nitidez **tres niveles de tamaño**: `motocicleta` (la más pequeña, muy
aislada) → un gran bloque de vehículos de tamaño medio → `{omnibus, articulado}` (los gigantes).
La estructura es puramente de **escala**: la apariencia (lo que separa combi de minibus, o
microbus de camión) **no aparece** porque la geometría no la ve.

---

## 5. La métrica clave: ¿cuánto ayuda agrupar?

macro-F1 del modelo **solo-geometría**, a nivel fino (9) vs colapsando a cada agrupamiento:

| nivel | macro-F1 |
|---|---:|
| **fino (9 clases)** | **0.603** |
| 6 grupos | 0.693 |
| 5 grupos | 0.689 |
| 4 grupos | 0.813 |
| **3 grupos** | **0.849** |

**Interpretación (resultado central):** colapsar a 3–4 grupos recupera +0.21–0.25 de macro-F1.
Es decir, la geometría ubica bien el **grupo de tamaño** pero **no al miembro dentro del grupo**.

> El error que importa es **intra-grupo** y es de **apariencia**, no de escala. Desde la cámara
> aérea, `auto`/`combi`/`minibus` miden casi lo mismo; lo que los diferencia (carrocería,
> ventanas, puertas, unión articulada) es **visual** → lo resuelve el **detector YOLO/RT-DETR**,
> no la geometría. (k=5 y k=6 bajan porque parten el bloque mediano justo donde el solapamiento es
> máximo, así que no ganan nada.)

---

## 6. Recomendación de estrategia de entrenamiento

**No fusionar la salida.** Entrenar el detector con las **9 clases** (YOLO maneja grano fino y, a
diferencia de la geometría, ve la apariencia). Usar los grupos como andamiaje:

1. **Muestreo balanceado + aumento dirigido (mayor palanca).** Para `articulado` (250), `omnibus`,
   `microbus`, `mototaxi`, `combi`: oversampling por clase + **copy-paste augmentation**. Sin esto,
   la métrica macro las penaliza brutalmente. Ya reflejado en `project.json → rare_classes`.
2. **Etiqueta jerárquica auxiliar ("agrupar bien hecho").** Cabeza/pérdida auxiliar de super-grupo.
   Grupo recomendado (semántico + soportado por el tamaño), 4 niveles:
   - **dos-ruedas:** `motocicleta, mototaxi`
   - **livianos ~auto:** `auto, combi, minibus`
   - **medianos/carga:** `microbus, camion`
   - **grandes:** `omnibus, articulado`

   Las raras heredan gradiente de sus hermanas abundantes del grupo y la cabeza fina las separa
   después. Es "agrupar para entrenar" sin perder las 9 clases.
3. **2ª etapa por apariencia donde duele.** Clasificador fino sobre el *crop* para desambiguar las
   parejas BC≈0.95 (`auto/combi/minibus`, `microbus/camion`, `omnibus/articulado`), fusionado con
   el **prior geométrico** (`geo_model/`) y el **consenso por tracking** (`TrackingPostprocessAgent`).
4. **Diagnóstico continuo.** Reportar macro-F1 grupo vs fino en validación: si el error es "grupo
   equivocado" → subir `imgsz`/mejorar detección; si es "miembro equivocado" → reforzar la 2ª etapa.
   Hoy, ~100 % del error es intra-grupo.

### Qué NO hacer
- ❌ Entrenar con 3–4 clases y entregar super-clases (AP = 0 en las fusionadas).
- ❌ Confiar la clase final solo a la geometría/ratio (resuelve grupo, no miembro).
- ❌ Ignorar `articulado`: con 250 cajas y precisión ínfima es el mayor riesgo del Private LB.

---

## Archivos generados (`analysis/out/`)

| Figura | Qué muestra |
|---|---|
| `area_violin.png` | Tamaño relativo al auto por clase (solape de los medianos). |
| `overlap_geo.png` | Matriz de solapamiento geométrico (Bhattacharyya). |
| `dendro_geo.png` | Dendrograma → super-grupos naturales por tamaño. |
| `confusion_holdout.png` | Confusión real en validación por vídeo (modelo geo). |
| `cooccurrence.png` | Qué clases aparecen juntas en el mismo frame. |
| `grouping_report.json` | Todos los números anteriores en JSON. |
