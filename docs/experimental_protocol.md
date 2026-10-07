# Protocolo experimental O1 — versión 1, 29/09/2026

**Actualización de alcance:** el pipeline de pseudomáscaras y segmentador común
se especifica en [weak_pipeline.md](weak_pipeline.md), según README/THESIS_CONTEXT
actualizados. La integración Roy sigue pendiente; la primera ejecución completa
usa Landslide4Sense. Las afirmaciones históricas sobre etapas no implementadas
deben interpretarse junto con esa guía y las pruebas actuales.

**Integración actualizada 03/10/2026:** se eligieron 14 canales y se reconstruyeron
806 muestras Roy. La política vigente, las 32 exclusiones, min–max común y los
manifiestos propuestos por solapamiento están en
[l4s_roy_14_v1.md](datasets/l4s_roy_14_v1.md). Se implementó soporte explícito para
fuentes mixtas, pero el mapeo físico pendiente mantiene deshabilitado este
experimento. Las restricciones y conteos O1 de los apartados siguientes son
históricos; el documento de integración recoge sus decisiones posteriores.

## Alcance y estado

La decisión autorizada es integrar L4S + L4S-PE conservando procedencia. Sustituye el plan inicial de experimentos separados. La autorización no demuestra compatibilidad: la integración ejecutable permanece bloqueada por identidad de canales, radiometría, anotaciones y geografía de Roy. No se entrenó O2 para redactar este protocolo.

Los cuatro clasificadores son ResNet18, ResNet50, DeiT-S/16 y MiT-B1. Reciben imagen y etiqueta binaria; ninguna máscara participa en la pérdida. «Positivo» significa al menos un píxel de clase landslide en la máscara original aceptada. La correspondencia landslide/huayco necesita justificación del fenómeno y no se asume equivalencia universal.

## Admisión y trazabilidad

`metadata/integrated_provisional.csv` es un inventario conjunto, NO un dataset integrado validado. Conserva el identificador original, un identificador único prefijado para Roy, dataset, rutas, etiqueta, área, hashes de imagen/máscara, estado, causas de revisión y partición de origen. Estado `pending_review` significa pendiente para el experimento integrado, aunque `local_structure_status=passed` certifique forma, lectura y etiqueta. `accepted` y `excluded` se reservan a decisiones fundamentadas; sus registros pueden tener cero filas. No se eliminan muestras para ejecutar el código.

El clasificador rechaza estados distintos de `accepted` y `training_eligible=false` cuando esos campos están presentes. Los CSV históricos de L4S siguen funcionando y no constituyen aprobación de integración. La restricción existente a un solo dataset se mantiene hasta resolver los bloqueos; no se disfraza Roy como L4S.

`python -m scripts.audit_o1` realiza una pasada de verificación y momentos descriptivos. Comprueba las máscaras binarias, dimensiones, etiquetas, área y hashes L4S frente a la auditoría anterior. Calcula hashes Roy para completar la comprobación entre fuentes. También verifica la transformación PNG→HDF5 actual. No repite la lectura exhaustiva de todos los TIFF: esa evidencia histórica se conserva como tal, sin certificar de nuevo su contenido. Cualquier discrepancia detiene la generación. Test se lee solo para integridad, jamás para calibración.

## Particiones

1. Mantener los 3.799/245/800 elementos originales de L4S. No redistribuir ni usar TestData para desarrollo.
2. Mantener Roy sin asignación hasta resolver equivalencia, anotaciones y agrupación geográfica. `Grupo` y `Fechas` no se interpretan automáticamente como evento o adquisición.
3. Para Roy, una vez disponibles huellas/escenas/eventos fiables, construir componentes de muestras relacionadas (mismo evento/escena, huellas solapadas o imágenes duplicadas). Componentes completos deben permanecer en una sola partición. Coordenadas iguales son una restricción mínima, no una prueba completa de independencia. Resolver máscaras conflictivas antes de admisión; no elegir la primera ni unirlas.
4. Congelar L4S y diseñar la asignación de componentes Roy usando solo metadatos y etiquetas de admisión, con semilla 42. Documentar el reparto viable antes de entrenar; no fijar 80/10/10 sin conocer el número de grupos negativos independientes. La viabilidad de un test Roy con ambas clases es una condición pendiente, no resuelta por un sorteo.
5. Si una relación cruza una partición L4S congelada, detener y revisar la admisión de Roy. Nunca mover una muestra test a train para resolverlo.
6. Registrar IDs, rutas, hashes, clase, procedencia, grupos y justificación por muestra. Verificar ausencia de solapamientos de IDs, rutas, hashes y grupos conocidos; informar por separado geografía desconocida.

Los CSV bajo `splits/provisional/` preservan únicamente las asignaciones L4S y mantienen las 838 filas Roy en `integrated_unassigned.csv`. Ninguna fila queda autorizada para entrenamiento integrado. No existen splits definitivos integrados. La división original no tiene semilla local; 42 es la semilla prevista para la futura asignación Roy.

## Condiciones comunes de comparación

- Mismo manifiesto aceptado, canales semánticos, resolución 128×128, normalización y datos de validación/test para los cuatro modelos.
- Experimento principal inicial: pesos aleatorios para todos, decisión explícita que permite entrada multicanal sin trasplante RGB no validado. Si se estudia preentrenamiento será un experimento adicional separado con origen, adaptación y presupuesto documentados; no se mezcla con el principal.
- Objetivo BCEWithLogitsLoss. Mantener inicialmente BCE sin ponderación como referencia; cualquier balanceo adicional se calcula solo con train y se aplica bajo la misma política a los cuatro modelos. No usar sobresampling o pesos implícitos. Reportar Recall/F1 y resultados por procedencia junto con Accuracy.
- Checkpoint: mínimo de pérdida de validación; empate conserva el primero. Test no decide checkpoint. Umbral de clasificación 0,5 fijo en el protocolo inicial.
- Semillas emparejadas 42, 43 y 44 como mínimo de repetición para describir variabilidad; no garantizan potencia estadística. Mismos splits para todas, semillas no regeneran particiones.
- AdamW y valores de `configs/train_l4s.json` son una plantilla técnica, no hiperparámetros finales. Antes de O2 ejecutar pilotos solo con train/val para medir memoria y estabilidad. Fijar una configuración común cuando sea viable; si batch físico cambia, registrar batch efectivo y reconocer el factor de confusión. El runner actual no implementa acumulación, scheduler ni early stopping: deben permanecer desactivados hasta una implementación y validación explícitas.
- Registrar antes del ajuste el mismo número de configuraciones y la misma regla de presupuesto máximo de épocas/actualizaciones para cada modelo; definir estos límites después del piloto hardware y antes de ver resultados test. No otorgar ensayos adicionales a una arquitectura por resultados desfavorables. No se establece aquí una tasa/épocas definitiva sin evidencia de piloto.
- Un experimento integra configuración, hash de manifiestos, hashes de datos de la auditoría, versión del código, versiones del entorno, semilla, optimizer/loss, checkpoint, historial y hardware. El runner existente guarda hashes de CSV, no sustituye los hashes de imágenes de la auditoría.

## Evaluación y estadística

Aplicar `docs/evaluation_protocol.md`. Presentar métricas por dataset y globales; nunca interpretar el global dominado por una fuente como generalización a la otra. Comparación emparejada sobre las mismas muestras y semillas. Reportar métricas de cada semilla y media/desviación entre semillas, no seleccionar la mejor semilla para test.

Si existen grupos geográficos válidos, intervalos descriptivos de diferencias emparejadas mediante bootstrap por grupos completos (2.000 réplicas, semilla 42, percentiles 2,5/97,5). Es una regla prospectiva, no ejecutada. No hacer bootstrap independiente por parche cuando hay dependencia espacial desconocida; en ese caso reportar diferencias descriptivas y limitar inferencia estadística. AUC de una réplica con una sola clase es indefinida y se registra la cantidad válida. No afirmar significación ni causalidad CNN/ViT con evidencia insuficiente.

Análisis de tamaño: conservar `area_ratio`; estudiar su distribución en positivos de train definitivo. Si se adoptan terciles, fijar P33/P66 exclusivamente en train antes de evaluar, registrar cortes y regla de igualdad (pequeño ≤P33, mediano >P33 y ≤P66, grande >P66). Si hay cortes degenerados no forzar tres grupos. Reportar Recall de positivos y métricas espaciales por tamaño, con denominadores. No usar percentiles de test para definir grupos.

## Recursos computacionales

Registrar parámetros, tiempo total de entrenamiento, tiempo por época y memoria pico CUDA con el mismo equipo y precisión. Para latencia de inferencia, evaluar batch=1, `eval`, sin gradientes, mismos datos de validación de comprobación, entradas residentes en dispositivo; separar lectura/preprocesamiento de forward. Tras 20 calentamientos, medir 100 ejecuciones sincronizadas, informar mediana y P95, dispositivo y versiones. Es un protocolo de medición, no un benchmark ya realizado. Si cambia hardware o precisión, no atribuir la diferencia temporal a la familia del modelo. Los tiempos del smoke CPU no son resultados de eficiencia.

## Criterio de cierre

R1 y R2 integrados continúan PARCIALES mientras no existan admisión, splits y preprocesamiento común validado. R3 es un marco definido para ejecución posterior: no equivale a implementar atribución ni a completar O3. Los bloqueos y resultados actuales están en `O1_COMPLETION_REPORT.md`.
# Experimento conjunto propuesto

Ver [l4s_roy_14_v1.md](datasets/l4s_roy_14_v1.md) para la reconstrucción ya ejecutada
y la incertidumbre de bandas pendiente. Las particiones L4S existentes se conservan;
Roy se agrupa por solapamiento y se informa por dominio.
