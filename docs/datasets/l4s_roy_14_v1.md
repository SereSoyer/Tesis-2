# Reconstrucción Roy y experimento conjunto de 14 canales

Fecha: 2026-10-03. El usuario eligió un experimento nuevo con Landslide4Sense y
Roy juntos, conservando **14 canales**, y autorizó reconstruir Roy desde TIFF.
La alternativa previa de diez canales queda como antecedente, no como decisión vigente.

**Actualización 2026-10-04: admisión completada y entrenamiento conjunto habilitado.**
Configuraciones activas: `configs/experiments/l4s_roy_14_v1/`. Manifiestos activos:
`splits/l4s_roy_14_v1/`. Registro: `metadata/l4s_roy_14_v1_admission.json`.
Se conservaron las propuestas, archivos y asignaciones existentes. Ver
[comandos de transferencia y entrenamiento](../REPRODUCIBILIDAD.md).

## Resultado local

Se leyeron los 11.732 TIFF de los 838 puntos, sus georreferencias y los 827 PNG
disponibles. Se reconstruyeron **806 imágenes 128×128×14** con sus máscaras binarias:
**778 positivas y 28 negativas**. Se conservaron los originales y los experimentos L4S.

| Motivo de exclusión conservadora | Muestras |
|---|---:|
| PNG original ausente | 11 |
| Imágenes duplicadas con máscaras contradictorias, ambos miembros | 4 |
| NoData de anotación no preservado por la máscara binaria, point_0122 | 1 |
| Máscara negativa en conflicto con etiqueta del autor, point_0905 | 1 |
| NoData en alguno de los 14 canales del recorte, adicionales a los anteriores | 15 |
| Total de muestras distintas apartadas | **32** |

Son 16 recortes con NoData, pero point_0465 también carece de PNG: no contarlo dos
veces. Los motivos detallados de `excluded.csv` pueden solaparse e incluyen los
motivos heredados de la auditoría anterior. No se imputaron valores ni se descartaron
huaycos pequeños. Las máscaras reconstruidas son numéricamente idénticas a las
máscaras HDF5 originales de esas 806 muestras; no se resolvieron conflictos
seleccionando una máscara arbitraria ni ajustando el recorte por IoU.

Archivos generados:

- `data/prepared/roy14_v1/img/` y `mask/`: HDF5 derivados con valores TIFF, sin
  normalización persistida, y atributos de procedencia.
- `results/audits/roy14_reconstruction_v1/`: `summary.json`, `policy.json`,
  `admission.csv`, `reconstructed.csv`, `excluded.csv`, `overlap_edges.csv`.
- `splits/proposed/l4s_roy_14_v1/{train,val,test}.csv`: manifiestos conjuntos propuestos.
- `configs/l4s_roy_14_v1/`: cuatro configuraciones de clasificación y `pipeline.json`.

**La reconstrucción está terminada; B13/B14 ya pueden adoptarse como pendiente en
radianes y elevación en metros con la evidencia disponible.** La fuente exacta y
el datum no están certificados. La admisión se formalizó con esa limitación;
el entrenamiento conjunto nuevo está habilitado. El mapeo B9/B10 tiene respaldo de los
cuatro sondeos EE, con el alcance detallado abajo. Los manifiestos propuestos históricos tienen `status=pending_review`
y `training_eligible=false`; las configuraciones, `training_enabled=false`.
Estos campos conservan el estado de la propuesta inicial. Las copias admitidas nuevas
tienen `status=accepted`, `training_eligible=true` y `band_mapping_status=empirically_supported_mapping`.
Los cuatro modelos y el pipeline fijan la admisión y sus manifiestos mediante hashes.

## Geometría adoptada

La configuración original declara `mask_area=[10,10,140,140]`. Se adopta la
interpretación de IRIS upstream: PNG 130×130 sobre TIFF `[10:140,10:140]`.
Por tanto, la máscara conservada PNG `[1:129,1:129].T` corresponde a la imagen
TIFF `[11:139,11:139].T`, para cada uno de los 14 canales. Los HDF5 distribuidos
de imagen correspondían al recorte `[10:138,10:138].T` en los canales auditados.

La evidencia de la convención está en el commit IRIS
`787073c83eca9feded9d438d1e7407f1103d9fad`,
[predict_mask](https://github.com/ESA-PhiLab/iris/blob/787073c83eca9feded9d438d1e7407f1103d9fad/iris/segmentation/__init__.py#L325).
No se recuperó el exportador final ni la versión modificada exacta del autor;
esto es una reconstrucción documentada bajo esa convención, no una certificación
independiente del registro original.

Se verificó que los 14 TIFF de cada punto comparten la misma malla de 10 m,
EPSG:32718, PixelIsArea y tamaño 150×150. No hubo reproyección ni interpolación:
para esta representación bastaron Pillow y las etiquetas GeoTIFF, sin añadir
rasterio. Si se incorpora otra proyección, resolución o transformación, este
procedimiento debe fallar y revisarse, no recortar silenciosamente.

## Uniformidad y limitación de las bandas

La configuración propuesta usa en ambas fuentes índices 0..13, entrada float32
CHW 14×128×128 y min–max **por canal y por imagen**. Una banda constante pasa a cero;
los valores no finitos producen error. En la reconstrucción se excluye cualquier
recorte con NoData explícito de TIFF, incluido el centinela negativo extremo.
No se interpreta cero como NoData.

Se aplica el mismo preprocesamiento a las dos fuentes, las cuatro arquitecturas
y las tres particiones. No se ajustan estadísticas con val/test. Este escalado
elimina diferencias absolutas entre parches y no convierte niveles atmosféricos,
fuentes de elevación o identidades espectrales en equivalentes.

Estado de la **correspondencia física**, separado del número de canales:

| Posiciones / archivos Roy | Evidencia y estado |
|---|---|
| 8/9, B9.tif y B10.tif | Sondeos EE respaldan B9/B10 como documentan tesis y R, frente a B8A/B9 en IRIS; coincidencia exacta de 25 píxeles por banda en point_0502 |
| 12/13, B13.tif y B14.tif | Sondeo topográfico respalda B13 pendiente en radianes y B14 elevación en metros; fuente y datum exactos no certificados |

Los valores observados de B13 (0..1,484) y B14 (434..4135) son compatibles con
pendiente en radianes y elevación, respectivamente, pero el rango no certifica
la variable ni la fuente. Los nombres TIFF y sus etiquetas inspeccionadas no
aportan una identificación física inequívoca de las bandas espectrales dudosas.

Actualización de la evidencia: la auditoría
[roy_band_identity_v1](../../results/audits/roy_band_identity_v1/README.md) encontró
cuadrículas de seis píxeles en B9 de los 838 puntos y en los 837 B10 no constantes,
coherentes con bandas de 60 m en la malla de 10 m. Junto con la selección explícita
del script R y la radiometría, favorece B9/B10 frente a B8A/B9. Sigue siendo
inferencia. El primer sondeo externo ya se realizó para point_0021: comparación
de 25 posiciones contra Sentinel-2B del 25/06/2019, baseline 05.00, con fuerte
evidencia a favor de B9/B10. B10 coincide en 23/25 valores y B9 tiene correlación
0,9992 con B9; las alternativas B8A/B9 tienen errores mucho mayores. Persisten
diferencias de representación en bandas de control. Ver
[análisis medido](../../results/audits/roy_ee_probe_point_0021_v1/README.md).
La extensión a los otros tres puntos ya se completó:
[comparación de cuatro sondeos](../../results/audits/roy_ee_probe_comparison_v1/README.md).
En point_0502 coinciden exactamente los 150 valores de las seis bandas consultadas
con el producto del 09/06/2020, baseline 02.09. point_0002 y point_0023 favorecen
B9/B10, aunque su adquisición exacta es ambigua por escenas con RMSE de control
cercano. Se respalda documentalmente B9.tif=B9 y B10.tif=B10; no se afirma que
se haya comparado cada píxel o cada archivo de Roy contra su fuente original.

Actualización 2026-10-04: el [sondeo topográfico](roy_topography_probe.md) ya se
ejecutó y su CSV fue analizado. El
[informe B13/B14](../../results/audits/roy_topography_comparison_v1/README.md)
verifica las 300 filas: 100 posiciones únicas contra SRTM V3, NASADEM y AW3D30 V4.1.
B13 como radianes tiene menor error en los doce grupos punto/fuente que como grados.
Con SRTM, RMSE agregado 0,07595 frente a 0,55455 rad. B14 sigue la elevación SRTM
(r=0,999889), con un sesgo Roy−SRTM de +34,45 m. La hipótesis independiente de
conversión EGM96 reduce el RMSE de 35,27 a 8,10 m, pero no prueba producto ni datum
originales. AW3D30 no equivale al DEM auxiliar de PALSAR RTC.

La recomendación es conservar el orden B1.tif..B14.tif, adoptando B13 pendiente en
radianes y B14 elevación en metros, con procedencia parcialmente conocida declarada.
Se cierra la duda de orden/unidades para esa adopción; no hace falta otro sondeo
para decidirlas. No se hizo una corrección de datum sobre los archivos.
No se marca `verified` la procedencia completa. La admisión vigente usa
`empirically_supported_mapping`, acompañada de evidencia y límites en el registro
de metadatos. Se crearon manifiestos admitidos y configuraciones consistentes,
conservando los manifiestos propuestos y sus asignaciones.

## Propuesta de particiones

Los 3.799/245/800 elementos de L4S conservan sus asignaciones originales. Las
huellas completas de Roy forman componentes conexas cuando se solapan o tocan;
se agrupan los 838 puntos antes de excluir, conservando las relaciones que pasan
por puntos apartados. Se encontraron 418 enlaces y 518 grupos con muestras retenidas.
Se asignan grupos completos, sin usar métricas de modelos.

Se fijó seed42, proporción objetivo Roy 70/15/15 y búsqueda de 2.000 asignaciones
por desviación de conteos por clase. La agrupación hace aproximadas las proporciones.

| Partición | L4S | Roy positivos | Roy negativos | Total conjunto |
|---|---:|---:|---:|---:|
| Train | 3799 | 546 | 20 | **4365** |
| Validación | 245 | 114 | 4 | **363** |
| Test | 800 | 118 | 4 | **922** |
| Total | 4844 | 778 | 28 | **5650** |

El conjunto propuesto contiene 3.691 positivos y 1.959 negativos. Roy sigue estando
muy desbalanceado: solo cuatro negativos propios en validación y cuatro en test.
Las métricas de falsas alarmas de Perú tendrán esa limitación aunque el total
conjunto tenga más negativos. Se conservan resultados globales y por procedencia.

Las huellas impiden el solapamiento observado de Roy entre particiones, pero no
certifican independencia de eventos o adquisiciones. `Grupo` y `Fechas` se conservan
sin reinterpretarlos como eventos verificados. Falta geografía verificable por
muestra L4S para certificar independencia geográfica cruzada. Su test ya fue
evaluado en el experimento anterior: esta extensión no crea un test L4S nuevo.

## Supervisión y comparación

Los clasificadores continúan recibiendo solo imagen y etiqueta de presencia; no
abren máscaras. Las máscaras se leen en la preparación para conservar anotaciones,
comprobar integridad y derivar la etiqueta binaria, como en L4S. No se emplea su
geometría para seleccionar parches de entrenamiento por rendimiento ni para la pérdida.
Los cuatro segmentadores comunes aprenden únicamente de pseudomáscaras.
La calibración del umbral y la selección del checkpoint de segmentación usan GT
de validación, según la decisión vigente del usuario, y se declaran como tal.

La configuración conjunta fija 50 épocas por clasificador y segmentador,
Grad-CAM adaptado a ResNet18/50, DeiT-S y MiT-B1, sin CRF, conservando las políticas
comunes del pipeline L4S. No se reutilizan sus checkpoints como si fueran resultados
del nuevo dataset. Para aislar el efecto de añadir Roy, se necesita un control
L4S-only con los mismos 14 canales **y min–max**: el experimento anterior usó otro
preprocesamiento. Aún no se han entrenado los modelos conjuntos.

## Reproducción y revisión

La reconstrucción ya está ejecutada localmente. En una copia limpia con las fuentes
y la auditoría previa presentes:

```bash
.venv/bin/python scripts/reconstruct_roy14.py --policy configs/reconstruct_roy14_v1.json
```

El script se niega a sobrescribir directorios. Una repetición requiere nuevas rutas
en una copia de la política; no borrar evidencia. `summary.json` se escribe al final
y su estado distingue finalización de una salida parcial.

Comprobación repetible de la copia actual, sin entrenar ni modificar admisiones:

```bash
.venv/bin/python scripts/validate_roy14.py
cat results/audits/roy14_reconstruction_v1/summary.json
```

La verificación conserva el resultado en `validation.json` y ejemplos solo de train
en `train_examples.png`. Comprueba archivos/arrays por hash, máscaras originales,
preprocesamiento de los 806 ejemplos, conservación de L4S, cobertura sin duplicación
y separación de grupos entre particiones. No certifica identidad física de bandas.
Las 34 pruebas automáticas del repositorio pasan; incluyen geometría conocida,
detección de NoData topográfico, rechazo del protocolo pendiente, integridad de
admisión y la cadena mixta de 14 canales sin máscaras GT de train disponibles.
La verificación de la admisión vigente está en `scripts/check_joint_experiment.py`;
`validate_roy14.py` conserva su función de comprobar la reconstrucción/propuesta histórica.
