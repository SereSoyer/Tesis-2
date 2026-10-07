# Ficha técnica de Roy / L4S-PE

Estado: caracterización local comprobada; decisiones de entrenamiento pendientes.
Fecha de elaboración: 25 de septiembre de 2026.

**Actualización 03/10/2026:** esta ficha caracteriza los originales. Ya existe una
copia derivada de 806 muestras reconstruidas con 14 canales, con exclusiones y
particiones propuestas documentadas en [l4s_roy_14_v1.md](l4s_roy_14_v1.md).
La correspondencia física de algunas bandas sigue pendiente.

## 1. Identificación y alcance

L4S-PE es la colección peruana asociada al repositorio `lanslide4sat_pl`, desarrollado originalmente para segmentación supervisada de deslizamientos. En este proyecto se identifica también como «dataset de Roy».

El documento local `resumen.odt` sitúa el área de estudio en Junín, Pasco y Huánuco y describe anotación mediante una versión modificada de IRIS. Estas afirmaciones proceden de la documentación del autor; no equivalen a verificar la ubicación administrativa o la validación experta de cada muestra. La referencia bibliográfica definitiva está pendiente: el README del repositorio conserva una cita `@article{tbd}`.

La adaptación a esta tesis contempla clasificación binaria y posterior localización débilmente supervisada. Las máscaras permiten derivar etiquetas de presencia y evaluar localización; no se incorporarán a la función de pérdida del clasificador. La correspondencia entre la clase general «landslide» y el término «huayco» deberá justificarse; no se consideran automáticamente sinónimos.

## 2. Inventario comprobado

| Representación | Ruta relativa | Cantidad | Forma observada | Observación |
|---|---|---:|---|---|
| Imagen procesada | `data/roy/l4s/image_ls4pe/` | 838 | 128×128×14 | HDF5, clave `img`, float64 |
| Máscara procesada | `data/roy/l4s/mask_ls4pe/` | 838 | 128×128 | HDF5, clave `mask`, valores 0/1 |
| Bandas originales | `data/roy/landslide_ok/point_XXXX/input/` | 11 732 | 150×150 por TIFF | 14 archivos por punto |
| Imagen HDF5 original | `data/roy/landslide_ok/point_XXXX/point_XXXX.h5` | 838 | 14×150×150 | Rango global por archivo [0,1] |
| Anotación original | `data/roy/landslide_ok/point_XXXX/target/manual.png` | 827 | 130×130×3 | RGB |
| Metadatos por punto | `data/roy/landslide_ok/point_XXXX/metadata.json` | 838 | — | Identificador, coordenadas, sensor declarado y resolución |

`l4s` contiene aquí L4S-PE; no debe confundirse con Landslide4Sense internacional. Los 838 pares procesados están emparejados por identificador. No hay imágenes procesadas sin máscara HDF5. Faltan 11 PNG originales, no 11 máscaras procesadas.

Todos los TIFF inspeccionados tienen paso de píxel 10×10, coherente con la resolución declarada de 10 m. Es una malla de salida: no demuestra que todos los canales tengan resolución nativa de 10 m.

## 3. Canales y preprocesamiento

La colección tiene 14 canales; no debe describirse como «14 bandas espectrales de Sentinel-2». La documentación los presenta como 12 entradas espectrales y dos variables topográficas.

| Posición en archivos | Interpretación documental | Estado |
|---|---|---|
| B1–B8 | Canales Sentinel-2 B1–B8 | Coincidencia entre listas del R e IRIS; falta trazabilidad completa de exportación |
| B9–B10 | B9/B10 en el R; B8A/B9 en IRIS y un notebook | Identidad pendiente |
| B11–B12 | Canales Sentinel-2 B11/B12 | Coincidencia documental |
| B13 | Valores compatibles con pendiente en radianes | Python de ALOS dice pendiente; R e IRIS indican otro orden |
| B14 | Valores compatibles con elevación | Python de ALOS dice elevación; R e IRIS indican otro orden |

La fuente topográfica tampoco está cerrada: ODT y flujos Python declaran ALOS; el generador R y los metadatos declaran SRTM. No se seleccionarán definitivamente canales ni unidades basándose solo en rangos numéricos.

El exportador R implementa min–max por capa y parche. Los HDF5 originales tienen rango [0,1], mientras los HDF5 de `l4s` no. Aún no se ha demostrado la transformación numérica completa entre representaciones. El cargador original tiene normalización opcional sobre todo el tensor y actualmente desactivada. Esto no constituye una decisión de normalización para la tesis.

## 4. Máscaras y etiquetas de imagen

La configuración de IRIS define:

| RGB | Significado |
|---|---|
| (255,0,0) | Landslide |
| (0,255,0) | Clear / fondo |
| (50,50,255) | NoData |

Se encontraron 7 píxeles NoData en `point_0122`. Las máscaras HDF5 son binarias y no conservan una clase NoData separada. No se debe interpretar una máscara RGB mediante su suma total: tanto el verde como el azul tienen valores distintos de cero.

La etiqueta provisional por imagen procesada se calcula como:

```python
label = int((mask == 1).any())
```

| Representación | Positivas | Negativas | Total con máscara |
|---|---:|---:|---:|
| PNG originales 130×130 | 797 | 30 | 827 |
| HDF5 procesados 128×128 | 808 | 30 | 838 |

Los 11 puntos sin PNG tienen máscara HDF5 positiva. La distribución procesada es 96,42% positiva y 3,58% negativa. Un predictor siempre positivo alcanzaría 96,42% de accuracy sobre la colección completa; por ello esa métrica por sí sola no demuestra utilidad del clasificador. Las etiquetas derivadas están contadas, pero su aceptación experimental depende de aclarar la calidad y procedencia de las anotaciones.

## 5. Tamaño de las regiones anotadas

Se define `area_ratio = positive_pixels / total_pixels`. En esta caracterización el denominador es el total de píxeles del parche, no únicamente los píxeles válidos; cualquier cambio para excluir NoData debe registrarse.

Sobre las 808 máscaras HDF5 positivas:

| Estadística | Área afectada (%) |
|---|---:|
| Mínimo | 0,1160 |
| P25 | 1,1520 |
| P33 | 1,4343 |
| Mediana / P50 | 2,1606 |
| P66 | 3,3119 |
| P75 | 4,0436 |
| Media | 3,0404 |
| Máximo | 20,3491 |

La mínima anotación positiva tiene 19 píxeles. Sobre los 838 HDF5, incluyendo negativos, la ocupación global es 2,9315%. Sobre los 827 PNG completos, incluyendo negativos, media=2,8734%, mediana=2,0178% y máximo=19,9053%. Son poblaciones y tamaños distintos; sus estadísticas no son intercambiables.

El 1,8% mencionado en el ODT no se reproduce con estas máscaras. No se fijan aún umbrales small/medium/large.

## 6. Calidad y trazabilidad

- **NaN:** 16 imágenes procesadas; 153 472 valores en canal de índice 12 y 150 656 en índice 13. No se detectaron infinitos. La estrategia de tratamiento está pendiente.
- **Anotaciones originales faltantes:** point_0024, point_0039, point_0075, point_0079, point_0191, point_0465, point_0605, point_0631, point_0786, point_0847 y point_0966.
- **Imágenes idénticas:** point_0523/point_0524 y point_0588/point_0598. Sus máscaras tienen diferentes cantidades de píxeles positivos: 21/28 y 775/779, respectivamente.
- **Otras coordenadas duplicadas:** point_0010/point_0423 y point_0014/point_0631. No se ha completado una búsqueda de todos los solapamientos geográficos.
- **Correspondencia espacial:** las 827 máscaras coinciden con `red_png[1:129,1:129].T`; 822 imágenes coinciden exactamente con `tif_stack[10:138,10:138,:].transpose(1,0,2)`. Las 16 restantes requieren revisión de NoData y valores.
- **Posible desplazamiento:** si `mask_area=[10,10,140,140]` ubica el PNG con origen 10 en el TIFF, el recorte de máscara queda desplazado un píxel en ambos ejes respecto a la imagen. Es una inferencia condicionada, no una corrección autorizada ni aplicada. No se encontró el exportador final en el repositorio.

## 7. Validación documental y particiones

La hoja `data` del Excel contiene 1001 puntos. Los 838 con `Final=si` son exactamente los puntos locales. Todos disponen de Grupo (1–16) e intervalos en Fechas. No se ha documentado si Grupo identifica evento, escena, lote o zona. Fechas no es la fecha exacta de adquisición.

Entre los 838, `label=si` aparece 603 veces y `valid=si` 550 veces; estos campos no reproducen la clasificación derivada de las máscaras. De los 30 negativos HDF5, 29 tienen label/valid=no; point_0905 tiene ambos en si y observación agri. Los 11 PNG faltantes también tienen Final=si. Ninguno de estos campos demuestra por sí solo exclusión o aprobación experta definitiva.

El código original usa train/val aleatorio 80/20 con seed 42. No se encontraron manifiestos oficiales persistidos de L4S-PE ni un test independiente identificable. Para la tesis se crearán particiones después de revisar geografía, duplicados, trazabilidad y viabilidad de los negativos. No se reutilizará automáticamente random_split ni se tratará Grupo como partición oficial.

## 8. Texto base para metodología

> Se realizó una caracterización de la copia local del conjunto L4S-PE, originalmente preparado para segmentación de deslizamientos. Se identificaron 838 pares de imágenes multicanal y máscaras binarias en formato HDF5. Las imágenes procesadas presentan dimensiones de 128×128 píxeles y 14 canales. Para adaptar la colección a un esquema de supervisión a nivel de imagen, se derivó una etiqueta binaria a partir de la presencia de píxeles anotados como deslizamiento, obteniéndose provisionalmente 808 muestras positivas y 30 negativas. Las máscaras se reservan para la derivación de etiquetas y la evaluación espacial posterior, sin utilizarlas como objetivos de la función de pérdida del clasificador. La inspección identificó valores no finitos, duplicados y diferencias documentales en la procedencia y orden de los canales; por ello, la selección definitiva de muestras, el preprocesamiento y las particiones se establecerán una vez resueltas las incidencias relevantes.

Este párrafo describe el estado actual, no experimentos completados. Añadir la referencia bibliográfica definitiva del dataset antes de incorporarlo a la versión final de la tesis.

## 9. Pendientes priorizados

| Prioridad | Pregunta | Evidencia necesaria / acción |
|---|---|---|
| 1 | ¿Qué versión generó los HDF5 finales? | Exportador y listado de bandas del autor; resolver SRTM/ALOS y B9/B10 |
| 1 | ¿Imagen y máscara representan el mismo recorte? | Semántica de IRIS y script final; comprobar numéricamente |
| 1 | ¿Los 30 negativos son anotaciones completas? | Significado e historial de label/valid/Final; revisión del caso point_0905 |
| 2 | ¿Por qué faltan 11 PNG? | Recuperación o explicación de su procedencia |
| 2 | ¿Qué significan Grupo y Fechas? | Diccionario; contrastar huellas y duplicados antes de dividir |
| 2 | ¿Cómo se transformaron valores y NoData? | Comparación TIFF/HDF5 completa; definir estrategia reproducible |
| 3 | ¿Qué fenómenos incluye landslide? | Definición de clase y referencia final para justificar su relación con huaycos |

## 10. Evidencia y reproducción

- [Metadatos por muestra](../../metadata/roy.csv).
- [Auditoría de archivos y estadísticas](../../results/audits/roy_inspection/README.md).
- [Estadísticas numéricas](../../results/audits/roy_inspection/summary.json).
- [Comprobación de orientación y recortes](../../results/audits/roy_inspection/alignment_summary.json).
- [Revisión del repositorio y documentos](../../results/audits/roy_repository_review/README.md).
- [Cruce del Excel con muestras locales](../../results/audits/roy_repository_review/validation_join.csv).
- `references/lanslide4sat_pl/README.md` (recurso original externo, no distribuido), `references/lanslide4sat_pl/resumen.odt` (recurso original externo, no distribuido) y `references/lanslide4sat_pl/data_validation.xlsx` (recurso original externo, no distribuido).

Desde la raíz del proyecto, los scripts `scripts/inspect_roy.py` y `scripts/check_roy_alignment.py` reproducen la auditoría de imágenes/máscaras y sobrescriben sus informes derivados. No modifican las fuentes. La extracción y cruce del Excel de la revisión previa se conserva como resultado, pero todavía no dispone de un script dedicado de reproducción.
