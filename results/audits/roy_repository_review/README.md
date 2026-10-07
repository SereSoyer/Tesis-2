# Revisión del repositorio local lanslide4sat_pl

Fecha: 2026-09-25. Referencia Git local: f846da4. Revisión documental y estática, sin ejecutar entrenamientos ni operaciones de Earth Engine/GCS. Se leyeron README, resumen.odt (texto completo extraído), las tres hojas de data_validation.xlsx, config.yaml, los 9 scripts Python/R, código y markdown de los tres notebooks, .gitignore y LICENSE. Se inspeccionaron visualmente los 12 PNG de graph y la estructura XML del SVG. No se certifica que los resultados publicados sean reproducibles con esta copia: faltan datos organizados como espera el código, checkpoints y dependencias declaradas.

## 1. Excel: cruce por identificador

La hoja `data` contiene 1001 puntos distintos, numerados 1–1001, además de filas vacías. El ODT describe una selección inicial de 1000 ROIs: existe una discrepancia de inventario de un punto que no afecta a la coincidencia de los 838 seleccionados.

Campos: point, label, valid, Valida con la imagen, clouds, fecha, obs, Final, Fechas, Grupo. `fecha` está vacío; `Fechas` contiene intervalos, no identificadores de escena ni fechas exactas de adquisición.

Estados Final sobre 1001 registros:

| Estado | Cantidad |
|---|---:|
| si | 838 |
| Cambiar fecha | 154 |
| CAMBIAR DEM | 7 |
| REVISAR | 2 |

El conjunto `Final=si` coincide exactamente con los identificadores locales de `l4s` y los metadatos auditados: no sobra ni falta ninguno. Los 838 seleccionados tienen Grupo (valores 1–16) y Fechas. No hay diccionario que defina Grupo como evento, escena o partición; varios grupos contienen distintas ventanas temporales. No usarlo como split oficial sin más evidencia.

Entre los 838 puntos: label=si en 603 y no en 235; valid=si en 550 y no en 288. Estas columnas no son equivalentes a la etiqueta binaria derivada de las máscaras actuales (808/30). Pueden reflejar etapas de revisión, pero el orden temporal no está documentado.

Los 30 negativos HDF5 tienen Final=si. En 29, label y valid dicen no; point_0905 dice si en ambos y tiene observación `agri`. Otras observaciones incluyen agricultura/agri, fecha, falta imagen, sin imagen y revisar. El Excel aporta contexto, pero no permite certificar automáticamente que los 30 sean negativos revisados y válidos para la tarea actual.

Los 11 puntos sin PNG también tienen Final=si y máscara HDF5 positiva. Siete tienen label/valid=si, cuatro tienen no: point_0024, point_0191, point_0605 y point_0847. No aparece una explicación explícita de por qué faltan sus PNG. No hay evidencia de exclusión oficial de esos 11.

`Hoja 2`: 840 filas con ventanas de fechas y sus seriales Excel; columna A con 567 entradas y 565 valores únicos, sin encabezado explícito. No es un manifiesto inequívoco de particiones.
`Hoja1`: lista auxiliar con encabezado img(point), 499 entradas no vacías y repetición; no contiene etiquetas train/val/test. No se encontraron fórmulas ni comentarios XML en el libro. Los valores se extrajeron directamente del XLSX sin recalcular ni alterar el original.

`validation_join.csv` conserva todos los puntos del Excel, su presencia local y las etiquetas HDF5/PNG disponibles. No reemplaza ni corrige las etiquetas de entrenamiento.

## 2. Qué declara resumen.odt

Declara Sentinel-2 + ALOS PALSAR, área de estudio Junín/Pasco/Huánuco, selección visual inicial de 1000 ROIs y reducción a 838 después de control de calidad por nubes, ruido o características distintas de deslizamientos. Atribuye el etiquetado a una versión modificada de IRIS basada en el flujo CloudSen12. Describe exportación HDF y clasificación espacial binaria, entrenamiento supervisado de segmentación, combinación L4S/L4S-PE y reparto aleatorio 80/20 con semilla.

Esto es evidencia documental del repositorio, no verificación independiente del artículo o de cada anotación. No identifica aquí un validador experto por muestra.

Inconsistencias del ODT que no deben copiarse a la tesis:

- Llama 150×150 y 128×128 «metros» en algunos párrafos. Los archivos son píxeles; con malla de 10 m, 150 píxeles representan aproximadamente 1500 m.
- Alterna 80/20 con 80/10; train.py y config.yaml implementan 80/20.
- Habla de 14 bandas espectrales y de seis canales de resolución nativa 10 m, aunque dos canales son topográficos.
- Reporta 1.8% de ocupación L4S-PE (24.71 km²). La copia actual produce 2.8734% sobre los 827 PNG completos y 2.9315% sobre los 838 HDF5 binarios. Debe aclararse versión/denominador; no sustituir el conteo local por la cifra narrativa.
- Indica normalización [0,1], pero la configuración activa del cargador tiene normalize=False y los HDF5 procesados no están normalizados a ese rango.

## 3. Fuentes topográficas y bandas: existen varios flujos

`get_data.R` llama a `download_srtm`; `get_data_utils.R` usa USGS/SRTMGL1_003, agrega elevación seguida de pendiente y convierte la pendiente a radianes. Sus metadatos dicen Sentinel2/SRTM. Selecciona COPERNICUS/S2 con B1..B12 (sin B8A), fechas 2018-01-01 a 2020-01-30. El código usa nombres point con padding de 3 dígitos, mientras los archivos locales tienen 4; también mezcla rutas landslide y l4s. No reconstruye exactamente la versión entregada.

`download_img_gee.py`, `upload_model_gee.ipynb` y `upload_model_gee_pl.ipynb` sí contienen uso explícito de archivos AP_* desde GCS, denominados ALOS PALSAR, con `B13=slope` y `B14=DEM`. Son flujos de descarga/inferencia para regiones, no una demostración de que generaron cada TIFF local. La pendiente en esas rutas no tiene la conversión a radianes del R.

`upload_model_gee_pl.ipynb` usa además COPERNICUS/S2_SR y selecciona B8A y B9 en las posiciones 9 y 10, coincidiendo con la configuración de IRIS, mientras el R selecciona B9/B10. Por tanto, ambas convenciones aparecen realmente en el repo; no puede certificarse identidad de esos canales por nombre de archivo o rango.

Corrección al informe inicial: el R NO confirma pendiente antes de elevación. La evidencia explícita de ese orden procede del código Python/notebooks de ALOS y de los rangos observados, en conflicto con el R y con nombres IRIS. Se requiere trazabilidad del exportador de esta versión.

## 4. Normalización, canales y NoData

- Exportador R `export_h5`: min–max por capa del parche, aplicado al raster completo, incluyendo capas topográficas; no documenta protección frente a banda constante. Falta demostrar numéricamente equivalencia con todos los HDF5 originales.
- Dataset PyTorch: lee claves img/mask de HDF5 128×128; transforma HWC a CHW; reemplaza NaN de imágenes por 1e-6; selecciona [1,2,3,7,12,13] (B2,B3,B4,B8,B13,B14 por posición). normalize=False en config. Si se activa, el min–max se calcula sobre todo el tensor seleccionado, no por banda.
- La rama ls4_norm se activa si la ruta COMPLETA contiene `image`. Con `data/roy/l4s/image_ls4pe/...` se activaría también para Roy si se adaptase la lectura conservando esa ruta. Debe sustituirse por procedencia explícita en un futuro cargador.
- La lista comentada de 14 canales usa 1..14, incompatible con índices NumPy 0..13. También debe revisarse correspondencia entre los factores con claves 1..14 y los índices directos del array.
- El notebook `upload_model_gee_pl` usa otra combinación de seis entradas: B2,B3,B4,B13,B14,NDVI, con min–max por banda. No debe confundirse con la configuración PyTorch que incluye B8.
- Inferencia GeoTIFF: relleno de NaN por vecino más cercano, distinto del reemplazo usado en entrenamiento.
- Las pérdidas consumen la máscara binaria completa; no hay máscara de validez NoData ni exclusión espacial explícita. El pipeline actual no conserva una tercera clase NoData como objetivo.

## 5. Entrenamiento y evaluación

U-Net propio o smp.Unet con encoder configurable. mit_b0 se utiliza como encoder de U-Net, no como un clasificador ViT comparable al propuesto en la tesis. La rama mit_b0 aplica una convolución a 3 canales mientras el constructor mantiene in_channels configurable: requiere revisión si se activa con 6/14.

La pérdida suma BCE y Dice. El código usa BCELoss(weight=tensor([5])); ese peso único no expresa un peso diferente exclusivo de la clase positiva, aunque la narrativa lo describe como ponderación de desbalance. Dice aplana el lote completo. No se ejecutó entrenamiento para validar el comportamiento numérico de versiones concretas de librerías.

train.py aplica random_split 80/20, seed 42 desde utils.py. No guarda un manifiesto de IDs ni agrupa por Grupo, escena, coordenadas o duplicados. No se encontró un split oficial persistido de L4S-PE. El comentario sobre ValidData en eval_pl no demuestra una partición oficial peruana.

Los notebooks son exploratorios: variables/imports ausentes (por ejemplo val_loader comentado, ee_utils no incluido, Sequence/tf/norm_dicc sin definición suficiente en el notebook de upload), rutas absolutas del autor, checkpoints no presentes y umbrales sobre logits sin sigmoid en algunas celdas. eval_img_gee.py sí usa sigmoid. No constituyen un protocolo único reproducible ejecutable de principio a fin.

## 6. Recorte y contenido faltante

No se encontró en los scripts o en el código de las celdas el exportador manual.png → mask_ls4pe ni el recorte que genera los 838 HDF5 finales. Los split_in_patches y cortes a 128 de inferencia GEE dividen escenas grandes: no prueban la transformación de las anotaciones IRIS de 150/130 a 128.

README menciona requirements.txt y ee_utils.py, ausentes en esta copia. .gitignore excluye datos, checkpoints, modelos, logs y directorios de validación/prueba; por eso no se pueden reconstruir los experimentos solo con los archivos presentes. No se descargó información externa ni se ejecutaron las celdas con exportaciones.

Las figuras incluyen diagramas de segmentación, ejemplo L4S con RGB/slope/DEM/label, registro temporal de ROIs, curvas de entrenamiento y comparaciones visuales. results.png distingue normalización por conjunto y por imagen, evidencia adicional de varios experimentos, sin manifiestos asociados. Las figuras no resuelven la procedencia de cada muestra ni la alineación exacta.

## Conclusión para la tesis

Queda resuelta la pertenencia de los 838 puntos a la selección Final=si y se recuperan grupos/ventanas temporales. Se confirma que el resumen sí sostiene ALOS y que existen implementaciones de ALOS, pero conviven con SRTM y con dos selecciones espectrales. No se certifican todavía fuente del DEM local, orden espectral completo, validez definitiva de negativos ni alineación; no se cambian etiquetas, canales ni splits automáticamente.
