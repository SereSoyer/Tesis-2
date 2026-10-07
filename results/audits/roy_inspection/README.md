# Inspección local de Roy / L4S-PE

Fecha: 2026-09-25. Auditoría de los archivos disponibles, sin modificar fuentes ni ejecutar entrenamiento. La procedencia se describe según el README y los scripts locales; no se verificó externamente el artículo ni la validación experta.

## Inventario verificado

- `data/roy/l4s/image_ls4pe`: 838 HDF5, clave `img`, forma (128,128,14), float64.
- `data/roy/l4s/mask_ls4pe`: 838 HDF5, clave `mask`, forma (128,128), valores 0 y 1. Emparejamiento por identificador, sin faltantes.
- `landslide_ok`: 838 HDF5 originales (14,150,150), rango global por archivo [0,1]; 11 732 TIFF (14 por muestra), todos 150×150; 827 PNG RGB de anotación (130,130,3). Los metadatos declaran Sentinel2/SRTM y resolución 10; todos los TIFF tienen paso de píxel 10×10. Esto no establece la resolución nativa de cada sensor.
- El nombre `l4s` no identifica aquí la colección internacional: sus subdirectorios contienen los 838 puntos de L4S-PE.

## Etiquetas derivadas, provisionales

808 positivas (96.42%) y 30 negativas (3.58%), usando presencia de píxeles exactamente iguales a 1. No se usan máscaras como objetivos de entrenamiento. Un clasificador que siempre prediga positivo alcanzaría 96.42% de accuracy en esta colección completa; accuracy sola sería insuficiente.

Entre las positivas, la fracción afectada tiene mínimo 0.116% (19 píxeles), mediana 2.161%, media 3.040% y máximo 20.349%. P25=1.152%, P33=1.434%, P66=3.312%, P75=4.044%. No se establecen categorías de tamaño todavía.

## Bandas y valores

Los índices de array son cero-based. Los canales 0 a 11 tienen valores finitos globales entre 0 y 12937 (rangos individuales en summary.json). El canal 12 tiene rango finito [0,1.4837092161], y el 13 [435,4128]. Son compatibles respectivamente con pendiente en radianes y elevación; esta interpretación coincide con el orden explícito del flujo Python de ALOS, pero contradice el orden elevación/pendiente del exportador R y los nombres de las dos últimas capas en landslides.json.

Conflictos pendientes:

- get_data_utils.R selecciona B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12; landslides.json describe las posiciones 9 y 10 como B8A y B9. No se certifica su identidad espectral solo por sus rangos.
- IRIS declara B13=elevation y B14=slope; los valores observados sugieren el orden opuesto.
- Hay 153472 valores no finitos en canal 12 y 150656 en canal 13, repartidos en 16 imágenes. La segunda auditoría confirma que son NaN; no hay infinitos ni valores inferiores a -1e30 en los HDF5 procesados.
- Los TIFF declaran NoData aproximadamente -3.4e38. No se debe confundir ausencia de NaN con ausencia de NoData en los originales.
- Los HDF5 originales están en [0,1], pero los procesados no. No deben intercambiarse directamente.

## Anotaciones y correspondencia espacial

IRIS define verde=(0,255,0) como Clear, rojo=(255,0,0) como Landslide y azul=(50,50,255) como No data. Se encontraron 7 píxeles azules, todos en point_0122. La máscara binaria no preserva una clase NoData separada.

Faltan PNG originales para 11 muestras, todas con máscara HDF5 positiva: point_0024, point_0039, point_0075, point_0079, point_0191, point_0465, point_0605, point_0631, point_0786, point_0847 y point_0966. No se asume que esas máscaras sean inválidas; su trazabilidad local está incompleta.

Comprobaciones exactas (slices Python, extremo final excluido):

- Las 827 máscaras disponibles cumplen `mask == red_png[1:129,1:129].T`.
- 822/838 imágenes cumplen `img == tif_stack[10:138,10:138,:].transpose(1,0,2)`. La correspondencia exacta de las otras 16 queda pendiente; no debe interpretarse como corrupción sin estudiar el tratamiento de NoData.
- IRIS declara `mask_area=[10,10,140,140]`. Si este campo sitúa el PNG en la imagen original con ese origen, las máscaras procesadas representan [11:139,11:139], mientras las imágenes representan [10:138,10:138]. Esto implica un posible desplazamiento de un píxel en ambos ejes, condicionado a confirmar esa semántica de IRIS. No se corrigió automáticamente.
- summary.json contiene la primera prueba de recortes sin transposición. Sus fallos no indican por sí mismos archivos erróneos; consultar alignment_summary.json para la prueba posterior de orientación.

## Duplicados y particiones

Dos grupos tienen arrays de imagen idénticos por SHA-256:

- point_0523 / point_0524: máscaras con 21 y 28 píxeles positivos.
- point_0588 / point_0598: máscaras con 775 y 779 píxeles positivos.

También comparten coordenadas point_0010 / point_0423 y point_0014 / point_0631. Ausencia de duplicación exacta no descarta solapamiento espacial. Antes de generar splits se deben revisar huellas geográficas y agrupar puntos relacionados. No se generaron particiones.

## Repositorio original

`lanslide4sat_pl` implementa segmentación supervisada con máscaras y U-Net; no constituye el pipeline de clasificación de esta tesis. Su configuración espera `data/TrainData/img` y `mask`, estructura distinta de los directorios actuales. Empareja listas ordenadas, sin validación explícita de identificadores.

La configuración activa usa índices [1,2,3,7,12,13]. El comentario que propone [1,...,14] para 14 canales sería inválido en NumPy: los índices válidos son 0,...,13. Los factores ls4_norm tienen claves 1,...,14 mientras el acceso al array usa directamente channel; esa convención requiere revisión antes de reutilizar la normalización de L4S. La división original es aleatoria train/val, sin test independiente ni agrupación espacial explícita.

## Resultado y siguiente paso

El inventario está auditado y los metadatos provisionales están en `metadata/roy.csv`. Antes del entrenamiento, resolver identidad de bandas y correspondencia espacial, definir tratamiento de NaN/NoData, aclarar las 11 máscaras sin PNG y diseñar particiones considerando duplicados, proximidad geográfica y escasez de negativos. La documentación local atribuye la anotación a IRIS, pero no identifica aquí quién validó cada máscara ni confirma que todos los eventos correspondan específicamente a huaycos.

Reproducción desde la raíz del proyecto:

```bash
.venv/bin/python scripts/inspect_roy.py
.venv/bin/python scripts/check_roy_alignment.py
```

Dependencias utilizadas: numpy 2.5.3, h5py 3.16.0, Pillow 12.3.0. No se necesita semilla: la auditoría no usa aleatoriedad. Los informes se sobrescriben al repetirla.
