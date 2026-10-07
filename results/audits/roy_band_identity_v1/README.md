# Evidencia independiente para B9.tif y B10.tif de Roy

Fecha: 2026-10-03. Auditoría de solo lectura, sin modificar imágenes, particiones,
configuraciones ni admisión. No se usaron máscaras, resultados de modelos ni IoU.

## Hallazgo

Se analizaron 5.028 TIFF: B1, B7, B8, B9, B10 y B11 en los 838 puntos.
En cada eje se contaron los cambios entre vecinos válidos y se comprobó si todos
caen en líneas de una cuadrícula de período seis, eligiendo su fase. En una malla
de salida de 10 m, ese patrón es compatible con una fuente de 60 m remuestreada
por vecino más próximo. No demuestra por sí solo la longitud de onda.

| Archivo | Media de valores crudos (DN) | Cuadrícula de 6 píxeles exacta, ambos ejes |
|---|---:|---:|
| B1.tif, control documental de 60 m | 933,40 | 838/838 |
| B7.tif, control documental de 20 m | 2312,39 | 0/838 |
| B8.tif, control documental de 10 m | 2315,98 | 0/838 |
| B9.tif, identidad discutida | 861,37 | 838/838 |
| B10.tif, identidad discutida | 12,80 | 837/838 |
| B11.tif, control documental de 20 m | 1261,35 | 0/838 |

La excepción B10 de point_0918 es constante, con valor 1 en todos los píxeles:
no permite estimar la cuadrícula, y no contradice el patrón de los otros puntos.
Las etiquetas GDAL de descripción de los 5.028 archivos dicen solamente `layer`.

Según la [especificación oficial del catálogo Sentinel-2 L1C de Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2),
B8A es infrarrojo cercano estrecho de 20 m; B9 es la banda de vapor de agua de
60 m y B10 la banda cirrus de 60 m. El [producto L2A](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
no contiene B10.

`references/lanslide4sat_pl/scripts/get_data_utils.R`, función `download_sen2`,
selecciona explícitamente B1..B8,B9,B10,B11,B12 de `COPERNICUS/S2`.
`scripts/get_data.R` exporta esas capas en orden como B1.tif..B12.tif.
La configuración IRIS las denomina B8A/B9, en contradicción con ese flujo.

**Inferencia:** código de descarga/exportación, estructura de muestreo y valores
favorecen que B9.tif sea B9 y B10.tif sea B10. Es evidencia indirecta fuerte; no
se recuperó una escena original que permita certificarlo por comparación de píxeles.
Un procesado adicional podría haber cambiado muestreo o escala. No se cambia
`band_mapping_status` ni se habilita el entrenamiento con esta auditoría.

## Cómo resolverlo sin consultar al autor

1. Usar coordenadas por punto y ventanas `Fechas` del Excel para buscar escenas
   Sentinel-2 L1C. Los metadatos locales no contienen un identificador Sentinel
   original: `scene_id=point_XXXX` es un identificador interno.
2. Identificar la adquisición correcta comparando varias bandas no discutidas
   (B2/B3/B4/B8) y estructura espacial; la cercanía temporal no basta.
3. Obtener B8A, B9 y B10 de esa adquisición; llevarlas a la misma proyección,
   origen y malla que los TIFF originales. Comparar TIFF completos antes de
   normalización. Usar un método de remuestreo documentado acorde a la evidencia.
4. Comparar cada candidato mediante error absoluto/RMSE, correlación, porcentaje
   de coincidencias y ajuste radiométrico conocido. Una correlación alta aislada
   no identifica la banda. Respetar la máscara de validez del producto fuente.
5. Repetir inicialmente en muestras de distintos grupos y ventanas temporales,
   y extender la trazabilidad a las muestras admitidas. No extrapolar una sola
   coincidencia como prueba de las 838 imágenes.
6. Registrar producto, fecha, nivel de procesamiento, banda, transformación y
   evidencia por muestra. Si no puede identificarse inequívocamente, mantener
   el resultado como inferencia documentada.

Se consultó sin credenciales el [catálogo STAC de Copernicus](https://documentation.dataspace.copernicus.eu/APIs/STAC.html)
para point_0001 (lon=-75.7666, lat=-8.9017; agosto–octubre de 2019).
`point_0001_l1c_candidates.json` conserva URL, parámetros y respuesta. De los 20
elementos devueltos, 18 corresponden a MGRS-18LVR; los otros dos son MGRS-01LAL
y no deben aceptarse por el simple resultado de una consulta bbox. Verificar
huella y proyección es parte necesaria de la selección. Ninguna escena ha sido
identificada aún como origen del TIFF.

Los productos devueltos son versiones reprocesadas N0500. Sus assets declaran
escala 0,0001 y offset −0,1: comparar DN directamente con una exportación antigua
puede producir diferencias de 1000, además de cambios de procesamiento. Debe
usarse el metadato del producto, no ajustar una escala libre hasta forzar igualdad.
El catálogo es accesible; las descargas enlazadas declaran autenticación CDSE.
No se descargaron imágenes ni se utilizaron credenciales.

## Reproducción

`scripts/audit_roy_band_identity.py` genera `statistics.csv` y `summary.json` en
una salida nueva, con hashes de todos los TIFF leídos. Ejemplo para repetir:

```bash
.venv/bin/python scripts/audit_roy_band_identity.py --output results/audits/roy_band_identity_v2
```

La detección de cuadrícula se comprobó con arrays sintéticos de períodos 6 y 2,
incluyendo desplazamiento de fase. El inventario actual cubre los 838 puntos.
