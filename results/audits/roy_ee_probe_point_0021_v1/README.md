# Primer sondeo Earth Engine: point_0021

Fecha: 2026-10-03. Se recibió `roy_band_probe_point_0021.csv` del usuario,
con 25 posiciones únicas y una escena consultada de Sentinel-2B del 25/06/2019.
Se verificaron los hashes de los TIFF fuente y los 150 valores de referencia
del CSV contra esos TIFF y contra `roy_ee_probe_v1/reference.json`.

Producto: `S2B_MSIL1C_20190625T151709_N0500_R125_T18LVN_20230508T061432`.
Colección: `COPERNICUS/S2_HARMONIZED`. Baseline: 05.00. Ventana consultada:
24/06/2019 a 26/06/2019 inclusivos. Los 25 puntos esperados están presentes.

## Comparación medida

RMSE y MAE se calculan directamente sobre DN armonizados del CSV y valores
crudos del TIFF, sin normalizar ni ajustar desplazamientos/escalas para mejorar
la coincidencia. Error firmado = candidato Earth Engine − referencia Roy.

| Comparación | RMSE | MAE | Correlación Pearson | Píxeles idénticos |
|---|---:|---:|---:|---:|
| Roy B2 / Sentinel B2 | 12,159 | 10,16 | 0,99318 | 2/25 |
| Roy B3 / Sentinel B3 | 19,636 | 15,96 | 0,98318 | 2/25 |
| Roy B4 / Sentinel B4 | 16,925 | 14,12 | 0,99146 | 0/25 |
| Roy B8 / Sentinel B8 | 143,028 | 121,80 | 0,98021 | 0/25 |
| Roy B9 / Sentinel B9 | **20,650** | **17,80** | **0,99920** | 1/25 |
| Roy B9 / Sentinel B8A | 1334,043 | 1264,92 | 0,84795 | 0/25 |
| Roy B10 / Sentinel B10 | **0,283** | **0,08** | **0,99873** | **23/25** |
| Roy B10 / Sentinel B9 | 1316,826 | 1283,28 | 0,71407 | 0/25 |

B10 coincide exactamente en 23 posiciones; las dos restantes difieren en +1 DN.
La evidencia favorece con mucha fuerza B9.tif=B9 y B10.tif=B10 para este punto,
frente a la interpretación B8A/B9 de la configuración IRIS.
Esto es coherente con la selección del script R y la estructura espacial observada
en la auditoría de los 838 puntos.

Las bandas de control respaldan correspondencia de la escena por su estructura
y valores, pero no son idénticas a la representación original. La referencia EE
tiene baseline 05.00. Copernicus documenta cambios radiométricos y geométricos en
el [reprocesamiento del archivo histórico](https://sentiwiki.copernicus.eu/web/s2-processing).
Eso es una explicación posible de parte de los residuos, **no una atribución
demostrada en este análisis**. No se confirmó el baseline original ni se estimó
un desplazamiento espacial a partir del CSV.

## Alcance y siguiente paso

- Se analizaron los 25 píxeles de un punto; aún no se comparó el parche completo.
- El CSV contiene una escena. Por sí solo no prueba que la consulta haya incluido
  todas las posibles adquisiciones o versiones del producto.
- No se generaliza esta coincidencia como certificación de todas las muestras.
- No se cambia `band_mapping_status` ni se habilita el entrenamiento conjunto.
- La identidad/fuente topográfica no se estudió con este sondeo.

Continuar el mismo script EE con `POINT_INDEX = 1`, `2` y `3`, por separado, y
conservar los CSV de point_0002, point_0023 y point_0502. Esto amplía la evidencia
a otros grupos y fechas sin cambiar el procedimiento ni usar rendimiento de modelos.

## Archivos y reproducción

`input.csv` es copia exacta del archivo recibido. `analysis.json` conserva hashes,
metadatos, cobertura y todas las métricas sin redondear. Repetir en una salida nueva:

```bash
.venv/bin/python scripts/analyze_roy_ee_probe.py \
  --csv results/audits/roy_ee_probe_point_0021_v1/input.csv \
  --output results/audits/roy_ee_probe_point_0021_v2
```

Se comprobaron los cálculos con un ejemplo numérico conocido, correlación
indefinida para bandas constantes, rechazo de referencias alteradas y filas
duplicadas, y señalización de cobertura incompleta. No se ejecutó entrenamiento.
