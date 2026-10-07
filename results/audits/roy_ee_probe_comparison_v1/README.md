# Comparación de los cuatro sondeos de Earth Engine

Fecha: 2026-10-03. Se analizaron los cuatro CSV entregados por el usuario con el
mismo procedimiento: referencias verificadas contra TIFF, cobertura por escena
y comparación de identidades después de ordenar escenas por B2/B3/B4/B8.

**Conclusión:** hay respaldo empírico sólido para el mapeo posicional
`B9.tif → Sentinel-2 B9`, `B10.tif → Sentinel-2 B10`. La coincidencia exacta de
point_0502 añade evidencia directa a la selección del script R y a la estructura
espacial observada en los 838 puntos. La interpretación B8A/B9 de la configuración
IRIS no explica los valores de ese sondeo.

## Alcance de la comprobación

- Cuatro puntos, grupos del autor 1, 8, 13 y 16; todos del train propuesto.
- 100 posiciones únicas, 25 por punto. Los valores se repiten al comparar varias
  adquisiciones; las 1.700 filas CSV no son 1.700 observaciones independientes.
- 68 pares punto/escena: 1, 30, 19 y 18 en point_0021, 0002, 0023 y 0502.
- Todas las escenas devueltas tienen 25 posiciones y referencias locales correctas.
- Los archivos originales, particiones y configuraciones de entrenamiento no cambian.

## Coincidencia exacta: point_0502

Escena: `20200609T151709_20200609T151945_T18LVP`.
Producto: `S2B_MSIL1C_20200609T151709_N0209_R125_T18LVP_20200609T201206`.
Fecha: 09/06/2020. Baseline: **02.09**.

Los **150 valores** de B2/B3/B4/B8/B9/B10 (25 por banda) coinciden exactamente con
los TIFF de Roy. La selección se hizo mediante bandas de control y tiene RMSE=0;
la siguiente escena tiene RMSE de control=71,691. Las bandas dudosas no se usaron
para escoger esta escena.

| Hipótesis sobre point_0502 | RMSE |
|---|---:|
| B9.tif es B9 | **0** |
| B9.tif es B8A | 2310,167 |
| B10.tif es B10 | **0** |
| B10.tif es B9 | 952,484 |

La igualdad se comprobó en los píxeles muestreados, no en los 22.500 píxeles del
parche ni en toda la colección. Sí identifica una correspondencia exacta en esas
posiciones con un producto Sentinel concreto.

## Otros puntos y ambigüedad de adquisición

Esta tabla conserva **la primera escena por RMSE de control**, según la regla
preparada antes de recibir los CSV. No se cambia el criterio para escoger una
escena que dé mejores resultados en B9/B10.

| Punto | Primera fecha por controles | RMSE controles | RMSE B9/B9 | RMSE B9/B8A | RMSE B10/B10 | RMSE B10/B9 |
|---|---|---:|---:|---:|---:|---:|
| point_0021 | 25/06/2019 | 72,933 | 20,650 | 1334,043 | 0,283 | 1316,826 |
| point_0002 | 19/07/2020 | 57,912 | 164,444 | 1146,551 | 4,604 | 795,577 |
| point_0023 | 10/07/2019 | 119,608 | 345,235 | 2212,183 | 1,865 | 1094,808 |
| point_0502 | 09/06/2020 | **0** | **0** | 2310,167 | **0** | 952,484 |

- **point_0021:** mantiene el resultado anterior: B10 coincide en 23/25 píxeles,
  con dos diferencias de +1; B9/B9 tiene correlación 0,9992. La escena es compatible,
  pero la representación de controles no es idéntica.
- **point_0002:** las dos primeras candidatas tienen errores de control 57,912 y
  61,558, y la tercera 62,225. No hay identificación inequívoca de adquisición.
  B10 en la primera candidata tiene correlación baja (0,139), por lo que su error
  absoluto pequeño no prueba una coincidencia temporal. La escena del 09/06/2020,
  quinta por controles (RMSE=67,693), tiene B9/B9 RMSE=13,855 y B10/B10 RMSE=0,4
  (21/25 idénticos). Es una pista adicional posterior, no una nueva regla de selección.
- **point_0023:** las dos primeras candidatas tienen errores de control 119,608 y
  122,066. La segunda, del 05/06/2019, tiene B9/B9 RMSE=10,274 y B10/B10 RMSE=0,2
  (24/25 idénticos), mientras la primera tiene correlación B10/B10 baja (0,074).
  La adquisición exacta queda abierta; tampoco se declara que todas las bandas
  tengan coincidencia exacta en esta muestra.

La mezcla de versiones de procesamiento y las diferencias espaciales/radiométricas
son aspectos a investigar; estos CSV no demuestran la causa concreta de los residuos.
La evidencia de identidad espectral y la identificación exacta de cada adquisición
tienen alcances diferentes y se registran por separado.

## Decisión documental y pendientes

Se respalda el orden B1..B12 del flujo R para las posiciones dudosas B9/B10 y se
recomienda conservarlo en el esquema conjunto de 14 canales. No se intercambian
archivos ni se modifica la radiometría basándose en este informe.

Los resultados no certifican individualmente los 838 archivos ni la igualdad
completa de parches. Esas limitaciones deben constar en la tesis. La incertidumbre
principal pendiente para cerrar las 14 entradas es el orden y la fuente de
**B13/B14 (pendiente/elevación)**, que estos sondeos no han evaluado.
El entrenamiento conjunto sigue deshabilitado mientras se cierra esa definición.

## Evidencia reproducible

`comparison.csv` contiene la tabla sin redondear y los hashes de los informes y
CSV fuente. `summary.json` conserva alcance, mapeo respaldado y limitaciones.
Cada carpeta `roy_ee_probe_point_XXXX_v1` contiene su `input.csv` íntegro y
`analysis.json` con **todas** las escenas y métricas, no solo la primera.

El mismo analizador probado en el primer punto se aplicó a los tres archivos
nuevos: `scripts/analyze_roy_ee_probe.py`. Verificó integridad de referencias,
ausencia de duplicados, valores finitos, ventanas/colección y cobertura; no se
entrenó ningún modelo ni se consultaron máscaras.
