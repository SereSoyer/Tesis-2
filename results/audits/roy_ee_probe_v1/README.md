# Consulta preparada para Earth Engine Code Editor

Fecha: 2026-10-03. Estado: **los cuatro puntos fueron ejecutados y analizados**.
point_0502 coincide exactamente en los 150 valores de las seis bandas consultadas;
los otros puntos aportan evidencia adicional, con incertidumbre de adquisición en
point_0002 y point_0023. Ver [comparación conjunta](../roy_ee_probe_comparison_v1/README.md).
No se han realizado consultas EE autenticadas desde este proyecto ni se necesitan
contraseñas/claves compartidas.

## Ejecutar el primer punto

1. Abrir https://code.earthengine.google.com/ con la cuenta propia y un proyecto
   habilitado para Earth Engine. El editor gestiona la autenticación de la sesión:
   [documentación oficial](https://developers.google.com/earth-engine/guides/auth).
2. Crear un script y pegar **todo** `roy_band_probe.js` de esta carpeta.
3. Conservar `POINT_INDEX = 0` y pulsar **Run**. La consulta inicial corresponde
   a point_0021, grupo 16, del 24 al 26 de junio de 2019 inclusive.
4. Esperar el resultado en **Console**, expandir las candidatas y abrir el enlace
   **DESCARGAR CSV**. Guardar el archivo descargado para analizarlo localmente.
   Si aparece un error o `SIN ESCENAS`, conservar ese mensaje: no ampliar fechas
   o cambiar colección silenciosamente.
5. Tras revisar el primer resultado, repetir con índices 1, 2 y 3 de uno en uno.

| Índice | Punto | Grupo del autor | Ventana inclusiva |
|---|---|---|---|
| 0 | point_0021 | 16 | 2019-06-24 a 2019-06-26 |
| 1 | point_0002 | 1 | 2020-05-01 a 2020-09-30 |
| 2 | point_0023 | 13 | 2019-05-01 a 2019-07-31 |
| 3 | point_0502 | 8 | 2020-06-01 a 2020-08-30 |

No se sube el dataset ni se exporta a Drive. El script incluye 25 puntos de muestreo
por parche y sus valores TIFF, y consulta imágenes públicas. El enlace CSV descarga
una tabla pequeña mediante
[FeatureCollection.getDownloadURL](https://developers.google.com/earth-engine/apidocs/ee-featurecollection-getdownloadurl).

## Qué compara

Los cuatro puntos pertenecen al train propuesto, seleccionados por ID y diversidad
de grupo/ventana. No se usaron máscaras ni desempeño de modelos. Se toman centros
de píxel de una cuadrícula fija de 5×5 posiciones sobre los TIFF originales
150×150, sin transponer, recortar o normalizar los valores de referencia.

La consulta usa `COPERNICUS/S2_HARMONIZED` (L1C), EPSG:32718 y la transformación
afín exacta de cada TIFF. Se conserva el remuestreo predeterminado por vecino
más próximo. No se filtran nubes ni se modifica la radiometría para forzar ajustes.
La colección harmonized documenta la compensación de cambios de rango desde
baseline 04.00; el CSV registra baseline y producto para revisar ese aspecto.

Las escenas se ordenan únicamente por RMSE en **B2/B3/B4/B8**, con preferencia
por las que conservan los 25 puntos válidos. Las cuatro comparaciones restantes
se reportan después y no intervienen en elegir la escena:

- `B9_como_B9` y `B10_como_B10`: hipótesis del script R.
- `B9_como_B8A` y `B10_como_B9`: hipótesis de la configuración IRIS.

Los números de esas columnas también son RMSE, en los valores crudos del producto.
Un cero indica igualdad solo en los puntos consultados. Una comparación aproximada
requiere examinar georreferenciación, versión de procesamiento, fechas y validez,
antes de interpretar identidades. Faltan la confirmación sobre parches completos
y la extensión a la colección; este sondeo no habilita automáticamente entrenamiento.

La ventana del Excel se interpreta inclusiva; se suma un día al límite final para
`filterDate`. No se trunca la búsqueda: si hay más de 200 escenas, se detiene y avisa.
Si no hay datos válidos para una escena, no aparecerá entre los resultados de píxeles.

## Reproducción local

`scripts/prepare_roy_ee_probe.py` genera esta carpeta a partir de los TIFF locales,
el informe de reconstrucción y `scripts/earthengine/roy_band_probe.template.js`.
`reference.json` conserva coordenadas, valores y hashes de los TIFF leídos.

Se comprobó la sintaxis JavaScript con Node, la selección de escena con casos
sintéticos (incluyendo píxeles faltantes y distinta hipótesis de bandas) y los
600 valores locales de referencia, además de sus centros UTM. Los cuatro CSV
remotos se recibieron y verificaron contra los TIFF locales. Las pruebas locales
del generador no sustituyen esa comparación con las respuestas de Earth Engine.
