# Comprobación topográfica de Roy

Preparación local: 2026-10-04. **Consulta ya ejecutada por el usuario; CSV recibido
y analizado.** El [resultado](../../results/audits/roy_topography_comparison_v1/README.md)
respalda adoptar B13 como pendiente en radianes y B14 como elevación en metros,
con fuente exacta y datum no certificados. No hace falta repetir el sondeo.
Lo siguiente conserva el protocolo utilizado. No habilita entrenamiento ni altera los derivados.

## Evidencia disponible

- Los valores y la tabla de la tesis respaldan B13 como pendiente y B14 como elevación.
  El script `references/lanslide4sat_pl/scripts/download_img_gee.py` también nombra
  B13 pendiente y B14 DEM, aunque calcula la pendiente en grados. Los rangos de los
  TIFF y su relación con el gradiente de B14 favorecen **radianes**.
- `scripts/get_data_utils.R` del repositorio de referencia tiene una ruta SRTM que
  convierte pendiente a radianes, pero añade primero elevación y después pendiente.
  Otra ruta descarga mosaicos COG del autor. No se asume que todos estos scripts
  hayan generado los TIFF disponibles.
- La URL del mosaico COG `gs://rgee_dev/COG/AP_26505_FBS_F6970_RT1.cog.tif`,
  citada en el repositorio, devolvió HTTP 403 en una consulta pública. No se verificó
  directamente ese archivo ni se intentó eludir su control de acceso.
- Los productos PALSAR RTC de ASF pueden incluir un DEM auxiliar procedente de
  SRTM/NED. Por ello, “ALOS PALSAR” y “SRTM” no siempre describen orígenes excluyentes;
  tampoco demuestran igualdad numérica, resolución nativa o datum. Fuente:
  [guía de uso de datos de ASF](https://docs.asf.alaska.edu/datasets/using_ASF_data/).

El generador deja en `reference.json` un diagnóstico reproducible: gradiente central
de B14 sobre la malla exportada de 10 m y comparación con B13 interpretada como
radianes o como grados. Ambos errores se expresan en radianes. Se excluyen NoData
y sus vecindarios. No se espera igualdad exacta si la pendiente se calculó antes
del remuestreo; este diagnóstico no identifica por sí solo la fuente del DEM.

## Consulta externa reproducible

Preparar en una ruta nueva, sin sobrescribir auditorías:

```bash
.venv/bin/python scripts/prepare_roy_topography_probe.py
```

Copiar `results/audits/roy_topography_probe_v1/roy_topography_probe.js` completo en
[Earth Engine Code Editor](https://code.earthengine.google.com/) y pulsar **Run**.
No hay que cambiar un índice: consulta los cuatro puntos juntos. El enlace
**DESCARGAR CSV** aparece en la consola y entrega `roy_topography_probe.csv`.

Se conservan los cuatro IDs y las 25 posiciones por punto del sondeo espectral:
point_0021, point_0002, point_0023 y point_0502, todos del train propuesto. Se comparan
centros exactos de píxel del TIFF original, sin transposición, normalización ni
desplazamientos ajustados. Las referencias contienen hashes de B13/B14. No se usan
anotaciones, resultados de modelos ni muestras val/test para escoger posiciones.

Fuentes fijadas antes de ver el resultado:

| Fuente pública | Papel en la comprobación |
|---|---|
| [SRTM V3](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003) | Fuente nombrada explícitamente por la ruta R; elevación en metros |
| [NASADEM HGT](https://developers.google.com/earth-engine/datasets/catalog/NASA_NASADEM_HGT_001) | Contraste con un reprocesamiento de SRTM; elevación entera referida a EGM96 |
| [ALOS AW3D30 V4.1](https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_AW3D30_V4_1) | Contraste con DSM óptico PRISM; **no es** el DEM auxiliar PALSAR RTC ni prueba del producto usado por Roy |

Para cada fuente se calcula [pendiente con cuatro vecinos](https://developers.google.com/earth-engine/apidocs/ee-terrain-slope)
en su malla nativa, fijando explícitamente la proyección antes del muestreo final
en la malla Roy. La API entrega grados y el script conserva también radianes.
AW3D30 excluye su máscara de nubes/nieve (MSK=1) y registra las teselas consultadas.
Se utiliza vecino más próximo sin aplicar correcciones de altura ajustadas a Roy.
Se esperan 300 filas: 100 posiciones × 3 fuentes. Una fuente sin cobertura válida
puede producir menos filas: el resumen muestra esa falta, no la interpreta como acierto.

## Criterio de interpretación

Revisar primero integridad del CSV, referencias y cobertura. Comparar B14 con la
elevación, incluyendo RMSE, sesgo y coincidencias exactas. Contrastar las dos unidades
propuestas de B13 contra pendiente expresando ambos errores en radianes. No comparar
un RMSE en grados con otro en radianes como si fueran la misma magnitud.

Una coincidencia apoya esa correspondencia en los píxeles consultados. Una
aproximación o correlación elevada no certifica el producto exacto: datum vertical,
remuestreo, relleno de vacíos y malla del cálculo pueden cambiar los valores.
AW3D30 V4.1 es una referencia diagnóstica actual y no una reconstrucción histórica.
Si ninguna fuente coincide, registrar el resultado sin inventar una identificación.
El siguiente paso sería adoptar una procedencia parcialmente conocida de forma
explícita o reconstruir la topografía con una fuente declarada; no repetir ajustes
hasta obtener artificialmente una coincidencia.

Esto es una auditoría del dataset: no modifica el pipeline CNN/ViT, su supervisión,
el segmentador común, los splits ni las máscaras originales.
