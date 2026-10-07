# Resultado del sondeo B13/B14 de Roy

Fecha: 2026-10-04. CSV recibido del usuario: `roy_topography_probe.csv`.

**Conclusión para el preprocesamiento:** hay evidencia suficiente para adoptar
**B13 = pendiente en radianes** y **B14 = elevación en metros**, conservando ese
orden. Se cierra la duda de identidad/unidades a este nivel de evidencia. La
procedencia exacta, el datum original y la cadena de remuestreo no están certificados.
No se etiqueta como verificación exhaustiva de toda la colección.

## Integridad

Las 300 filas esperadas están presentes: cuatro puntos train × 25 posiciones × tres
fuentes. Son **100 posiciones únicas**, no 300 píxeles independientes. Se verificaron
los ocho hashes TIFF, las referencias B13/B14 y las coordenadas; no hay duplicados
ni grupos incompletos. Los identificadores de fuentes y el método de muestreo
coinciden con la consulta preparada. Grados/radianes devueltos son consistentes.

Se conserva el CSV íntegro en `input.csv`, resultados detallados en `analysis.json`
y la tabla por punto/fuente en `comparison.csv`. El resultado no usa máscaras,
predicciones de modelos ni ejemplos de val/test.

## Pendiente: la hipótesis de radianes está respaldada

Errores sobre las 100 posiciones; **ambas hipótesis se comparan en radianes**:

| Referencia | RMSE si B13 está en radianes | RMSE si B13 está en grados |
|---|---:|---:|
| SRTM V3 | **0,07595 rad** | 0,55455 rad |
| NASADEM HGT | 0,10176 rad | 0,56168 rad |
| ALOS AW3D30 V4.1 | 0,12319 rad | 0,57732 rad |

La hipótesis de radianes tiene menor error en los doce grupos punto/fuente, no solo
en el agregado. Con SRTM la correlación de pendiente es 0,9076 en el agregado y
0,8489–0,9507 dentro de cada punto. La escala angular y la relación espacial apoyan
pendiente en radianes; la correlación por sí sola no distingue grados de radianes.
No existe igualdad exacta con las pendientes consultadas. El algoritmo y la malla
originales de cálculo siguen sin verificarse.

## Elevación: diferencia compatible con referencia vertical distinta

B14 sigue la elevación de SRTM con correlación 0,999889 en el agregado y
0,999164–0,999373 dentro de cada punto. Roy está **34,45 m por encima** de SRTM en
promedio. No hay coincidencias exactas de elevación con ninguna de las tres fuentes.

Para examinar el sesgo se calculó una hipótesis geodésica independiente:
`h = H + N`, usando la ondulación del geoide EGM96 de la cuadrícula local
`/usr/share/proj/egm96_15.gtx`. Se interpoló en las 100 coordenadas mediante GDAL,
sin estimar una constante a partir de Roy ni desplazar imágenes. La ondulación
obtenida está entre 31,52 y 32,96 m. Las muestras, hash de la cuadrícula, versión
GDAL y método quedan registrados en `analysis.json`.

| Referencia | RMSE elevación sin corrección | RMSE bajo hipótesis `H + N` |
|---|---:|---:|
| SRTM V3 | 35,27 m | **8,10 m** |
| NASADEM HGT | 35,08 m | 8,84 m |
| ALOS AW3D30 V4.1 | 34,38 m | 12,42 m |

La mejora es compatible con alturas elipsoidales en Roy. No demuestra por sí sola
su datum ni identifica un producto único: también mejora el contraste con las
otras referencias. El RMSE residual y la ausencia de igualdad impiden certificar
el archivo original. **La corrección solo se calculó para el diagnóstico; no se
aplicó a los TIFF ni a los HDF5.**

La hipótesis tiene respaldo documental: ASF explica que el DEM auxiliar de PALSAR
RTC procede de DEMs como SRTM y se adapta para el procesamiento, incluyendo una
conversión de alturas. Así, la mención “ALOS PALSAR” en la tesis y una relación
con SRTM pueden coexistir. Fuentes primarias:
[guía ASF/NASA](https://www.earthdata.nasa.gov/s3fs-public/2024-10/ASF-SAR-ALOS-PALSAR-RTC-User-Guide.pdf?VersionId=r96by4i5NOP4_lkz8zk6UhAMXk4oyvl9),
[aclaración del equipo ASF](https://forum.earthdata.nasa.gov/viewtopic.php?t=4094) y
[guía de uso ASF](https://docs.asf.alaska.edu/datasets/using_ASF_data/).
Esto es una **inferencia de procedencia compatible con la documentación**, no una
certificación de que el COG privado citado por Roy produjo estos valores.
AW3D30 es un DSM óptico PRISM diferente del DEM auxiliar de PALSAR RTC.

## Decisión y alcance

Se recomienda conservar B13/B14 y los catorce canales reconstruidos, documentando:

- B13 pendiente, unidad adoptada radianes; B14 elevación, unidad adoptada metros.
- Identidad/unidades respaldadas por sondeo empírico y evidencia local/documental.
- Fuente exacta y datum pendientes de certificación; compatibilidad con DEM auxiliar
  SRTM/PALSAR RTC, sin afirmar una resolución nativa de 12,5 m ni una fuente verificada.
- El min–max por canal/imagen ya propuesto elimina cambios de escala positivos y
  desplazamientos constantes. No elimina diferencias espaciales de fuente,
  interpolación, datum variable o algoritmo de pendiente.

No hace falta pedir otro CSV para decidir el orden de estos dos canales. Queda
formalizar la admisión del experimento con esta limitación explícita y verificar
configuraciones/manifiestos. Este informe no altera los originales, los derivados,
las asignaciones de split ni los experimentos anteriores. Las configuraciones
conjuntas siguen deshabilitadas y no se ha ejecutado entrenamiento conjunto.

## Reproducción

Análisis completo en este entorno (GDAL disponible en el Python del sistema):

```bash
/usr/bin/python3 scripts/analyze_roy_topography_probe.py \
  --csv results/audits/roy_topography_comparison_v1/input.csv \
  --geoid /usr/share/proj/egm96_15.gtx \
  --output results/audits/roy_topography_comparison_v2
```

Usar una ruta nueva: el script rechaza sobrescribir informes. Sin `--geoid`, el
análisis básico también funciona con `.venv/bin/python` y no necesita GDAL.

Se comprobó reproducción del análisis básico entre entornos con tolerancia numérica
de 1e-12. Pruebas de datos alterados confirmaron rechazo de referencias modificadas,
duplicados, fuentes distintas y unidades inconsistentes; un grupo ausente se
registra como incompleto. Una pendiente sintética verifica la comparación de ambas
hipótesis en radianes. Estas comprobaciones validan el análisis, no la autenticidad
del producto original inaccesible.
