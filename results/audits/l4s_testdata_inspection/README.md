# Inspección de Landslide4Sense — TestData

Auditoría local del 25/09/2026. Se revisaron los 1600 HDF5 presentes: 800 imágenes y 800 máscaras, emparejadas por sufijo numérico. No se modificaron archivos originales ni se entrenaron modelos con test.

## Resultados

| Propiedad | Resultado |
|---|---|
| Imágenes | 800; todas con clave img |
| Forma de imagen | 128×128×14 (HWC) |
| Tipo de imagen | float64 |
| Máscaras | 800; clave mask; 128×128; uint8 |
| Valores de máscara | Exclusivamente 0 y 1 |
| Positivas (algún píxel = 1) | 536 (67%) |
| Negativas | 264 (33%) |
| Archivos con error de lectura | 0 |
| Pares faltantes / dimensiones incompatibles | 0 |
| NaN / infinitos en imágenes | 0 |
| Grupos de imágenes idénticas por hash de tensor | 0 |
| Atributos de archivo HDF5 de imágenes | Vacíos en los 800 |

La copia local incluye máscaras de test. Esto corrige la afirmación anterior basada en la descripción del repositorio/Zenodo, que indicaba disponibilidad de máscaras solo en train. La inspección local no establece quién añadió/publicó las máscaras ni cuándo: conviene conservar la procedencia del ZIP.

## Valores, NoData y canales

Los valores observados son no negativos y no están limitados a [0,1]. El máximo global es aproximadamente 60.2351 (índice de canal 9). Los rangos individuales están en summary.json. No se han identificado físicamente canales o unidades solo mediante esos rangos.

La presencia de valores finitos no descarta NoData codificado como cero u otro sentinel. Muchos canales tienen exactamente 512 ceros acumulados; pendiente revisar su distribución y significado. No se eliminaron ceros ni se aplicó normalización en la auditoría. Las estadísticas de test son descriptivas: no se usarán para ajustar el preprocesamiento, hiperparámetros o umbrales.

## Tamaño de regiones

Sobre las 536 muestras positivas, área afectada / 16384 píxeles:

- Mínimo: 0.00610% (1 píxel).
- P25: 0.37842%.
- P33: 0.52216%.
- Mediana/P50: 1.14136%.
- P66: 2.02087%.
- P75: 2.87323%.
- Media: 2.81867%.
- Máximo: 88.34839%.

No se definen grupos de tamaño a partir de test. La igualdad de dimensiones no demuestra alineación geográfica; falta revisión espacial de muestras y contraste documental. La ausencia de duplicados exactos internos tampoco descarta solapamiento o fuga entre particiones.

## Implicaciones

Esta copia permite derivar etiquetas y, una vez confirmado el protocolo, evaluar clasificación y localización sobre test. Se conserva el papel TestData; no se divide ni se integra al entrenamiento. No es necesario planear un test interno solo por la supuesta ausencia de máscaras oficiales, pero hay que revisar TrainData y ValidData reales antes de cerrar las particiones.

La estructura es compatible con el lector configurable existente usando image_key=img y layout=HWC. No se extrapolan los conteos, rangos ni clases a TrainData. Roy permanece separado.

## Archivos generados y reproducción

- `metadata/landslide4sense_testdata.csv`: 800 filas, etiqueta, área, forma, rutas y hash por imagen. IDs prefijados con el split para evitar colisiones con image_1 de train/val.
- `results/audits/l4s_testdata_inspection/summary.json`: estadísticas completas.
- `scripts/inspect_l4s.py`: auditoría parametrizada por TrainData, ValidData o TestData.

```bash
.venv/bin/python scripts/inspect_l4s.py --directory TestData
```

El script sobrescribe únicamente sus metadatos e informes derivados. Las etiquetas de las máscaras se usan en la auditoría; el Dataset de clasificación sigue sin abrir máscaras.
