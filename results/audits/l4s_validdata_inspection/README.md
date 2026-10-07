# Inspección de ValidData — 28/09/2026

Se auditaron los 245 pares locales con `scripts/inspect_l4s.py --directory ValidData`.

- Imágenes HDF5: clave img, 128×128×14, float64.
- Máscaras: clave mask, 128×128, uint8, únicamente 0/1.
- Etiquetas derivadas de presencia de píxeles positivos: 146 positivas y 99 negativas.
- Sin faltantes, errores de lectura, NaN/Inf ni máscaras inválidas.
- Ningún duplicado exacto dentro de validación ni entre los tres manifiestos auditados (train: 3799, val: 245, test: 800). Informe: `results/audits/l4s_duplicate_check.json`.
- Los 245 ejemplos pasan `scripts/check_l4s_preprocessing.py --split val`, conservando los valores tras conversión float32 y cambio HWC→CHW; lote 8×14×128×128. Informe: `results/audits/l4s_preprocessing_val.json`.

Los CSV con etiquetas y área afectada se guardan en `metadata/landslide4sense_validdata.csv`. La carga de clasificación no abre máscaras. No se entrenó ningún modelo ni se ajustó normalización usando validación.

La copia local sí incluye máscaras de validación. Los archivos de imagen no contienen atributos de archivo que permitan verificar independencia geográfica. Igualdad de dimensiones no prueba alineación espacial; ausencia de duplicados exactos no descarta parches solapados. Se conservan ceros y el escalado distribuido, sin inferir que cero signifique NoData. Las estadísticas por canal están en summary.json.

Las tres particiones originales ya están caracterizadas. Todavía deben persistirse sus manifiestos en las rutas de splits esperadas por el entrenador y ejecutarse una prueba técnica corta con datos reales antes del entrenamiento formal.
