# Inspección de TrainData — 28/09/2026

Se ejecutó `scripts/inspect_l4s.py --directory TrainData` sobre los archivos locales.

- 3.799 imágenes HDF5 con clave `img`, forma HWC 128×128×14 y dtype float64.
- 3.799 máscaras HDF5 de 128×128, uint8 y valores exclusivamente 0/1.
- Correspondencia completa de identificadores imagen/máscara; sin errores de lectura, máscaras inválidas, NaN ni infinitos.
- Etiquetas por imagen derivadas de presencia de al menos un píxel positivo: 2.231 positivas y 1.568 negativas.
- Sin duplicados exactos de imágenes dentro de TrainData ni entre TrainData y el manifiesto auditado de TestData (800 imágenes). Ver `results/audits/l4s_duplicate_check.json`; reproducir con `scripts/check_l4s_duplicates.py` después de auditar los splits.
- ValidData no está disponible en la raíz del proyecto en esta inspección. No se ha creado una partición alternativa ni iniciado entrenamiento.

Los rangos por canal y la distribución de área positiva se guardan en `summary.json`. Hay valores cero y bandas constantes en algunas imágenes: índice 12 en 114 imágenes e índice 13 en 115. Estos hechos no permiten concluir por sí solos que exista NoData. Se conservan los valores distribuidos sin imputaciones ni normalización adicional.

La igualdad de dimensiones no demuestra alineación geográfica. Los archivos de imagen no contienen atributos de archivo con información de región, escena o evento; la revisión de duplicados exactos no descarta parches solapados o dependencia geográfica. El orden semántico de bandas no se deduce únicamente de sus índices.

El CSV `metadata/landslide4sense_traindata.csv` conserva etiqueta, rutas, área afectada, dimensiones y hash de cada imagen. Las máscaras se leen para esta caracterización y generación de etiquetas, no como objetivos del entrenamiento de clasificación.

También se ejecutó `scripts/check_l4s_preprocessing.py --split train`: las 3.799 imágenes conservan exactamente la selección original convertida a float32 y reorganizada a CHW, sin máscaras en la carga. Lote comprobado: 8×14×128×128. Informe: `results/audits/l4s_preprocessing_train.json`. No se ajustaron estadísticas ni se entrenó un modelo.
