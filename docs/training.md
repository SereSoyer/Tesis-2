# Entrenamiento por épocas y registro de experimentos

Implementado en `src/training/train.py` y `src/training/epochs.py`. Admite los cuatro clasificadores existentes, uno por ejecución. Recibe imágenes multiespectrales y etiquetas binarias; no abre máscaras. No crea particiones ni usa test. Todavía no se han ejecutado experimentos formales.

## Configuración y ejecución

`configs/train_l4s.json` es una plantilla técnica de dos épocas, CPU y pesos aleatorios. Los hiperparámetros son provisionales, no un protocolo comparativo definitivo. Requiere `splits/l4s_train.csv` y `splits/l4s_val.csv`, ya preparados conservando la división original. No copia TestData ni sustituye archivos ausentes. Los CSV deben conservar `source_split` (data/landslide4sense/TrainData/ValidData) para Landslide4Sense. Las imágenes y los CSV se resuelven respecto a `--root`; el JSON de configuración y `--output`, respecto al directorio de ejecución.

Cuando estén listas las particiones:

```bash
.venv/bin/python -m src.training.train --config configs/train_l4s.json --root . --output results/experiments/l4s_resnet18_seed42
```

Para otro modelo, copiar el JSON y cambiar `model` por `resnet50`, `deit_small` o `mit_b1`; utilizar una carpeta de salida nueva. En el servidor, ajustar `device` a `cuda` o `cuda:0`, y los recursos según el hardware. CUDA no disponible produce error, sin cambiar silenciosamente a CPU.

Roy puede usar el mismo entrenador con su dataset, CSV y preprocesamiento explícitos una vez resueltas sus incidencias y particiones. No se genera una configuración experimental de Roy que dé por resueltas esas decisiones.

## Comportamiento

- Entrenamiento: `model.train()`, AdamW y BCEWithLogitsLoss sin pesos de clase; barajado reproducible y todos los lotes, incluido el último incompleto.
- Validación: `model.eval()`, sin gradientes, sin barajado ni actualización de parámetros.
- Loss: promedio ponderado por número de imágenes. Métricas: acumuladas sobre toda la época; sigmoid y umbral configurable (0.5 en la plantilla).
- Las métricas de train corresponden a predicciones durante las actualizaciones, no a una segunda evaluación del modelo final de la época. Validación sí utiliza un único estado del modelo.
- Mejor checkpoint: mínimo de loss de validación en la plantilla. También permite maximizar accuracy, precision, recall, F1 o ROC-AUC; esta última exige ambas clases. En empates se conserva el primero. La selección no usa test.
- Scheduler y early stopping están explícitamente desactivados (`null`). El runner rechaza otros valores; ejecuta el número configurado de épocas. No implementa AMP, entrenamiento distribuido, augmentation ni balanceo implícito.
- Rechaza train/val con IDs o rutas de imagen compartidos y mezcla de datasets. Estas comprobaciones no detectan por sí solas duplicados de contenido, proximidad geográfica, eventos comunes o parches solapados: la auditoría previa de splits sigue siendo necesaria.

## Archivos guardados

Cada ejecución crea una carpeta exclusiva; no sobrescribe otra sin `--resume`.

| Archivo | Contenido |
|---|---|
| `config.json` | Configuración completa vigente |
| `environment.json` | Plataforma, dispositivo y versiones |
| `manifest_sha256.json` | Hashes de los CSV de entrada |
| `splits/train.csv`, `splits/val.csv` | Copias exactas de metadatos, incluidos area_ratio y otros campos existentes |
| `history.json` | Pérdidas, métricas, tiempos y learning rate por época y split |
| `summary.json` | Épocas completadas, mejor época/valor, parámetros y tiempo acumulado de los ciclos |
| `last.pt` | Estado de la última época completa para reanudar |
| `best.pt` | Estado seleccionado exclusivamente por validación |

Los checkpoints contienen pesos, optimizador, configuración, historial y estados aleatorios de Python, NumPy, PyTorch, CUDA y generadores de los DataLoader. Se escriben primero a un archivo temporal y se reemplazan de forma atómica. El historial incluido en `last.pt` es la referencia para reanudar si se interrumpió la escritura de los informes JSON. Los hashes verifican los CSV, no el contenido de las imágenes: conservar los datos originales sin cambios.

El tiempo por split incluye lectura de datos y cálculo de métricas; no representa latencia pura de inferencia. El pico de memoria CUDA corresponde a la sesión actual y queda null en CPU. La medición formal de eficiencia se abordará por separado.

## Reanudar

Aumentar `epochs` si la ejecución ya completó el total anterior; el valor es el total objetivo, no el número de épocas adicionales. Mantener el resto de la configuración:

```bash
.venv/bin/python -m src.training.train --config configs/train_l4s.json --root . --output results/experiments/l4s_resnet18_seed42 --resume
```

Continúa en la época siguiente a `last.pt`; una época interrumpida se repite. Exige igual configuración (excepto epochs), hashes de manifiestos y entorno registrado. No permite cambiar dispositivo/versions al reanudar. La reproducibilidad exacta entre hardware distinto no está garantizada. Cargar únicamente checkpoints propios o de confianza, pues incluyen estados Python serializados.

## Verificación

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Pruebas con HDF5 sintéticos y un clasificador pequeño con dropout: dos épocas, último lote incompleto, selección de mejor checkpoint, lectura de informes, recuperación de best.pt, igualdad exacta de pesos/métricas entre ejecución continua y reanudada en CPU, validación sin actualizar parámetros y rechazo de particiones ausentes o compartidas. Las rutas de máscara apuntan a archivos inexistentes para comprobar que no se requieren. Esto valida el mecanismo; no constituye evidencia predictiva sobre Roy o Landslide4Sense ni una prueba en GPU.
