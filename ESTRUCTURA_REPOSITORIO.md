# Estructura del repositorio y evidencias para la tesis

Este documento permite localizar los productos asociados con los objetivos
específicos. Puede citarse desde el Anexo D junto con el enlace del repositorio
y el commit de la versión entregada.

## Organización

```text
src/          Lectores, modelos, entrenamiento, atribución y evaluación
scripts/      Preparación, auditorías y ejecución por etapas
tests/        Comprobaciones automatizadas
configs/      Configuraciones activas y antecedentes identificados
splits/       Particiones admitidas y baseline L4S
metadata/     Admisión y metadatos de las fuentes
docs/         Procedimientos y decisiones metodológicas
results/
  audits/     Reportes y tablas de auditoría seleccionados
  analysis/   Métricas reportadas y análisis de su alcance
```

Las imágenes, anotaciones, checkpoints y resultados masivos permanecen fuera de
Git. Los manifiestos conservan las rutas y hashes de esos archivos externos.

## Productos y evidencia disponible

| RE | Producto o comprobación | Ubicación |
|---|---|---|
| O1-R1 | Población admitida y particiones | [Admisión](metadata/l4s_roy_14_v1_admission.json), [manifiestos](splits/l4s_roy_14_v1/) |
| O1-R1 | Reconstrucción y exclusiones de Roy | [Decisiones](docs/datasets/l4s_roy_14_v1.md), [auditoría](results/audits/roy14_reconstruction_v1/) |
| O1-R2 | Preprocesamiento común | [Implementación](src/preprocessing/multispectral.py), [documentación](docs/preprocessing.md) |
| O1-R3 | Métricas y casos especiales | [Protocolo](docs/evaluation_protocol.md), [evaluación](src/evaluation/) |
| O2-R1 | Clasificadores y componentes espaciales | [Clasificadores](src/models/classifiers.py), [Grad-CAM](src/localization/gradcam.py), [pseudoetiquetas](src/pipeline/pseudo.py), [segmentador](src/segmentation/model.py) |
| O2-R1 | Comprobación técnica registrada | [Reporte conjunto](results/audits/l4s_roy_14_v1_admission_check.json) |
| O2-R2 | Hiperparámetros y selección | [Configuración activa](configs/experiments/l4s_roy_14_v1/), [ejecutor](scripts/run_joint_experiment.py) |
| O2-R3 | Procedimiento de reproducción | [Guía pública](docs/REPRODUCIBILIDAD.md), [dependencias](requirements.txt), [pruebas](tests/) |
| O3-R1 | Clasificación reportada | [CSV global](results/analysis/l4s_roy_14_v1_test_seed42_user_report/comparison.csv), [CSV por fuente](results/analysis/l4s_roy_14_v1_test_seed42_user_report/comparison_by_dataset.csv) |
| O3-R2 | Pseudoetiquetas y segmentación reportadas | [CSV de comparación](results/analysis/l4s_roy_14_v1_test_seed42_user_report/comparison.csv) |
| O3-R3 y O3-R4 | Comparación preliminar y limitaciones | [Análisis](results/analysis/l4s_roy_14_v1_test_seed42_user_report/README.md) |

## Alcance de las comprobaciones

- El reporte de modelos registra propagación hacia adelante y atrás para cuatro
  clasificadores y un segmentador, en CPU y sin actualizar pesos. No demuestra
  rendimiento predictivo ni convergencia.
- Las métricas remotas son cifras reportadas, con comprobación aritmética. Los
  archivos originales de predicciones y checkpoints no se verificaron directamente.
- La existencia de `tests/` no equivale a una ejecución aprobada: para acreditarla
  debe conservarse el log, la fecha, el entorno y la versión.
- El commit de publicación identifica esta entrega del código; no certifica que
  sea exactamente la versión usada por una corrida remota anterior.

## Fuentes externas y antecedentes

La preparación desde cero requiere las fuentes originales documentadas en
`docs/datasets/`, incluidas las anotaciones y los recursos de procedencia.
Los directorios locales `data/`, `archives/` y `references/` no se distribuyen.
Las propuestas de partición duplicadas y algunos reportes históricos también
quedan fuera. Algunas notas históricas describen esos archivos: su mención no
implica que estén incluidos en esta publicación ligera.

Para ejecutar el protocolo admitido, seguir la
[guía de reproducción](docs/REPRODUCIBILIDAD.md) y las configuraciones activas.
