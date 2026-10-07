# Reproducción del experimento admitido

Esta guía utiliza rutas relativas y omite direcciones, cuentas y claves del servidor.
El experimento activo es `l4s_roy_14_v1_50ep_seed42`.

## 1. Entorno

Desde la raíz del repositorio, crear un entorno aislado con Python compatible con
las dependencias y registrar las versiones instaladas:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` no sustituye el registro exacto del entorno de una corrida previa.
La configuración activa usa CUDA; instalar PyTorch compatible con la GPU y el entorno.
El ejecutor usa `fcntl`, por lo que requiere un entorno compatible, como Linux.

## 2. Datos externos obligatorios

Los datasets y pesos no están incluidos. Proporcionar los archivos admitidos,
sin regenerar particiones, en las ubicaciones indicadas por los manifiestos:

```text
data/landslide4sense/TrainData/
data/landslide4sense/ValidData/
data/landslide4sense/TestData/
data/prepared/roy14_v1/img/
data/prepared/roy14_v1/mask/
```

Los nombres y hashes están en `splits/l4s_roy_14_v1/`. Una descarga genérica de
Landslide4Sense no garantiza disponer de todas las referencias locales de
validación/prueba: comprobar la procedencia y coincidencia de cada archivo.
Roy requiere la reconstrucción descrita en [la ficha](datasets/l4s_roy_14_v1.md).
Respetar las condiciones de acceso y atribución de las fuentes originales.

Los scripts de preparación conservan dependencias de recursos originales externos.
No ejecutar nuevamente admisión o generación de particiones para eludir un archivo
ausente o un hash distinto. Los datos de esta versión no se redistribuyen aquí.

## 3. Comprobación previa

```bash
python -m unittest discover -s tests -v
python scripts/check_joint_experiment.py --models --device cuda
```

La comprobación conjunta requiere los datos externos. Revisa manifiestos, archivos
y arrays; el paso de modelos usa un lote de ambas fuentes y objetivos sintéticos
para el segmentador, sin optimización. No obtiene métricas de prueba.

## 4. Entrenamiento y validación

```bash
python -u scripts/run_joint_experiment.py --phase train-val
```

Alternativamente, ejecutar primero solo clasificadores:

```bash
python -u scripts/run_joint_experiment.py --phase classifiers
```

El ejecutor procesa las cuatro ramas, conserva salidas y recupera estados compatibles.
Repetir el comando continúa hasta el presupuesto configurado; no añade otras 50 épocas.
No modificar código o configuración mientras se reanuda una corrida.

## 5. Evaluación final

Después de completar validación y congelar decisiones:

```bash
python -u scripts/run_joint_experiment.py --phase test
```

No seleccionar umbrales o variantes usando prueba. El test del experimento ya fue
observado en el reporte disponible; declarar esa circunstancia en análisis posteriores.

## 6. Productos

- Clasificadores: `results/experiments/l4s_roy_14_v1_MODELO_50ep_seed42/`.
- Etapas espaciales: `results/joint/l4s_roy_14_v1_50ep_seed42/`.
- Conservar configuración efectiva, entorno, historiales, checkpoints, umbrales,
  predicciones por muestra y métricas, además de los CSV agregados.

Las salidas completas quedan excluidas de Git por tamaño. Seleccionar informes
ligeros para una entrega posterior, verificando su procedencia. El reporte actual
es una transcripción de cifras del servidor con análisis aritmético, no una auditoría
de sus archivos originales.

El [índice de estructura y evidencias](../ESTRUCTURA_REPOSITORIO.md) relaciona cada
producto con su objetivo. No se publican datasets ni checkpoints de terceros.
