# Pipeline ejecutable: cuatro clasificadores → mapas → pseudomáscaras → segmentador común

Esta primera versión implementa el alcance de README, THESIS_CONTEXT y AGENTS actualizados.
Landslide4Sense es la fuente ejecutable actual; Roy permanece almacenado y deshabilitado.
No modifica datos ni particiones. No reproduce LGAGNet ni todos los módulos de Deng.
No se han ejecutado aquí entrenamientos largos ni evaluación sobre test real.

## Decisiones de la primera versión

- Cuatro ramas: ResNet18, ResNet50, DeiT-S sin destilación y MiT-B1 clasificador.
- Grad-CAM dirigido al logit positivo en todas las ramas, sin usar máscaras.
  ResNet: `backbone.layer4`; DeiT: `backbone.blocks[-1].norm1`, excluyendo CLS
  y reorganizando tokens según el tamaño de parche real; MiT: `encoder.last_hidden_state`.
  La resolución nativa puede variar; compartir el algoritmo no elimina ese factor.
- Guardar el mapa firmado nativo, el mapa tras ReLU interpolado a la imagen (`raw`),
  el mapa normalizado, la probabilidad y magnitud del gradiente por muestra.
  Los mapas constantes se convierten a cero y quedan marcados. Gradientes espaciales
  nulos o valores no finitos detienen el proceso; no producen pseudomáscaras silenciosas.
- Normalización min–max por mapa, no por banda de entrada. Umbrales 0,05…0,95.
  Selección por IoU macro en todas las imágenes GT positivas de validación,
  incluyendo fallos de clasificación como predicciones vacías. Empate: mayor umbral.
  Se elige un umbral por rama bajo la misma política. No se calibra con train/test.
- Compuerta de presencia: p<0,5 → pseudomáscara vacía. Se aplica en todos los splits.
  No se usa el label real para forzar máscaras vacías durante generación. Se preserva
  además la máscara umbralizada sin compuerta para análisis independiente.
- Refinamiento `none`: no se ha implementado CRF. No se guarda un resultado como
  "refinado" cuando no existe refinamiento. Añadirlo requiere una ablación explícita.
- Segmentador compartido: variante DeepLabv3+ con encoder ResNet50 aleatorio,
  output stride 16, ASPP de tasas 6/12/18, proyección de bajo nivel a 48 canales y
  decoder a 256 canales. Usa convoluciones estándar y GroupNorm en ASPP/decoder;
  conserva BatchNorm del encoder. Evita fallo de BatchNorm en la rama pooled 1×1
  cuando el último lote tiene una muestra. No se presenta como checkpoint oficial
  ni reproducción exacta de la implementación de los papers.
- BCEWithLogitsLoss no ponderada, AdamW y umbral espacial final 0,5. Sin augmentation,
  scheduler ni early stopping en esta primera implementación. Las cuatro ramas usan
  exactamente la misma configuración y semilla del segmentador, con pesos nuevos
  en cada ejecución. El clasificador no se ajusta durante generación de mapas.

La plantilla del segmentador tiene **2 épocas para piloto**, no entrenamiento formal.
Fijar su presupuesto definitivo, recursos y semillas antes del experimento comparativo.
Los JSON de clasificadores tienen 50 épocas como baseline inicial ya acordado; no
garantizan convergencia. No sustituir el checkpoint existente de ResNet18: indicar
su ruta real en `branches.resnet18.checkpoint` de `configs/pipeline_l4s.json`.

## Uso de etiquetas y validez del experimento

La loss del clasificador recibe únicamente etiquetas de imagen. La loss de entrenamiento
del segmentador recibe únicamente pseudomáscaras de **train**. Son lectores separados;
el lector de pseudomáscaras jamás abre `mask_path` aunque esté presente en los CSV.

**Sí se usan máscaras reales de validación** para calibrar los umbrales y seleccionar
el checkpoint del segmentador (mayor IoU macro en positivos; primer empate). Esta
supervisión espacial de validación debe declararse al presentar resultados. No es
un experimento sin ninguna anotación espacial disponible durante el desarrollo.
No se usan máscaras reales de train para optimizar pesos. Test solo se evalúa mediante
comandos explícitos después de congelar modelos, umbrales y procedimiento.

Una referencia totalmente supervisada queda fuera de esta primera implementación.
El mismo segmento de código no cambia silenciosamente de pseudomáscaras a GT.

## Configuración y checkpoints

`configs/pipeline_l4s.json` reúne las cuatro ramas y un único bloque `segmentation`.
Todos los caminos de datos y checkpoint se resuelven respecto a `--root`; los
directorios de salida y las rutas a artefactos son respecto al directorio de trabajo.
Los checkpoints deben ser propios/de confianza: se usa `torch.load(weights_only=False)`.

Las rutas por defecto son:

```
results/experiments/l4s_resnet18_50ep_seed42/best.pt
results/experiments/l4s_resnet50_50ep_seed42/best.pt
results/experiments/l4s_deit_small_50ep_seed42/best.pt
results/experiments/l4s_mit_b1_50ep_seed42/best.pt
```

Para entrenar un clasificador aún faltante, por ejemplo DeiT:

```bash
python -m src.training.train --config configs/train_l4s_deit_small.json --root . --output results/experiments/l4s_deit_small_50ep_seed42
```

ResNet50 y MiT tienen sus respectivos JSON con las mismas condiciones iniciales.
Antes de ejecutar en el servidor, activar el entorno y usar la GPU asignada.
No entrenar automáticamente los cuatro si ya existen checkpoints válidos.

## Ejecutar una rama después de su clasificación

Desde la raíz del proyecto, definir una de las cuatro ramas. El directorio debe ser
nuevo. Se rechaza sobrescribir salidas; si un paso falla, conservarlo para diagnóstico
y usar otra ruta después de resolver la causa.

```bash
BRANCH=resnet18
RUN=results/pipeline/$BRANCH

python -m src.pipeline.maps --config configs/pipeline_l4s.json --branch "$BRANCH" --split val --root . --output "$RUN/maps_val"
python -m src.pipeline.pseudo calibrate --maps "$RUN/maps_val" --root . --output "$RUN/calibration"
python -m src.pipeline.maps --config configs/pipeline_l4s.json --branch "$BRANCH" --split train --root . --output "$RUN/maps_train"

python -m src.pipeline.pseudo generate --maps "$RUN/maps_train" --calibration "$RUN/calibration/threshold.json" --output "$RUN/pseudo_train"
python -m src.pipeline.pseudo generate --maps "$RUN/maps_val" --calibration "$RUN/calibration/threshold.json" --output "$RUN/pseudo_val"
python -m src.pipeline.pseudo evaluate --pseudo "$RUN/pseudo_val" --root . --output "$RUN/localization_val"

python -m src.segmentation.train train --config configs/pipeline_l4s.json --pseudo "$RUN/pseudo_train" --root . --output "$RUN/segmenter"
python -m src.segmentation.train evaluate --checkpoint "$RUN/segmenter/best.pt" --split val --root . --output "$RUN/segmentation_val"
```

Repetir con `resnet50`, `deit_small` y `mit_b1` después de sus clasificadores.
No cambiar `segmentation`, preprocessing, semillas ni splits entre ramas.
Los mapas de train proceden del clasificador entrenado en train (no son out-of-fold);
su calidad de entrenamiento no debe confundirse con generalización.

Reanudar segmentación interrumpida: mismo comando `train` y directorio con `--resume`.
Requiere la misma configuración completa, artefactos, versiones, código y hardware.
Repite una época incompleta. No amplía automáticamente un experimento ya terminado.

## Test final explícito

Una vez congelado todo, para cada rama:

```bash
python -m src.pipeline.maps --config configs/pipeline_l4s.json --branch "$BRANCH" --split test --root . --output "$RUN/maps_test"
python -m src.pipeline.pseudo generate --maps "$RUN/maps_test" --calibration "$RUN/calibration/threshold.json" --output "$RUN/pseudo_test"
python -m src.pipeline.pseudo evaluate --pseudo "$RUN/pseudo_test" --root . --output "$RUN/localization_test"
python -m src.segmentation.train evaluate --checkpoint "$RUN/segmenter/best.pt" --split test --root . --output "$RUN/segmentation_test"
```

`maps_test/classification.json` contiene las métricas de clasificación de test;
el manifiesto conserva probabilidades por muestra. No se recalibra ningún umbral.

## Comparación

Tras completar las cuatro ramas para el mismo split:

```bash
python -m src.pipeline.compare --split val --runs results/pipeline/resnet18 results/pipeline/resnet50 results/pipeline/deit_small results/pipeline/mit_b1 --output results/comparison_val_seed42
```

Cambiar `val` por `test` solo tras evaluación final. Produce CSV y JSON con clasificación,
IoU/Dice de pseudomáscaras y segmentación. Rechaza ramas duplicadas, poblaciones,
configuraciones comunes o condiciones del clasificador distintas (salvo arquitectura).
Las tablas son descriptivas para una semilla: no constituyen significación estadística
ni prueban causalidad exclusiva de la familia del backbone.

## Artefactos y métricas

- Mapas/pseudomáscaras: NPZ por muestra; CSV conserva ID, label, procedencia, área,
  probabilidad, umbral y hashes; metadata conserva checkpoint, split y configuración.
- Segmentación: best/last, estado del optimizador y RNG, historial, entorno, hashes de
  código y manifiestos; validación usa GT y está etiquetada como tal.
- Evaluación: métricas por muestra y agregadas; probabilidades y máscaras finales.
  IoU/Dice macro incluyen todos los positivos; micro acumula TP/FP/FN sobre todas
  las imágenes. Vacío/vacío tiene overlap indefinido, no un IoU ficticio de 1.
  Pixel F1 equivale al Dice micro. Se informa tasa de falsas regiones en negativos.
- `area_ratio` se conserva para análisis posterior. No se fijan automáticamente
  cortes de tamaño ni se calculan percentiles de test.

Los hashes de imágenes en CSV provienen de auditorías previas; conservar originales.
Los nuevos artefactos NPZ se verifican al consumirlos. No convertir esa verificación
de integridad en una afirmación de independencia geográfica.

## Verificación y límites

```bash
python -m unittest discover -s tests -v
```

Pruebas de señal espacial sintética, los cuatro adaptadores reales, forward/backward
del segmentador con lote de una muestra, normalización, calibración sin acceso a test,
trazabilidad y flujo completo con máscaras GT de train inexistentes (redes pequeñas
en la prueba integral). No equivale a validar calidad de localización en imágenes reales
ni rendimiento en GPU. El usuario conserva su entrenamiento remoto; este código no
accede automáticamente al servidor.

Verificación local realizada: 24 pruebas pasan. Además, se cargaron un negativo
y un positivo reales de TrainData, con tensor 2×14×128×128; los cuatro adaptadores
produjeron mapas finitos y gradientes espaciales no nulos, y el segmentador produjo
salidas finitas 2×128×128. Se usaron pesos aleatorios, sin optimización ni lectura
de máscaras. El informe está en `results/audits/pipeline_smoke.json`; no representa
calidad predictiva ni un benchmark de GPU.

Fuentes: [Grad-CAM](https://arxiv.org/abs/1610.02391),
[adaptación de tokens ViT](https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md),
[DeepLabv3+](https://arxiv.org/abs/1802.02611).
