# Primer resultado conjunto de test: L4S + Roy, 14 canales

Fecha de recepción: 2026-10-04. Experimento reportado:
`l4s_roy_14_v1_50ep_seed42`.

Los dos CSV se transcribieron del mensaje del usuario con los resultados del
servidor. No se descargaron ni verificaron directamente sus checkpoints,
predicciones o marcas de finalización. El usuario informó que completó train,
validación y test. Este informe analiza las cifras entregadas, no certifica la
ejecución remota. El primer bloque pegado es global; el segundo incluye `dataset`.

Se contrastaron los conteos con el manifiesto local admitido de test y las
definiciones locales de métricas. Las métricas de clasificación admiten matrices
de confusión enteras consistentes con las poblaciones; la agregación global de
clasificación y de métricas macro positivas coincide con los resultados por
fuente. Los conteos inferidos y comprobaciones están en `analysis.json`.

## Comparación de las cuatro ramas

Todas las métricas de la tabla se expresan en porcentajes.
`IoU macro positivo` promedia el IoU por imagen cuya máscara real contiene
deslizamiento. `IoU micro` acumula TP, FP y FN de píxeles en todas las imágenes,
incluidas las negativas. No son métricas intercambiables.

| Rama | F1 clasificación | IoU macro positivo, pseudomáscaras | IoU macro positivo, segmentación | IoU micro, segmentación | F1 píxel, segmentación |
|---|---:|---:|---:|---:|---:|
| ResNet18 | 88,13 | 3,68 | 4,68 | 5,80 | 10,96 |
| ResNet50 | 86,83 | 5,27 | 7,57 | 10,56 | 19,10 |
| DeiT-S | 86,77 | 1,04 | 1,14 | 1,77 | 3,48 |
| MiT-B1 | 86,12 | 2,32 | 2,62 | 2,18 | 4,26 |

ResNet18 obtiene el mayor F1 de clasificación; ResNet50 obtiene las mayores
métricas espaciales incluidas en los CSV. La clasificación no determina la
calidad de localización. Las dos ramas CNN superan espacialmente a las dos ramas
Transformer en esta corrida, pero una semilla y este protocolo no bastan para
declarar una superioridad general de familias de arquitecturas.

Las pseudomáscaras ya presentan poco solapamiento con GT antes de entrenar el
segmentador. DeepLabv3+ aumenta el IoU macro positivo global de ResNet50 en
2,29 puntos porcentuales. Eso señala la localización/pseudomáscara como una etapa
prioritaria de diagnóstico, sin identificar por sí solo la causa del problema.
No hay mejora uniforme: en MiT-B1 el IoU micro baja de 2,32 % a 2,18 %; en DeiT-S
el IoU macro positivo de L4S baja ligeramente de 1,087 % a 1,082 %.

## Resultados por fuente

IoU macro positivo de la segmentación final, en porcentajes:

| Rama | Landslide4Sense | Roy |
|---|---:|---:|
| ResNet18 | 4,28 | 6,50 |
| ResNet50 | 6,71 | 11,43 |
| DeiT-S | 1,08 | 1,43 |
| MiT-B1 | 2,45 | 3,38 |

Las cifras espaciales siguen siendo bajas en ambas fuentes. El resultado global
contiene más imágenes de L4S: 800 frente a 122 de Roy. Para el macro positivo, los
pesos implícitos son 536 imágenes positivas L4S y 118 positivas Roy.

## Desbalance de clasificación en Roy

El test admitido contiene 118 positivos y 4 negativos Roy. Un clasificador que
siempre predijera presencia obtendría Accuracy = 96,72 % y F1 positivo = 98,33 %,
superiores a los valores correspondientes de las cuatro ramas en Roy. El F1
positivo cercano al 97 % no demuestra que se distinga bien presencia de ausencia.

Matrices de clasificación inferidas de las métricas y conteos locales:

| Rama | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| ResNet18 | 115 | 4 | 3 | 0 |
| ResNet50 | 114 | 4 | 4 | 0 |
| DeiT-S | 114 | 4 | 4 | 0 |
| MiT-B1 | 104 | 2 | 14 | 2 |

Las tres primeras ramas clasifican como positivas las cuatro imágenes negativas.
MiT-B1 acierta dos, con menor sensibilidad. Solo cuatro negativos ofrecen escasa
evidencia para medir discriminación en Roy; deben acompañarse F1 y Accuracy con
conteos, sensibilidad, especificidad y ROC-AUC. Estas matrices se reconstruyeron
aritméticamente, no se leyeron de predicciones remotas.

## Diagnóstico siguiente, usando validación

Conservar esta corrida y sus resultados de test. Las cifras agregadas no permiten
decidir si la causa principal son mapas casi vacíos, demasiado extensos, errores
de atribución, resolución espacial o aprendizaje del clasificador.

1. Revisar ejemplos de validación por fuente y tamaño del deslizamiento: imagen,
   mapa normalizado, pseudomáscara, predicción del segmentador y GT juntos.
2. Revisar precisión/recall de píxeles y área predicha frente a real. Comparar la
   evaluación principal con `ungated_ablation` en
   `MODEL/localization_val/summary.json` para separar el efecto del filtro de
   clasificación del de la localización.
3. Examinar `MODEL/calibration/threshold.json` y su curva completa. DeiT-S y MiT-B1
   seleccionaron 0,05, el límite inferior de la rejilla configurada. Es una pista
   para inspeccionar mapas y calibración, no prueba de que bajar el umbral mejore.
4. Contrastar los mapas constantes, gradientes, resolución nativa de atribución
   e historiales de entrenamiento en validación antes de elegir un cambio.

Cualquier variante debe tener configuración e identificador nuevos y mantener la
comparación CNN/ViT con el mismo segmentador. El test ya fue observado: no usarlo
para elegir umbrales o variantes ni presentarlo luego como una evaluación nunca
consultada. Para una conclusión robusta harán falta repeticiones comparables.

La comparación con el experimento histórico de solo L4S no aísla el efecto de
añadir Roy: también cambió la normalización. Sería necesario un control L4S con
el mismo preprocesamiento y presupuesto para estudiar ese efecto.

Este análisis no modifica código de entrenamiento, umbrales, particiones ni
resultados originales.
