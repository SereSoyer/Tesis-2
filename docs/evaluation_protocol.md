# Métricas de clasificación

La implementación de localización, pseudomáscaras y evaluación final se describe en
[weak_pipeline.md](weak_pipeline.md). Conserva la regla de calibración espacial
de este documento. Añade selección del segmentador mediante IoU macro de positivos
GT de validación, con uso de máscaras reales de validación declarado explícitamente.

Implementadas en src/evaluation/classification.py con scikit-learn. Reciben exclusivamente etiquetas binarias y probabilidades de presencia por imagen. No reciben imágenes ni máscaras.

- Accuracy: proporción de imágenes correctamente clasificadas.
- Precision, recall y F1: clase positiva 1 (presencia).
- ROC-AUC: calculada sobre probabilidades continuas, no etiquetas umbralizadas.
- Matriz de confusión: filas=real, columnas=predicción, orden [0,1], estructura [[TN,FP],[FN,TP]].

Umbral por defecto 0.5, registrado en el resultado; positivo si p >= umbral. Es una convención inicial, no un umbral optimizado ni una decisión de localización espacial. Si se ajusta posteriormente, hacerlo en validación y congelarlo antes de test.

Cuando precision, recall o F1 no tienen denominador, se devuelve 0 y se registra el nombre en undefined_metrics. ROC-AUC devuelve None (null en JSON) si falta una clase; nunca se inventa un AUC=0. Se rechazan entradas vacías, NaN/Inf, etiquetas no binarias, formas incompatibles y probabilidades fuera de [0,1].

ClassificationMetrics acumula batches en CPU y calcula métricas sobre el conjunto completo. No promediar F1 ni ROC-AUC de lotes. Crear/resetear un acumulador para cada época y split; convertir logits mediante sigmoid antes de update. No conserva grafos de autograd. El llamador debe garantizar que cada muestra aparezca una sola vez en evaluación.

```python
from src.evaluation import ClassificationMetrics
metrics = ClassificationMetrics(threshold=0.5)
# Dentro del bucle de validación, con model.eval() y sin gradientes:
metrics.update(batch['label'], logits.sigmoid())
# Al terminar todos los lotes:
report = metrics.compute()
```

El módulo está probado con casos sintéticos de resultados conocidos; no representa evaluación de modelos entrenados. El entrenador src/training/train.py integra estas métricas por época para train y validación. Accuracy, precision, recall, F1 y AUC se expresan en [0,1], no en porcentaje.

Referencia: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

## Extensión O1 — protocolo espacial y comparativo, 29/09/2026

Estado: marco de evaluación definido; extracción de mapas todavía no implementada. Las métricas binarias espaciales sí están en `src/evaluation/spatial.py` y se verifican mediante casos sintéticos. No son resultados de localización de modelos entrenados.

### Clasificación y desbalance

La clase 1 es presencia según una máscara aceptada, no confirmación visual independiente de huayco. Umbral 0,5 fijo para el experimento inicial. Acumular probabilidades y etiquetas por conjunto, no promediar F1/AUC por lote. Reportar ambas clases en matriz y conteos; ROC-AUC indefinida devuelve null. Con desbalance, acompañar Accuracy con Recall/F1 y resultados por procedencia. Predictores constantes son referencias descriptivas; no seleccionar estrategias mirando test. La política de balanceo e inicialización se define en `experimental_protocol.md`.

### Métodos candidatos y criterio de representación

- ResNet18/50: Grad-CAM sobre salida espacial del último bloque convolucional (`backbone.layer4`), dirigido al logit de clase positiva antes de sigmoid. Si se comparan capas de mayor resolución, declarar el mismo presupuesto de selección por familia y usar solo validación.
- DeiT-S/16: atribución por gradiente sobre tokens espaciales de una capa anterior a la última mezcla de atención (candidata `backbone.blocks[-1].norm1`), excluyendo token CLS y reorganizando 64 tokens en 8×8 para entrada 128. Verificar gradientes no nulos y correspondencia de tokens; no usar directamente los tokens finales cuya relación con la cabeza CLS puede producir gradiente espacial nulo. Attention rollout es una alternativa exploratoria, no necesariamente específica de clase.
- MiT-B1: Grad-CAM sobre la salida espacial final del encoder que alimenta el promedio global, sin decoder. Verificar hook y forma N,C,H,W. No aplicar sin adaptación el rollout de CLS de DeiT: MiT no usa esa misma estructura y tiene reducción espacial jerárquica.

Estas selecciones son candidatas implementables, no módulos ya ejecutados. Antes de evaluar: verificar forma de activaciones, dependencia respecto al logit, gradientes finitos/no triviales y localización de un estímulo sintético. Registrar capa y método. No concluir que una diferencia espacial se debe solo a CNN/ViT cuando intervienen métodos o resoluciones diferentes.

Fuentes metodológicas primarias: [Selvaraju et al., Grad-CAM](https://arxiv.org/abs/1610.02391) y [Abnar y Zuidema, Attention Flow](https://aclanthology.org/2020.acl-main.385/). La elección concreta de capas anterior es una propuesta de este protocolo que deberá verificarse en el código, no una afirmación de los artículos sobre estos modelos particulares.

### De mapa a región

1. Congelar checkpoint seleccionado por clasificación en validación.
2. Obtener mapa del logit positivo, aplicar ReLU al mapa firmado para estudiar evidencia favorable a la clase; documentar si el método produce solo valores no negativos.
3. Recuperar resolución 128×128 mediante interpolación bilineal, `align_corners=False`. No transformar ejes arbitrariamente. Las coordenadas de salida deben coincidir con la imagen y con una máscara cuya alineación esté validada.
4. Normalizar por imagen usando (H−min H)/(max H−min H). Mapa constante → todo cero, registrar `constant_map=true`; NaN/Inf → error, no mapa vacío silencioso. Esta normalización de mapas no es normalización de entrada radiométrica.
5. Umbralizar con H≥τ. Como regla prospectiva de búsqueda, evaluar τ en {0,05;0,10;…;0,95}: una malla común de resolución 0,05, no umbrales ajustados usando test. Conservar curvas completas de validación.
6. Seleccionar por combinación modelo/método un único τ que maximice IoU macro en positivos de validación, incluyendo falsos negativos de clasificación como máscaras vacías. En empates elegir mayor τ para favorecer la región más conservadora. Si no hay positivos o todos los mapas son degenerados, declarar calibración fallida. No elegir un umbral por imagen según su máscara.
7. Congelar τ, método y normalización antes de test. Reportar sus valores para comparar justamente; el mismo procedimiento de selección no implica el mismo umbral numérico para todos los modelos.

Salida principal del sistema: si p<0,5, región vacía; si p≥0,5, región umbralizada. Esto contabiliza fallos de detección en la localización. Presentar opcionalmente el mapa sin esta compuerta como análisis separado de atribución; nunca reportar solo los verdaderos positivos como desempeño total.

### Fórmulas y casos vacíos

Para píxeles válidos: TP=|P∩G|, FP=|P\G|, FN=|G\P|, TN=|complemento(P∪G)|.

- IoU = TP/(TP+FP+FN).
- Dice = 2TP/(2TP+FP+FN).
- Pixel Precision = TP/(TP+FP).
- Pixel Recall = TP/(TP+FN).

`spatial_metrics` exige máscaras binarias 2D alineadas y ofrece `valid_mask` explícita cuando hay NoData confirmado. No deduce validez del fondo. Si no queda ningún píxel válido, produce error y esa imagen requiere revisión, no una puntuación ficticia.

| Referencia | Predicción | Convención |
|---|---|---|
| No vacía | Vacía | IoU=Dice=Recall=0; Precision=null |
| Vacía | No vacía | IoU=Dice=Precision=0; Recall=null |
| Vacía | Vacía | IoU, Dice, Precision y Recall=null; registrar acierto de ausencia por clasificación |
| No vacía | No vacía | Fórmulas; cualquier denominador cero queda null |

Se reportan valores indefinidos y denominadores. Para macro IoU/Dice usar todas las imágenes GT positivas, incluyendo predicción vacía; no favorecer métodos que solo devuelven regiones en casos fáciles. Para cada métrica adicional indicar cantidad definida. Para micro acumular TP/FP/FN sobre todos los píxeles válidos de todas las imágenes, incluidos negativos, y aplicar la fórmula una sola vez. Reportar negativos aparte: fracción con alguna falsa región y proporción de área falsamente positiva. Una predicción vacía en un negativo no aumenta artificialmente el IoU macro de landslides.

Activaciones parciales son evidencia discriminativa incompleta, no segmentaciones supervisadas. Activaciones fuera del fenómeno cuentan como FP, aunque tengan una explicación contextual. Evaluar por procedencia y por tamaño usando cortes fijados en train; nunca recalibrar en test. No se calcula IoU de Roy hasta resolver correspondencia espacial y política NoData de referencia.
# Evaluación de la integración propuesta

El soporte para fuentes mixtas conserva métricas conjuntas y añade resultados
separados por dataset; ver [l4s_roy_14_v1.md](datasets/l4s_roy_14_v1.md).
La propuesta todavía no está habilitada para entrenamiento. No comparar directamente
el efecto de Roy contra el experimento L4S anterior sin controlar el cambio de
normalización, aunque ambos conserven 14 canales.
