# Weakly Supervised Landslide Detection and Localization

## Acceso para la tesis

- [Estructura del repositorio y evidencias por resultado](ESTRUCTURA_REPOSITORIO.md).
- [Guía pública de reproducción](docs/REPRODUCIBILIDAD.md).
- [Configuraciones activas](configs/experiments/l4s_roy_14_v1/).
- [Particiones admitidas](splits/l4s_roy_14_v1/).

Esta publicación incluye código, configuraciones, manifiestos y reportes ligeros.
Los datasets, ZIP, pesos, entornos, salidas completas de entrenamiento y borradores
de tesis quedan fuera de Git. Los datos externos se preparan siguiendo la guía.
Los archivos locales excluidos no se eliminan.

**Integración Roy (04/10/2026):** reconstruidas y verificadas 806 muestras desde TIFF,
con **14 canales**; 778 positivas y 28 negativas. Se prepararon manifiestos conjuntos
con L4S (5.650 muestras), conservando sus particiones, y configuraciones para los
cuatro modelos. Los sondeos Earth Engine respaldan B9/B10, con coincidencia exacta
en point_0502, y permiten adoptar B13 como pendiente en radianes y B14 como elevación
en metros. La fuente topográfica exacta y el datum permanecen sin certificar;
ver el [análisis B13/B14](results/audits/roy_topography_comparison_v1/README.md).
**Admisión completada:** entrenamiento conjunto habilitado en
`configs/experiments/l4s_roy_14_v1/`, con manifiestos en `splits/l4s_roy_14_v1/`.
El reporte de admisión registra la comprobación de las 5.650 imágenes y los cinco
modelos con entradas de ambas fuentes, sin pasos de optimización. La documentación
histórica menciona 34 pruebas aprobadas; para acreditar una nueva ejecución se debe
conservar su log y versión. El usuario ya reportó train/validación/test
completos en el servidor y entregó las tablas de la corrida con semilla 42.
Ver [primer análisis del test conjunto](results/analysis/l4s_roy_14_v1_test_seed42_user_report/README.md):
ResNet50 obtiene el mayor IoU macro positivo final (7,57 %), con calidad espacial
aún baja. El análisis conserva las tablas recibidas; no verifica directamente
los checkpoints remotos.
Ver [preparación, entrenamiento y reanudación](docs/REPRODUCIBILIDAD.md).
Ver [resultados, exclusiones y siguientes pasos](docs/datasets/l4s_roy_14_v1.md).

**Implementación ejecutable:** ver [guía del pipeline completo](docs/weak_pipeline.md)
y [configuración común](configs/pipeline_l4s.json). Incluye los cuatro clasificadores
elegidos, Grad-CAM adaptado, pseudomáscaras y una variante DeepLabv3+ común.
La configuración L4S histórica conserva su piloto de dos épocas. El experimento
conjunto nuevo usa 50 épocas por clasificador y 50 por segmentador. Las propuestas
antiguas de Roy siguen archivadas/deshabilitadas; no son las configuraciones activas.

## CNN vs Vision Transformers

This repository contains the experimental implementation of the thesis:

"Evaluación comparativa de modelos CNN y Vision Transformers para la detección y localización débilmente supervisada de huaycos en imágenes satelitales."

The project studies whether convolutional neural networks (CNNs) and Vision Transformers (ViTs), trained under the same weakly supervised setting, differ not only in their ability to detect the presence of landslides/huaycos but also in their ability to spatially localize affected regions.

The central experimental idea is:

Image-level supervision
        ↓
CNN / Vision Transformer
        ↓
Classification
        ↓
Localization map
        ↓
Pseudo-label generation
        ↓
Same segmentation model
        ↓
Final segmentation
        ↓
Quantitative comparison

The comparison is designed so that the backbone used to generate the weak localization is the principal independent variable. The downstream segmentation architecture and, as far as possible, the post-processing and evaluation pipeline are kept fixed.

---

## Research Question

How does the use of CNN and Vision Transformer architectures affect:

1. image-level landslide detection performance,
2. the spatial quality of weakly supervised localization maps,
3. the quality of pseudo-labels derived from those maps, and
4. the final segmentation performance obtained when the pseudo-labels are used to train the same segmentation architecture?

The thesis does not assume beforehand that either CNNs or Vision Transformers will perform better.

---

## Weak Supervision Setting

The main weakly supervised training stage uses only image-level labels:

- landslide / huayco present
- landslide / huayco absent

Pixel-level masks must NOT be used to train the CNN or Vision Transformer localization models.

If ground-truth masks are available in the dataset, they are reserved for:

- validation of weak localization,
- quantitative comparison of pseudo-labels,
- evaluation of final segmentation,
- optional fully supervised upper-bound experiments.

This separation is critical to preserve the weakly supervised nature of the experiment.

---

## Main Experimental Pipeline

### Stage 1 — Dataset preparation

Prepare a common train/validation/test partition.

Each sample may contain:

- satellite image,
- image-level binary label,
- ground-truth mask, when available.

The image-level label is used during weakly supervised training.

Ground-truth masks are excluded from the weak training process.

All dataset splits must remain identical for CNN and ViT experiments.

---

### Stage 2 — Image-level classification

Train at least one representative CNN and one representative Vision Transformer under the same classification task.

Baseline architecture families:

CNN:
- ResNet-family model

Transformer:
- ViT or DeiT-family model

Both models solve:

image → landslide / non-landslide

Whenever possible, training conditions should be comparable:

- same dataset split,
- same input information,
- equivalent augmentation strategy,
- documented optimization settings,
- early stopping based on the same validation criterion,
- fixed random seeds.

Classification metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC when applicable

---

### Stage 3 — Weak localization

After training the image classifiers, spatial localization maps are generated without using segmentation masks.

CNN branch:

Image
  ↓
CNN classifier
  ↓
Grad-CAM / CAM-compatible localization
  ↓
Continuous activation map

ViT branch:

Image
  ↓
Vision Transformer classifier
  ↓
Transformer-compatible attribution / attention localization
  ↓
Continuous localization map

The exact localization method used for each architecture must be explicitly documented.

Raw localization maps must be preserved before thresholding or post-processing.

---

### Stage 4 — Localization evaluation

Localization maps are compared with ground-truth masks when those masks are available.

This stage evaluates whether a classifier that correctly predicts the presence of a landslide also correctly identifies its spatial extent.

Relevant metrics include:

- IoU
- Dice coefficient
- pixel-level precision
- pixel-level recall
- pixel-level F1

Threshold-dependent evaluations should record the threshold selection procedure.

Thresholds must not be chosen using the test set.

---

### Stage 5 — Pseudo-label generation

Continuous localization maps are transformed into binary pseudo-masks.

General pipeline:

Localization map
        ↓
Normalization
        ↓
Thresholding
        ↓
Optional spatial refinement
        ↓
Pseudo-mask

Spatial refinement techniques such as Dense CRF may be evaluated.

If refinement is used, the same refinement protocol should be applied to both CNN and ViT branches unless a specific ablation experiment states otherwise.

The following outputs should be preserved:

- raw localization map,
- normalized localization map,
- thresholded mask,
- refined pseudo-mask.

This makes it possible to determine where improvements or degradations occur.

---

### Stage 6 — Common segmentation model

The pseudo-labels generated by the two architecture families are used to train the SAME semantic segmentation architecture.

Example:

CNN pseudo-labels
        ↓
    DeepLabv3+
        ↓
CNN-derived segmentation model


ViT pseudo-labels
        ↓
    DeepLabv3+
        ↓
ViT-derived segmentation model

The segmentation architecture, training procedure, loss function and evaluation protocol must remain equal between the two branches.

This isolates the effect of pseudo-label quality.

The study is therefore not primarily:

DeepLabv3+ vs another segmentation model.

It is:

pseudo-labels generated from CNN localization
vs
pseudo-labels generated from ViT localization.

---

## Optional Fully Supervised Upper Bound

When ground-truth masks are available, an additional segmentation model may be trained directly with the real masks.

Ground-truth masks
        ↓
same segmentation architecture
        ↓
fully supervised segmentation

This experiment provides an upper reference point for estimating the performance loss introduced by weak supervision.

It must be clearly identified as fully supervised and must not be mixed with the main weakly supervised comparison.

---

## Comparison Levels

The project evaluates CNN and ViT at three levels.

LEVEL 1 — Detection

CNN classifier
vs
ViT classifier

Metrics:
Accuracy, Precision, Recall, F1, AUC.


LEVEL 2 — Weak localization / pseudo-label quality

CNN localization
vs
ViT localization

Metrics:
IoU, Dice, pixel Precision, pixel Recall, pixel F1.


LEVEL 3 — Segmentation from pseudo-labels

same segmenter trained using CNN pseudo-labels
vs
same segmenter trained using ViT pseudo-labels

Metrics:
IoU, Dice, Precision, Recall, F1 and other segmentation metrics defined in the experimental protocol.

---

## Key Experimental Principle

Classification performance and localization performance must be analyzed separately.

A model can correctly classify an image while only highlighting a small discriminative part of the landslide.

Therefore:

high classification accuracy != high localization quality

One of the principal goals of the thesis is to determine whether CNNs and Vision Transformers exhibit different behavior regarding this trade-off.

---

## Research Contribution

The intended contribution is not the proposal of a completely new neural architecture.

The contribution is a controlled comparative study of CNN and Vision Transformer architectures across the complete weakly supervised pipeline:

architecture
    ↓
image classification
    ↓
weak localization
    ↓
pseudo-label generation
    ↓
segmentation training
    ↓
final spatial performance

The study evaluates whether differences in architectural representation translate into differences in:

- landslide detection,
- localization completeness,
- pseudo-label quality,
- downstream segmentation quality.

The experiment is motivated by previous weakly supervised landslide extraction approaches based primarily on convolutional architectures, but extends the analysis toward a controlled CNN-versus-Transformer comparison.

---

## Reproducibility

Every experiment should store at minimum:

- architecture,
- dataset version,
- train/validation/test split,
- random seed,
- image bands/channels used,
- input resolution,
- preprocessing configuration,
- augmentation configuration,
- optimizer,
- learning rate,
- scheduler,
- batch size,
- number of epochs,
- early-stopping configuration,
- loss function,
- pretrained weights,
- localization method,
- pseudo-label threshold,
- refinement configuration,
- evaluation metrics,
- model checkpoint,
- experiment identifier.

Experimental results should be reproducible from configuration files whenever possible.

---

## Scope

The primary scope is:

- binary detection of landslide/huayco presence,
- weakly supervised spatial localization,
- pseudo-label generation,
- segmentation from pseudo-labels,
- CNN vs Vision Transformer comparison.

The project is NOT primarily focused on:

- developing a new semantic segmentation architecture,
- reproducing LGAGNet in full,
- reproducing the complete method of Deng et al.,
- optimizing one architecture until it dominates the other,
- using ground-truth masks during weak classifier training.

The central requirement is a fair and reproducible comparison.
