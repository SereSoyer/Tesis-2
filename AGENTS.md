# AGENTS.md

## Project Purpose

This repository supports a university thesis comparing CNN and Vision Transformer architectures for weakly supervised landslide/huayco detection and localization in satellite imagery.

All implementation decisions must preserve the central experimental question.

Do not transform the project into a generic segmentation project.

Do not transform the project into a reproduction of a specific paper unless explicitly requested.

---

## Central Experimental Design

The required pipeline is:

image-level labels
        ↓
CNN / Vision Transformer
        ↓
classification
        ↓
weak localization maps
        ↓
pseudo-label generation
        ↓
same segmentation architecture
        ↓
segmentation evaluation

The core comparison is:

CNN-generated weak localization and pseudo-labels
vs
ViT-generated weak localization and pseudo-labels.

---

## Critical Rule: Prevent Ground-Truth Leakage

Ground-truth segmentation masks MUST NOT be used to train the weakly supervised CNN or ViT classifiers.

During weak classifier training, allowed supervision is:

image + image-level label

Ground-truth masks may only be used for:

1. validation/evaluation of localization,
2. validation/evaluation of pseudo-labels,
3. evaluation of final segmentation,
4. explicitly labeled fully supervised upper-bound experiments.

Any code path that accidentally exposes masks to the classifier training pipeline must be treated as a methodological bug.

---

## Fair Comparison Requirement

CNN and ViT experiments must use the same:

- dataset split,
- class definition,
- train/validation/test samples,
- input information,
- evaluation samples,
- metric implementations.

Whenever possible they should also use equivalent:

- preprocessing,
- augmentation,
- optimization budget,
- early-stopping policy,
- number of training trials.

Architecture-specific differences are allowed when technically necessary but must be documented.

Do not silently give one architecture additional information or supervision.

---

## Experiment Structure

Experiments are divided into three principal stages.

### Experiment A: Classification

Train CNN and ViT models using image-level binary labels.

Expected outputs:

- model checkpoint,
- validation history,
- test predictions,
- confusion matrix data,
- Accuracy,
- Precision,
- Recall,
- F1,
- ROC-AUC when applicable.

---

### Experiment B: Weak Localization and Pseudo-labels

Generate localization maps from the trained classifiers.

CNN baseline:
- Grad-CAM or another explicitly configured CAM-compatible method.

ViT baseline:
- a documented Transformer-compatible attribution/localization method such as attention rollout or another selected attribution method.

Do not assume raw Transformer attention is automatically equivalent to Grad-CAM.

The selected ViT localization method must be documented in configuration and experiment metadata.

For every sample, preserve:

- original image identifier,
- raw localization map,
- normalized localization map,
- selected threshold,
- binary pseudo-mask,
- refined pseudo-mask if refinement is enabled.

Evaluate localization against ground-truth masks only when available.

---

### Experiment C: Segmentation from Pseudo-labels

Use pseudo-labels from each weak localization branch to train the SAME segmentation architecture.

Example:

CNN pseudo-labels -> DeepLabv3+
ViT pseudo-labels -> DeepLabv3+

The segmentation model must not change between branches in the primary comparison.

Keep fixed:

- segmentation architecture,
- loss,
- optimizer policy,
- augmentations,
- train/validation/test split,
- evaluation implementation.

The independent variable should remain the source of the pseudo-label.

---

## Optional Upper Bound

A fully supervised segmentation baseline may use real masks:

ground-truth mask -> same segmentation model

This run must be isolated under a clearly labeled experiment type such as:

fully_supervised_upper_bound

Never mix its results with weakly supervised training results without explicitly identifying the difference in supervision.

---

## Metrics

Classification metrics:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC when defined

Localization / pseudo-label metrics:

- IoU
- Dice
- pixel Precision
- pixel Recall
- pixel F1

Segmentation metrics:

- IoU
- Dice
- Precision
- Recall
- F1

Additional metrics can be added, but existing thesis metrics must not be removed without an explicit methodological decision.

---

## Threshold Selection

Pseudo-label thresholds must be selected using training/validation information only.

Never optimize a threshold on the test set.

The test set must remain untouched until final evaluation.

Support experiments such as:

fixed threshold
validation-selected threshold
threshold sensitivity analysis

Record the threshold used for every experiment.

---

## Refinement

Pseudo-label refinement may include Dense CRF or another explicitly approved post-processing method.

For the principal CNN-versus-ViT comparison, use the same refinement method and parameters for both branches wherever technically possible.

Recommended ablation:

raw map
→ thresholded pseudo-mask
→ refined pseudo-mask

This allows the contribution of post-processing to be separated from the contribution of the backbone.

---

## Dataset Split Integrity

Dataset partitions must be deterministic and version-controlled.

Do not regenerate train/validation/test splits automatically unless explicitly requested.

Avoid leakage between splits.

If geographically or temporally related samples may be highly correlated, preserve grouping information and report it.

All models compared within an experiment must use the exact same split manifest.

---

## Reproducibility

Prefer configuration-driven experiments.

Every experiment must have a unique ID and save:

experiment_id
timestamp
git commit if available
dataset_version
split_version
architecture
pretrained_weights
seed
input_channels
input_resolution
preprocessing
augmentation
optimizer
learning_rate
scheduler
batch_size
epochs
early_stopping
loss
checkpoint_path
localization_method
pseudo_label_threshold
refinement_method
metrics

Avoid hard-coding experimental hyperparameters inside model implementation files.

---

## Recommended Repository Separation

Keep conceptual stages separated.

data/
    dataset loading and metadata handling

splits/
    immutable train/val/test manifests

models/
    CNN and ViT classification models

training/
    classifier training logic

localization/
    Grad-CAM, Transformer attribution and map generation

pseudo_labels/
    normalization, thresholding and refinement

segmentation/
    common segmentation model and training

evaluation/
    classification, localization and segmentation metrics

configs/
    experiment configurations

scripts/
    executable experiment entrypoints

outputs/
    experiment artifacts and generated results

Do not tightly couple localization logic with a single architecture.

Use common interfaces where practical.

---

## Coding Principles

Prefer clear, reproducible research code over premature optimization.

Functions should have one clear responsibility.

Avoid hidden transformations inside dataset loaders.

Do not apply preprocessing differently between CNN and ViT unless required and documented.

Do not overwrite experimental outputs.

Do not silently change a dataset split, threshold, metric implementation or pretrained model.

Any methodological change must be visible in configuration.

---

## Result Interpretation

Do not equate classification performance with localization performance.

A classifier may obtain high F1 while producing poor spatial localization.

Results should therefore always be analyzed at the appropriate level:

classification
localization
pseudo-label quality
segmentation

Do not claim that one architecture is superior based only on one metric or one experimental seed.

Generate the measurements; interpretation belongs to the thesis analysis.

---

## Relation to Previous Work

Two methodological references motivate the pipeline:

Deng et al.:
weak image-level supervision
→ improved CAM
→ pseudo-label
→ CRF
→ DeepLabv3+.

Zhang et al. / LGAGNet:
weak image-level supervision
→ CAM refinement using structural/local-global information
→ pseudo-label
→ CRF
→ segmentation.

This project does NOT require reproducing those methods in full.

Their role is to motivate the weak-localization-to-pseudo-label-to-segmentation strategy.

The thesis extends this direction by studying CNN and Vision Transformer backbones under a controlled common pipeline.

---

## Non-Goals

Unless explicitly requested, do NOT:

- implement LGAGNet in full,
- reproduce every module from Deng et al.,
- replace the common segmenter with architecture-specific segmenters,
- use real masks during weak classifier training,
- tune the test set,
- introduce additional datasets into experiments without documenting them,
- modify the scientific question to CNN-vs-segmentation-model or ViT-vs-segmentation-model.

The main thesis question must remain recognizable in every major experiment.