"""Image-level binary metrics. No masks or threshold fitting."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

def _vector(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError('Expected a nonempty finite vector of shape (N,) or (N,1)')
    return array

def _validate(labels, probabilities):
    y, p = _vector(labels), _vector(probabilities)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Matching shapes and binary labels are required')
    if np.any((p < 0) | (p > 1)):
        raise ValueError('Expected probabilities in [0,1]; apply sigmoid to logits first')
    return y.astype(np.int64), p

def classification_metrics(labels, probabilities, *, threshold=0.5):
    """Return JSON-safe metrics; class 1 is positive, prediction uses p >= threshold.

    Undefined precision/recall/F1 use zero with explicit flags. Single-class
    ROC-AUC is None. Compute over the entire epoch/split, not per-batch means.
    """
    if not np.isscalar(threshold) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be a finite scalar in [0,1]')
    y, p = _validate(labels, probabilities)
    pred = (p >= threshold).astype(np.int64)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())
    both_classes = np.unique(y).size == 2
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, p)) if both_classes else None,
        'confusion_matrix': cm.tolist(), 'confusion_matrix_labels': [0, 1],
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'n_samples': int(y.size), 'n_positive': int(y.sum()), 'n_negative': int((y==0).sum()),
        'threshold': float(threshold), 'positive_class': 1,
        'undefined_metrics': [name for name, missing in (
            ('precision', tp+fp == 0), ('recall', tp+fn == 0),
            ('f1', 2*tp+fp+fn == 0), ('roc_auc', not both_classes)) if missing],
        'zero_division': 0,
    }

class ClassificationMetrics:
    """Collect detached batches for exact whole-epoch metrics (including ROC-AUC)."""
    def __init__(self, threshold=0.5):
        classification_metrics([0,1],[0.,1.],threshold=threshold)
        self.threshold = threshold
        self.reset()

    def reset(self):
        self._labels, self._probabilities = [], []

    def update(self, labels, probabilities):
        y, p = _validate(labels, probabilities)
        self._labels.append(y.copy())
        self._probabilities.append(p.copy())

    def compute(self):
        if not self._labels:
            raise ValueError('No samples accumulated')
        return classification_metrics(np.concatenate(self._labels), np.concatenate(self._probabilities), threshold=self.threshold)
