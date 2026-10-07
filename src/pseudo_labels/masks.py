import numpy as np


def normalize_map(raw):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or not raw.size or not np.isfinite(raw).all():
        raise ValueError('Expected finite nonempty 2D map')
    lo, hi = float(raw.min()), float(raw.max())
    constant = hi == lo
    return (np.zeros_like(raw) if constant else (raw - lo) / (hi - lo)), constant


def pseudo_mask(normalized, probability, threshold, classification_threshold=0.5):
    """Gate by predicted presence; never gate by a ground-truth image label."""
    m = np.asarray(normalized)
    if m.ndim != 2 or not m.size or not np.isfinite(m).all() or ((m < 0) | (m > 1)).any():
        raise ValueError('Expected normalized finite 2D map')
    if not 0 < threshold < 1 or not 0 <= classification_threshold <= 1:
        raise ValueError('Invalid threshold')
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Invalid classification probability')
    return ((m >= threshold) & (probability >= classification_threshold)).astype(np.uint8)
