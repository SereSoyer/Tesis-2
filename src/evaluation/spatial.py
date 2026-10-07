"""Binary spatial metrics with explicit undefined cases; no attribution model."""
import numpy as np


def spatial_metrics(target, prediction, valid_mask=None):
    """Return pixel counts and ratios on a single aligned, nonempty 2D grid.

    Empty/empty overlap is undefined (None), not a perfect landslide detection.
    Any zero denominator is recorded. A known validity mask may exclude NoData;
    never derive validity by treating ground-truth background as invalid.
    """
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.ndim != 2 or target.shape != prediction.shape or not target.size:
        raise ValueError('Expected matching nonempty 2D masks')
    if not np.isin(target, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError('Masks must be finite and binary')
    valid = np.ones(target.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask)
    if valid.shape != target.shape or not np.isin(valid, [0, 1]).all():
        raise ValueError('Invalid validity mask')
    valid = valid.astype(bool)
    if not valid.any():
        raise ValueError('No valid pixels')
    t, p = target[valid].astype(bool), prediction[valid].astype(bool)
    tp, fp, fn, tn = (int(v.sum()) for v in (t & p, ~t & p, t & ~p, ~t & ~p))
    values = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'valid_pixels': int(valid.sum())}
    undefined = []
    for name, numerator, denominator in (
        ('iou', tp, tp+fp+fn), ('dice', 2*tp, 2*tp+fp+fn),
        ('pixel_precision', tp, tp+fp), ('pixel_recall', tp, tp+fn)):
        values[name] = numerator/denominator if denominator else None
        if not denominator:
            undefined.append(name)
    values['undefined_metrics'] = undefined
    return values
