"""Explicit per-image preprocessing; no dataset-wide statistics are fitted here."""
import numpy as np
import torch

class Preprocess:
    def __init__(self, channels, nan_policy='error', normalization='none', mean=None, std=None):
        if not channels or len(set(channels)) != len(channels) or any(type(c) is not int or c < 0 for c in channels):
            raise ValueError('channels must be unique nonnegative integer indices')
        if nan_policy not in ('error', 'zero') or normalization not in ('none', 'per_image_minmax', 'standardize'):
            raise ValueError('Unsupported preprocessing policy')
        self.channels, self.nan_policy, self.normalization = channels, nan_policy, normalization
        self.mean, self.std = mean, std
        if normalization == 'standardize':
            if mean is None or std is None or len(mean) != len(channels) or len(std) != len(channels):
                raise ValueError('Provide train-only mean/std for each selected channel')
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(np.asarray(std) <= 0):
                raise ValueError('Invalid mean/std')

    def __call__(self, array):
        if array.ndim != 3 or max(self.channels) >= array.shape[-1]:
            raise ValueError('Expected HWC image with requested channels')
        x = array[..., self.channels].astype(np.float32).transpose(2, 0, 1).copy()
        bad = ~np.isfinite(x)
        if bad.any() and self.nan_policy == 'error':
            raise ValueError('Nonfinite pixels: define an explicit NoData policy')
        # NaN/Inf are excluded from per-image extrema, then filled with zero.
        if self.normalization == 'per_image_minmax':
            for band, invalid in zip(x, bad):
                valid = band[~invalid]
                if valid.size == 0:
                    raise ValueError('Entire selected band is nonfinite')
                low, high = valid.min(), valid.max()
                band[~invalid] = (valid-low)/(high-low) if high > low else 0
        elif self.normalization == 'standardize':
            x = (x-np.asarray(self.mean, dtype=np.float32)[:,None,None])/np.asarray(self.std, dtype=np.float32)[:,None,None]
        x[bad] = 0
        if not np.isfinite(x).all():
            raise ValueError('Preprocessing produced nonfinite values')
        return torch.from_numpy(np.ascontiguousarray(x))
