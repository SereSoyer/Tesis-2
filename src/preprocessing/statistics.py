"""Stable streaming channel moments. Callers determine an explicit fit scope."""
import numpy as np


class ChannelMoments:
    def __init__(self, channels):
        self.count = np.zeros(channels, dtype=np.int64)
        self.bad = self.count.copy()
        self.mean = np.zeros(channels, dtype=np.float64)
        self.m2 = self.mean.copy()
        self.low = np.full(channels, np.inf)
        self.high = np.full(channels, -np.inf)

    def update(self, image):
        if image.ndim != 3 or image.shape[-1] != len(self.count):
            raise ValueError('Channel shape mismatch')
        for i in range(len(self.count)):
            band = image[..., i].astype(np.float64)
            values = band[np.isfinite(band)]
            self.bad[i] += band.size-values.size
            n = values.size
            if not n:
                continue
            mean = values.mean()
            delta = mean-self.mean[i]
            total = self.count[i]+n
            self.m2[i] += ((values-mean)**2).sum()+delta**2*self.count[i]*n/total
            self.mean[i] += delta*n/total
            self.count[i] = total
            self.low[i] = min(self.low[i], values.min())
            self.high[i] = max(self.high[i], values.max())

    def report(self):
        return [dict(channel_index=i, finite_count=int(n), nonfinite_count=int(self.bad[i]),
                     nonfinite_percent=float(100*self.bad[i]/(n+self.bad[i])) if n+self.bad[i] else None,
                     mean=float(self.mean[i]) if n else None,
                     std=float(np.sqrt(self.m2[i]/n)) if n else None,
                     minimum=float(self.low[i]) if n else None,
                     maximum=float(self.high[i]) if n else None)
                for i, n in enumerate(self.count)]
