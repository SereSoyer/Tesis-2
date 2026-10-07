"""Export a real L4S patch as RGB JPEG without annotations.

Official channel order: B1..B12, slope, DEM. RGB = B4/B3/B2
(zero-based indices 3/2/1). Per-channel 2nd/98th percentile stretch
is display-only; original arrays and training preprocessing are untouched.
"""
from pathlib import Path
import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT/'data/landslide4sense/TrainData/img/image_1.h5'
    destination = ROOT/'docs/o1_evidence/landslide4sense_rgb.jpg'
    with h5py.File(source, 'r') as stream:
        image = stream['img'][:]
    rgb = image[..., [3, 2, 1]].astype(np.float64)
    if not np.isfinite(rgb).all():
        raise ValueError('Nonfinite RGB values')
    for channel in range(3):
        band = rgb[..., channel]
        low, high = np.percentile(band, [2, 98])
        rgb[..., channel] = np.clip((band-low)/(high-low), 0, 1) if high > low else 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(rgb*255).astype(np.uint8)).save(destination, quality=98, subsampling=0)
    print(destination.relative_to(ROOT))


if __name__ == '__main__':
    main()
