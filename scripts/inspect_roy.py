"""Read-only audit of Roy / L4S-PE sources; writes derived reports only.
Run from repository root: .venv/bin/python scripts/inspect_roy.py
Dependencies: numpy, h5py, Pillow.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/audits/roy_inspection'
OUT.mkdir(parents=True, exist_ok=True)
(ROOT / 'metadata').mkdir(exist_ok=True)

def read_h5(path, key):
    with h5py.File(path) as f:
        return f[key][:]

def counts(values):
    return dict(Counter(map(str, values)))

images = {p.stem:p for p in (ROOT/'data/roy/l4s/image_ls4pe').glob('*.h5')}
masks = {p.stem:p for p in (ROOT/'data/roy/l4s/mask_ls4pe').glob('*.h5')}
rows, errors, png_missing = [], [], []
shapes, dtypes, png_shapes, colors = Counter(), Counter(), Counter(), Counter()
band_min, band_max = np.full(14, np.inf), np.full(14, -np.inf)
band_bad = np.zeros(14, dtype=np.int64)
mask_values = Counter()
image_hashes, coordinates = defaultdict(list), defaultdict(list)
image_offsets, mask_offsets = Counter(), Counter()
tif_shapes, tif_resolutions, tif_nodata = Counter(), Counter(), Counter()
original_shapes, original_ranges = Counter(), Counter()
tif_missing, alignment_failures = [], []
for sid in sorted(images.keys() & masks.keys()):
    try:
        x, y = read_h5(images[sid], 'img'), read_h5(masks[sid], 'mask')
        shapes[str(x.shape)] += 1; dtypes[str(x.dtype)] += 1
        vals, nums = np.unique(y, return_counts=True)
        mask_values.update({str(v):int(n) for v,n in zip(vals, nums)})
        valid_mask = bool(np.isin(y, [0,1]).all())
        positive = int(np.count_nonzero(y == 1))
        folder = ROOT/'data/roy/landslide_ok'/sid
        meta = json.loads((folder/'metadata.json').read_text())
        coordinates[tuple(meta['location'])].append(sid)
        image_hashes[hashlib.sha256(x.tobytes()).hexdigest()].append(sid)
        band_bad += (~np.isfinite(x)).sum(axis=(0,1))
        for b in range(14):
            finite = x[:,:,b][np.isfinite(x[:,:,b])]
            if finite.size:
                band_min[b] = min(band_min[b], float(finite.min()))
                band_max[b] = max(band_max[b], float(finite.max()))
        png = folder/'target/manual.png'
        matching_masks = []
        if png.exists():
            a = np.array(Image.open(png).convert('RGB'))
            png_shapes[str(a.shape)] += 1
            c,n = np.unique(a.reshape(-1,3),axis=0,return_counts=True)
            colors.update({str(tuple(map(int,v))):int(k) for v,k in zip(c,n)})
            red = np.all(a == [255,0,0], axis=-1)
            for i in range(red.shape[0]-y.shape[0]+1):
                for j in range(red.shape[1]-y.shape[1]+1):
                    if np.array_equal(red[i:i+y.shape[0],j:j+y.shape[1]],y):
                        matching_masks.append([i,j]); mask_offsets[str((i,j))] += 1
            if not matching_masks: alignment_failures.append(sid)
        else: png_missing.append(sid)
        # Identify crop and channel correspondence using actual TIFF values.
        tifs = []
        for b in range(1,15):
            p = folder/f'input/B{b}.tif'
            if not p.exists(): tif_missing.append(str(p.relative_to(ROOT))); continue
            with Image.open(p) as im:
                tifs.append(np.array(im))
                tif_shapes[str(im.size)] += 1
                tif_resolutions[str(im.tag_v2.get(33550))] += 1
                tif_nodata[str(im.tag_v2.get(42113))] += 1
        matching_images = []
        if len(tifs)==14:
            t = np.stack(tifs, axis=-1)
            # Candidate offsets found with one nonconstant band, verify every channel.
            for i in range(t.shape[0]-x.shape[0]+1):
                for j in range(t.shape[1]-x.shape[1]+1):
                    if np.array_equal(t[i:i+128,j:j+128,0], x[:,:,0], equal_nan=True):
                        if np.array_equal(t[i:i+128,j:j+128,:], x, equal_nan=True):
                            matching_images.append([i,j]); image_offsets[str((i,j))] += 1
        original = read_h5(folder/f'{sid}.h5','img')
        original_shapes[str(original.shape)] += 1
        original_ranges[str((float(np.nanmin(original)),float(np.nanmax(original))))] += 1
        rows.append(dict(sample_id=sid,dataset='roy_l4s_pe',image_path=str(images[sid].relative_to(ROOT)),mask_path=str(masks[sid].relative_to(ROOT)),height=x.shape[0],width=x.shape[1],channels=x.shape[2],dtype=str(x.dtype),mask_dtype=str(y.dtype),mask_binary=valid_mask,label=int(positive>0) if valid_mask else '',positive_pixels=positive,total_pixels=y.size,area_ratio=positive/y.size,nonfinite_values=int((~np.isfinite(x)).sum()),latitude=meta['location'][0],longitude=meta['location'][1],declared_resolution=meta.get('resolution'),spacecraft_id=meta.get('spacecraft_id'),original_png_exists=png.exists(),png_matching_offsets=json.dumps(matching_masks),tif_matching_offsets=json.dumps(matching_images)))
    except Exception as exc:
        errors.append({'sample_id':sid,'error':repr(exc)})

with (ROOT/'metadata/roy.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
areas = np.array([r['area_ratio'] for r in rows if r['label']==1])
report = dict(image_count=len(images),mask_count=len(masks),audited_pairs=len(rows),errors=errors,images_without_masks=sorted(images.keys()-masks.keys()),masks_without_images=sorted(masks.keys()-images.keys()),image_shapes=dict(shapes),image_dtypes=dict(dtypes),mask_values=dict(mask_values),labels=counts(r['label'] for r in rows),png_missing=png_missing,png_shapes=dict(png_shapes),png_colors=dict(colors),png_without_matching_crop=alignment_failures,png_matching_offsets=dict(mask_offsets),tif_matching_offsets=dict(image_offsets),images_without_exact_tif_crop=[r['sample_id'] for r in rows if r['tif_matching_offsets']=='[]'],tif_missing=tif_missing,tif_shapes=dict(tif_shapes),tif_pixel_scales=dict(tif_resolutions),tif_nodata=dict(tif_nodata),original_h5_shapes=dict(original_shapes),original_h5_ranges=dict(original_ranges),duplicate_images=[v for v in image_hashes.values() if len(v)>1],duplicate_coordinates=[v for v in coordinates.values() if len(v)>1],positive_area_ratio={'min':float(areas.min()),'max':float(areas.max()),'mean':float(areas.mean()),**{f'p{p}':float(np.percentile(areas,p)) for p in [25,33,50,66,75]}},band_statistics=[{'index_zero_based':i,'file':f'B{i+1}.tif','min':float(band_min[i]),'max':float(band_max[i]),'nonfinite':int(band_bad[i])} for i in range(14)],versions={'numpy':np.__version__,'h5py':h5py.__version__})
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
