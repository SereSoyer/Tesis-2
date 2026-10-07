"""Read-only spectral identity clues; this audit does not certify band identity."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


def transition_grid(array, valid, period):
    """Fraction of changes outside the best regularly spaced grid, per axis.

    Nearest-neighbor upsampling of a 60 m grid to aligned 10 m pixels can
    concentrate changes every six pixels. This is evidence of sampling structure,
    not proof of wavelength; other processing can create or destroy the pattern.
    """
    result = {}
    for axis in (0,1):
        a, b = (array[:-1,:],array[1:,:]) if axis == 0 else (array[:,:-1],array[:,1:])
        good = (valid[:-1,:]&valid[1:,:]) if axis == 0 else (valid[:,:-1]&valid[:,1:])
        changes = (a != b)&good
        counts = changes.sum(axis=1-axis)
        phases = np.bincount((np.arange(len(counts))+1)%period, weights=counts, minlength=period)
        result[f'axis{axis}_offgrid_{period}'] = float(1-phases.max()/phases.sum()) if phases.sum() else None
    return result


def audit(root, output):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    folders = sorted((root/'data/roy/landslide_ok').glob('point_*'))
    folders = [p for p in folders if p.is_dir()]
    for folder in folders:
        for name in ('B1','B7','B8','B9','B10','B11'):
            path = folder/'input'/f'{name}.tif'
            with Image.open(path) as image:
                array = np.array(image)
                descriptions = []
                metadata = image.tag_v2.get(42112)
                if metadata:
                    descriptions = [e.text for e in ET.fromstring(metadata).findall('Item')
                                    if e.attrib.get('name') == 'DESCRIPTION']
                description = str(image.tag_v2.get(270, ''))
            valid = np.isfinite(array)&(array > -1e30)
            values = array[valid]
            row = dict(sample_id=folder.name, file=name+'.tif', sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       minimum=float(values.min()), median=float(np.median(values)),
                       mean=float(values.mean(dtype=np.float64)), maximum=float(values.max()),
                       invalid_pixels=int((~valid).sum()), gdal_descriptions=json.dumps(descriptions),
                       image_description=description)
            for period in (2,6): row.update(transition_grid(array, valid, period))
            rows.append(row)
    with (output/'statistics.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(samples=len(folders), files=len(rows), bands={}, status='indirect_evidence_only',
                   band_mapping_changed=False, training_enabled=False,
                   caution='Grid repetition and radiometry do not prove wavelength or source-scene identity.')
    for name in ('B1','B7','B8','B9','B10','B11'):
        band = [r for r in rows if r['file'] == name+'.tif']
        summary['bands'][name] = dict(mean_dn=float(np.mean([r['mean'] for r in band])),
            median_image_median=float(np.median([r['median'] for r in band])),
            min_dn=min(r['minimum'] for r in band), max_dn=max(r['maximum'] for r in band),
            descriptions=sorted({r['gdal_descriptions'] for r in band}),
            grid6_exact_both_axes=sum(all(r[f'axis{a}_offgrid_6'] == 0 for a in (0,1)) for r in band),
            grid6_offgrid_median=[float(np.median([r[f'axis{a}_offgrid_6'] for r in band
                if r[f'axis{a}_offgrid_6'] is not None])) for a in (0,1)])
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.')
    parser.add_argument('--output',default='results/audits/roy_band_identity_v1')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    print(json.dumps(audit(root,root/args.output),indent=2))
