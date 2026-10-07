"""Check a returned EE probe against local TIFFs and report both band hypotheses."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

CONTROL = ['B2','B3','B4','B8']
REFERENCE = CONTROL + ['B9','B10']
SATELLITE = CONTROL + ['B8A','B9','B10']
PAIRS = [(b,b) for b in CONTROL] + [('B9','B9'),('B9','B8A'),('B10','B10'),('B10','B9')]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(reference, candidate):
    x, y = np.asarray(reference,dtype=np.float64), np.asarray(candidate,dtype=np.float64)
    if not x.size or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Metrics require finite corresponding nonempty values')
    delta = y-x
    return dict(n=int(x.size), rmse=float(np.sqrt(np.mean(delta**2))),
                mae=float(np.mean(np.abs(delta))), bias_candidate_minus_reference=float(delta.mean()),
                max_absolute_error=float(np.max(np.abs(delta))), exact_pixels=int(np.sum(delta == 0)),
                pearson_r=float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else None)


def analyze(root, csv_path, reference_path):
    reference = json.loads(reference_path.read_text())
    probes = {p['id']:p for p in reference['probes']}
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Empty probe CSV')
    required = {'sample_id','scene_id','pixel_id','row','col','product_id','date','baseline',
                'mgrs_tile','collection','window_start','window_end_exclusive'}
    required.update('ref_'+b for b in REFERENCE)
    required.update('gee_'+b for b in SATELLITE)
    if not required.issubset(rows[0]):
        raise ValueError('Unexpected CSV schema')
    groups, checked, seen = defaultdict(list), {}, set()
    for row in rows:
        sid, scene = row['sample_id'], row['scene_id']
        if sid not in probes or not scene:
            raise ValueError('Unknown sample or empty scene identifier')
        probe = probes[sid]
        if (row['collection'] != 'COPERNICUS/S2_HARMONIZED'
                or row['window_start'] != probe['start']
                or row['window_end_exclusive'] != probe['end_exclusive']):
            raise ValueError('Returned collection/window differs from the prepared probe')
        if sid not in checked:
            checked[sid] = {}
            for band in REFERENCE:
                path = root/'data/roy/landslide_ok'/sid/'input'/f'{band}.tif'
                if digest(path) != reference['provenance'][sid]['source_hashes'][band]:
                    raise ValueError(f'Source TIFF changed after preparing probe: {path}')
                with Image.open(path) as image:
                    checked[sid][band] = np.array(image)
        expected = {p['id']:p for p in probe['pixels']}
        if row['pixel_id'] not in expected:
            raise ValueError('Unexpected reference pixel')
        pixel = expected[row['pixel_id']]
        r, c = int(row['row']), int(row['col'])
        if (r,c) != (pixel['row'],pixel['col']):
            raise ValueError('Pixel coordinates do not match reference')
        key = (sid,scene,row['pixel_id'])
        if key in seen:
            raise ValueError('Duplicate sample/scene/pixel')
        seen.add(key)
        for band in REFERENCE:
            value = float(row['ref_'+band])
            if value != pixel['values'][band] or value != float(checked[sid][band][r,c]):
                raise ValueError(f'Returned reference differs from local TIFF: {sid}/{band}/{row["pixel_id"]}')
        if not all(np.isfinite(float(row['gee_'+band])) for band in SATELLITE):
            raise ValueError('Nonfinite Earth Engine values')
        groups[(sid,scene)].append(row)
    results = []
    metadata_keys = ['product_id','date','baseline','mgrs_tile','collection','window_start','window_end_exclusive']
    for (sid,scene), group in groups.items():
        for key in metadata_keys:
            if len({r[key] for r in group}) != 1:
                raise ValueError(f'Inconsistent scene metadata: {key}')
        comparisons = {f'{a}_vs_{b}':metrics([float(r['ref_'+a]) for r in group],
                                           [float(r['gee_'+b]) for r in group]) for a,b in PAIRS}
        controls = metrics([float(r['ref_'+b]) for r in group for b in CONTROL],
                           [float(r['gee_'+b]) for r in group for b in CONTROL])
        expected_ids = {p['id'] for p in probes[sid]['pixels']}
        actual_ids = {r['pixel_id'] for r in group}
        results.append(dict(sample_id=sid,scene_id=scene,metadata={k:group[0][k] for k in metadata_keys},
            returned_pixels=len(group),expected_pixels=len(expected_ids),complete=actual_ids == expected_ids,
            missing_pixel_ids=sorted(expected_ids-actual_ids),control_pooled=controls,comparisons=comparisons))
    results.sort(key=lambda r:(r['sample_id'],not r['complete'],r['control_pooled']['rmse'],r['scene_id']))
    return dict(status='measured_evidence_not_automatic_admission',input_csv_sha256=digest(csv_path),
        prepared_reference_sha256=digest(reference_path),rows=len(rows),samples=sorted(checked),
        source_hashes_and_reference_values_verified=True,ranking='Complete coverage first, then control B2/B3/B4/B8 RMSE only.',
        scenes=results,band_mapping_changed=False,training_enabled=False,
        limitations=['Sparse pixel probe, not a full-patch comparison.',
                    'Returned CSV alone does not prove that all candidate scenes were searched.',
                    'Residual error does not identify its processing/geometric cause.',
                    'Do not generalize a single point to every Roy sample or to topographic layers.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',required=True)
    parser.add_argument('--root',default='.')
    parser.add_argument('--reference',default='results/audits/roy_ee_probe_v1/reference.json')
    parser.add_argument('--output',required=True,help='New report directory; existing outputs are never overwritten')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    csv_path, output = Path(args.csv).resolve(), (root/args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = analyze(root,csv_path,root/args.reference)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(csv_path,output/'input.csv')
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
