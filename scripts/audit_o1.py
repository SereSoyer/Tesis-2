"""Revalidate current arrays once; generate a BLOCKED integration inventory.

Run from repository root: python -m scripts.audit_o1.
Never writes raw data, assigns Roy splits, or fits integrated normalization.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from src.datasets.manifests import (array_digest, duplicate_groups, read_manifest,
                                    validate_manifest, validate_partition_integrity)
from src.preprocessing.statistics import ChannelMoments


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--config', default='configs/o1_audit.json')
    args = parser.parse_args()
    root = args.root.resolve()
    cfg = json.loads((root/args.config).read_text())
    if cfg['integrated_training_authorized'] is not False:
        raise ValueError('This auditor cannot approve integration')
    rows, origins = [], {}
    for path in cfg['sources']:
        origins[path] = hashlib.sha256((root/path).read_bytes()).hexdigest()
        rows.extend(read_manifest(root/path))
    validate_manifest(rows, root)
    review = {int(r['point']): r for r in read_manifest(root/'results/audits/roy_repository_review/validation_join.csv')}
    alignment = {r['sample_id']: r for r in read_manifest(root/'results/audits/roy_inspection/alignment.csv')}
    moments = {s: ChannelMoments(14) for s in cfg['statistics_scopes']}
    nonfinite, issues = [], []
    for i, row in enumerate(rows):
        sid = row['sample_id']
        row['original_sample_id'] = sid
        if row['dataset'] == 'roy_l4s_pe':
            row['sample_id'] = 'roy_l4s_pe__'+sid
        with h5py.File(root/row['image_path'], 'r') as stream:
            x = stream['img'][:]
        with h5py.File(root/row['mask_path'], 'r') as stream:
            mask = stream['mask'][:]
        if tuple(x.shape) != tuple(cfg['expected_shape']) or mask.shape != x.shape[:2]:
            raise ValueError(f'Shape mismatch: {sid}')
        if not np.isin(mask, [0, 1]).all():
            raise ValueError(f'Nonbinary mask: {sid}')
        positive = int((mask == 1).sum())
        if int(row['label']) != int(positive > 0) or int(row['positive_pixels']) != positive:
            raise ValueError(f'Label or area changed: {sid}')
        if not np.isclose(float(row['area_ratio']), positive/mask.size, rtol=0, atol=1e-12):
            raise ValueError(f'area_ratio changed: {sid}')
        digest = array_digest(x)
        if row.get('image_sha256') and digest != row['image_sha256']:
            raise ValueError(f'Image changed since earlier audit: {sid}')
        row.update(image_sha256=digest, mask_sha256=array_digest(mask),
                   status='pending_review', training_eligible='false',
                   local_structure_status='passed', geographic_group='unavailable',
                   geographic_independence='unverified')
        row['source_split'] = row.get('source_split', 'unassigned')
        row['split'] = {'TrainData':'train','ValidData':'val','TestData':'test'}.get(row['source_split'], 'unassigned')
        row['split_status'] = 'original_preserved_provisional_integration' if row['split'] != 'unassigned' else 'unassigned'
        reasons = ['cross_dataset_channel_equivalence_unresolved', 'geographic_independence_unverified']
        count = int((~np.isfinite(x)).sum())
        prior_count = int(row.get('nonfinite_values') or int(row.get('nan', 0))+int(row.get('inf', 0)))
        if count != prior_count:
            raise ValueError(f'Nonfinite audit changed: {sid}')
        row['nonfinite_values'] = count
        if count:
            reasons.append('nonfinite_source_review')
            nonfinite.append(dict(sample_id=row['sample_id'], counts_by_channel=[int((~np.isfinite(x[...,b])).sum()) for b in range(14)]))
        if row['dataset'] == 'landslide4sense':
            if row['split'] == 'train':
                moments['l4s_original_train_descriptive'].update(x)
            else:
                reasons.append('evaluation_mask_provenance_unconfirmed')
        else:
            moments['roy_unsplit_descriptive_not_for_fit'].update(x)
            reasons.extend(['exporter_radiometry_unresolved', 'spatial_alignment_unresolved', 'annotation_acceptance_unresolved'])
            entry = review[int(sid.split('_')[-1])]
            for field in ('label','valid','Final','Grupo','Fechas','obs'):
                row['author_'+field] = entry[field]
            row['coordinate_key'] = row['latitude']+','+row['longitude']
            original = root/'data/roy/landslide_ok'/sid
            png = original/'target/manual.png'
            if png.exists():
                a = np.array(Image.open(png).convert('RGB'))
                match = bool(np.array_equal(mask, np.all(a[1:129,1:129] == [255,0,0], axis=-1).T))
                if not match:
                    raise ValueError(f'PNG transformation changed: {sid}')
                row['png_transform_match'] = 'true'
                row['png_nodata_pixels'] = int(np.all(a == [50,50,255], axis=-1).sum())
                if row['png_nodata_pixels']:
                    reasons.append('png_nodata_not_preserved_in_binary_mask')
            else:
                row['png_transform_match'] = 'unavailable'
                reasons.append('original_png_missing')
            if row['label'] == '0':
                reasons.append('negative_annotation_unconfirmed')
            row['historical_tif_crop_match'] = alignment[sid]['image_equals_tif_crop_10_138_transposed']
        row['review_reasons'] = ';'.join(reasons)
        if (i+1) % 500 == 0:
            print(f'Verified {i+1}/{len(rows)} pairs', flush=True)
    by_id = {r['sample_id']:r for r in rows}
    image_groups = duplicate_groups(rows, 'image_sha256')
    coordinate_groups = duplicate_groups(rows, 'coordinate_key')
    for kind, groups in [('identical_image', image_groups), ('same_coordinate', coordinate_groups)]:
        for index, group in enumerate(groups):
            for sid in group:
                by_id[sid]['review_reasons'] += ';'+kind
                by_id[sid][kind+'_group'] = f'{kind}_{index:04d}'
            issues.append(dict(kind=kind, samples=group,
                               distinct_masks=len({by_id[s]['mask_sha256'] for s in group}),
                               treatment='pending_review_no_automatic_exclusion'))
    partitions = {s:[r for r in rows if r['split']==s] for s in ('train','val','test')}
    validate_partition_integrity(partitions)
    out = root/cfg['output_directory']
    write_csv(root/cfg['output_manifest'], rows)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    for split, subset in partitions.items():
        write_csv(root/cfg['provisional_splits_directory']/f'integrated_{split}.csv', subset, fields)
    write_csv(root/cfg['provisional_splits_directory']/'integrated_unassigned.csv', [r for r in rows if r['split']=='unassigned'], fields)
    for status in ('accepted','excluded','pending_review'):
        write_csv(out/f'{status}.csv', [r for r in rows if r['status']==status], fields)
    write_json(out/'channel_statistics.json', dict(normalization_fit=False,
        note='Positional descriptive moments; Roy is unsplit; no validation/test statistics computed.',
        scopes={name: value.report() for name,value in moments.items()}))
    inventory = []
    for dataset in sorted({r['dataset'] for r in rows}):
        subset = [r for r in rows if r['dataset']==dataset]
        inventory.append(dict(dataset=dataset, samples=len(subset), masks=len(subset),
                              positive=sum(r['label']=='1' for r in subset), negative=sum(r['label']=='0' for r in subset),
                              nonfinite_samples=sum(int(r['nonfinite_values'])>0 for r in subset)))
    write_csv(out/'inventory.csv', inventory)
    report = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), reference_date=cfg['reference_date'],
        purpose=cfg['purpose'], integrated_ready=False, normalization_fit=False,
        source_manifest_sha256=origins, inventory=inventory,
        status_counts=dict(Counter(r['status'] for r in rows)), split_seed=None, future_split_seed=cfg['seed'],
        partitions={s:dict(samples=len(v), labels=dict(Counter(r['label'] for r in v)),
                           datasets=dict(Counter(r['dataset'] for r in v))) for s,v in partitions.items()},
        issues=issues, nonfinite=nonfinite, mask_label_pairs_verified=len(rows),
        cross_assigned_split_exact_overlap=False,
        limitations=['No semantic integration approval','No geographic independence certificate',
                     'Historical TIFF audit reused; current PNG transformation checked; no alignment correction',
                     'Current content hashes stored, raw images and masks read only'])
    write_json(out/'summary.json', report)
    print(json.dumps(dict(inventory=inventory, status=report['status_counts'], duplicate_groups=len(image_groups)), indent=2))


if __name__ == '__main__':
    main()
