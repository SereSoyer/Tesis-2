"""Read-only Roy admission audit; candidate status is not training approval."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

COMMON_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]


def digest(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.shape).encode() + str(array.dtype).encode()
                          + array.tobytes()).hexdigest()


def footprint(path):
    """Read north-up EPSG:32718 TIFF bounds without resampling the image."""
    with Image.open(path) as image:
        tags = image.tag_v2
        scale, tie = tags[33550], tags[33922]
        keys = tags[34735]
        entries = {keys[i]: tuple(keys[i+1:i+4]) for i in range(4, len(keys), 4)}
        if entries.get(3072) != (0, 1, 32718) or entries.get(1025) != (0, 1, 1):
            raise ValueError('Expected projected EPSG:32718, PixelIsArea TIFF')
        if 34264 in tags or tuple(scale[:2]) != (10., 10.) or tuple(tie[:3]) != (0., 0., 0.):
            raise ValueError('Unexpected TIFF georeferencing')
        x, y = float(tie[3]), float(tie[4])
        return (x, y-image.height*scale[1], x+image.width*scale[0], y)


def spatial_groups(rows):
    """Connected components of intersecting full TIFF footprints, including edges.

    Use all source records before exclusion: an excluded intermediate patch must
    not disconnect related retained patches. This controls observed overlap,
    not independence of unknown acquisitions/events or distant related sites.
    """
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    links = []
    for i, a in enumerate(rows):
        for j in range(i):
            b = rows[j]
            if (max(a['xmin'], b['xmin']) <= min(a['xmax'], b['xmax'])
                    and max(a['ymin'], b['ymin']) <= min(a['ymax'], b['ymax'])):
                parent[find(i)] = find(j)
                links.append((a['sample_id'], b['sample_id']))
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[find(i)].append(row['sample_id'])
    names = {sid: 'roy_overlap_' + min(ids) for ids in components.values() for sid in ids}
    for row in rows:
        row['geographic_group'] = names[row['sample_id']]
    return links


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(root, output):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f'Review output already exists: {output}')
    with (root/'results/audits/roy_repository_review/validation_join.csv').open() as stream:
        review = {f"point_{int(r['point']):04d}": r for r in csv.DictReader(stream)}
    with (root/'metadata/roy.csv').open() as stream:
        inventory = list(csv.DictReader(stream))
    rows = []
    for original in inventory:
        row = dict(original)
        sid = row['sample_id']
        folder = root/'data/roy/landslide_ok'/sid
        with h5py.File(root/row['image_path']) as stream:
            x = stream['img'][:]
        with h5py.File(root/row['mask_path']) as stream:
            y = stream['mask'][:]
        if x.shape != (128, 128, 14) or y.shape != (128, 128) or not np.isin(y, [0, 1]).all():
            raise ValueError(f'Unexpected image/mask structure: {sid}')
        if int(y.any()) != int(row['label']) or int(y.sum()) != int(row['positive_pixels']):
            raise ValueError(f'Stale label/area metadata: {sid}')
        selected = x[..., COMMON_CHANNELS]
        row.update(image_sha256=digest(x), mask_sha256=digest(y), selected_sha256=digest(selected),
                   selected_nonfinite=int((~np.isfinite(selected)).sum()),
                   selected_nodata_sentinel=int((selected < -1e30).sum()))
        bounds = footprint(folder/'input/B1.tif')
        row.update(zip(('xmin', 'ymin', 'xmax', 'ymax'), bounds))
        same_crop = True
        for channel in COMMON_CHANNELS:
            path = folder/f'input/B{channel+1}.tif'
            if footprint(path) != bounds:
                raise ValueError(f'Band georeferencing mismatch: {sid}, {channel}')
            with Image.open(path) as image:
                tif = np.array(image)
            same_crop &= np.array_equal(x[..., channel], tif[10:138, 10:138].T, equal_nan=True)
        row['selected_matches_tiff_crop'] = bool(same_crop)
        png = folder/'target/manual.png'
        row.update(png_crop_matches=False, png_nodata_pixels=0, png_unknown_pixels=0)
        if png.exists():
            with Image.open(png) as image:
                rgb = np.array(image.convert('RGB'))
            if rgb.shape != (130, 130, 3):
                raise ValueError(f'Unexpected annotation dimensions: {sid}')
            red = np.all(rgb == (255, 0, 0), axis=-1)
            green = np.all(rgb == (0, 255, 0), axis=-1)
            blue = np.all(rgb == (50, 50, 255), axis=-1)
            row['png_crop_matches'] = bool(np.array_equal(y, red[1:129, 1:129].T))
            row['png_nodata_pixels'] = int(blue.sum())
            row['png_unknown_pixels'] = int((~(red | green | blue)).sum())
        source = review[sid]
        row.update(author_final=source['Final'], author_label=source['label'],
                   author_valid=source['valid'], author_observation=source['obs'],
                   author_group=source['Grupo'], author_date_window=source['Fechas'])
        row.update(status='pending_review', training_eligible=False,
                   candidate=False, review_reasons='')
        rows.append(row)
    duplicates = defaultdict(list)
    for row in rows:
        duplicates[row['selected_sha256']].append(row)
    conflict_ids = {r['sample_id'] for group in duplicates.values() if len(group) > 1
                    and len({r['mask_sha256'] for r in group}) > 1 for r in group}
    links = spatial_groups(rows)
    for row in rows:
        reasons = []
        if row['sample_id'] in conflict_ids: reasons.append('identical_input_conflicting_masks')
        if not row['png_crop_matches']: reasons.append('missing_or_mismatched_original_png')
        if row['png_nodata_pixels'] or row['png_unknown_pixels']: reasons.append('annotation_nodata_or_unknown')
        if row['selected_nonfinite'] or row['selected_nodata_sentinel']: reasons.append('invalid_selected_channels')
        if not row['selected_matches_tiff_crop']: reasons.append('selected_channels_untraced')
        if row['author_final'].strip().lower() != 'si': reasons.append('author_final_not_selected')
        if int(row['label']) == 0 and row['author_label'].strip().lower() == 'si':
            reasons.append('negative_annotation_conflicts_with_author_label')
        row['candidate'] = not reasons
        row['review_reasons'] = '|'.join(reasons) or 'global_alignment_and_protocol_decision_pending'
    candidates = [r for r in rows if r['candidate']]
    excluded = [r for r in rows if not r['candidate']]
    groups = defaultdict(list)
    for row in candidates:
        groups[row['geographic_group']].append(row)
    summary = dict(purpose='review_only_not_training_approval', source_count=len(rows),
        proposed_channels_zero_based=COMMON_CHANNELS,
        proposed_band_names=['B1','B2','B3','B4','B5','B6','B7','B8','B11','B12'],
        selected_channel_nonfinite=sum(r['selected_nonfinite'] for r in rows),
        selected_channels_tiff_matches=sum(r['selected_matches_tiff_crop'] for r in rows),
        candidates=len(candidates), candidate_labels=dict(Counter(r['label'] for r in candidates)),
        excluded=len(excluded), excluded_ids=[r['sample_id'] for r in excluded],
        exclusion_counts=dict(Counter(s for r in excluded for s in r['review_reasons'].split('|'))),
        selected_input_duplicate_groups=[[r['sample_id'] for r in g] for g in duplicates.values() if len(g)>1],
        overlap_edges=len(links), candidate_geographic_groups=len(groups),
        largest_group=max(map(len, groups.values())),
        groups_with_negatives=sum(any(int(r['label']) == 0 for r in g) for g in groups.values()),
        global_pending=['alignment_policy','common_band_and_scaling_policy','annotation_acceptance',
                        'grouped_split_design','cross_dataset_geographic_independence_unverified'])
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'admission_review.csv', rows)
    write_csv(output/'candidates.csv', candidates)
    write_csv(output/'excluded.csv', excluded)
    write_csv(output/'overlap_edges.csv', [dict(sample_a=a, sample_b=b) for a,b in links])
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary
