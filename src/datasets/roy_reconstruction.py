"""Reconstruct all 14 positional TIFF channels; never assume spectral equivalence.

Builds structurally checked data and proposed joint manifests. Semantic admission
remains separate and every generated training config/manifest is disabled.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from src.datasets.integration import grouped_assignment, sha256
from src.datasets.manifests import array_digest, validate_partition_integrity
from src.datasets.roy_review import footprint, spatial_groups, write_csv
from src.preprocessing.multispectral import Preprocess

VERSION = 'l4s_roy_14_v1'
SOURCES = ['landslide4sense', 'roy_l4s_pe']


def validate_policy(policy):
    """Fail before creating any output if the requested geometry/protocol differs."""
    expected = dict(dataset_version=VERSION, reconstruction_authorized=True,
        training_enabled=False, band_mapping_status='pending_conflicting_sources',
        image_crop=[11,139,11,139], mask_crop=[1,129,1,129], transpose_both=True,
        alignment_policy='iris_mask_area_reconstructed_v1',
        nodata_policy='exclude_sample_no_imputation', normalization='per_image_minmax',
        channel_order=[f'B{i}.tif' for i in range(1,15)])
    for key, value in expected.items():
        if policy.get(key) != value:
            raise ValueError(f'Reconstruction policy mismatch: {key}; review before changing methodology')


def read_tiff_stack(folder, policy):
    """Read the retained crop and its explicit validity from each TIFF channel."""
    planes, bad_counts, statistics, georeferences = [], [], [], []
    r0, r1, c0, c1 = policy['image_crop']
    for filename in policy['channel_order']:
        path = folder/'input'/filename
        bounds = footprint(path)
        georeferences.append(bounds)
        with Image.open(path) as image:
            if image.size != (150,150):
                raise ValueError(f'Unexpected TIFF shape: {path}')
            plane = np.array(image)[r0:r1,c0:c1].astype(np.float64)
            nodata = image.tag_v2.get(42113)
        if policy['transpose_both']:
            plane = plane.T
        invalid = ~np.isfinite(plane) | (plane < -1e30)
        if nodata is not None:
            sentinel = float(str(nodata).strip('\x00'))
            if np.isfinite(sentinel):
                # Source storage can round the textual sentinel to float32.
                invalid |= np.isclose(plane, sentinel, rtol=1e-7, atol=0)
        bad_counts.append(int(invalid.sum()))
        finite = plane[~invalid]
        statistics.append(dict(file=filename, minimum=float(finite.min()) if finite.size else None,
            maximum=float(finite.max()) if finite.size else None, invalid_pixels=int(invalid.sum())))
        planes.append(plane)
    if len(set(georeferences)) != 1:
        raise ValueError(f'TIFF bands have different georeferences: {folder.name}')
    return np.stack(planes, axis=-1), bad_counts, statistics, georeferences[0]


def read_annotation(folder, policy):
    path = folder/'target/manual.png'
    if not path.exists():
        return None, 'missing_original_png'
    with Image.open(path) as image:
        rgb = np.array(image.convert('RGB'))
    if rgb.shape != (130,130,3):
        return None, 'unexpected_png_shape'
    red = np.all(rgb == (255,0,0), axis=-1)
    green = np.all(rgb == (0,255,0), axis=-1)
    if not (red | green).all():
        return None, 'annotation_nodata_or_unknown'
    r0,r1,c0,c1 = policy['mask_crop']
    mask = red[r0:r1,c0:c1]
    return (mask.T if policy['transpose_both'] else mask).astype(np.uint8), None


def write_union_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    write_csv(path, [{key: row.get(key, '') for key in fields} for row in rows])


def safe_output(root, relative):
    path = Path(relative)
    destination = (root/path).resolve()
    if path.is_absolute() or not destination.is_relative_to(root) or destination == root:
        raise ValueError('Output must be a named directory inside the project')
    if destination.exists():
        raise FileExistsError(f'Output already exists: {destination}')
    return destination


def reconstruct(root, policy_path):
    root = Path(root).resolve()
    policy = json.loads(Path(policy_path).read_text())
    validate_policy(policy)
    iris_path = root/'data/roy/landslide_ok/landslides.json'
    if json.loads(iris_path.read_text())['segmentation']['mask_area'] != [10,10,140,140]:
        raise ValueError('IRIS mask_area differs from the authorized reconstruction')
    dirs = {key: safe_output(root, policy[key+'_directory']) for key in
            ('prepared_data','split','config','report')}
    if len(set(dirs.values())) != 4:
        raise ValueError('Output directories must be distinct')
    review_path = root/policy['review_directory']/'admission_review.csv'
    with review_path.open() as stream:
        review = list(csv.DictReader(stream))
    if len(review) != 838 or len({r['sample_id'] for r in review}) != 838:
        raise ValueError('Expected the reviewed 838 unique source records')
    original_manifests = {s:root/f'splits/l4s_{s}.csv' for s in ('train','val','test')}
    original_hashes = {s:sha256(p) for s,p in original_manifests.items()}
    for directory in dirs.values(): directory.mkdir(parents=True, exist_ok=False)
    for sub in ('img','mask'): (dirs['prepared_data']/sub).mkdir()
    records, admitted = [], []
    prep = Preprocess(list(range(14)), normalization=policy['normalization'], nan_policy='error')
    global_min, global_max = np.full(14,np.inf), np.full(14,-np.inf)
    for number, source in enumerate(review,1):
        sid = source['sample_id']
        folder = root/'data/roy/landslide_ok'/sid
        with h5py.File(root/source['image_path']) as stream: original_x = stream['img'][:]
        with h5py.File(root/source['mask_path']) as stream: original_y = stream['mask'][:]
        if (array_digest(original_x) != source['image_sha256']
                or array_digest(original_y) != source['mask_sha256']):
            raise ValueError(f'Source HDF5 changed after review: {sid}')
        x, bad, statistics, bounds = read_tiff_stack(folder, policy)
        y, annotation_issue = read_annotation(folder, policy)
        reasons = [] if source['candidate'] == 'True' else source['review_reasons'].split('|')
        if annotation_issue: reasons.append(annotation_issue)
        if sum(bad): reasons.append('invalid_pixels_in_14_channel_crop')
        if y is not None and not np.array_equal(y, original_y):
            reasons.append('reconstructed_mask_differs_from_reviewed_mask')
        row = dict(source, reasons='|'.join(sorted(set(reasons))), structural_status='excluded' if reasons else 'reconstructed',
                   band_mapping_status=policy['band_mapping_status'], dataset_version=VERSION,
                   status='pending_review', training_eligible='false',
                   invalid_counts_14=json.dumps(bad), band_statistics=json.dumps(statistics),
                   original_image_path=source['image_path'], original_mask_path=source['mask_path'],
                   original_image_sha256=source['image_sha256'], original_mask_sha256=source['mask_sha256'],
                   image_tiff_crop='[11:139,11:139].T', mask_png_crop='[1:129,1:129].T',
                   alignment_policy=policy['alignment_policy'], source_split='RoyOriginal')
        row.update(zip(('xmin','ymin','xmax','ymax'),bounds))
        row['source_file_sha256'] = json.dumps({str(p.relative_to(root)):sha256(p) for p in
            [*(folder/'input'/name for name in policy['channel_order']),
             *([folder/'target/manual.png'] if (folder/'target/manual.png').exists() else [])]}, sort_keys=True)
        if not reasons:
            normalized = prep(x)
            if normalized.shape != (14,128,128) or not np.isfinite(normalized.numpy()).all():
                raise ValueError(f'Unexpected preprocessing output: {sid}')
            global_min = np.minimum(global_min, x.min(axis=(0,1)))
            global_max = np.maximum(global_max, x.max(axis=(0,1)))
            for sub,key,array in (('img','img',x),('mask','mask',y)):
                path = dirs['prepared_data']/sub/f'{sid}.h5'
                with h5py.File(path,'x') as stream:
                    stream.create_dataset(key,data=array,compression='gzip')
                    stream.attrs.update(dataset_version=VERSION, alignment_policy=policy['alignment_policy'],
                                        band_mapping_status=policy['band_mapping_status'])
                    stream.attrs['source_channel_files'] = json.dumps(policy['channel_order'])
                row['image_path' if sub == 'img' else 'mask_path'] = str(path.relative_to(root))
            row.update(image_sha256=array_digest(x), mask_sha256=array_digest(y),
                       selected_sha256=array_digest(x), selected_nonfinite=0,
                       selected_nodata_sentinel=0, nonfinite_values=0,
                       mask_dtype=str(y.dtype), dtype=str(x.dtype),
                       label=str(int(y.any())), positive_pixels=int(y.sum()),
                       total_pixels=y.size, area_ratio=float(y.mean()),
                       normalized_min=float(normalized.min()), normalized_max=float(normalized.max()))
            row['image_file_sha256'] = sha256(root/row['image_path'])
            row['mask_file_sha256'] = sha256(root/row['mask_path'])
            admitted.append(row)
        records.append(row)
        if number%100 == 0: print(f'Roy: revisadas {number}/{len(review)}; reconstruidas {len(admitted)}', flush=True)
    # Recompute overlap components using all current TIFF footprints, even excluded samples.
    links = spatial_groups(records)
    groups_by_id = {r['sample_id']:r['geographic_group'] for r in records}
    for row in admitted: row['geographic_group'] = groups_by_id[row['sample_id']]
    exact = defaultdict(list)
    for row in admitted: exact[row['image_sha256']].append(row)
    if any(len(group)>1 for group in exact.values()):
        raise ValueError('New duplicates after reconstruction; inspect before writing split proposals')
    assignments = grouped_assignment(admitted, policy['seed'], policy['roy_split_ratios'], policy['split_search_trials'])
    partitions = {}
    for split,path in original_manifests.items():
        with path.open() as stream:
            partitions[split] = [dict(r, partition=split, dataset_version=VERSION,
                status='pending_review', training_eligible='false', geographic_group='',
                alignment_policy='distributed_l4s_unchanged', band_mapping_status=policy['band_mapping_status'])
                for r in csv.DictReader(stream)]
        for row in admitted:
            if assignments[row['sample_id']] == split:
                row['partition'] = split
                partitions[split].append(row)
    validate_partition_integrity(partitions)
    for split,rows in partitions.items(): write_union_csv(dirs['split']/f'{split}.csv', rows)
    write_union_csv(dirs['report']/'admission.csv', records)
    write_union_csv(dirs['report']/'reconstructed.csv', admitted)
    write_union_csv(dirs['report']/'excluded.csv', [r for r in records if r['reasons']])
    write_csv(dirs['report']/'overlap_edges.csv', [dict(sample_a=a,sample_b=b) for a,b in links])
    cfg = json.loads((root/'configs/pipeline_l4s.json').read_text())
    cfg.update(dataset=VERSION, source_datasets=SOURCES, dataset_version=VERSION,
               split_version=VERSION+f'_seed{policy["seed"]}', seed=policy['seed'],
               training_enabled=False, disabled_reason=policy['disabled_reason'],
               band_mapping_status=policy['band_mapping_status'])
    cfg['preprocessing'].update(channels=list(range(14)), normalization=policy['normalization'], nan_policy='error')
    cfg['segmentation']['epochs'] = 50
    cfg['manifests'] = {s:str((dirs['split']/f'{s}.csv').relative_to(root)) for s in partitions}
    for model in cfg['branches']:
        cfg['branches'][model]['checkpoint'] = f'results/experiments/{VERSION}_{model}_50ep_seed{policy["seed"]}/best.pt'
        classifier = json.loads((root/f'configs/train_l4s_{model}.json').read_text())
        for key in ('dataset','source_datasets','dataset_version','split_version','seed','training_enabled',
                    'disabled_reason','band_mapping_status','preprocessing'):
            classifier[key] = cfg[key]
        classifier['manifests'] = {s:cfg['manifests'][s] for s in ('train','val')}
        (dirs['config']/f'train_{model}.json').write_text(json.dumps(classifier,indent=2)+'\n')
    (dirs['config']/'pipeline.json').write_text(json.dumps(cfg,indent=2)+'\n')
    for split,path in original_manifests.items():
        if sha256(path) != original_hashes[split]:
            raise ValueError('Original L4S manifest changed during reconstruction')
    summary = dict(status='reconstruction_complete_joint_training_pending_band_mapping',
        dataset_version=VERSION, policy=policy, source_count=len(records), reconstructed=len(admitted),
        labels=dict(Counter(r['label'] for r in admitted)), excluded=len(records)-len(admitted),
        excluded_ids=[r['sample_id'] for r in records if r['reasons']],
        exclusion_counts=dict(Counter(reason for r in records for reason in r['reasons'].split('|') if reason)),
        overlap_edges=len(links), geographic_groups=len({r['geographic_group'] for r in admitted}),
        raw_min_per_channel=global_min.tolist(), raw_max_per_channel=global_max.tolist(),
        proposed_counts={s:{d:dict(Counter(r['label'] for r in rows if r['dataset']==d)) for d in SOURCES}
                         for s,rows in partitions.items()},
        original_l4s_split_sha256=original_hashes,
        proposed_split_sha256={s:sha256(dirs['split']/f'{s}.csv') for s in partitions},
        iris_config_sha256=sha256(iris_path), review_sha256=sha256(review_path),
        outputs={k:str(v.relative_to(root)) for k,v in dirs.items()},
        caveats=['Band names are positional TIFF filenames; physical identities are not certified.',
                 'IRIS upstream geometry is an adopted reconstruction convention.',
                 'Min-max does not establish radiometric equivalence.',
                 'Observed Roy overlap is grouped; event/acquisition independence is not certified.',
                 'L4S split assignments are unchanged; its test has already been evaluated.'])
    (dirs['report']/'policy.json').write_text(json.dumps(policy,indent=2)+'\n')
    # A completed summary is written last. Partial outputs must not be treated as ready.
    (dirs['report']/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
