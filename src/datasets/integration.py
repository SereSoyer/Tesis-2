"""Build an explicitly approved, versioned L4S/Roy experiment without edits to sources."""
import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from src.datasets.manifests import array_digest, validate_partition_integrity
from src.datasets.protocol import CHANNELS, INTEGRATED, SOURCES, check_rows
from src.datasets.roy_review import write_csv


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grouped_assignment(rows, seed, ratios, trials=2000):
    """Choose group allocations by size/class balance only, never model outcomes.

    Membership of each geographic connected component is immutable. Candidate
    assignments use a fixed RNG and select the smallest squared relative count
    deviation from the requested per-class/per-split counts; ties keep the first.
    """
    if len(ratios) != 3 or min(ratios) <= 0 or not np.isclose(sum(ratios), 1):
        raise ValueError('Provide three positive train/val/test ratios summing to one')
    groups = defaultdict(list)
    for row in rows:
        groups[row['geographic_group']].append(row)
    names = sorted(groups)
    class_counts = np.array([[sum(int(r['label']) == c for r in groups[g]) for c in (0, 1)]
                             for g in names])
    target = np.array(ratios)[:, None] * class_counts.sum(axis=0)[None, :]
    rng = np.random.default_rng(seed)
    cuts = [int(len(names)*ratios[0]), int(len(names)*(ratios[0]+ratios[1]))]
    best = None
    for _ in range(trials):
        allocations = np.split(rng.permutation(len(names)), cuts)
        counts = np.array([class_counts[a].sum(axis=0) for a in allocations])
        if (counts == 0).any():
            continue
        score = float(np.square((counts-target)/np.maximum(target, 1)).sum())
        if best is None or score < best[0]:
            best = score, allocations
    if best is None:
        raise ValueError('No grouped allocation with both classes in every split')
    result = {}
    for split, indices in zip(('train', 'val', 'test'), best[1]):
        for index in indices:
            for row in groups[names[index]]:
                result[row['sample_id']] = split
    return result


def reconstruct_roy(folder):
    """Apply local IRIS mask_area [10,10,140,140] + one-pixel inner margin.

    The transposition matches the distributed HDF5 orientation for BOTH arrays.
    Keep source band positions; only common CHANNELS enter model preprocessing.
    This reconstruction is an explicit research decision, not a recovered author
    exporter. Original TIFF/PNG/HDF5 remain unchanged.
    """
    planes = []
    for index in range(1, 15):
        with Image.open(folder/f'input/B{index}.tif') as image:
            plane = np.array(image)
        if plane.shape != (150, 150):
            raise ValueError('Unexpected TIFF shape')
        planes.append(plane[11:139, 11:139].T)
    x = np.stack(planes, axis=-1).astype(np.float64)
    with Image.open(folder/'target/manual.png') as image:
        rgb = np.array(image.convert('RGB'))
    if rgb.shape != (130, 130, 3):
        raise ValueError('Unexpected PNG shape')
    red, green = (np.all(rgb == color, axis=-1) for color in ((255, 0, 0), (0, 255, 0)))
    if not (red | green).all():
        raise ValueError('NoData/unknown annotation colors require separate validity handling')
    y = red[1:129, 1:129].T.astype(np.uint8)
    selected = x[..., CHANNELS]
    if not np.isfinite(selected).all() or (selected < -1e30).any():
        raise ValueError('Invalid selected bands after reconstruction')
    return x, y


def prepare(root, policy_path):
    root = Path(root).resolve()
    policy = json.loads(Path(policy_path).read_text())
    required = dict(common_bands='common10', normalization='per_image_minmax',
                    alignment='iris_mask_area_reconstructed_v1', admission='conservative_821_v1')
    if policy.get('decisions') != required or policy.get('training_authorized') is not True:
        raise ValueError('Methodological decisions remain pending; do not prepare trainable manifests')
    review_dir = root/policy['review_directory']
    iris_config = root/'data/roy/landslide_ok/landslides.json'
    if json.loads(iris_config.read_text())['segmentation']['mask_area'] != [10, 10, 140, 140]:
        raise ValueError('IRIS mask_area changed; review reconstruction geometry')
    with (review_dir/'admission_review.csv').open() as stream:
        review = list(csv.DictReader(stream))
    candidates = [r for r in review if r['candidate'] == 'True']
    if len(candidates) != 821:
        raise ValueError('Source population changed; review the admission policy again')
    assignments = grouped_assignment(candidates, policy['seed'], policy['roy_split_ratios'])
    data_dir = root/policy['prepared_data_directory']
    split_dir = root/policy['split_directory']
    config_dir = root/policy['config_directory']
    report_dir = root/policy['report_directory']
    for directory in (data_dir, split_dir, config_dir, report_dir):
        if directory.exists():
            raise FileExistsError(f'Versioned output already exists: {directory}')
    for directory in (data_dir/'img', data_dir/'mask', split_dir, config_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=False)
    partitions = {}
    source_hashes = {}
    for split in ('train', 'val', 'test'):
        path = root/f'splits/l4s_{split}.csv'
        source_hashes[str(path.relative_to(root))] = sha256(path)
        with path.open() as stream:
            partitions[split] = [dict(r, partition=split, dataset_version=INTEGRATED,
                                     status='accepted', training_eligible='true',
                                     geographic_group='', alignment_policy='distributed_l4s_unchanged')
                                 for r in csv.DictReader(stream)]
    for source in candidates:
        sid = source['sample_id']
        original_image = root/source['image_path']
        original_mask = root/source['mask_path']
        with h5py.File(original_image) as stream:
            original_x = stream['img'][:]
        with h5py.File(original_mask) as stream:
            original_y = stream['mask'][:]
        if array_digest(original_x) != source['image_sha256'] or array_digest(original_y) != source['mask_sha256']:
            raise ValueError(f'Source changed after admission review: {sid}')
        folder = root/'data/roy/landslide_ok'/sid
        x, y = reconstruct_roy(folder)
        if not np.array_equal(y, original_y):
            raise ValueError(f'Reconstructed mask differs from the reviewed mask: {sid}')
        image_path, mask_path = data_dir/'img'/f'{sid}.h5', data_dir/'mask'/f'{sid}.h5'
        for path, key, array in ((image_path, 'img', x), (mask_path, 'mask', y)):
            with h5py.File(path, 'w') as stream:
                stream.create_dataset(key, data=array, compression='gzip')
                stream.attrs['dataset_version'] = INTEGRATED
                stream.attrs['alignment_policy'] = required['alignment']
        row = dict(source, image_path=str(image_path.relative_to(root)), mask_path=str(mask_path.relative_to(root)),
                   original_image_path=source['image_path'], original_mask_path=source['mask_path'],
                   original_image_sha256=source['image_sha256'], original_mask_sha256=source['mask_sha256'],
                   image_sha256=array_digest(x), mask_sha256=array_digest(y),
                   selected_sha256=array_digest(x[..., CHANNELS]),
                   selected_nonfinite=0, nonfinite_values=int((~np.isfinite(x)).sum()),
                   image_tiff_crop='[11:139,11:139].T', mask_png_crop='[1:129,1:129].T',
                   source_split='RoyOriginal', partition=assignments[sid], dataset_version=INTEGRATED,
                   status='accepted', training_eligible='true', alignment_policy=required['alignment'])
        row['source_file_sha256'] = json.dumps({str(p.relative_to(root)): sha256(p)
            for p in [*(folder/'input'/f'B{i}.tif' for i in range(1,15)), folder/'target/manual.png']}, sort_keys=True)
        partitions[row['partition']].append(row)
    validate_partition_integrity(partitions)
    fields = sorted({key for rows in partitions.values() for row in rows for key in row})
    for split, rows in partitions.items():
        write_csv(split_dir/f'{split}.csv', [{key: row.get(key, '') for key in fields} for row in rows])
    cfg = json.loads((root/'configs/pipeline_l4s.json').read_text())
    cfg.update(dataset=INTEGRATED, source_datasets=SOURCES, dataset_version=INTEGRATED,
               split_version=INTEGRATED+f'_seed{policy["seed"]}', seed=policy['seed'])
    cfg['preprocessing'].update(channels=CHANNELS, normalization='per_image_minmax', nan_policy='error')
    cfg['segmentation']['epochs'] = 50
    cfg['manifests'] = {s: str((split_dir/f'{s}.csv').relative_to(root)) for s in partitions}
    for split, rows in partitions.items():
        check_rows(cfg, rows, split)
    for model in cfg['branches']:
        exp = f'results/experiments/{INTEGRATED}_{model}_50ep_seed{policy["seed"]}'
        cfg['branches'][model]['checkpoint'] = exp + '/best.pt'
        classifier = json.loads((root/f'configs/train_l4s_{model}.json').read_text())
        classifier.update(dataset=INTEGRATED, source_datasets=SOURCES, dataset_version=INTEGRATED,
                          split_version=cfg['split_version'], seed=policy['seed'],
                          preprocessing=copy.deepcopy(cfg['preprocessing']),
                          manifests={s: cfg['manifests'][s] for s in ('train', 'val')})
        (config_dir/f'train_{model}.json').write_text(json.dumps(classifier, indent=2)+'\n')
    (config_dir/'pipeline.json').write_text(json.dumps(cfg, indent=2)+'\n')
    summary = dict(dataset_version=INTEGRATED, policy=policy, source_manifest_sha256=source_hashes,
        iris_config_sha256=sha256(iris_config),
        review_sha256=sha256(review_dir/'admission_review.csv'),
        split_sha256={s: sha256(split_dir/f'{s}.csv') for s in partitions},
        counts={s: {d: dict(Counter(r['label'] for r in rows if r['dataset'] == d)) for d in SOURCES}
                for s, rows in partitions.items()},
        limitations=['Unknown L4S geography prevents certified cross-dataset geographic independence.',
                     'IRIS alignment reconstruction follows documented upstream convention, not recovered author exporter.',
                     'Per-image min-max does not make sensors/acquisitions radiometrically equivalent.',
                     'Roy image labels and masks are inherited annotations, not newly expert-validated.',
                     'L4S test has already been inspected in a previous experiment.'])
    (report_dir/'preparation.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary
