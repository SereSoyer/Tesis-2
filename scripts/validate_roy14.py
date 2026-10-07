"""Verify reconstructed files and proposed splits, without admitting them to training."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.integration import sha256
from src.datasets.manifests import array_digest, read_manifest, validate_partition_integrity
from src.datasets.protocol import allowed_sources
from src.preprocessing.multispectral import Preprocess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(root, report):
    summary = json.loads((report/'summary.json').read_text())
    require(summary['status'] == 'reconstruction_complete_joint_training_pending_band_mapping',
            'Reconstruction is not complete or its protocol has changed')
    rows = read_manifest(report/'reconstructed.csv')
    require(len(rows) == summary['reconstructed'], 'Reconstructed count differs')
    prep = Preprocess(list(range(14)), normalization='per_image_minmax', nan_policy='error')
    for row in rows:
        sid = row['sample_id']
        arrays = {}
        for prefix, key in [('image', 'img'), ('mask', 'mask')]:
            path = root/row[prefix+'_path']
            require(sha256(path) == row[prefix+'_file_sha256'], f'File changed: {path}')
            with h5py.File(path) as stream:
                arrays[prefix] = stream[key][:]
                require(stream.attrs['band_mapping_status'] == 'pending_conflicting_sources',
                        f'Unexpected band metadata: {sid}')
            require(array_digest(arrays[prefix]) == row[prefix+'_sha256'],
                    f'Decoded array differs: {sid}/{prefix}')
        x, y = arrays['image'], arrays['mask']
        require(x.shape == (128,128,14) and y.shape == (128,128), f'Unexpected shape: {sid}')
        require(np.isfinite(x).all() and not (x < -1e30).any(), f'Invalid image: {sid}')
        require(np.isin(y, [0,1]).all() and int(y.any()) == int(row['label']), f'Invalid mask/label: {sid}')
        require(int(y.sum()) == int(row['positive_pixels']), f'Area changed: {sid}')
        with h5py.File(root/row['original_mask_path']) as stream:
            require(np.array_equal(y, stream['mask'][:]), f'Original mask changed: {sid}')
        tensor = prep(x)
        require(tensor.shape == (14,128,128) and float(tensor.min()) >= 0
                and float(tensor.max()) <= 1, f'Invalid normalized tensor: {sid}')
    split_dir = root/summary['outputs']['split']
    partitions = {}
    for split in ('train','val','test'):
        path = split_dir/f'{split}.csv'
        require(sha256(path) == summary['proposed_split_sha256'][split], f'Proposal changed: {split}')
        partitions[split] = read_manifest(path)
        require(all(r['status'] == 'pending_review' and r['training_eligible'] == 'false'
                    for r in partitions[split]), 'Pending proposal must not be admitted implicitly')
        original = root/f'splits/l4s_{split}.csv'
        require(sha256(original) == summary['original_l4s_split_sha256'][split], 'L4S manifest changed')
        baseline = {r['sample_id']:r for r in read_manifest(original)}
        current = {r['sample_id']:r for r in partitions[split] if r['dataset'] == 'landslide4sense'}
        require(baseline.keys() == current.keys(), f'L4S assignments changed: {split}')
        for sid, row in baseline.items():
            require(all(current[sid][k] == v for k,v in row.items()), f'L4S metadata changed: {sid}')
    validate_partition_integrity(partitions)
    proposed_roy = [r for partition in partitions.values() for r in partition if r['dataset'] == 'roy_l4s_pe']
    require(Counter(r['sample_id'] for r in proposed_roy) == Counter(r['sample_id'] for r in rows),
            'Proposals do not cover each reconstructed sample exactly once')
    for path in (root/summary['outputs']['config']).glob('*.json'):
        cfg = json.loads(path.read_text())
        require(cfg['training_enabled'] is False, f'Pending config was enabled: {path}')
        try:
            allowed_sources(cfg)
        except ValueError as error:
            require('Dataset preparation pending' in str(error), f'Unexpected block: {error}')
        else:
            raise ValueError(f'Pending configuration is trainable: {path}')
    return dict(status='passed', samples_checked=len(rows), normalized_shape=[14,128,128],
        labels=dict(Counter(r['label'] for r in rows)), split_counts={s:len(r) for s,r in partitions.items()},
        reconstructed_file_and_array_hashes_match=True, original_masks_unchanged=True,
        original_l4s_manifests_and_assignments_unchanged=True, known_groups_disjoint=True,
        joint_training_enabled=False, physical_band_equivalence_verified=False)


def gallery(root, report, output):
    """Fixed train-only examples chosen by label/area, never model performance."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    require(not output.exists(), f'Output already exists: {output}')
    rows = [r for r in read_manifest(report/'reconstructed.csv') if r['partition'] == 'train']
    positives = sorted((r for r in rows if r['label'] == '1'), key=lambda r:int(r['positive_pixels']))
    selected = [positives[0], positives[len(positives)//2], positives[-1],
                next(r for r in rows if r['label'] == '0')]
    prep = Preprocess([3,2,1], normalization='per_image_minmax')
    fig, axes = plt.subplots(4,3,figsize=(10,12), layout='constrained')
    for i,row in enumerate(selected):
        with h5py.File(root/row['image_path']) as stream: rgb = prep(stream['img'][:]).numpy().transpose(1,2,0)
        with h5py.File(root/row['mask_path']) as stream: mask = stream['mask'][:]
        axes[i,0].imshow(rgb)
        axes[i,1].imshow(rgb)
        overlay = np.zeros((*mask.shape,4)); overlay[mask == 1] = (1,0,0,.45)
        axes[i,1].imshow(overlay)
        axes[i,2].imshow(mask,cmap='gray',vmin=0,vmax=1)
        axes[i,0].set_title(f"{row['sample_id']} | {int(mask.sum())} px positivos")
        axes[i,1].set_title('Anotación superpuesta')
        axes[i,2].set_title('Máscara original conservada')
        for ax in axes[i]: ax.axis('off')
    fig.suptitle('Roy reconstruido · ejemplos de train · convención geométrica IRIS\nRGB con min–max por banda; esta galería no certifica las identidades físicas')
    fig.savefig(output,dpi=140)
    plt.close(fig)
    return [r['sample_id'] for r in selected]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--report', default='results/audits/roy14_reconstruction_v1')
    parser.add_argument('--output', help='Optional new JSON file to preserve the check')
    parser.add_argument('--gallery', help='Optional new PNG with train examples')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = validate(root, root/args.report)
    if args.gallery:
        result['gallery_train_ids'] = gallery(root, root/args.report, root/args.gallery)
    if args.output:
        with (root/args.output).open('x') as stream: json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
