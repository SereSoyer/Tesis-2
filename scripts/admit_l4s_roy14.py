"""Admit the reviewed joint dataset without changing samples, arrays or split assignments."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_roy14 import validate as validate_reconstruction
from src.datasets.manifests import read_manifest, validate_manifest, validate_partition_integrity
from src.datasets.protocol import EMPIRICAL_MAPPING, check_rows
from src.pipeline.artifacts import sha256, write_rows

VERSION = 'l4s_roy_14_v1'
MODELS = ['resnet18', 'resnet50', 'deit_small', 'mit_b1']
CONFIG_DIR = f'configs/experiments/{VERSION}'
ADMISSION = f'metadata/{VERSION}_admission.json'


def admit_rows(rows, split):
    admitted = []
    for row in rows:
        if (row['status'] != 'pending_review' or row['training_eligible'] != 'false'
                or row['partition'] != split or row['dataset_version'] != VERSION
                or row['band_mapping_status'] != 'pending_conflicting_sources'):
            raise ValueError('Only the unchanged reviewed proposal may be admitted')
        admitted.append(dict(row, status='accepted', training_eligible='true',
                             band_mapping_status=EMPIRICAL_MAPPING,
                             admission_id=VERSION + '_admission_v1'))
    return admitted


def admit(root):
    root = Path(root).resolve()
    split_dir, config_dir, admission_path = root / 'splits' / VERSION, root / CONFIG_DIR, root / ADMISSION
    for path in (split_dir, config_dir, admission_path):
        if path.exists():
            raise FileExistsError(f'Admission already exists; preserve it: {path}')
    reconstruction = root / 'results/audits/roy14_reconstruction_v1'
    validation = validate_reconstruction(root, reconstruction)
    summary = json.loads((reconstruction / 'summary.json').read_text())
    topo = root / 'results/audits/roy_topography_comparison_v1'
    decision = json.loads((topo / 'decision.json').read_text())
    evidence = json.loads((topo / 'analysis.json').read_text())
    if (decision['status'] != 'mapping_supported_source_partially_inferred'
            or decision['analysis_sha256'] != sha256(topo / 'analysis.json')
            or not evidence['complete'] or not evidence['source_hashes_and_reference_values_verified']):
        raise ValueError('Topographic evidence is missing, changed or incomplete')
    spectral = root / 'results/audits/roy_ee_probe_comparison_v1/summary.json'
    if not spectral.is_file():
        raise ValueError('Missing spectral audit')
    partitions = {}
    for split in ('train', 'val', 'test'):
        original = root / summary['outputs']['split'] / f'{split}.csv'
        if sha256(original) != summary['proposed_split_sha256'][split]:
            raise ValueError('Proposed split changed')
        rows = admit_rows(read_manifest(original), split)
        validate_manifest(rows, root, require_accepted=True)
        # File hashes enable checking a server copy without original TIFFs or training masks being decoded.
        for row in rows:
            for kind in ('image', 'mask'):
                value = sha256(root / row[f'{kind}_path'])
                if row.get(f'{kind}_file_sha256') and value != row[f'{kind}_file_sha256']:
                    raise ValueError(f'Changed data file: {row[f"{kind}_path"]}')
                row[f'{kind}_file_sha256'] = value
        partitions[split] = rows
    validate_partition_integrity(partitions)
    split_dir.mkdir(parents=True, exist_ok=False)
    for split, rows in partitions.items():
        write_rows(split_dir / f'{split}.csv', rows)
    manifest_hashes = {s: sha256(split_dir / f'{s}.csv') for s in partitions}
    counts = {s: {source: dict(Counter(r['label'] for r in rows if r['dataset'] == source))
                  for source in ['landslide4sense', 'roy_l4s_pe']} for s, rows in partitions.items()}
    record = dict(
        status='admitted', dataset_version=VERSION, admission_id=VERSION + '_admission_v1',
        timestamp=datetime.now(timezone.utc).isoformat(), band_mapping_status=EMPIRICAL_MAPPING,
        authorization='User requested unification for the first full joint training after accepting order and units.',
        channel_order=[f'B{i}' for i in range(1, 13)] + ['slope', 'elevation'],
        roy_topography_units={'B13': 'radian', 'B14': 'metre'},
        preprocessing={'normalization': 'per_image_minmax', 'nan_policy': 'error', 'channels': list(range(14))},
        samples=sum(map(len, partitions.values())), split_counts={s: len(r) for s, r in partitions.items()},
        class_counts_by_source=counts, manifest_sha256=manifest_hashes,
        proposal_manifest_sha256=summary['proposed_split_sha256'],
        reconstruction_validation=validation,
        evidence_sha256={str(p.relative_to(root)): sha256(p) for p in
                         [reconstruction / 'summary.json', spectral, topo / 'analysis.json', topo / 'decision.json']},
        arrays_modified=False, split_assignments_modified=False, gt_training_policy='Image labels only; segmenter targets pseudo masks only.',
        caveats=[
            'Exact original topographic product, vertical datum and resampling chain are not certified.',
            'Empirical channel mapping is supported by sampled checks, not exhaustive source certification.',
            'Reconstruction follows the documented IRIS geometry convention.',
            'Min-max provides the same preprocessing, not identical physical provenance.',
            'Roy overlap groups are disjoint; event/acquisition and cross-source geographic independence are not certified.',
            'L4S test was evaluated previously; Roy has only four negatives in val and four in test.',
            'Ground-truth validation masks are used for pseudo-threshold and segmenter checkpoint selection.',
            'One seed is the first formal run, not statistical proof of CNN/ViT superiority.',
        ])
    with admission_path.open('x') as stream:
        json.dump(record, stream, indent=2); stream.write('\n')
    config_dir.mkdir(parents=True, exist_ok=False)
    for name in [f'train_{model}.json' for model in MODELS] + ['pipeline.json']:
        cfg = json.loads((root / f'configs/{VERSION}' / name).read_text())
        cfg.update(training_enabled=True, band_mapping_status=EMPIRICAL_MAPPING,
                   data_admission={'path': ADMISSION, 'sha256': sha256(admission_path)},
                   experiment_id=VERSION + '_50ep_seed42')
        cfg.pop('disabled_reason', None)
        cfg['manifests'] = {s: f'splits/{VERSION}/{s}.csv' for s in cfg['manifests']}
        cfg['manifest_sha256'] = {s: manifest_hashes[s] for s in cfg['manifests']}
        if name == 'pipeline.json':
            cfg['run_directory'] = f'results/joint/{VERSION}_50ep_seed42'
        for split in cfg['manifests']:
            check_rows(cfg, partitions[split], split)
        with (config_dir / name).open('x') as stream:
            json.dump(cfg, stream, indent=2); stream.write('\n')
    return dict(status='admitted', samples=record['samples'], split_counts=record['split_counts'],
                counts=counts, configs=CONFIG_DIR, admission=ADMISSION)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    args = parser.parse_args()
    print(json.dumps(admit(args.root), indent=2))
