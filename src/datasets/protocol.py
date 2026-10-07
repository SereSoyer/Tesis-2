"""Explicit dataset admission and source-partition checks shared by all stages."""
import hashlib
import json
from pathlib import Path
from src.datasets.manifests import validate_partition_integrity

INTEGRATED = 'l4s_roy_common10_v1'
INTEGRATED14 = 'l4s_roy_14_v1'
SOURCES = ['landslide4sense', 'roy_l4s_pe']
CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]
EMPIRICAL_MAPPING = 'empirically_supported_mapping'


def allowed_sources(cfg):
    if cfg.get('training_enabled', True) is not True:
        raise ValueError('Dataset preparation pending: ' + cfg.get('disabled_reason', 'training disabled'))
    if cfg['dataset'] in (INTEGRATED, INTEGRATED14):
        if cfg.get('source_datasets') != SOURCES:
            raise ValueError('Integrated experiment requires explicit source_datasets')
        p = cfg['preprocessing']
        expected_channels = CHANNELS if cfg['dataset'] == INTEGRATED else list(range(14))
        if (p['channels'] != expected_channels or p['normalization'] != 'per_image_minmax'
                or p['nan_policy'] != 'error'):
            raise ValueError('Integrated v1 requires the documented common-ten-band or fourteen-band preprocessing')
        if (cfg['dataset'] == INTEGRATED14 and cfg.get('band_mapping_status') not in
                ('verified', 'adopted_thesis_table1', EMPIRICAL_MAPPING)):
            raise ValueError('Fourteen-channel band mapping decision is pending')
        if cfg.get('band_mapping_status') == EMPIRICAL_MAPPING:
            if not cfg.get('data_admission') or set(cfg.get('manifest_sha256', {})) != set(cfg['manifests']):
                raise ValueError('Empirical mapping requires a pinned admission and manifests')
        return SOURCES
    if cfg.get('source_datasets'):
        raise ValueError('Mixed sources require an explicitly supported integration protocol')
    return [cfg['dataset']]


def check_manifest_integrity(cfg, root, split):
    """Pin newly admitted data while keeping historical configurations compatible."""
    allowed_sources(cfg)
    root = Path(root).resolve()

    def checked(path, expected):
        resolved = (root / path).resolve()
        if Path(path).is_absolute() or not resolved.is_relative_to(root):
            raise ValueError('Admission paths must remain inside root')
        if hashlib.sha256(resolved.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Admitted file changed: {path}')
        return resolved

    if cfg.get('data_admission'):
        ref = cfg['data_admission']
        record = json.loads(checked(ref['path'], ref['sha256']).read_text())
        if (record['status'] != 'admitted' or record['dataset_version'] != cfg['dataset']
                or record['band_mapping_status'] != cfg['band_mapping_status']
                or record['manifest_sha256'][split] != cfg['manifest_sha256'][split]):
            raise ValueError('Configuration differs from dataset admission')
    if cfg.get('manifest_sha256'):
        checked(cfg['manifests'][split], cfg['manifest_sha256'][split])


def check_rows(cfg, rows, split):
    sources = allowed_sources(cfg)
    if split not in ('train', 'val', 'test'):
        raise ValueError('Unknown partition')
    for row in rows:
        if row['dataset'] not in sources:
            raise ValueError('Manifest dataset does not match configuration')
        if row['dataset'] == 'landslide4sense':
            expected = {'train': 'TrainData', 'val': 'ValidData', 'test': 'TestData'}[split]
            if row.get('source_split') != expected:
                raise ValueError('Landslide4Sense source_split does not match requested split')
        if cfg['dataset'] in (INTEGRATED, INTEGRATED14):
            if (row.get('partition') != split or row.get('status') != 'accepted'
                    or str(row.get('training_eligible')).lower() != 'true'
                    or row.get('dataset_version') != cfg['dataset']):
                raise ValueError('Integrated manifest requires accepted versioned partition assignments')
            if (cfg['dataset'] == INTEGRATED14 and row.get('band_mapping_status') != cfg['band_mapping_status']):
                raise ValueError('Manifest and configuration band mapping decisions differ')
            if row['dataset'] == 'roy_l4s_pe':
                if (row.get('source_split') != 'RoyOriginal'
                        or not row.get('geographic_group', '').startswith('roy_overlap_')
                        or row.get('alignment_policy') != 'iris_mask_area_reconstructed_v1'):
                    raise ValueError('Roy requires traced reconstruction and geographic grouping')
        if split != 'test' and row.get('source_split', '').lower() in ('test', 'testdata'):
            raise ValueError('Test samples are not allowed in training/validation')


def check_partitions(partitions):
    validate_partition_integrity(partitions)
