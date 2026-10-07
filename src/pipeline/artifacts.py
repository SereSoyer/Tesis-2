"""Artifact integrity and data boundaries shared by the pipeline stages."""
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from src.datasets.classification import ClassificationDataset
from src.datasets.protocol import allowed_sources, check_rows, check_manifest_integrity
from src.preprocessing.multispectral import Preprocess
from src.evaluation.spatial import spatial_metrics


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative_file(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Missing or nonrelative artifact: {value}')
    return path


def read_rows(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Manifest must contain samples with unique IDs')
    return rows


def write_rows(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setup(seed, threads, device):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested device unavailable')
    return device


def environment(device):
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
                                         text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return dict(timestamp=datetime.now(timezone.utc).isoformat(), git_commit=commit,
                code_sha256={str(p.relative_to(Path(__file__).resolve().parents[2])): sha256(p)
                             for p in sorted(Path(__file__).resolve().parents[1].rglob('*.py'))},
                platform=platform.platform(), device=str(device), cuda=torch.version.cuda,
                hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                versions={p: importlib.metadata.version(p) for p in
                          ('torch', 'torchvision', 'timm', 'transformers', 'numpy', 'h5py', 'scikit-learn')})


def dataset(cfg, root, split):
    """Classification reader only: never opens a mask, even when its path exists."""
    check_manifest_integrity(cfg, root, split)
    p = cfg['preprocessing']
    ds = ClassificationDataset(Path(root)/cfg['manifests'][split], root,
                               Preprocess(p['channels'], p['nan_policy'], p['normalization'],
                                          p.get('mean'), p.get('std')), p['image_key'], p['layout'],
                               allowed_datasets=allowed_sources(cfg))
    if cfg['dataset'] not in ('landslide4sense', 'l4s_roy_common10_v1', 'l4s_roy_14_v1'):
        raise ValueError('Spatial pipeline requires an audited dataset protocol')
    check_rows(cfg, ds.rows, split)
    return ds


def ground_truth(root, row):
    """Call only from explicit validation/evaluation, never pseudo-label training."""
    with h5py.File(relative_file(root, row['mask_path']), 'r') as stream:
        target = stream['mask'][:]
    if target.ndim != 2 or not np.isin(target, [0, 1]).all():
        raise ValueError('Ground truth must be a binary 2D mask')
    if int(target.any()) != int(row['label']):
        raise ValueError('Ground truth disagrees with image label')
    return target


def load_artifact(path, stage):
    path = Path(path)
    meta = json.loads((path/'metadata.json').read_text())
    if meta['stage'] != stage or meta['manifest_sha256'] != sha256(path/'manifest.csv'):
        raise ValueError('Wrong stage or modified artifact manifest')
    return meta, read_rows(path/'manifest.csv')


def load_arrays(directory, row):
    path = relative_file(directory, row['artifact_path'])
    if sha256(path) != row['artifact_sha256']:
        raise ValueError('Spatial artifact was modified')
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def check_checkpoint_data(state, cfg, split):
    for key in ('dataset', 'preprocessing', 'image_size'):
        if state['config'][key] != cfg[key]:
            raise ValueError(f'Checkpoint/input mismatch: {key}')
    if state['config'].get('source_datasets') != cfg.get('source_datasets'):
        raise ValueError('Checkpoint/input source datasets mismatch')
    # A trained classifier must see exactly its original train/val data downstream.
    if split in ('train', 'val'):
        if state['manifest_sha256'][split] != sha256(Path(cfg['_root'])/cfg['manifests'][split]):
            raise ValueError('Classifier split differs from pipeline split')


def aggregate_spatial(records):
    positive = [r for r in records if int(r['label']) == 1]
    counts = {k: sum(r[k] for r in records) for k in ('tp', 'fp', 'fn', 'tn')}
    tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
    negative = [r for r in records if int(r['label']) == 0]
    result = dict(counts=counts, samples=len(records), positive_samples=len(positive),
                macro_iou_positive=float(np.mean([r['iou'] for r in positive])) if positive else None,
                macro_dice_positive=float(np.mean([r['dice'] for r in positive])) if positive else None,
                micro_iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                micro_dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                pixel_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                pixel_precision=tp/(tp+fp) if tp+fp else None,
                pixel_recall=tp/(tp+fn) if tp+fn else None,
                negative_samples=len(negative),
                negative_false_region_rate=sum(r['fp'] > 0 for r in negative)/len(negative) if negative else None)
    sources = sorted({r.get('dataset', 'unknown') for r in records})
    if len(sources) > 1:
        result['by_dataset'] = {source: aggregate_spatial(
            [r for r in records if r.get('dataset', 'unknown') == source]) for source in sources}
    return result


def spatial_record(row, target, prediction):
    return dict(sample_id=row['sample_id'], dataset=row['dataset'], label=int(row['label']),
                area_ratio=row.get('area_ratio', ''), **spatial_metrics(target, prediction))
