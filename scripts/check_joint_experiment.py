"""Check admitted mixed data, pinned splits and the four-model protocol before training."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import torch
from scripts.admit_l4s_roy14 import CONFIG_DIR, MODELS
from src.datasets.manifests import array_digest, validate_manifest, validate_partition_integrity
from src.pipeline.artifacts import dataset, sha256
from src.training.train import validate_config
from src.segmentation.train import validate as validate_segmentation


def check(root, config_dir=CONFIG_DIR, full=True, models=False, device='cpu'):
    root = Path(root).resolve()
    cfg = json.loads((root / config_dir / 'pipeline.json').read_text())
    validate_segmentation(cfg)
    if list(cfg['branches']) != MODELS:
        raise ValueError('Expected the four selected classifier branches')
    datasets = {s: dataset(cfg, root, s) for s in ('train', 'val', 'test')}
    partitions = {s: ds.rows for s, ds in datasets.items()}
    validate_partition_integrity(partitions)
    admission = json.loads((root / cfg['data_admission']['path']).read_text())
    baseline = None
    for name in MODELS:
        classifier = json.loads((root / config_dir / f'train_{name}.json').read_text())
        validate_config(classifier)
        if classifier['model'] != name:
            raise ValueError('Classifier name differs from configuration')
        for key in ('dataset', 'preprocessing', 'image_size', 'seed', 'source_datasets', 'data_admission', 'band_mapping_status'):
            if classifier[key] != cfg[key]:
                raise ValueError(f'Classifier/pipeline mismatch: {name}/{key}')
        for s in ('train', 'val'):
            if (classifier['manifests'][s] != cfg['manifests'][s]
                    or classifier['manifest_sha256'][s] != cfg['manifest_sha256'][s]):
                raise ValueError('Classifier and pipeline use different splits')
        protocol = {k: v for k, v in classifier.items() if k != 'model'}
        if baseline is not None and protocol != baseline:
            raise ValueError('Classifier training policies differ across branches')
        baseline = protocol
    counts = {}
    for split, ds in datasets.items():
        validate_manifest(ds.rows, root, require_accepted=True)
        if len(ds) != admission['split_counts'][split]:
            raise ValueError('Admitted population changed')
        counts[split] = {source: dict(Counter(r['label'] for r in ds.rows if r['dataset'] == source))
                         for source in cfg['source_datasets']}
        if counts[split] != admission['class_counts_by_source'][split]:
            raise ValueError('Admitted class counts changed')
        if not full:
            continue
        for i, row in enumerate(ds.rows):
            for kind in ('image', 'mask'):
                if sha256(root / row[kind + '_path']) != row[kind + '_file_sha256']:
                    raise ValueError(f'Data file changed: {row[kind + "_path"]}')
            with h5py.File(root / row['image_path']) as stream:
                array = stream[cfg['preprocessing']['image_key']][:]
            if array.shape != (128, 128, 14) or array_digest(array) != row['image_sha256']:
                raise ValueError(f'Image shape/content changed: {row["sample_id"]}')
            if not np.isfinite(array).all() or (array < -1e30).any():
                raise ValueError('Invalid pixels must not be silently normalized')
            tensor = ds.preprocess(array)
            if tensor.shape != (14, 128, 128) or not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
                raise ValueError('Invalid common preprocessing')
            if (i + 1) % 1000 == 0:
                print(f'Integridad {split}: {i + 1}/{len(ds)}', flush=True)
    model_checks = {}
    if models:
        from src.models.classifiers import build_model
        from src.segmentation.model import DeepLabV3Plus
        torch.set_num_threads(cfg['threads'])
        torch.manual_seed(cfg['seed'])
        train = datasets['train']
        indices = [next(i for i, r in enumerate(train.rows) if r['dataset'] == source)
                   for source in cfg['source_datasets']]
        # Real images from both datasets; no ground-truth mask opened for these checks.
        samples = [train[i] for i in indices]
        x = torch.stack([s['image'] for s in samples]).to(device)
        y = torch.stack([s['label'] for s in samples]).to(device)
        for name in MODELS + ['deeplabv3plus']:
            model = (DeepLabV3Plus(14) if name == 'deeplabv3plus' else build_model(name, 14, 128)).to(device).train()
            output = model(x)
            target = torch.zeros_like(output) if name == 'deeplabv3plus' else y
            expected_shape = (2, 128, 128) if name == 'deeplabv3plus' else (2,)
            if tuple(output.shape) != expected_shape or not torch.isfinite(output).all():
                raise ValueError(f'Invalid model output: {name}')
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise ValueError(f'Invalid gradients: {name}')
            model_checks[name] = {'forward_backward': 'passed', 'shape': list(output.shape)}
            print(f'Entrada conjunta y gradientes: {name} OK', flush=True)
            del model, output, loss, gradients
            gc.collect()
        model_checks['targets'] = 'Image labels for classifiers; synthetic zeros for segmenter; no optimizer steps.'
    return dict(status='passed', integrity='full_file_hashes_and_image_arrays' if full else 'manifests_and_paths_only',
                samples=admission['samples'], split_counts=admission['split_counts'], counts=counts,
                normalization='per_image_minmax', channels=14, joint_training_enabled=True,
                model_checks=model_checks, models_device=device if models else None,
                training_started=False, test_model_evaluation_performed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--configs', default=CONFIG_DIR)
    parser.add_argument('--quick', action='store_true', help='Skip data contents; retain manifest, path and protocol checks')
    parser.add_argument('--models', action='store_true', help='Forward/backward mixed train batch through all five real models')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--output', help='Optional new JSON report; existing reports are preserved')
    args = parser.parse_args()
    report = check(args.root, args.configs, not args.quick, args.models, args.device)
    if args.output:
        path = Path(args.root) / args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps(report, indent=2))
