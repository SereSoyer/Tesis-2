"""Configurable epoch runner with atomic, resumable experiment checkpoints."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil

import numpy as np
import torch
from src.datasets.classification import ClassificationDataset, make_loader
from src.datasets.protocol import allowed_sources, check_rows, check_partitions, check_manifest_integrity
from src.preprocessing.multispectral import Preprocess
from src.models.classifiers import build_model, MODELS
from src.training.epochs import run_epoch


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def validate_config(cfg):
    enabled = cfg.get('training_enabled', True)
    if type(enabled) is not bool:
        raise ValueError('training_enabled must be a boolean')
    if not enabled:
        raise ValueError(f"Training disabled for {cfg.get('dataset', 'unknown')}: "
                         f"{cfg.get('disabled_reason', 'Dataset preparation pending')}")
    if cfg['model'] not in MODELS or cfg['pretrained'] is not False:
        raise ValueError('Choose a supported model with pretrained=false')
    for key in ('epochs', 'batch_size', 'threads', 'image_size'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(cfg['num_workers']) is not int or cfg['num_workers'] < 0:
        raise ValueError('num_workers must be nonnegative')
    if type(cfg['seed']) is not int or not 0 <= cfg['seed'] < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if cfg['optimizer'] != 'AdamW' or cfg['loss'] != 'BCEWithLogitsLoss':
        raise ValueError('Only AdamW and unweighted BCEWithLogitsLoss are implemented')
    if cfg['scheduler'] is not None or cfg['early_stopping'] is not None:
        raise ValueError('scheduler and early_stopping must be null in this base runner')
    for key in ('learning_rate', 'weight_decay', 'threshold'):
        if not np.isfinite(cfg[key]):
            raise ValueError(f'{key} must be finite')
    if cfg['learning_rate'] <= 0 or cfg['weight_decay'] < 0 or not 0 <= cfg['threshold'] <= 1:
        raise ValueError('Invalid optimizer values or threshold')
    if cfg['checkpoint_metric'] not in ('loss', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'):
        raise ValueError('Unsupported validation checkpoint metric')
    if set(cfg['manifests']) != {'train', 'val'}:
        raise ValueError('Provide only train/val manifests; this runner does not evaluate test')


def prepare_data(cfg, root):
    for split in cfg['manifests']:
        check_manifest_integrity(cfg, root, split)
    p = cfg['preprocessing']
    prep = Preprocess(p['channels'], p['nan_policy'], p['normalization'], p.get('mean'), p.get('std'))
    datasets = {split: ClassificationDataset(root/path, root, prep, p['image_key'], p['layout'],
                                             allowed_datasets=allowed_sources(cfg))
                for split, path in cfg['manifests'].items()}
    for split, ds in datasets.items():
        check_rows(cfg, ds.rows, split)
    check_partitions({split: ds.rows for split, ds in datasets.items()})
    for field in ('sample_id', 'image_path'):
        def values(ds):
            return {str((root/r[field]).resolve()) if field == 'image_path' else r[field] for r in ds.rows}
        if values(datasets['train']) & values(datasets['val']):
            raise ValueError(f'Train/val overlap in {field}')
    if cfg['checkpoint_metric'] == 'roc_auc' and len({r['label'] for r in datasets['val'].rows}) < 2:
        raise ValueError('ROC-AUC checkpoint selection requires both validation classes')
    loaders = {split: make_loader(ds, cfg['batch_size'], split == 'train', cfg['seed'], cfg['num_workers'])
               for split, ds in datasets.items()}
    # Sanity check before creating any experiment output; no optimizer update.
    for loader in loaders.values():
        batch = next(iter(loader))
        if tuple(batch['image'].shape[1:]) != (len(p['channels']), cfg['image_size'], cfg['image_size']):
            raise ValueError('Input shape differs from configuration; no implicit resize')
        loader.generator.manual_seed(cfg['seed'])
    return loaders


def rng_state(loaders):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                loaders={k: v.generator.get_state() for k, v in loaders.items()})


def restore_rng(state, loaders):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])
    for key, loader in loaders.items():
        loader.generator.set_state(state['loaders'][key])


def run_experiment(cfg, root, output, resume=False):
    cfg = copy.deepcopy(cfg)
    root, output = Path(root).resolve(), Path(output).resolve()
    validate_config(cfg)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    device = torch.device(cfg['device'])
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only cpu/cuda devices are supported')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if not resume and output.exists():
        raise FileExistsError(f'Experiment already exists: {output}; use a new output or --resume')
    loaders = prepare_data(cfg, root)
    hashes = {k: hashlib.sha256((root/v).read_bytes()).hexdigest() for k, v in cfg['manifests'].items()}
    environment = {'platform': platform.platform(), 'device': str(device),
                   'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                   'cuda_version': torch.version.cuda,
                   'versions': {p: importlib.metadata.version(p) for p in
                                ('torch', 'torchvision', 'numpy', 'h5py', 'timm', 'transformers', 'scikit-learn')}}
    model = build_model(cfg['model'], len(cfg['preprocessing']['channels']), cfg['image_size'], cfg['pretrained']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    history, start_epoch, best_value, best_epoch = [], 1, None, None
    if resume:
        # Only load checkpoints produced by this runner from a trusted experiment directory.
        state = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
        old_cfg, new_cfg = dict(state['config']), dict(cfg)
        old_cfg.pop('epochs'); new_cfg.pop('epochs')
        if old_cfg != new_cfg or state['manifest_sha256'] != hashes or state['environment'] != environment:
            raise ValueError('Resume requires unchanged configuration, manifests and environment (except epochs)')
        if cfg['epochs'] <= state['epoch']:
            raise ValueError('epochs must exceed the last completed epoch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        history, start_epoch = state['history'], state['epoch'] + 1
        best_value, best_epoch = state['best_value'], state['best_epoch']
        if best_epoch == state['epoch']:
            # Recover if interrupted between writing last.pt and best.pt.
            save_checkpoint(output/'best.pt', state)
        else:
            if not (output/'best.pt').is_file():
                raise FileNotFoundError('Missing best.pt; retain both checkpoint files')
            best = torch.load(output/'best.pt', map_location='cpu', weights_only=False)
            if best['epoch'] != best_epoch or best['manifest_sha256'] != hashes:
                raise ValueError('best.pt does not match the resumed experiment')
            del best
        restore_rng(state['rng'], loaders)
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/'splits').mkdir()
        for split, path in cfg['manifests'].items():
            shutil.copyfile(root/path, output/'splits'/f'{split}.csv')
    write_json(output/'config.json', cfg)
    write_json(output/'environment.json', environment)
    write_json(output/'manifest_sha256.json', hashes)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    shape = (len(cfg['preprocessing']['channels']), cfg['image_size'], cfg['image_size'])
    for epoch in range(start_epoch, cfg['epochs']+1):
        train = run_epoch(model, loaders['train'], device, cfg['threshold'], optimizer, shape)
        val = run_epoch(model, loaders['val'], device, cfg['threshold'], expected_shape=shape)
        value = val[cfg['checkpoint_metric']]
        if value is None:
            raise ValueError('Checkpoint metric is undefined on validation')
        improved = best_value is None or (value < best_value if cfg['checkpoint_metric'] == 'loss' else value > best_value)
        if improved:
            best_value, best_epoch = value, epoch
        history.append({'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr'], 'train': train, 'val': val})
        state = dict(epoch=epoch, config=cfg, environment=environment, manifest_sha256=hashes,
                     model=model.state_dict(), optimizer=optimizer.state_dict(), history=history,
                     best_value=best_value, best_epoch=best_epoch, rng=rng_state(loaders))
        save_checkpoint(output/'last.pt', state)
        if improved:
            save_checkpoint(output/'best.pt', state)
        write_json(output/'history.json', history)
        write_json(output/'summary.json', {'completed_epochs': epoch, 'best_epoch': best_epoch,
                   'checkpoint_metric': cfg['checkpoint_metric'], 'best_validation_value': best_value,
                   'parameters': sum(p.numel() for p in model.parameters()),
                   'epoch_seconds_total': sum(h[s]['seconds'] for h in history for s in ('train', 'val')),
                   'peak_cuda_memory_bytes_current_session': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None})
        print(f'Epoch {epoch}/{cfg["epochs"]}: train loss={train["loss"]:.6f}, val loss={val["loss"]:.6f}, val F1={val["f1"]:.4f}', flush=True)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--root', default='.', help='Root for image and manifest paths')
    parser.add_argument('--output', required=True, help='New experiment directory, normally under results/experiments/')
    parser.add_argument('--resume', action='store_true', help='Continue output/last.pt at the next epoch')
    args = parser.parse_args()
    run_experiment(json.loads(Path(args.config).read_text()), args.root, args.output, args.resume)


if __name__ == '__main__':
    main()
