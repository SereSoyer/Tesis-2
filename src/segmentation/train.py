"""Train a shared DeepLabv3+ variant on pseudo-targets; select on validation GT.

    The use of pixel-level validation masks for model selection is explicit.
    Ground-truth training masks are never read. Test is a separate CLI action.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from src.datasets.classification import make_loader
from src.segmentation.data import PseudoTrainingDataset, ReferenceEvaluationDataset
from src.segmentation.model import DeepLabV3Plus
from src.pipeline.artifacts import (setup, environment, sha256, aggregate_spatial,
                                    spatial_record, dataset)
from src.training.train import write_json, save_checkpoint, rng_state, restore_rng


def validate(cfg):
    if cfg.get('training_enabled', True) is not True:
        raise ValueError('Dataset preparation pending: ' + cfg.get('disabled_reason', 'training disabled'))
    s = cfg['segmentation']
    if s['model'] != 'deeplabv3plus_resnet50_gn_decoder' or s['pretrained'] is not False:
        raise ValueError('Use the documented common segmenter with random initialization')
    if s['loss'] != 'BCEWithLogitsLoss' or s['optimizer'] != 'AdamW':
        raise ValueError('Unsupported segmentation optimizer/loss')
    if s['scheduler'] is not None or s['early_stopping'] is not None or s['augmentation'] != 'none':
        raise ValueError('Scheduler/early stopping/augmentation not implemented in this baseline')
    if s['checkpoint_metric'] != 'val_gt_macro_iou_positive':
        raise ValueError('Explicit common validation-GT selection criterion required')
    for key in ('epochs', 'batch_size'):
        if type(s[key]) is not int or s[key] < 1:
            raise ValueError(f'Invalid {key}')
    if not np.isfinite(s['learning_rate']) or s['learning_rate'] <= 0:
        raise ValueError('Invalid learning rate')
    if not np.isfinite(s['weight_decay']) or s['weight_decay'] < 0 or not 0 < s['threshold'] < 1:
        raise ValueError('Invalid decay/threshold')


def assert_disjoint(train, val):
    for field in ('sample_id', 'image_path', 'image_sha256', 'geographic_group'):
        a = {r.get(field) for r in train if r.get(field)}
        b = {r.get(field) for r in val if r.get(field)}
        if a & b:
            raise ValueError(f'Split overlap in {field}')


def evaluation(model, loader, device, threshold, rows, output=None):
    model.eval()
    lookup = {r['sample_id']: r for r in rows}
    records = []
    with torch.no_grad():
        for batch in loader:
            probabilities = model(batch['image'].to(device)).sigmoid().cpu().numpy()
            if probabilities.shape != tuple(batch['target'].shape) or not np.isfinite(probabilities).all():
                raise ValueError('Invalid segmentation output')
            for i, sample_id in enumerate(batch['sample_id']):
                prediction = (probabilities[i] >= threshold).astype(np.uint8)
                record = spatial_record(lookup[sample_id], batch['target'][i].numpy(), prediction)
                if output is not None:
                    path = f'predictions/{len(records):06d}.npz'
                    np.savez_compressed(output/path, probability=probabilities[i], prediction=prediction)
                    record.update(artifact_path=path, artifact_sha256=sha256(output/path))
                records.append(record)
    return aggregate_spatial(records), records


def train(cfg, root, pseudo, output, resume=False):
    cfg = copy.deepcopy(cfg)
    validate(cfg)
    root, pseudo, output = Path(root).resolve(), Path(pseudo).resolve(), Path(output)
    s = cfg['segmentation']
    device = setup(cfg['seed'], cfg['threads'], cfg['device'])
    train_ds = PseudoTrainingDataset(cfg, root, pseudo)
    val_ds = ReferenceEvaluationDataset(cfg, root, 'val')
    assert_disjoint(train_ds.rows, val_ds.images.rows)
    if not any(int(r['label']) for r in val_ds.images.rows):
        raise ValueError('Validation needs positives for checkpoint criterion')
    # Ensure the calibration did not use a different validation population.
    if train_ds.meta['calibration']['validation_manifest_sha256'] != sha256(root/cfg['manifests']['val']):
        raise ValueError('Pseudo-label calibration used a different validation split')
    loaders = {k: make_loader(ds, s['batch_size'], k == 'train', cfg['seed'], cfg['num_workers'])
               for k, ds in [('train', train_ds), ('val', val_ds)]}
    model = DeepLabV3Plus(len(cfg['preprocessing']['channels'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=s['learning_rate'], weight_decay=s['weight_decay'])
    provenance = dict(pseudo_manifest=sha256(pseudo/'manifest.csv'),
                      pseudo_metadata=sha256(pseudo/'metadata.json'),
                      train=sha256(root/cfg['manifests']['train']), val=sha256(root/cfg['manifests']['val']))
    env = environment(device)
    history, first_epoch, best_value, best_epoch = [], 1, None, None
    if resume:
        state = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
        if state['kind'] != 'pseudo_segmentation' or state['config'] != cfg or state['provenance'] != provenance:
            raise ValueError('Resume requires identical config and pseudo-label/data provenance')
        for key in ('device', 'cuda', 'hardware', 'versions', 'code_sha256'):
            if state['environment'][key] != env[key]:
                raise ValueError('Resume environment changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        history, first_epoch = state['history'], state['epoch'] + 1
        best_value, best_epoch = state['best_value'], state['best_epoch']
        if first_epoch > s['epochs']:
            raise ValueError('Run already completed')
        if best_epoch == state['epoch']:
            save_checkpoint(output/'best.pt', state)
        else:
            best = torch.load(output/'best.pt', map_location='cpu', weights_only=False)
            if best['epoch'] != best_epoch or best['provenance'] != provenance:
                raise ValueError('Best checkpoint does not match resumed experiment')
        restore_rng(state['rng'], loaders)
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/'train_pseudo_manifest.csv').write_bytes((pseudo/'manifest.csv').read_bytes())
        (output/'val_manifest.csv').write_bytes((root/cfg['manifests']['val']).read_bytes())
    write_json(output/'config.json', cfg)
    write_json(output/'environment.json', env)
    write_json(output/'pseudo_source.json', train_ds.meta)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(first_epoch, s['epochs'] + 1):
        model.train()
        if device.type == 'cuda': torch.cuda.synchronize(device)
        start = time.perf_counter()
        total_loss, count = 0., 0
        for batch in loaders['train']:
            image, target = batch['image'].to(device), batch['target'].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            if logits.shape != target.shape:
                raise ValueError('Segmentation target shape mismatch')
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise ValueError('Missing/nonfinite gradients')
            optimizer.step()
            total_loss += float(loss.detach()) * len(image)
            count += len(image)
        report, _ = evaluation(model, loaders['val'], device, s['threshold'], val_ds.images.rows)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        value = report['macro_iou_positive']
        if best_value is None or value > best_value:
            best_value, best_epoch = value, epoch
        history.append(dict(epoch=epoch, train_pseudo_loss=total_loss/count,
                            validation_gt=report, seconds=time.perf_counter()-start))
        state = dict(kind='pseudo_segmentation', config=cfg, provenance=provenance, environment=env,
                     branch=train_ds.meta['branch'], epoch=epoch, history=history, best_value=best_value,
                     best_epoch=best_epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     rng=rng_state(loaders))
        save_checkpoint(output/'last.pt', state)
        if best_epoch == epoch: save_checkpoint(output/'best.pt', state)
        write_json(output/'history.json', history)
        write_json(output/'summary.json', dict(kind='pseudo_segmentation', branch=train_ds.meta['branch'],
            completed_epochs=epoch, best_epoch=best_epoch, best_validation_iou=best_value,
            parameters=sum(p.numel() for p in model.parameters()),
            validation_gt_used_for_selection=True, training_targets='pseudo_masks_only',
            seconds_total=sum(h['seconds'] for h in history),
            peak_cuda_memory_bytes_current_session=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None))
        print(f'Epoch {epoch}/{s["epochs"]}: pseudo loss={total_loss/count:.6f}, val GT IoU={value:.4f}', flush=True)
    return history


def evaluate_checkpoint(checkpoint, root, split, output):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['kind'] != 'pseudo_segmentation': raise ValueError('Wrong checkpoint type')
    cfg = state['config']
    validate(cfg)
    device = setup(cfg['seed'], cfg['threads'], cfg['device'])
    ds = ReferenceEvaluationDataset(cfg, root, split)
    for original in ('train', 'val'):
        if sha256(Path(root)/cfg['manifests'][original]) != state['provenance'][original]:
            raise ValueError('Training/validation manifest changed')
        if split != original:
            assert_disjoint(dataset(cfg, root, original).rows, ds.images.rows)
    model = DeepLabV3Plus(len(cfg['preprocessing']['channels'])).to(device)
    model.load_state_dict(state['model'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'predictions').mkdir()
    loader = make_loader(ds, cfg['segmentation']['batch_size'], False, cfg['seed'], cfg['num_workers'])
    summary, records = evaluation(model, loader, device, cfg['segmentation']['threshold'], ds.images.rows, output)
    write_json(output/'per_sample.json', records)
    write_json(output/'summary.json', dict(summary, branch=state['branch'], split=split,
        checkpoint_sha256=sha256(checkpoint), config=cfg, environment=environment(device),
        evaluation_manifest_sha256=sha256(Path(root)/cfg['manifests'][split])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train', 'evaluate'])
    p.add_argument('--config')
    p.add_argument('--pseudo')
    p.add_argument('--checkpoint')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--root', default='.')
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.action == 'train':
        if not a.config or not a.pseudo: p.error('--config and --pseudo required')
        train(json.loads(Path(a.config).read_text()), a.root, a.pseudo, a.output, a.resume)
    else:
        if not a.checkpoint: p.error('--checkpoint required')
        evaluate_checkpoint(a.checkpoint, a.root, a.split, a.output)


if __name__ == '__main__':
    main()
