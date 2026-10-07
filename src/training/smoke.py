"""One real-data optimization step per model, not a training experiment."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
from src.datasets.classification import ClassificationDataset, make_loader
from src.preprocessing.multispectral import Preprocess
from src.models.classifiers import build_model

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/smoke_roy.json')
    args = parser.parse_args()
    cfg = json.loads((ROOT/args.config).read_text())
    if cfg['optimizer'] != 'AdamW' or cfg['loss'] != 'BCEWithLogitsLoss' or cfg['optimizer_steps'] != 1:
        raise ValueError('Smoke runner supports AdamW, BCEWithLogitsLoss and exactly one step')
    if any(cfg[k] is not None for k in ('scheduler', 'early_stopping', 'checkpoint_criterion')):
        raise ValueError('Smoke runner does not implement experiment scheduling/checkpoints')
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    prep = Preprocess(cfg['channels'], cfg['nan_policy'], cfg['normalization'])
    dataset = ClassificationDataset(ROOT/cfg['metadata'], ROOT, prep, cfg['image_key'], cfg['layout'])
    batch = next(iter(make_loader(dataset, cfg['batch_size'], False, cfg['seed'], cfg['num_workers'])))
    x, y = batch['image'].to(cfg['device']), batch['label'].to(cfg['device'])
    if tuple(x.shape[2:]) != (cfg['image_size'],cfg['image_size']):
        raise ValueError('Image size differs from configuration; no implicit resize')
    report = {'config':cfg, 'manifest_sha256':hashlib.sha256((ROOT/cfg['metadata']).read_bytes()).hexdigest(),
              'sample_ids':batch['sample_id'], 'labels':y.tolist(), 'input_shape':list(x.shape),
              'hardware':platform.platform(), 'versions':{p:importlib.metadata.version(p) for p in ('torch','torchvision','timm','transformers','numpy','h5py')}, 'models':[]}
    for name in cfg['models']:
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        model = build_model(name, len(cfg['channels']), cfg['image_size'], cfg['pretrained']).to(cfg['device']).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if logits.shape != y.shape:
            raise ValueError(f'Wrong output shape: {logits.shape}')
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        grads = [p for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(p.grad).all() for p in grads):
            raise ValueError('Missing/nonfinite gradients')
        tracked = next(p for p in grads if p.grad.abs().sum() > 0)
        before = tracked.detach().clone()
        optimizer.step()
        changed = not torch.equal(before, tracked.detach())
        if not changed: raise ValueError('Optimizer did not update weights')
        result = {'name':name,'parameters':sum(p.numel() for p in model.parameters()),'loss':loss.item(),
                  'output_shape':list(logits.shape),'finite_gradients':True,'weights_updated':changed,
                  'step_seconds':time.perf_counter()-start}
        report['models'].append(result)
        print(json.dumps(result), flush=True)
        del optimizer, model, grads, tracked, before, logits, loss
    path = ROOT/cfg['output'];path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')

if __name__ == '__main__': main()
