"""Export Grad-CAM and classification predictions from an existing checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.models.classifiers import build_model
from src.localization.gradcam import GradCAM
from src.pseudo_labels.masks import normalize_map
from src.evaluation.classification import classification_metrics
from src.training.train import write_json
from src.datasets.protocol import allowed_sources
from src.pipeline.artifacts import (sha256, dataset, setup, environment, write_rows,
                                    check_checkpoint_data)


def generate(cfg, branch, split, root, output):
    allowed_sources(cfg)
    root, output = Path(root).resolve(), Path(output)
    device = setup(cfg['seed'], cfg['threads'], cfg['device'])
    spec = cfg['branches'][branch]
    if spec['method'] != 'gradcam':
        raise ValueError('This implementation supports explicitly configured Grad-CAM only')
    checkpoint = root/spec['checkpoint']
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    check_checkpoint_data(state, dict(cfg, _root=str(root)), split)
    if state['config']['model'] != branch:
        raise ValueError('Checkpoint architecture does not match selected branch')
    ds = dataset(cfg, root, split)
    model = build_model(branch, len(cfg['preprocessing']['channels']), cfg['image_size']).to(device)
    model.load_state_dict(state['model'])
    cam = GradCAM(model, branch)
    if cam.layer_name != spec['target_layer']:
        cam.close()
        raise ValueError('Configured target layer differs from implemented adapter')
    output.mkdir(parents=True, exist_ok=False)
    (output/'maps').mkdir()
    rows, probabilities = [], []
    try:
        for i in range(len(ds)):
            sample = ds[i]
            x = sample['image'][None].to(device)
            if x.shape[1:] != (len(cfg['preprocessing']['channels']), cfg['image_size'], cfg['image_size']):
                raise ValueError('Unexpected input shape')
            result = cam(x)
            raw = result['raw'][0].cpu().numpy()
            normalized, constant = normalize_map(raw)
            probability = float(result['logits'][0].sigmoid().cpu())
            path = f'maps/{i:06d}.npz'
            np.savez_compressed(output/path, raw=raw, normalized=normalized,
                                signed_native=result['signed_native'][0].cpu().numpy())
            row = dict(ds.rows[i], probability=probability, constant_map=constant,
                       gradient_l1=float(result['gradient_l1'][0].cpu()), artifact_path=path,
                       artifact_sha256=sha256(output/path))
            rows.append(row)
            probabilities.append(probability)
            if (i+1) % 100 == 0:
                print(f'{branch} {split}: {i+1}/{len(ds)}', flush=True)
    finally:
        cam.close()
    write_rows(output/'manifest.csv', rows)
    write_json(output/'classification.json', classification_metrics(
        [int(r['label']) for r in rows], probabilities, threshold=cfg['classification_threshold']))
    sources = sorted({r['dataset'] for r in rows})
    if len(sources) > 1:
        write_json(output/'classification_by_dataset.json', {
            source: classification_metrics([int(r['label']) for r in rows if r['dataset'] == source],
                [r['probability'] for r in rows if r['dataset'] == source],
                threshold=cfg['classification_threshold']) for source in sources})
    write_json(output/'metadata.json', dict(stage='maps', branch=branch, split=split,
        experiment_id=str(output), checkpoint_sha256=sha256(checkpoint), checkpoint_epoch=state['epoch'],
        source_manifest_sha256=sha256(root/cfg['manifests'][split]),
        classifier_config=state['config'],
        manifest_sha256=sha256(output/'manifest.csv'), config=cfg, environment=environment(device),
        method=spec['method'], target_layer=spec['target_layer'], masks_opened=False))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--branch', required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    p.add_argument('--root', default='.')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    generate(json.loads(Path(a.config).read_text()), a.branch, a.split, a.root, a.output)


if __name__ == '__main__':
    main()
