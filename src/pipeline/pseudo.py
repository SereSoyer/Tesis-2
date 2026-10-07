"""Calibrate on validation only; materialize pseudo-masks without opening GT."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.pseudo_labels.masks import pseudo_mask
from src.pipeline.artifacts import (load_artifact, load_arrays, ground_truth, sha256,
                                    write_rows, spatial_record, aggregate_spatial)
from src.training.train import write_json
from src.datasets.protocol import check_rows


def calibrate(maps, root, output):
    maps, output = Path(maps), Path(output)
    meta, rows = load_artifact(maps, 'maps')
    if meta['split'] != 'val':
        raise ValueError('Threshold calibration requires validation; train/test prohibited')
    cfg = meta['config']
    check_rows(cfg, rows, 'val')
    grid = cfg['pseudo_labels']['threshold_grid']
    if not grid or len(set(grid)) != len(grid) or any(not 0 < t < 1 for t in grid):
        raise ValueError('Invalid threshold grid')
    positives = [r for r in rows if int(r['label']) == 1]
    if not positives:
        raise ValueError('Calibration requires positive validation samples')
    totals = {t: [] for t in grid}
    nonconstant = 0
    for row in positives:
        arrays = load_arrays(maps, row)
        m = arrays['normalized']
        nonconstant += int(m.max() != m.min())
        gt = ground_truth(root, row)
        for t in grid:
            pred = pseudo_mask(m, float(row['probability']), t, cfg['classification_threshold'])
            totals[t].append(spatial_record(row, gt, pred)['iou'])
    if not nonconstant:
        raise ValueError('All positive validation maps are degenerate; calibration failed')
    curve = [{'threshold': t, 'macro_iou_positive': float(np.mean(totals[t]))} for t in sorted(grid)]
    best = max(curve, key=lambda r: (r['macro_iou_positive'], r['threshold']))
    if cfg['pseudo_labels']['refinement'] != 'none':
        raise ValueError('Refinement not implemented; choose none explicitly')
    result = dict(stage='calibration', branch=meta['branch'], threshold=best['threshold'],
                  criterion='macro_iou_positive_validation', tie_break='largest_threshold',
                  validation_gt_used=True, positive_samples=len(positives), curve=curve,
                  checkpoint_sha256=meta['checkpoint_sha256'],
                  validation_manifest_sha256=meta['source_manifest_sha256'],
                  maps_manifest_sha256=meta['manifest_sha256'],
                  classification_threshold=cfg['classification_threshold'],
                  preprocessing=cfg['preprocessing'], method=meta['method'],
                  target_layer=meta['target_layer'], refinement='none')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'threshold.json', result)
    return result


def materialize(maps, calibration, output):
    maps, calibration, output = Path(maps), Path(calibration), Path(output)
    meta, rows = load_artifact(maps, 'maps')
    selected = json.loads(calibration.read_text())
    if selected['stage'] != 'calibration' or selected['branch'] != meta['branch']:
        raise ValueError('Wrong calibration branch/stage')
    for key in ('checkpoint_sha256', 'method', 'target_layer'):
        if selected[key] != meta[key]:
            raise ValueError('Calibration belongs to a different classifier or attribution method')
    for key in ('classification_threshold', 'preprocessing'):
        if selected[key] != meta['config'][key]:
            raise ValueError('Calibration and map preprocessing differ')
    if selected['refinement'] != 'none' or meta['config']['pseudo_labels']['refinement'] != 'none':
        raise ValueError('Unsupported refinement')
    output.mkdir(parents=True, exist_ok=False)
    (output/'masks').mkdir()
    result = []
    for i, row in enumerate(rows):
        arrays = load_arrays(maps, row)
        binary = pseudo_mask(arrays['normalized'], float(row['probability']),
                             selected['threshold'], selected['classification_threshold'])
        path = f'masks/{i:06d}.npz'
        # Keep ungated mask as an ablation artifact; train uses gated pseudo_mask.
        thresholded = (arrays['normalized'] >= selected['threshold']).astype(np.uint8)
        np.savez_compressed(output/path, raw=arrays['raw'], normalized=arrays['normalized'],
                            thresholded=thresholded, pseudo_mask=binary)
        result.append(dict(row, artifact_path=path, artifact_sha256=sha256(output/path),
                           pseudo_threshold=selected['threshold'], refinement='none'))
    write_rows(output/'manifest.csv', result)
    write_json(output/'metadata.json', dict(meta, stage='pseudo_labels',
        experiment_id=str(output), maps_manifest_sha256=meta['manifest_sha256'],
        manifest_sha256=sha256(output/'manifest.csv'), calibration=selected,
        calibration_sha256=sha256(calibration), masks_opened=False))
    return output


def evaluate(pseudo, root, output):
    meta, rows = load_artifact(pseudo, 'pseudo_labels')
    if meta['split'] not in ('val', 'test'):
        raise ValueError('Spatial evaluation accepts val or explicit final test only')
    records, ungated = [], []
    for row in rows:
        target, arrays = ground_truth(root, row), load_arrays(pseudo, row)
        records.append(spatial_record(row, target, arrays['pseudo_mask']))
        ungated.append(spatial_record(row, target, arrays['thresholded']))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'per_sample.json', records)
    write_json(output/'ungated_per_sample.json', ungated)
    write_json(output/'summary.json', dict(aggregate_spatial(records), split=meta['split'],
        branch=meta['branch'], threshold=meta['calibration']['threshold'],
        ungated_ablation=aggregate_spatial(ungated),
        source_manifest_sha256=meta['source_manifest_sha256'],
        manifest_sha256=meta['manifest_sha256']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['calibrate', 'generate', 'evaluate'])
    p.add_argument('--maps', help='Completed map directory for calibrate/generate')
    p.add_argument('--pseudo', help='Completed pseudo-label directory for evaluate')
    p.add_argument('--calibration', help='threshold.json for generate')
    p.add_argument('--root', default='.')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.action == 'calibrate':
        if not a.maps: p.error('--maps required')
        calibrate(a.maps, a.root, a.output)
    elif a.action == 'generate':
        if not a.maps or not a.calibration: p.error('--maps and --calibration required')
        materialize(a.maps, a.calibration, a.output)
    else:
        if not a.pseudo: p.error('--pseudo required')
        evaluate(a.pseudo, a.root, a.output)


if __name__ == '__main__':
    main()
