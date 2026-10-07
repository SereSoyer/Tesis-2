"""Compare completed branches on exactly the same evaluation population/config."""
import argparse
import json
from pathlib import Path
from src.pipeline.artifacts import load_artifact, write_rows
from src.training.train import write_json


def compare(runs, split, output):
    if len(runs) < 2:
        raise ValueError('Comparison requires at least two branches')
    records, source_records, reference, seen = [], [], None, set()
    for directory in map(Path, runs):
        meta, samples = load_artifact(directory/f'maps_{split}', 'maps')
        classification = json.loads((directory/f'maps_{split}/classification.json').read_text())
        localization = json.loads((directory/f'localization_{split}/summary.json').read_text())
        segmentation = json.loads((directory/f'segmentation_{split}/summary.json').read_text())
        branch = meta['branch']
        if branch in seen or localization['branch'] != branch or segmentation['branch'] != branch:
            raise ValueError('Duplicate or mismatched branches')
        seen.add(branch)
        manifest = meta['source_manifest_sha256']
        if (meta['split'] != split or localization['split'] != split or segmentation['split'] != split
                or localization['source_manifest_sha256'] != manifest
                or segmentation['evaluation_manifest_sha256'] != manifest):
            raise ValueError('Evaluation populations differ')
        config = segmentation['config']
        signature = {k: config[k] for k in ('segmentation', 'preprocessing', 'image_size',
                                           'seed', 'manifests', 'classification_threshold', 'pseudo_labels')}
        signature['ids'] = [r['sample_id'] for r in samples]
        classifier_config = dict(meta['classifier_config'])
        classifier_config.pop('model')
        classifier_config.setdefault('training_enabled', True)
        signature['classifier_training'] = classifier_config
        signature['dataset'] = config.get('dataset')
        signature['source_datasets'] = config.get('source_datasets')
        if reference is None: reference = signature
        elif reference != signature: raise ValueError('Common comparison protocol differs between branches')
        records.append(dict(branch=branch, split=split,
            **{f'classification_{k}': classification[k] for k in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')},
            pseudo_threshold=localization['threshold'],
            **{f'localization_{k}': localization[k] for k in ('macro_iou_positive', 'macro_dice_positive', 'micro_iou', 'pixel_f1')},
            **{f'segmentation_{k}': segmentation[k] for k in ('macro_iou_positive', 'macro_dice_positive', 'micro_iou', 'pixel_f1')}))
        if config.get('source_datasets'):
            by_source = json.loads((directory/f'maps_{split}/classification_by_dataset.json').read_text())
            for source in config['source_datasets']:
                source_records.append(dict(branch=branch, split=split, dataset=source,
                    **{f'classification_{k}': by_source[source][k] for k in ('accuracy','precision','recall','f1','roc_auc')},
                    **{f'localization_{k}': localization['by_dataset'][source][k] for k in ('macro_iou_positive','macro_dice_positive','micro_iou','pixel_f1')},
                    **{f'segmentation_{k}': segmentation['by_dataset'][source][k] for k in ('macro_iou_positive','macro_dice_positive','micro_iou','pixel_f1')}))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output/'comparison.csv', records)
    if source_records:
        write_rows(output/'comparison_by_dataset.csv', source_records)
    write_json(output/'comparison.json', dict(results=records, shared_protocol=reference,
        interpretation='Descriptive single-seed comparison; method/layer resolution remain possible confounders.'))
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--split', choices=['val', 'test'], required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    compare(a.runs, a.split, a.output)


if __name__ == '__main__': main()
