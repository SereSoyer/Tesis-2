"""Persist audited original L4S partitions without random reassignment or training."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    prepared = []
    seen_ids, seen_paths, seen_hashes = set(), set(), set()
    for split, source, expected in [('train', 'TrainData', 3799), ('val', 'ValidData', 245), ('test', 'TestData', 800)]:
        path = ROOT/'metadata'/f'landslide4sense_{source.lower()}.csv'
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected:
            raise ValueError(f'Unexpected count for {source}')
        for row in rows:
            if row['dataset'] != 'landslide4sense' or row['source_split'] != source:
                raise ValueError('Incorrect dataset/source partition')
            if row['label'] not in ('0', '1') or row['mask_binary'] != 'True' or row['shape_matches'] != 'True':
                raise ValueError('Invalid mask/label audit')
            if (row['height'], row['width'], row['channels']) != ('128', '128', '14') or int(row['nan']) or int(row['inf']):
                raise ValueError('Unexpected shape or nonfinite values')
            for field in ('image_path', 'mask_path'):
                p = (ROOT/row[field]).resolve()
                if not p.is_relative_to(ROOT) or not p.is_file():
                    raise ValueError(f'Missing or invalid {field}')
            for field, seen in [('sample_id', seen_ids), ('image_path', seen_paths), ('image_sha256', seen_hashes)]:
                value = row[field]
                if not value or value in seen:
                    raise ValueError(f'Duplicate or missing {field}; review before persisting')
                seen.add(value)
        target = ROOT/'splits'/f'l4s_{split}.csv'
        data = path.read_bytes()
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(f'Refusing to replace different split: {target}')
        prepared.append((target, data, dict(split=split, source_split=source, samples=len(rows),
                         positive=sum(r['label']=='1' for r in rows), negative=sum(r['label']=='0' for r in rows),
                         sha256=hashlib.sha256(data).hexdigest())))
    (ROOT/'splits').mkdir(exist_ok=True)
    for target, data, _ in prepared:
        target.write_bytes(data)
        if target.read_bytes() != data:
            raise IOError('Split copy verification failed')
    report = dict(method='Preserve original partitions; byte-identical copies of audited metadata CSVs',
                  random_split=False, split_seed=None, training_performed=False,
                  limitations='Exact duplicates checked from audit hashes; geographic independence and overlapping patches are not established. Raw images must remain unchanged since auditing.',
                  partitions=[item[2] for item in prepared])
    (ROOT/'results/audits/l4s_splits.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
