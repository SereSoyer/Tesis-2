"""Compare exact decoded-image hashes from existing L4S audit manifests."""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    groups = defaultdict(list)
    manifests = {}
    for split in ('traindata', 'validdata', 'testdata'):
        path = ROOT/'metadata'/f'landslide4sense_{split}.csv'
        if not path.exists():
            continue
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        manifests[split] = len(rows)
        for row in rows:
            groups[row['image_sha256']].append({k: row[k] for k in ('sample_id', 'source_split', 'label', 'image_path')})
    repeated = [rows for rows in groups.values() if len(rows) > 1]
    report = {
        'manifests': manifests,
        'within_split_duplicate_groups': [g for g in repeated if len({r['source_split'] for r in g}) == 1],
        'cross_split_duplicate_groups': [g for g in repeated if len({r['source_split'] for r in g}) > 1],
        'scope': 'Exact decoded array hashes including shape and dtype. Does not establish geographic independence or rule out overlapping patches. Uses existing audit manifests.'
    }
    path = ROOT/'results/audits/l4s_duplicate_check.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
