"""Manifest integrity checks; inventory approval is separate from training approval."""
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_manifest(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def array_digest(array):
    """Match the legacy L4S decoded-array hash (dtype and shape included)."""
    return hashlib.sha256(str(array.shape).encode() + str(array.dtype).encode()
                          + np.ascontiguousarray(array).tobytes()).hexdigest()


def duplicate_groups(rows, field):
    groups = defaultdict(list)
    for row in rows:
        if row.get(field) not in ('', None, 'unavailable'):
            groups[row[field]].append(row['sample_id'])
    return [sorted(ids) for ids in groups.values() if len(ids) > 1]


def validate_manifest(rows, root, require_accepted=False):
    """Reject invalid paths/IDs/labels and provisional training inputs."""
    root = Path(root).resolve()
    if not rows:
        raise ValueError('Empty manifest')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample_id')
    for row in rows:
        if str(row['label']) not in ('0', '1'):
            raise ValueError('Nonbinary label')
        for key in ('image_path', 'mask_path'):
            path = Path(row[key])
            resolved = (root/path).resolve()
            if path.is_absolute() or not resolved.is_relative_to(root):
                raise ValueError('Paths must remain inside root')
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
        if require_accepted and (row.get('status') != 'accepted'
                                 or str(row.get('training_eligible')).lower() != 'true'):
            raise ValueError('Provisional or excluded sample cannot enter training')


def validate_partition_integrity(partitions):
    """Reject IDs, paths, exact content and known groups crossing splits.

    Missing geography does not establish independence. Callers must retain that
    limitation and must not synthesize group identifiers from unknown events.
    """
    for field in ('sample_id', 'image_path', 'image_sha256', 'geographic_group'):
        owners = {}
        for split, rows in partitions.items():
            for row in rows:
                value = row.get(field)
                if value in (None, '', 'unavailable'):
                    continue
                if value in owners and owners[value] != split:
                    raise ValueError(f'Cross-split overlap in {field}')
                owners[value] = split
