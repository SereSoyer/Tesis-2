"""Pseudo-only training targets and separate, explicitly supervised evaluation."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.pipeline.artifacts import dataset, load_artifact, load_arrays, ground_truth, sha256
from src.datasets.protocol import check_rows


class PseudoTrainingDataset(Dataset):
    def __init__(self, cfg, root, pseudo_directory):
        self.meta, rows = load_artifact(pseudo_directory, 'pseudo_labels')
        if self.meta['split'] != 'train':
            raise ValueError('Only train pseudo-labels may be training targets')
        check_rows(cfg, rows, 'train')
        self.images = dataset(cfg, root, 'train')
        if self.meta['source_manifest_sha256'] != sha256(Path(root)/cfg['manifests']['train']):
            raise ValueError('Pseudo-labels belong to another training split')
        for key in ('preprocessing', 'image_size', 'dataset'):
            if self.meta['config'][key] != cfg[key]:
                raise ValueError('Pseudo-label input configuration mismatch')
        if [r['sample_id'] for r in rows] != [r['sample_id'] for r in self.images.rows]:
            raise ValueError('Pseudo-label IDs/order differ from training data')
        self.rows, self.directory = rows, pseudo_directory

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = self.images[index]
        target = load_arrays(self.directory, self.rows[index])['pseudo_mask']
        if target.shape != tuple(sample['image'].shape[-2:]) or not np.isin(target, [0, 1]).all():
            raise ValueError('Pseudo-mask must be binary and aligned to input')
        # Ground-truth mask_path is deliberately never read in this class.
        return dict(image=sample['image'], target=torch.tensor(target, dtype=torch.float32),
                    sample_id=sample['sample_id'])


class ReferenceEvaluationDataset(Dataset):
    def __init__(self, cfg, root, split):
        if split not in ('val', 'test'):
            raise ValueError('Real-mask loader is restricted to validation/final evaluation')
        self.images, self.root = dataset(cfg, root, split), root

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = self.images[index]
        target = ground_truth(self.root, self.images.rows[index])
        if target.shape != tuple(sample['image'].shape[-2:]):
            raise ValueError('Reference shape mismatch')
        return dict(sample, target=torch.tensor(target, dtype=torch.float32))
