"""Classification-only HDF5 reader. Pixel masks are never opened."""
import csv
import random
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ClassificationDataset(Dataset):
    def __init__(self, csv_path, root, preprocess, image_key='img', layout='HWC', allowed_datasets=None):
        self.root = Path(root).resolve()
        with open(csv_path, newline='') as stream:
            self.rows = list(csv.DictReader(stream))
        if not self.rows or not {'sample_id','image_path','label','dataset'} <= self.rows[0].keys():
            raise ValueError('CSV requires sample_id,image_path,label,dataset and at least one row')
        if any(('status' in r and r['status'] != 'accepted') or
               ('training_eligible' in r and r['training_eligible'].lower() != 'true')
               for r in self.rows):
            raise ValueError('Provisional or excluded samples cannot enter the classifier')
        present = {r['dataset'] for r in self.rows}
        if allowed_datasets is None:
            if len(present) != 1:
                raise ValueError('Datasets must be processed separately unless explicitly configured')
        elif not present <= set(allowed_datasets):
            raise ValueError('Unapproved dataset in manifest')
        if len({r['sample_id'] for r in self.rows}) != len(self.rows):
            raise ValueError('Duplicate sample_id')
        if any(r['label'] not in ('0','1') for r in self.rows):
            raise ValueError('Labels must be 0 or 1')
        if layout not in ('HWC','CHW'):
            raise ValueError('layout must be HWC or CHW')
        self.preprocess, self.image_key, self.layout = preprocess, image_key, layout
        for row in self.rows:
            path = Path(row['image_path'])
            if path.is_absolute() or not (self.root/path).resolve().is_relative_to(self.root):
                raise ValueError('image_path must be relative to dataset root')
            if not (self.root/path).is_file():
                raise FileNotFoundError(self.root/path)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with h5py.File(self.root/row['image_path'], 'r') as file:
            array = file[self.image_key][:]
        if self.layout == 'CHW':
            array = np.moveaxis(array, 0, -1)
        return {'image': self.preprocess(array), 'label': torch.tensor(float(row['label'])), 'sample_id': row['sample_id']}

def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

def make_loader(dataset, batch_size=2, training=False, seed=42, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=training,
                      num_workers=num_workers, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed))
