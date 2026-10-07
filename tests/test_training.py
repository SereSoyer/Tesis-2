import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch
from src.datasets.classification import make_loader
from src.training.epochs import run_epoch
from src.training.train import run_experiment


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Dropout(0.2), torch.nn.Linear(32, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class TrainingTests(unittest.TestCase):
    def test_disabled_dataset_stops_before_loading_data(self):
        cfg = json.loads(Path('configs/train_roy_pending.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('src.training.train.prepare_data') as prepare:
                with self.assertRaisesRegex(ValueError, 'Training disabled for roy_l4s_pe'):
                    run_experiment(cfg, root, root/'output')
                prepare.assert_not_called()
            self.assertFalse((root/'output').exists())

    def fixture(self, root):
        cfg = json.loads(Path('configs/train_l4s.json').read_text())
        cfg.update(dataset='synthetic', image_size=4, batch_size=3, threads=1)
        cfg['preprocessing']['channels'] = [0, 1]
        cfg['manifests'] = {'train': 'train.csv', 'val': 'val.csv'}
        rng = np.random.default_rng(42)
        for split in ('train', 'val'):
            with (root/f'{split}.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['sample_id', 'dataset', 'image_path', 'label', 'mask_path'])
                writer.writeheader()
                for i in range(5):
                    name = f'{split}_{i}'
                    with h5py.File(root/f'{name}.h5', 'w') as image:
                        image['img'] = rng.normal(size=(4, 4, 2)).astype('float32')
                    writer.writerow(dict(sample_id=name, dataset='synthetic', image_path=f'{name}.h5',
                                         label=i % 2, mask_path='MASK_DOES_NOT_EXIST.h5'))
        return cfg

    @patch('src.training.train.build_model', side_effect=lambda *a: TinyClassifier())
    def test_resume_matches_uninterrupted_and_preserves_best(self, _):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.fixture(root)
            full = run_experiment(cfg, root, root/'full')
            first = copy.deepcopy(cfg); first['epochs'] = 1
            run_experiment(first, root, root/'resumed')
            (root/'resumed/best.pt').unlink()  # Simulate interruption before best checkpoint write.
            resumed = run_experiment(cfg, root, root/'resumed', resume=True)
            for a, b in zip(full, resumed):
                for split in ('train', 'val'):
                    self.assertEqual({k:v for k,v in a[split].items() if k != 'seconds'},
                                     {k:v for k,v in b[split].items() if k != 'seconds'})
                    self.assertEqual(a[split]['n_samples'], 5)
            load = lambda p: torch.load(p, weights_only=False, map_location='cpu')
            a, b = load(root/'full/last.pt'), load(root/'resumed/last.pt')
            for key in a['model']:
                self.assertTrue(torch.equal(a['model'][key], b['model'][key]))
            best = load(root/'resumed/best.pt')
            expected = min(resumed, key=lambda h: h['val']['loss'])
            self.assertEqual(best['epoch'], expected['epoch'])
            self.assertEqual(json.loads((root/'resumed/history.json').read_text()), resumed)
            self.assertTrue((root/'resumed/splits/train.csv').is_file())
            with self.assertRaises(FileExistsError):
                run_experiment(cfg, root, root/'resumed')
            changed = copy.deepcopy(cfg); changed.update(epochs=3, threshold=0.7)
            with self.assertRaisesRegex(ValueError, 'unchanged'):
                run_experiment(changed, root, root/'resumed', resume=True)
            (root/'val.csv').write_text((root/'val.csv').read_text() + '\n')
            changed = copy.deepcopy(cfg); changed['epochs'] = 3
            with self.assertRaisesRegex(ValueError, 'unchanged'):
                run_experiment(changed, root, root/'resumed', resume=True)

    def test_validation_no_updates_and_weighted_loss(self):
        model = TinyClassifier()
        samples = [{'image': torch.ones(2,4,4)*i, 'label': torch.tensor(float(i%2))} for i in range(5)]
        loader = make_loader(samples, batch_size=3)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        report = run_epoch(model, loader, torch.device('cpu'))
        model.eval()
        with torch.no_grad():
            expected = torch.nn.functional.binary_cross_entropy_with_logits(
                model(torch.stack([s['image'] for s in samples])), torch.stack([s['label'] for s in samples]))
        self.assertAlmostEqual(report['loss'], expected.item(), places=6)
        self.assertFalse(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(torch.equal(v, before[k]) for k,v in model.state_dict().items()))

    @patch('src.training.train.build_model', side_effect=lambda *a: TinyClassifier())
    def test_missing_and_overlapping_splits_rejected(self, _):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.fixture(root)
            cfg['manifests']['val'] = 'missing.csv'
            with self.assertRaises(FileNotFoundError):
                run_experiment(cfg, root, root/'output')
            self.assertFalse((root/'output').exists())
            cfg['manifests']['val'] = 'train.csv'
            with self.assertRaisesRegex(ValueError, 'overlap'):
                run_experiment(cfg, root, root/'output')
            self.assertFalse((root/'output').exists())


if __name__ == '__main__':
    unittest.main()
