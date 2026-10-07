import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from scripts.admit_l4s_roy14 import admit_rows, VERSION
from scripts.run_joint_experiment import training_state
from src.datasets.protocol import check_manifest_integrity, EMPIRICAL_MAPPING, SOURCES
from src.pipeline.artifacts import sha256, write_rows


class JointAdmissionTests(unittest.TestCase):
    def test_admission_preserves_sample_labels_paths_groups_and_partitions(self):
        row = dict(sample_id='roy_example', label='1', image_path='source.h5', mask_path='mask.h5',
                   geographic_group='roy_overlap_example', partition='train', dataset_version=VERSION,
                   status='pending_review', training_eligible='false', band_mapping_status='pending_conflicting_sources')
        before = copy.deepcopy(row)
        admitted = admit_rows([row], 'train')[0]
        self.assertEqual(row, before)
        for key in ('sample_id', 'label', 'image_path', 'mask_path', 'geographic_group', 'partition'):
            self.assertEqual(admitted[key], row[key])
        self.assertEqual(admitted['status'], 'accepted')
        self.assertEqual(admitted['band_mapping_status'], EMPIRICAL_MAPPING)
        with self.assertRaisesRegex(ValueError, 'unchanged reviewed proposal'):
            admit_rows([row], 'val')

    def test_changed_accepted_population_or_admission_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_rows(root / 'train.csv', [dict(sample_id='a', label='1')])
            cfg = dict(dataset=VERSION, training_enabled=True, source_datasets=SOURCES,
                       preprocessing=dict(channels=list(range(14)), normalization='per_image_minmax', nan_policy='error'),
                       band_mapping_status=EMPIRICAL_MAPPING, manifests={'train': 'train.csv'},
                       manifest_sha256={'train': sha256(root / 'train.csv')})
            admission = dict(status='admitted', dataset_version=VERSION, band_mapping_status=EMPIRICAL_MAPPING,
                             manifest_sha256=cfg['manifest_sha256'])
            (root / 'admission.json').write_text(json.dumps(admission))
            cfg['data_admission'] = dict(path='admission.json', sha256=sha256(root / 'admission.json'))
            check_manifest_integrity(cfg, root, 'train')
            write_rows(root / 'train.csv', [dict(sample_id='a', label='0')])
            with self.assertRaisesRegex(ValueError, 'Admitted file changed'):
                check_manifest_integrity(cfg, root, 'train')
            write_rows(root / 'train.csv', [dict(sample_id='a', label='1')])
            admission['status'] = 'pending'
            (root / 'admission.json').write_text(json.dumps(admission))
            with self.assertRaisesRegex(ValueError, 'Admitted file changed'):
                check_manifest_integrity(cfg, root, 'train')

    def test_runner_resumes_remaining_budget_and_validates_completed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'experiment'
            cfg = dict(epochs=50, manifest_sha256={'train': 'train_hash', 'val': 'val_hash'})
            self.assertEqual(training_state(output, cfg), 'new')
            output.mkdir()
            state = dict(epoch=42, best_epoch=25, config=cfg, manifest_sha256=cfg['manifest_sha256'])
            torch.save(state, output / 'last.pt')
            self.assertEqual(training_state(output, cfg), 'resume')
            state['epoch'] = 50
            torch.save(state, output / 'last.pt')
            torch.save(dict(state, epoch=25), output / 'best.pt')
            self.assertEqual(training_state(output, cfg), 'complete')
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                training_state(output, dict(cfg, epochs=60))
            state['best_epoch'] = 50
            torch.save(state, output / 'last.pt')
            self.assertEqual(training_state(output, cfg), 'complete')
            self.assertEqual(torch.load(output / 'best.pt', weights_only=False)['epoch'], 50)
