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
from torch import nn
from src.localization.gradcam import GradCAM
from src.models.classifiers import build_model
from src.pseudo_labels.masks import normalize_map, pseudo_mask
from src.pipeline.artifacts import sha256, load_artifact, load_arrays, write_rows
from src.pipeline.maps import generate
from src.pipeline.pseudo import calibrate, materialize, evaluate
from src.segmentation.data import PseudoTrainingDataset, ReferenceEvaluationDataset
from src.segmentation.model import DeepLabV3Plus
from src.segmentation.train import train, evaluate_checkpoint
from src.pipeline.compare import compare
from src.training.train import write_json, save_checkpoint


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.layer4 = nn.Conv2d(2, 1, 1, bias=False)
        nn.init.ones_(self.backbone.layer4.weight)

    def forward(self, x):
        return self.backbone.layer4(x).mean((1, 2, 3))


class TinySegmenter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layer = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        return self.layer(x)[:, 0]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_gradcam_known_spatial_signal_and_no_parameter_gradients(self):
        model = TinyClassifier()
        image = torch.zeros(1, 2, 8, 8)
        image[:, :, 2:4, 5:7] = 2
        before = {k: v.clone() for k, v in model.state_dict().items()}
        cam = GradCAM(model, 'resnet18')
        raw = cam(image)['raw'][0].numpy()
        cam.close()
        expected = np.zeros((8, 8)); expected[2:4, 5:7] = 1
        normalized, constant = normalize_map(raw)
        np.testing.assert_array_equal(normalized, expected)
        self.assertFalse(constant)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))

    def test_four_real_classifier_adapters(self):
        for name in ('resnet18', 'resnet50', 'deit_small', 'mit_b1'):
            with self.subTest(name=name):
                torch.manual_seed(42)
                model = build_model(name, 2, image_size=32)
                cam = GradCAM(model, name)
                result = cam(torch.randn(1, 2, 32, 32))
                self.assertEqual(result['raw'].shape, (1, 32, 32))
                self.assertGreater(float(result['gradient_l1'][0]), 0)
                self.assertTrue(torch.isfinite(result['raw']).all())
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                cam.close()

    def test_deeplab_variant_forward_backward_singleton(self):
        model = DeepLabV3Plus(2).train()
        image = torch.randn(1, 2, 32, 32)
        logits = model(image)
        self.assertEqual(logits.shape, (1, 32, 32))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.decoder[-1].weight.grad).all())

    def test_constant_map_and_probability_gate(self):
        zeros, constant = normalize_map(np.ones((4, 4)))
        self.assertTrue(constant)
        self.assertFalse(pseudo_mask(zeros, 0.9, 0.5).any())
        self.assertFalse(pseudo_mask(np.ones((4, 4)), 0.2, 0.5).any())
        with self.assertRaises(ValueError): normalize_map(np.full((4, 4), np.nan))

    def fixture(self, root):
        cfg = json.loads(Path('configs/pipeline_l4s.json').read_text())
        cfg.update(device='cpu', image_size=8, threads=1)
        cfg['preprocessing']['channels'] = [0, 1]
        cfg['segmentation'].update(batch_size=2, epochs=2)
        cfg['manifests'] = {}
        for split, source in [('train', 'TrainData'), ('val', 'ValidData'), ('test', 'TestData')]:
            rows = []
            for i in range(3):
                sample_id = f'{split}_{i}'
                image = np.zeros((8, 8, 2), dtype=np.float32)
                image[:, 4:, :] = i + 1
                with h5py.File(root/f'{sample_id}.h5', 'w') as stream: stream['img'] = image
                mask = np.zeros((8, 8), dtype=np.uint8)
                if i != 0: mask[:, 4:] = 1
                # Deliberately absent training GT: pseudo-training must still work.
                if split != 'train':
                    with h5py.File(root/f'{sample_id}_mask.h5', 'w') as stream: stream['mask'] = mask
                rows.append(dict(sample_id=sample_id, dataset='landslide4sense', source_split=source,
                    label=int(mask.any()), image_path=f'{sample_id}.h5', mask_path=f'{sample_id}_mask.h5',
                    height=8, width=8, channels=2, area_ratio=float(mask.mean())))
            cfg['manifests'][split] = f'{split}.csv'
            write_rows(root/f'{split}.csv', rows)
        cfg['branches']['resnet18']['checkpoint'] = 'classifier.pt'
        torch.save(dict(config=dict(cfg, model='resnet18'), model=TinyClassifier().state_dict(), epoch=2,
                        manifest_sha256={k: sha256(root/v) for k, v in cfg['manifests'].items()}), root/'classifier.pt')
        return cfg

    @patch('src.pipeline.maps.build_model', side_effect=lambda *a: TinyClassifier())
    @patch('src.segmentation.train.DeepLabV3Plus', TinySegmenter)
    def test_end_to_end_no_training_gt_and_provenance(self, _):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.fixture(root)
            for split in ('train', 'val', 'test'):
                generate(cfg, 'resnet18', split, root, root/f'maps_{split}')
            for split in ('train', 'test'):
                with self.assertRaisesRegex(ValueError, 'validation'):
                    calibrate(root/f'maps_{split}', root, root/'invalid')
            selected = calibrate(root/'maps_val', root, root/'calibration')
            # All scores tied for these binary maps: largest threshold must win.
            self.assertEqual(selected['threshold'], 0.95)
            for split in ('train', 'val', 'test'):
                materialize(root/f'maps_{split}', root/'calibration/threshold.json', root/f'pseudo_{split}')
            with self.assertRaisesRegex(ValueError, 'Only train'):
                PseudoTrainingDataset(cfg, root, root/'pseudo_val')
            with self.assertRaises(ValueError): ReferenceEvaluationDataset(cfg, root, 'train')
            history = train(cfg, root, root/'pseudo_train', root/'segmenter')
            self.assertEqual(len(history), 2)
            evaluate(root/'pseudo_val', root, root/'localization_eval')
            evaluate_checkpoint(root/'segmenter/best.pt', root, 'test', root/'final_eval')
            self.assertEqual(json.loads((root/'final_eval/summary.json').read_text())['samples'], 3)
            self.assertFalse(any(root.glob('train_*_mask.h5')))
            ds = PseudoTrainingDataset(cfg, root, root/'pseudo_train')
            with open(root/'pseudo_train'/ds.rows[0]['artifact_path'], 'ab') as stream: stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'modified'): ds[0]

    @patch('src.pipeline.maps.build_model', side_effect=lambda *a: TinyClassifier())
    @patch('src.segmentation.train.DeepLabV3Plus', TinySegmenter)
    def test_segmentation_resume_matches_continuous(self, _):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.fixture(root)
            for split in ('train', 'val'):
                generate(cfg, 'resnet18', split, root, root/f'maps_{split}')
            calibrate(root/'maps_val', root, root/'calibration')
            materialize(root/'maps_train', root/'calibration/threshold.json', root/'pseudo')
            train(cfg, root, root/'pseudo', root/'full')

            def interrupt_second_epoch(path, state):
                if path.name == 'last.pt' and state['epoch'] == 2:
                    raise RuntimeError('simulated interruption')
                save_checkpoint(path, state)

            with patch('src.segmentation.train.save_checkpoint', side_effect=interrupt_second_epoch):
                with self.assertRaisesRegex(RuntimeError, 'simulated'):
                    train(cfg, root, root/'pseudo', root/'interrupted')
            train(cfg, root, root/'pseudo', root/'interrupted', resume=True)
            a = torch.load(root/'full/last.pt', weights_only=False)
            b = torch.load(root/'interrupted/last.pt', weights_only=False)
            for key in a['model']:
                self.assertTrue(torch.equal(a['model'][key], b['model'][key]))
            for x, y in zip(a['history'], b['history']):
                self.assertEqual(x['train_pseudo_loss'], y['train_pseudo_loss'])
                self.assertEqual(x['validation_gt'], y['validation_gt'])

    def test_comparison_rejects_changed_population_or_training_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.fixture(root)
            runs = []
            metric = dict(macro_iou_positive=0.4, macro_dice_positive=0.5, micro_iou=0.4, pixel_f1=0.5)
            for branch in ('resnet18', 'deit_small'):
                run = root/branch
                runs.append(run)
                for sub in ('maps_val', 'localization_val', 'segmentation_val'):
                    (run/sub).mkdir(parents=True)
                write_rows(run/'maps_val/manifest.csv', [{'sample_id': 'same'}])
                write_json(run/'maps_val/metadata.json', dict(stage='maps', split='val', branch=branch,
                    manifest_sha256=sha256(run/'maps_val/manifest.csv'), source_manifest_sha256='same',
                    classifier_config=dict(model=branch, epochs=50)))
                write_json(run/'maps_val/classification.json', {k: 0.5 for k in ('accuracy','precision','recall','f1','roc_auc')})
                write_json(run/'localization_val/summary.json', dict(metric, split='val', branch=branch,
                    source_manifest_sha256='same', threshold=0.5))
                write_json(run/'segmentation_val/summary.json', dict(metric, split='val', branch=branch,
                    evaluation_manifest_sha256='same', config=cfg))
            self.assertEqual(len(compare(runs, 'val', root/'comparison')), 2)
            path = runs[1]/'segmentation_val/summary.json'
            summary = json.loads(path.read_text())
            summary['config']['segmentation']['epochs'] = 99
            write_json(path, summary)
            with self.assertRaisesRegex(ValueError, 'protocol differs'):
                compare(runs, 'val', root/'invalid')
            summary['config'] = cfg
            summary['evaluation_manifest_sha256'] = 'different'
            write_json(path, summary)
            with self.assertRaisesRegex(ValueError, 'populations differ'):
                compare(runs, 'val', root/'invalid')


if __name__ == '__main__':
    unittest.main()
