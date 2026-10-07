import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from PIL import Image
from torch import nn

from src.datasets.classification import ClassificationDataset
from src.datasets.integration import grouped_assignment, reconstruct_roy, prepare
from src.datasets.protocol import CHANNELS, INTEGRATED, INTEGRATED14, EMPIRICAL_MAPPING, SOURCES, check_rows, check_partitions
from src.datasets.roy_review import spatial_groups
from src.pipeline.artifacts import write_rows, sha256
from src.pipeline.maps import generate
from src.pipeline.pseudo import calibrate, materialize, evaluate
from src.preprocessing.multispectral import Preprocess
from src.segmentation.train import train, evaluate_checkpoint


class Classifier(nn.Module):
    def __init__(self, channels=10):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.layer4 = nn.Conv2d(channels, 1, 1, bias=False)
        nn.init.ones_(self.backbone.layer4.weight)

    def forward(self, x):
        return self.backbone.layer4(x).mean((1, 2, 3))


class Segmenter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layer = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        return self.layer(x)[:, 0]


class IntegrationTests(unittest.TestCase):
    def test_overlap_components_are_transitive_and_group_splits_are_reproducible(self):
        rows = [dict(sample_id=str(i), xmin=i*9, xmax=i*9+10, ymin=0, ymax=10, label=str(i%2))
                for i in range(3)]
        spatial_groups(rows)
        self.assertEqual(len({r['geographic_group'] for r in rows}), 1)
        rows = [dict(sample_id=str(i), label=str(i%2), geographic_group=f'g{i//2}') for i in range(40)]
        a = grouped_assignment(rows, 42, [.7,.15,.15], trials=20)
        self.assertEqual(a, grouped_assignment(rows, 42, [.7,.15,.15], trials=20))
        self.assertTrue(all(a[str(i)] == a[str(i+1)] for i in range(0,40,2)))
        bad = {'train':[dict(sample_id='a', geographic_group='same')],
               'test':[dict(sample_id='b', geographic_group='same')]}
        with self.assertRaisesRegex(ValueError, 'geographic_group'):
            check_partitions(bad)

    def test_reconstruction_has_correct_pixel_alignment_and_rejects_unknown_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'input').mkdir(); (folder/'target').mkdir()
            rr, cc = np.indices((150, 150))
            image = (rr*150+cc).astype(np.float32)
            for band in range(1,15):
                Image.fromarray(image).save(folder/f'input/B{band}.tif')
            png = np.zeros((130,130,3), dtype=np.uint8)
            png[...,1] = 255
            png[2,3] = [255,0,0]  # original image row 12, col 13
            Image.fromarray(png).save(folder/'target/manual.png')
            x, y = reconstruct_roy(folder)
            self.assertEqual(y[2,1], 1)  # both arrays transposed after the same ROI crop
            self.assertEqual(x[2,1,0], image[12,13])
            self.assertEqual(y.sum(), 1)
            png[0,0] = [50,50,255]
            Image.fromarray(png).save(folder/'target/manual.png')
            with self.assertRaisesRegex(ValueError, 'NoData'):
                reconstruct_roy(folder)

    def test_pending_decisions_do_not_materialize(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp)/'policy.json'
            policy.write_text(json.dumps(dict(training_authorized=False, decisions={})))
            with self.assertRaisesRegex(ValueError, 'pending'):
                prepare(tmp, policy)
            self.assertEqual(list(Path(tmp).iterdir()), [policy])

    @patch('src.pipeline.maps.build_model', side_effect=lambda name, channels, *args: Classifier(channels))
    @patch('src.segmentation.train.DeepLabV3Plus', Segmenter)
    def test_mixed_end_to_end_no_train_gt_and_per_dataset_reporting(self, _):
        for version, channels in [(INTEGRATED, CHANNELS), (INTEGRATED14, list(range(14)))]:
            with self.subTest(version=version):
                self.mixed_pipeline(version, channels)

    def mixed_pipeline(self, version, channels):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = json.loads(Path('configs/pipeline_l4s.json').read_text())
            cfg.update(dataset=version, source_datasets=SOURCES, device='cpu', image_size=8, threads=1)
            cfg['preprocessing'].update(channels=channels, normalization='per_image_minmax')
            cfg['segmentation'].update(epochs=1, batch_size=2)
            cfg['manifests'] = {}
            for split in ('train','val','test'):
                rows = []
                for index in range(4):
                    source = SOURCES[index//2]
                    sid = f'{split}_{index}'
                    x = np.zeros((8,8,14), dtype=np.float32)
                    y = np.zeros((8,8), dtype=np.uint8)
                    if index%2: x[:,4:,:] = 3; y[:,4:] = 1
                    with h5py.File(root/f'{sid}.h5','w') as f: f['img'] = x
                    if split != 'train':
                        with h5py.File(root/f'{sid}_mask.h5','w') as f: f['mask'] = y
                    rows.append(dict(sample_id=sid, dataset=source,
                        source_split={'train':'TrainData','val':'ValidData','test':'TestData'}[split]
                                     if source == SOURCES[0] else 'RoyOriginal',
                        label=str(index%2), image_path=f'{sid}.h5', mask_path=f'{sid}_mask.h5',
                        partition=split, dataset_version=version, status='accepted', training_eligible='true',
                        band_mapping_status=EMPIRICAL_MAPPING if version == INTEGRATED14 else '',
                        geographic_group='roy_overlap_'+sid if source == SOURCES[1] else '',
                        alignment_policy='iris_mask_area_reconstructed_v1' if source == SOURCES[1] else '',
                        area_ratio=float(y.mean())))
                write_rows(root/f'{split}.csv', rows)
                cfg['manifests'][split] = f'{split}.csv'
            if version == INTEGRATED14:
                cfg['band_mapping_status'] = EMPIRICAL_MAPPING
                cfg['manifest_sha256'] = {s: sha256(root / p) for s, p in cfg['manifests'].items()}
                (root / 'admission.json').write_text(json.dumps(dict(status='admitted', dataset_version=version,
                    band_mapping_status=EMPIRICAL_MAPPING, manifest_sha256=cfg['manifest_sha256'])))
                cfg['data_admission'] = dict(path='admission.json', sha256=sha256(root / 'admission.json'))
            bad = copy.deepcopy(cfg)
            bad['preprocessing']['channels'] = list(range(14)) if version == INTEGRATED else CHANNELS
            with self.assertRaisesRegex(ValueError, 'common-ten'):
                check_rows(bad, rows, 'test')
            with self.assertRaisesRegex(ValueError, 'separately'):
                ClassificationDataset(root/'train.csv', root, Preprocess(channels))
            cfg['branches']['resnet18']['checkpoint'] = 'classifier.pt'
            torch.save(dict(config=dict(cfg, model='resnet18'), model=Classifier(len(channels)).state_dict(), epoch=1,
                            manifest_sha256={s:sha256(root/p) for s,p in cfg['manifests'].items()}), root/'classifier.pt')
            for split in ('train','val','test'):
                generate(cfg, 'resnet18', split, root, root/f'maps_{split}')
            with self.assertRaisesRegex(ValueError, 'validation'):
                calibrate(root/'maps_test', root, root/'bad_calibration')
            calibrate(root/'maps_val', root, root/'calibration')
            for split in ('train','val'):
                materialize(root/f'maps_{split}', root/'calibration/threshold.json', root/f'pseudo_{split}')
            evaluate(root/'pseudo_val', root, root/'localization')
            train(cfg, root, root/'pseudo_train', root/'segmenter')
            evaluate_checkpoint(root/'segmenter/best.pt', root, 'test', root/'evaluation')
            for path in ('localization/summary.json','evaluation/summary.json'):
                result = json.loads((root/path).read_text())
                self.assertEqual(result['samples'], 4)
                self.assertEqual(set(result['by_dataset']), set(SOURCES))
                self.assertTrue(all(v['samples'] == 2 for v in result['by_dataset'].values()))
            self.assertFalse(any(root.glob('train_*_mask.h5')))
            # Reclassifying a test Roy row as train must not pass partition guards.
            with self.assertRaises(ValueError):
                check_rows(cfg, rows, 'train')
