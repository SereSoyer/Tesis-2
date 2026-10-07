import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.datasets.classification import ClassificationDataset
from src.datasets.manifests import array_digest, duplicate_groups, validate_manifest, validate_partition_integrity
from src.evaluation.spatial import spatial_metrics
from src.preprocessing.multispectral import Preprocess
from src.preprocessing.statistics import ChannelMoments


class O1Tests(unittest.TestCase):
    def test_spatial_known_counts(self):
        result = spatial_metrics([[1,1],[0,0]], [[1,0],[1,0]])
        self.assertEqual([result[k] for k in ['tp','fp','fn','tn']], [1,1,1,1])
        self.assertAlmostEqual(result['iou'], 1/3)
        for key in ['dice','pixel_precision','pixel_recall']:
            self.assertEqual(result[key], 0.5)

    def test_spatial_empty_and_false_negative(self):
        empty = spatial_metrics(np.zeros((2,2)), np.zeros((2,2)))
        self.assertEqual(len(empty['undefined_metrics']), 4)
        missed = spatial_metrics(np.ones((2,2)), np.zeros((2,2)))
        self.assertEqual(missed['iou'], 0)
        self.assertEqual(missed['dice'], 0)
        self.assertEqual(missed['pixel_recall'], 0)
        self.assertIsNone(missed['pixel_precision'])
        false_alarm = spatial_metrics(np.zeros((2,2)), np.ones((2,2)))
        self.assertEqual(false_alarm['iou'], 0)
        self.assertIsNone(false_alarm['pixel_recall'])

    def test_spatial_invalid_and_validity(self):
        for target,pred in [(np.zeros((2,2)),np.zeros((3,3))), ([[np.nan]],[[0]]), ([[1]],[[0.5]])]:
            with self.assertRaises(ValueError): spatial_metrics(target,pred)
        self.assertEqual(spatial_metrics([[1,0]],[[1,1]],[[1,0]])['iou'],1)
        with self.assertRaises(ValueError): spatial_metrics([[1]],[[1]],[[0]])

    def test_statistics_match_numpy_and_ignore_nonfinite(self):
        x = np.arange(48,dtype=float).reshape(4,6,2)
        x[0,0,1] = np.nan
        stats = ChannelMoments(2)
        stats.update(x[:2]);stats.update(x[2:])
        for i,r in enumerate(stats.report()):
            self.assertAlmostEqual(r['mean'], np.nanmean(x[...,i]))
            self.assertAlmostEqual(r['std'], np.nanstd(x[...,i]))
        self.assertEqual(stats.report()[1]['nonfinite_count'],1)

    def test_preprocessing_preserves_spatial_indices(self):
        x = np.arange(3*4*14,dtype=float).reshape(3,4,14)
        y = Preprocess(list(range(14)))(x).numpy()
        np.testing.assert_array_equal(y.transpose(1,2,0),x.astype('float32'))
        x[0,0,0]=np.inf
        with self.assertRaises(ValueError): Preprocess(list(range(14)))(x)

    def test_duplicates_and_partition_leakage(self):
        a = dict(sample_id='a', image_path='a.h5', image_sha256=array_digest(np.zeros((2,2))))
        b = dict(sample_id='b', image_path='b.h5', image_sha256=a['image_sha256'])
        self.assertEqual(duplicate_groups([a,b],'image_sha256'),[['a','b']])
        with self.assertRaises(ValueError): validate_partition_integrity({'train':[a],'test':[b]})
        b.update(image_sha256='different', geographic_group='event')
        a['geographic_group']='event'
        with self.assertRaises(ValueError): validate_partition_integrity({'train':[a],'val':[b]})

    def test_provisional_manifest_rejected_by_classifier(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p=root/'rows.csv'
            row=dict(sample_id='a',dataset='l4s',image_path='image.h5',mask_path='mask.h5',label='1',status='pending_review',training_eligible='false')
            for name in ('image.h5','mask.h5'): (root/name).touch()
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
            validate_manifest([row],root)
            with self.assertRaises(ValueError): validate_manifest([row],root,True)
            with self.assertRaisesRegex(ValueError,'Provisional'): ClassificationDataset(p,root,Preprocess([0]))
            with self.assertRaises(ValueError): validate_manifest([row,row],root)
            row['image_path']='../outside.h5'
            with self.assertRaises(ValueError): validate_manifest([row],root)

if __name__ == '__main__': unittest.main()
