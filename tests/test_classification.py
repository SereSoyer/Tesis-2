import csv
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from src.datasets.classification import ClassificationDataset, make_loader
from src.preprocessing.multispectral import Preprocess

class DataTests(unittest.TestCase):
    def test_reader_ignores_missing_masks_and_preserves_channels_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            x=np.stack([np.full((32,32),i) for i in range(4)],axis=-1)
            with h5py.File(root/'image.h5','w') as f: f['img']=x
            with (root/'data.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['sample_id','dataset','image_path','mask_path','label']);w.writeheader()
                for i in range(4):w.writerow(dict(sample_id=str(i),dataset='fixture',image_path='image.h5',mask_path='does_not_exist.h5',label=i%2))
            ds=ClassificationDataset(root/'data.csv',root,Preprocess([3,1]))
            self.assertEqual(set(ds[0]),{'image','label','sample_id'})
            self.assertEqual(ds[0]['image'].shape,(2,32,32))
            self.assertTrue(torch.all(ds[0]['image'][0]==3))
            self.assertEqual(ds[1]['label'].item(),1)
            ids=lambda: [b['sample_id'] for b in make_loader(ds,2,True,73)]
            self.assertEqual(ids(),ids())

    def test_nonfinite_and_constant_bands(self):
        x=np.ones((4,4,2));x[0,0,0]=np.nan
        with self.assertRaises(ValueError):Preprocess([0,1])(x)
        out=Preprocess([0,1],'zero','per_image_minmax')(x)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.all(out==0))
        x[:,:,0]=np.nan
        with self.assertRaises(ValueError):Preprocess([0,1],'zero','per_image_minmax')(x)

if __name__=='__main__':unittest.main()
