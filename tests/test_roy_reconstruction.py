import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin

from src.datasets.protocol import allowed_sources, check_rows, SOURCES
from src.datasets.roy_reconstruction import validate_policy, read_tiff_stack, read_annotation
from src.pipeline.maps import generate
from src.segmentation.train import validate


class RoyReconstructionTests(unittest.TestCase):
    def policy(self):
        return json.loads(Path('configs/reconstruct_roy14_v1.json').read_text())

    def tags(self, x=400000):
        tags = TiffImagePlugin.ImageFileDirectory_v2()
        tags[33550] = (10.,10.,0.)
        tags[33922] = (0.,0.,0.,float(x),9000000.,0.)
        tags[34735] = (1,1,0,2,3072,0,1,32718,1025,0,1,1)
        tags[42113] = '-3.39999999999999996e+38'
        return tags

    def make_folder(self, root):
        (root/'input').mkdir(); (root/'target').mkdir()
        rr,cc = np.indices((150,150))
        plane = (150*rr+cc).astype(np.float32)
        for i in range(1,15):
            Image.fromarray(plane+i).save(root/f'input/B{i}.tif',tiffinfo=self.tags())
        rgb = np.zeros((130,130,3),dtype=np.uint8); rgb[...,1] = 255
        rgb[3,4] = (255,0,0)
        Image.fromarray(rgb).save(root/'target/manual.png')
        return plane

    def test_all_fourteen_channels_and_known_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); plane = self.make_folder(folder)
            x,bad,stats,bounds = read_tiff_stack(folder,self.policy())
            y,error = read_annotation(folder,self.policy())
            self.assertIsNone(error)
            self.assertEqual(x.shape,(128,128,14))
            self.assertEqual(sum(bad),0)
            self.assertEqual(bounds,(400000,8998500,401500,9000000))
            self.assertEqual(y[3,2],1)
            for i in range(14): self.assertEqual(x[3,2,i],plane[13,14]+i+1)

    def test_topography_nodata_is_detected_even_if_spectral_channels_are_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); plane = self.make_folder(folder)
            plane[12,12] = np.float32(-3.4e38)
            Image.fromarray(plane).save(folder/'input/B13.tif',tiffinfo=self.tags())
            x,bad,_,_ = read_tiff_stack(folder,self.policy())
            self.assertEqual(sum(bad[:12]),0)
            self.assertEqual(bad[12],1)
            self.assertLess(x[1,1,12],-1e30)  # never silently imputed
            Image.fromarray(plane).save(folder/'input/B14.tif',tiffinfo=self.tags(400010))
            with self.assertRaisesRegex(ValueError,'different georeferences'):
                read_tiff_stack(folder,self.policy())

    def test_policy_and_pipeline_cannot_enable_pending_data(self):
        cfg = self.policy()
        validate_policy(cfg)
        for field,value in [('image_crop',[10,138,10,138]),('training_enabled',True),('transpose_both',False)]:
            bad = copy.deepcopy(cfg); bad[field]=value
            with self.assertRaisesRegex(ValueError,'policy mismatch'): validate_policy(bad)
        disabled = dict(dataset='l4s_roy_14_v1', training_enabled=False, disabled_reason='band mapping pending')
        with self.assertRaisesRegex(ValueError,'pending'): allowed_sources(disabled)
        with self.assertRaisesRegex(ValueError,'pending'): validate(disabled)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'pending'):
                generate(disabled,'resnet18','train',tmp,Path(tmp)/'output')
            self.assertFalse((Path(tmp)/'output').exists())
        enabled = dict(disabled,training_enabled=True,source_datasets=SOURCES,
            preprocessing=dict(channels=list(range(14)),normalization='per_image_minmax',nan_policy='error'),
            band_mapping_status='pending_conflicting_sources')
        with self.assertRaisesRegex(ValueError,'mapping decision'): allowed_sources(enabled)
        enabled['band_mapping_status']='adopted_thesis_table1'
        self.assertEqual(allowed_sources(enabled),SOURCES)
        pending_row=dict(dataset='roy_l4s_pe',source_split='RoyOriginal',partition='train',
            status='pending_review',training_eligible='false',dataset_version='l4s_roy_14_v1')
        with self.assertRaisesRegex(ValueError,'accepted'): check_rows(enabled,[pending_row],'train')
