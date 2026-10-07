"""Verify all available split images retain their values after float32 conversion.
No training, normalization fitting or mask reads occur.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import h5py
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datasets.configured import configured_dataset, configured_loader

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/preprocessing_l4s.json')
    parser.add_argument('--split', choices=['train','val','test'], default='test')
    args=parser.parse_args()
    config=json.loads((ROOT/args.config).read_text())
    ds=configured_dataset(config,ROOT,args.split)
    for i,row in enumerate(ds.rows):
        sample=ds[i]
        with h5py.File(ROOT/row['image_path']) as f:raw=f[config['image_key']][:]
        if config['layout']=='CHW':raw=np.moveaxis(raw,0,-1)
        expected=raw[...,config['channels']].astype(np.float32).transpose(2,0,1)
        if not np.array_equal(sample['image'].numpy(),expected):
            raise ValueError(f'Unexpected value change: {row["sample_id"]}')
        if sample['image'].dtype != torch.float32 or not torch.isfinite(sample['image']).all():
            raise ValueError(f'Invalid output: {row["sample_id"]}')
        if set(sample) != {'image','label','sample_id'}:raise ValueError('Unexpected batch fields')
    batch=next(iter(configured_loader(config,ROOT,args.split)))
    result={'config':config,'split':args.split,'samples_verified':len(ds),
            'manifest_sha256':hashlib.sha256((ROOT/config['manifests'][args.split]).read_bytes()).hexdigest(),
            'batch_shape':list(batch['image'].shape),'dtype':str(batch['image'].dtype),
            'values_preserved_after_float32':True,'mask_files_read':False,'training_performed':False}
    out=ROOT/'results/audits'/f'l4s_preprocessing_{args.split}.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
