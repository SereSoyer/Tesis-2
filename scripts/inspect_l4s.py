"""Audit an L4S split without assigning it a new experimental role.
Run: .venv/bin/python scripts/inspect_l4s.py --directory TestData
Global statistics are descriptive only, never training normalization statistics.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',default='TestData',choices=['TestData','TrainData','ValidData'])
    args=parser.parse_args();base=ROOT/'data/landslide4sense'/args.directory
    out=ROOT/'results/audits'/('l4s_'+args.directory.lower()+'_inspection');out.mkdir(parents=True,exist_ok=True)
    images={p.stem.removeprefix('image_'):p for p in (base/'img').glob('image_*.h5')}
    masks={p.stem.removeprefix('mask_'):p for p in (base/'mask').glob('mask_*.h5')}
    if not images: raise ValueError('No images found')
    rows=[];errors=[];bands={};duplicates=defaultdict(list);mask_values=Counter();schemas=Counter();attrs=Counter()
    for sid,p in sorted(images.items(),key=lambda kv:int(kv[0])):
        try:
            with h5py.File(p) as f:
                schemas[str(tuple(f.keys()))]+=1;attrs[str(dict(f.attrs))]+=1
                x=f['img'][:]
            if x.ndim!=3:raise ValueError(f'Unexpected image shape {x.shape}')
            digest=hashlib.sha256(str(x.shape).encode()+str(x.dtype).encode()+x.tobytes()).hexdigest()
            duplicates[digest].append(sid)
            for b in range(x.shape[-1]):
                stat=bands.setdefault(b,dict(min=float('inf'),max=float('-inf'),nan=0,inf=0,zero=0,constant_images=0))
                a=x[:,:,b];finite=a[np.isfinite(a)]
                stat['nan']+=int(np.isnan(a).sum());stat['inf']+=int(np.isinf(a).sum());stat['zero']+=int((a==0).sum())
                if finite.size:
                    stat['min']=min(stat['min'],float(finite.min()));stat['max']=max(stat['max'],float(finite.max()))
                    stat['constant_images']+=int(finite.min()==finite.max())
            row=dict(sample_id='l4s_'+args.directory.lower()+'_'+sid,dataset='landslide4sense',source_split=args.directory,image_path=str(p.relative_to(ROOT)),mask_path='',label='',height=x.shape[0],width=x.shape[1],channels=x.shape[2],dtype=str(x.dtype),positive_pixels='',total_pixels=x.shape[0]*x.shape[1],area_ratio='',nan=int(np.isnan(x).sum()),inf=int(np.isinf(x).sum()),mask_shape='',mask_dtype='',mask_binary='',shape_matches='',image_sha256=digest)
            if sid in masks:
                with h5py.File(masks[sid]) as f:y=f['mask'][:]
                v,n=np.unique(y,return_counts=True);mask_values.update({str(k):int(c) for k,c in zip(v,n)})
                binary=bool(np.isin(y,[0,1]).all());matches=y.shape==x.shape[:2]
                row.update(mask_path=str(masks[sid].relative_to(ROOT)),mask_shape=str(y.shape),mask_dtype=str(y.dtype),mask_binary=binary,shape_matches=matches)
                if binary and matches:
                    pos=int((y==1).sum());row.update(label=int(pos>0),positive_pixels=pos,area_ratio=pos/y.size)
            rows.append(row)
        except Exception as exc:errors.append(dict(image=str(p.relative_to(ROOT)),error=repr(exc)))
    path=ROOT/'metadata'/('landslide4sense_'+args.directory.lower()+'.csv');path.parent.mkdir(exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    areas=np.array([r['area_ratio'] for r in rows if r['label']==1])
    summary=dict(source_split=args.directory,images=len(images),masks=len(masks),audited=len(rows),errors=errors,
      images_without_masks=sorted(images.keys()-masks.keys()),masks_without_images=sorted(masks.keys()-images.keys()),
      shapes=dict(Counter(str((r['height'],r['width'],r['channels'])) for r in rows)),dtypes=dict(Counter(r['dtype'] for r in rows)),
      mask_shapes=dict(Counter(r['mask_shape'] for r in rows)),mask_dtypes=dict(Counter(r['mask_dtype'] for r in rows)),mask_values=dict(mask_values),
      labels=dict(Counter(str(r['label']) for r in rows)),nonfinite_samples=[r['sample_id'] for r in rows if r['nan'] or r['inf']],
      invalid_masks=[r['sample_id'] for r in rows if r['mask_binary'] is False or r['shape_matches'] is False],
      duplicate_image_groups=[v for v in duplicates.values() if len(v)>1],bands=bands,image_keys=dict(schemas),image_file_attributes=dict(attrs),
      positive_area_ratio=({'min':float(areas.min()),'max':float(areas.max()),'mean':float(areas.mean()),**{f'p{p}':float(np.percentile(areas,p)) for p in [25,33,50,66,75]}} if areas.size else {}),
      note='Descriptive audit only. Shape agreement does not prove geospatial alignment. Test statistics must not fit training normalization.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
