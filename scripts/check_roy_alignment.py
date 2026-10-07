"""Verify the transposed crop hypothesis discovered by inspect_roy.py."""
from pathlib import Path
import csv,json
import numpy as np
import h5py
from PIL import Image
R=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted((R/'data/roy/l4s/image_ls4pe').glob('*.h5')):
 sid=p.stem; folder=R/'data/roy/landslide_ok'/sid
 with h5py.File(p) as f:x=f['img'][:]
 with h5py.File(R/'data/roy/l4s/mask_ls4pe'/p.name) as f:y=f['mask'][:]
 t=np.stack([np.array(Image.open(folder/f'input/B{b}.tif')) for b in range(1,15)],axis=-1)
 png=folder/'target/manual.png'
 row=dict(sample_id=sid,image_equals_tif_crop_10_138_transposed=bool(np.array_equal(x,t[10:138,10:138].transpose(1,0,2),equal_nan=True)),mask_equals_png_crop_1_129_transposed=None,png_nodata_pixels=None,image_nan=int(np.isnan(x).sum()),image_inf=int(np.isinf(x).sum()),image_nodata_sentinel=int((x < -1e30).sum()))
 if png.exists():
  a=np.array(Image.open(png).convert('RGB'))
  row['mask_equals_png_crop_1_129_transposed']=bool(np.array_equal(y,np.all(a[1:129,1:129]==[255,0,0],axis=-1).T))
  row['png_nodata_pixels']=int(np.all(a==[50,50,255],axis=-1).sum())
 rows.append(row)
with (R/'results/audits/roy_inspection/alignment.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summary=dict(samples=len(rows),image_matches=sum(r['image_equals_tif_crop_10_138_transposed'] for r in rows),mask_matches=sum(r['mask_equals_png_crop_1_129_transposed'] is True for r in rows),mask_mismatches=[r['sample_id'] for r in rows if r['mask_equals_png_crop_1_129_transposed'] is False],samples_with_nan=[r['sample_id'] for r in rows if r['image_nan']],samples_with_inf=[r['sample_id'] for r in rows if r['image_inf']],samples_with_nodata_sentinel=[r['sample_id'] for r in rows if r['image_nodata_sentinel']],png_nodata_samples=[r['sample_id'] for r in rows if r['png_nodata_pixels']])
(R/'results/audits/roy_inspection/alignment_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
