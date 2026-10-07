"""Prepare a small Code Editor query from train TIFF pixels; no GEE access needed."""
import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

IDS = ['point_0021', 'point_0002', 'point_0023', 'point_0502']
BANDS = ['B2','B3','B4','B8','B9','B10']
POSITIONS = [17,43,75,107,133]


def prepare(root, output):
    with (root/'results/audits/roy14_reconstruction_v1/reconstructed.csv').open() as stream:
        rows = {r['sample_id']:r for r in csv.DictReader(stream)}
    probes = []
    provenance = {}
    for sid in IDS:
        row = rows[sid]
        if row['partition'] != 'train':
            raise ValueError(f'Initial probes must come from train: {sid}')
        start, stop = [date.fromisoformat(t.strip()) for t in row['author_date_window'].split('|')]
        images, hashes, grids = {}, {}, []
        for band in BANDS:
            path = root/'data/roy/landslide_ok'/sid/'input'/f'{band}.tif'
            with Image.open(path) as image:
                images[band] = np.array(image)
                scale = tuple(image.tag_v2[33550]); tie = tuple(image.tag_v2[33922])
                grids.append((scale,tie))
            hashes[band] = hashlib.sha256(path.read_bytes()).hexdigest()
        if len(set(grids)) != 1 or scale[:2] != (10.,10.) or tie[:3] != (0.,0.,0.):
            raise ValueError(f'Unexpected TIFF grid: {sid}')
        if any(x.shape != (150,150) for x in images.values()):
            raise ValueError(f'Unexpected TIFF size: {sid}')
        xmin,ymax = tie[3:5]
        if xmin != float(row['xmin']) or ymax != float(row['ymax']):
            raise ValueError(f'Reviewed georeference differs: {sid}')
        pixels = []
        for r in POSITIONS:
            for c in POSITIONS:
                values = {b:float(images[b][r,c]) for b in BANDS}
                if any(not np.isfinite(v) or v < -1e30 for v in values.values()):
                    raise ValueError(f'Invalid reference pixel: {sid}/{r}/{c}')
                pixels.append(dict(id=f'r{r}_c{c}',row=r,col=c,x=xmin+(c+.5)*10,
                                   y=ymax-(r+.5)*10,values=values))
        probes.append(dict(id=sid,author_group=row['author_group'],start=str(start),
            end_exclusive=str(stop+timedelta(days=1)),crs='EPSG:32718',
            transform=[10,0,xmin,0,-10,ymax],bounds=[xmin,ymax-1500,xmin+1500,ymax],pixels=pixels))
        provenance[sid] = dict(source_hashes=hashes,split='train',author_date_window=row['author_date_window'])
    template = root/'scripts/earthengine/roy_band_probe.template.js'
    source = template.read_text().replace('__ROY_PROBES__',json.dumps(probes,separators=(',',':')))
    output.mkdir(parents=True,exist_ok=False)
    (output/'roy_band_probe.js').write_text(source)
    (output/'reference.json').write_text(json.dumps(dict(probes=probes,provenance=provenance,
        selection='Fixed IDs, four author groups; train only; no model scores or masks used.',
        sampling='Fixed 5x5 positions on original 150x150 TIFF; UTM pixel centers; no transpose/crop.',
        template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
        execution_status='prepared_locally_not_executed_in_earth_engine'),indent=2)+'\n')
    return dict(script=str(output/'roy_band_probe.js'),points=IDS,pixels_per_point=25,
                execution_status='prepared_locally_not_executed_in_earth_engine')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.')
    parser.add_argument('--output',default='results/audits/roy_ee_probe_v1')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    print(json.dumps(prepare(root,root/args.output),indent=2))
