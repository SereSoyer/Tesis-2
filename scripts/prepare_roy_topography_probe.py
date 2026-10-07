"""Prepare a read-only Earth Engine comparison of Roy's B13/B14 on train pixels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_roy_ee_probe import IDS, POSITIONS


def local_slope_diagnostic(slope, elevation):
    """Diagnostic only: slope from the exported 10 m DEM need not match native slope."""
    valid = np.isfinite(slope) & np.isfinite(elevation) & (slope > -1e30) & (elevation > -1e30)
    neighbors_valid = np.logical_and.reduce([
        valid[r:r + valid.shape[0] - 2, c:c + valid.shape[1] - 2]
        for r in range(3) for c in range(3)
    ])
    dx = (elevation[1:-1, 2:] - elevation[1:-1, :-2]) / 20.
    dy = (elevation[2:, 1:-1] - elevation[:-2, 1:-1]) / 20.
    calculated = np.arctan(np.hypot(dx, dy))[neighbors_valid]
    reference = slope[1:-1, 1:-1][neighbors_valid]
    if not len(reference):
        raise ValueError('No valid 3x3 neighborhoods')
    return {
        'n_valid_neighborhoods': int(len(reference)),
        'method': 'atan(hypot(central_dx, central_dy)), spacing 10 m, radians',
        'b13_as_radians_rmse_rad': float(np.sqrt(np.mean((reference - calculated) ** 2))),
        'b13_as_degrees_rmse_rad': float(np.sqrt(np.mean((np.deg2rad(reference) - calculated) ** 2))),
        'pearson_r': (float(np.corrcoef(reference, calculated)[0, 1])
                      if np.std(reference) > 0 and np.std(calculated) > 0 else None),
        'interpretation': 'Diagnostic, not certification of source, datum or original slope algorithm.',
    }


def prepare(root, output):
    with (root / 'results/audits/roy14_reconstruction_v1/reconstructed.csv').open() as stream:
        rows = {r['sample_id']: r for r in csv.DictReader(stream)}
    probes, provenance, diagnostics = [], {}, {}
    for sid in IDS:
        row = rows[sid]
        if row['partition'] != 'train':
            raise ValueError(f'Probe must belong to reviewed train: {sid}')
        arrays, hashes, grids = {}, {}, []
        for band in ['B13', 'B14']:
            path = root / 'data/roy/landslide_ok' / sid / 'input' / f'{band}.tif'
            with Image.open(path) as image:
                arrays[band] = np.asarray(image, dtype=np.float64)
                scale = tuple(image.tag_v2[33550])
                tie = tuple(image.tag_v2[33922])
                keys = tuple(image.tag_v2[34735])
                inline_keys = {keys[i]: keys[i + 3] for i in range(4, len(keys), 4)
                               if keys[i + 1] == 0 and keys[i + 2] == 1}
                if inline_keys.get(3072) != 32718 or inline_keys.get(1025) != 1:
                    raise ValueError(f'Expected EPSG:32718 PixelIsArea: {path}')
                grids.append((scale, tie))
            hashes[band] = hashlib.sha256(path.read_bytes()).hexdigest()
        if len(set(grids)) != 1 or scale[:2] != (10., 10.) or tie[:3] != (0., 0., 0.):
            raise ValueError(f'Unexpected or unequal grids: {sid}')
        if any(a.shape != (150, 150) for a in arrays.values()):
            raise ValueError(f'Unexpected TIFF shape: {sid}')
        xmin, ymax = tie[3:5]
        if xmin != float(row['xmin']) or ymax != float(row['ymax']):
            raise ValueError(f'Reviewed georeference differs: {sid}')
        pixels = []
        for r in POSITIONS:
            for c in POSITIONS:
                values = {b: float(a[r, c]) for b, a in arrays.items()}
                if any(not np.isfinite(v) or v <= -1e30 for v in values.values()):
                    raise ValueError(f'Invalid reference pixel: {sid}/{r}/{c}')
                pixels.append(dict(id=f'r{r}_c{c}', row=r, col=c,
                                   x=xmin + (c + .5) * 10, y=ymax - (r + .5) * 10,
                                   values=values))
        probes.append(dict(id=sid, author_group=row['author_group'], crs='EPSG:32718',
                           transform=[10, 0, xmin, 0, -10, ymax],
                           bounds=[xmin, ymax - 1500, xmin + 1500, ymax], pixels=pixels))
        provenance[sid] = dict(source_hashes=hashes, split='train')
        diagnostics[sid] = local_slope_diagnostic(arrays['B13'], arrays['B14'])
    template = root / 'scripts/earthengine/roy_topography_probe.template.js'
    source = template.read_text().replace('__ROY_PROBES__', json.dumps(probes, separators=(',', ':')))
    reference = dict(
        probes=probes, provenance=provenance, local_slope_diagnostics=diagnostics,
        selection='Same four train IDs and fixed 5x5 positions as spectral probe; no masks/model scores.',
        sampling='Original TIFF row/col, UTM pixel centers; no transpose/crop.',
        candidate_assets=['USGS/SRTMGL1_003', 'NASA/NASADEM_HGT/001', 'JAXA/ALOS/AW3D30/V4_1'],
        terrain_policy='Native grid 4-neighbor EE slope; nearest sampling to TIFF grid; no fitted offsets.',
        expected_rows=300,
        template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
        execution_status='prepared_locally_not_executed_in_earth_engine',
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / 'roy_topography_probe.js').write_text(source)
    (output / 'reference.json').write_text(json.dumps(reference, indent=2, allow_nan=False) + '\n')
    (output / 'README.md').write_text(
        '# Sondeo topográfico de Roy\n\n'
        'Pegar `roy_topography_probe.js` completo en Earth Engine Code Editor y ejecutar una vez.\n'
        'Los cuatro puntos se consultan juntos. Descargar el enlace **DESCARGAR CSV** de la consola.\n'
        'Salida: `roy_topography_probe.csv`, hasta 300 filas: 4 puntos × 25 píxeles × 3 fuentes.\n\n'
        'Estado: preparado localmente; aún no ejecutado en Earth Engine.\n'
        'No modifica TIFF, HDF5, admisiones, particiones ni configuraciones de entrenamiento.\n'
        'No obtiene ni transmite máscaras o credenciales.\n\n'
        'Consultar el [protocolo y sus límites](../../../docs/datasets/roy_topography_probe.md).\n'
    )
    return dict(script=str(output / 'roy_topography_probe.js'), points=IDS,
                expected_rows=300, execution_status=reference['execution_status'],
                local_slope_diagnostics=diagnostics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--output', default='results/audits/roy_topography_probe_v1')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    print(json.dumps(prepare(root, root / args.output), indent=2, allow_nan=False))
