"""Validate returned Roy B13/B14 probes and compare fixed topographic hypotheses."""
import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from analyze_roy_ee_probe import digest, metrics

ASSETS = {
    'SRTM_V3': 'USGS/SRTMGL1_003',
    'NASADEM_HGT': 'NASA/NASADEM_HGT/001',
    'ALOS_AW3D30_V4_1': 'JAXA/ALOS/AW3D30/V4_1',
}
METADATA = ['asset_id', 'source_ids', 'native_crs', 'native_transform',
            'native_scale_m', 'slope_grid', 'sampling']
NUMERIC = ['ref_B13', 'ref_B14', 'gee_elevation_m', 'gee_slope_deg', 'gee_slope_rad']


def sample_geoid(probes, path):
    """Independent EGM96 hypothesis; never estimate an offset from Roy elevations."""
    from osgeo import gdal, osr

    gdal.UseExceptions()
    grid = gdal.Open(str(path))
    inverse = gdal.InvGeoTransform(grid.GetGeoTransform())
    if inverse is None:
        raise ValueError('Noninvertible geoid grid')
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    if not target.IsSameGeogCS(osr.SpatialReference(wkt=grid.GetProjection())):
        raise ValueError('Expected geographic WGS84 geoid grid')
    band = grid.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    samples = []
    for probe in probes.values():
        source = osr.SpatialReference()
        source.SetFromUserInput(probe['crs'])
        source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(source, target)
        for pixel in probe['pixels']:
            lon, lat, _ = transform.TransformPoint(pixel['x'], pixel['y'])
            c, r = gdal.ApplyGeoTransform(inverse, lon, lat)
            c, r = c - .5, r - .5
            ci, ri = math.floor(c), math.floor(r)
            if ci < 0 or ri < 0 or ci + 1 >= grid.RasterXSize or ri + 1 >= grid.RasterYSize:
                raise ValueError('Probe outside geoid interpolation coverage')
            a = band.ReadAsArray(ci, ri, 2, 2).astype(np.float64)
            if not np.isfinite(a).all() or (nodata is not None and (a == nodata).any()):
                raise ValueError('Invalid geoid cells')
            dx, dy = c - ci, r - ri
            n = (a[0, 0] * (1 - dx) * (1 - dy) + a[0, 1] * dx * (1 - dy)
                 + a[1, 0] * (1 - dx) * dy + a[1, 1] * dx * dy)
            samples.append(dict(sample_id=probe['id'], pixel_id=pixel['id'],
                                longitude=lon, latitude=lat, egm96_N_m=float(n)))
    return dict(grid=str(path), grid_sha256=digest(path), gdal_version=gdal.VersionInfo(),
                grid_transform=list(grid.GetGeoTransform()),
                interpolation='Bilinear at pixel centers; no fitted or selected offsets.',
                hypothesis='h_ellipsoid = H_orthometric + N_EGM96; not a proven Roy datum.',
                source='Local PROJ egm96_15.gtx, 15 arc-minute geoid undulation grid.', samples=samples)


def comparisons(rows, geoid_values=None):
    if not rows:
        return None
    values = lambda key: np.array([float(row[key]) for row in rows], dtype=np.float64)
    result = {
        'b14_vs_elevation_m': metrics(values('ref_B14'), values('gee_elevation_m')),
        'b13_as_radians_vs_slope_rad': metrics(values('ref_B13'), values('gee_slope_rad')),
        'b13_as_degrees_vs_slope_rad': metrics(np.deg2rad(values('ref_B13')), values('gee_slope_rad')),
    }
    if geoid_values is not None:
        shifts = np.array([geoid_values[(r['sample_id'], r['pixel_id'])] for r in rows])
        result['b14_vs_elevation_plus_egm96_m'] = metrics(
            values('ref_B14'), values('gee_elevation_m') + shifts)
        result['egm96_N_m'] = dict(min=float(shifts.min()), max=float(shifts.max()), mean=float(shifts.mean()))
    return result


def analyze(root, csv_path, reference_path, geoid_path=None):
    reference = json.loads(reference_path.read_text())
    probes = {p['id']: p for p in reference['probes']}
    if set(reference['candidate_assets']) != set(ASSETS.values()):
        raise ValueError('Prepared candidates differ from this analysis')
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    required = {'candidate', 'sample_id', 'pixel_id', 'row', 'col', *METADATA, *NUMERIC}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('Empty CSV or unexpected schema')
    checked = {}
    # Check every prepared TIFF, including groups that might be absent from the CSV.
    for sid, probe in probes.items():
        if reference['provenance'][sid]['split'] != 'train':
            raise ValueError('Only prepared train probes may be analyzed')
        checked[sid] = {}
        for band in ['B13', 'B14']:
            path = root / 'data/roy/landslide_ok' / sid / 'input' / f'{band}.tif'
            if digest(path) != reference['provenance'][sid]['source_hashes'][band]:
                raise ValueError(f'TIFF changed after probe preparation: {path}')
            with Image.open(path) as image:
                checked[sid][band] = np.array(image)
            for pixel in probe['pixels']:
                if pixel['values'][band] != float(checked[sid][band][pixel['row'], pixel['col']]):
                    raise ValueError(f'Prepared reference differs from TIFF: {sid}/{band}')
    seen, groups = set(), defaultdict(list)
    for row in rows:
        sid, candidate = row['sample_id'], row['candidate']
        if sid not in probes or candidate not in ASSETS:
            raise ValueError('Unknown point or candidate')
        if (row['asset_id'] != ASSETS[candidate] or row['native_crs'] != 'EPSG:4326'
                or row['slope_grid'] != 'native_4_neighbors'
                or row['sampling'] != 'nearest_to_original_UTM_TIFF_grid'
                or not row['source_ids'] or not row['native_transform']
                or not np.isfinite(float(row['native_scale_m'])) or float(row['native_scale_m']) <= 0):
            raise ValueError('Returned provenance differs from prepared query')
        if candidate != 'ALOS_AW3D30_V4_1' and row['source_ids'] != row['asset_id']:
            raise ValueError('Unexpected image source identifier')
        pixels = {p['id']: p for p in probes[sid]['pixels']}
        if row['pixel_id'] not in pixels:
            raise ValueError('Unknown pixel')
        pixel = pixels[row['pixel_id']]
        if (int(row['row']), int(row['col'])) != (pixel['row'], pixel['col']):
            raise ValueError('Pixel coordinates differ from prepared query')
        key = sid, candidate, row['pixel_id']
        if key in seen:
            raise ValueError('Duplicate point/candidate/pixel')
        seen.add(key)
        for name in NUMERIC:
            if not np.isfinite(float(row[name])) or float(row[name]) <= -1e30:
                raise ValueError('Invalid numeric value in returned CSV')
        for band in ['B13', 'B14']:
            if float(row['ref_' + band]) != pixel['values'][band]:
                raise ValueError('Returned reference differs from TIFF')
        if not (0 <= float(row['gee_slope_deg']) <= 90):
            raise ValueError('Invalid GEE slope range')
        if not np.isclose(np.deg2rad(float(row['gee_slope_deg'])), float(row['gee_slope_rad']),
                          rtol=1e-6, atol=1e-8):
            raise ValueError('Returned degrees and radians are inconsistent')
        groups[(sid, candidate)].append(row)
    geoid = sample_geoid(probes, geoid_path) if geoid_path else None
    geoid_values = ({(p['sample_id'], p['pixel_id']): p['egm96_N_m'] for p in geoid['samples']}
                    if geoid else None)
    results = []
    for sid, probe in probes.items():
        expected = {p['id'] for p in probe['pixels']}
        for candidate in ASSETS:
            group = groups[(sid, candidate)]
            if any(len({r[k] for r in group}) > 1 for k in METADATA):
                raise ValueError('Inconsistent source metadata within point/candidate')
            actual = {r['pixel_id'] for r in group}
            results.append(dict(sample_id=sid, candidate=candidate, returned_pixels=len(group),
                                expected_pixels=len(expected), complete=expected == actual,
                                missing_pixel_ids=sorted(expected - actual),
                                metadata={k: group[0][k] for k in METADATA} if group else None,
                                comparisons=comparisons(group, geoid_values)))
    pooled = {source: comparisons([r for r in rows if r['candidate'] == source], geoid_values)
              for source in ASSETS}
    return dict(status='measured_evidence_not_automatic_admission', input_csv_sha256=digest(csv_path),
                prepared_reference_sha256=digest(reference_path), rows=len(rows),
                unique_positions=len({(r['sample_id'], r['pixel_id']) for r in rows}),
                complete=all(r['complete'] for r in results),
                source_hashes_and_reference_values_verified=True,
                slope_hypothesis_comparison_units='radians_for_both',
                groups=results, pooled=pooled, independent_geoid_diagnostic=geoid,
                band_mapping_changed=False, training_enabled=False,
                limitations=[
                    'Four train patches, 100 positions: not an audit of every Roy topographic pixel.',
                    'Correlation or lowest RMSE does not identify an exact DEM product.',
                    'Fixed EGM96 shift tests a datum hypothesis; it does not prove the original datum.',
                    'Native-grid GEE slope need not equal slope computed on another grid before export.',
                    'Original private COG, interpolation chain and vertical datum remain unverified.',
                ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--root', default='.')
    parser.add_argument('--reference', default='results/audits/roy_topography_probe_v1/reference.json')
    parser.add_argument('--geoid', help='Optional EGM96 GTX grid; requires GDAL in this Python interpreter')
    parser.add_argument('--output', required=True, help='New report directory; existing outputs are preserved')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    csv_path, output = Path(args.csv).resolve(), (root / args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = analyze(root, csv_path, root / args.reference, Path(args.geoid).resolve() if args.geoid else None)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(csv_path, output / 'input.csv')
    (output / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    table = []
    for group in result['groups']:
        row = {k: group[k] for k in ['sample_id', 'candidate', 'returned_pixels', 'complete']}
        for name, metric in (group['comparisons'] or {}).items():
            if name == 'egm96_N_m':
                continue
            for key in ['rmse', 'bias_candidate_minus_reference', 'pearson_r', 'exact_pixels']:
                row[f'{name}_{key}'] = metric[key]
        table.append(row)
    columns = list(dict.fromkeys(key for row in table for key in row))
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps({k: result[k] for k in ['rows', 'unique_positions', 'complete', 'pooled']}, indent=2))
