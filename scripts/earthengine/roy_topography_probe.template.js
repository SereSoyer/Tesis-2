// Auditoría de B13/B14. Pegar COMPLETO en Earth Engine Code Editor y pulsar Run UNA VEZ.
// Cuatro puntos train juntos; sin máscaras, entrenamiento ni cambios al dataset.
// Generado con scripts/prepare_roy_topography_probe.py.
var PROBES = __ROY_PROBES__;
var candidates = [
  {name:'SRTM_V3', asset:'USGS/SRTMGL1_003', band:'elevation'},
  {name:'NASADEM_HGT', asset:'NASA/NASADEM_HGT/001', band:'elevation'},
  {name:'ALOS_AW3D30_V4_1', asset:'JAXA/ALOS/AW3D30/V4_1', band:'DSM'}
];
var copiedProperties = ['sample_id','pixel_id','row','col','ref_B13','ref_B14'];
var outputBands = ['gee_elevation_m','gee_slope_deg','gee_slope_rad'];
var exportFields = ['candidate','asset_id','source_ids','native_crs','native_transform',
  'native_scale_m','slope_grid','sampling'].concat(copiedProperties,outputBands);
var tables = [];

PROBES.forEach(function(probe) {
  var roi = ee.Geometry.Rectangle(probe.bounds,probe.crs,false);
  var targetProjection = ee.Projection(probe.crs,probe.transform);
  var points = ee.FeatureCollection(probe.pixels.map(function(p) {
    return ee.Feature(ee.Geometry.Point([p.x,p.y],probe.crs),{
      sample_id:probe.id, pixel_id:p.id, row:p.row, col:p.col,
      ref_B13:p.values.B13, ref_B14:p.values.B14
    });
  }));
  Map.addLayer(roi,{color:'yellow'},probe.id);
  candidates.forEach(function(candidate) {
    var dem, nativeProjection, sourceIds;
    if (candidate.name === 'ALOS_AW3D30_V4_1') {
      // AW3D30 procede de PRISM; NO es el DEM auxiliar de productos PALSAR RTC.
      // La proyección por defecto de un mosaico debe fijarse antes de calcular pendiente.
      var tiles = ee.ImageCollection(candidate.asset).filterBounds(roi.buffer(2000))
        .sort('system:index');
      nativeProjection = ee.Image(tiles.first()).select(candidate.band).projection();
      sourceIds = tiles.aggregate_array('system:index').join('|');
      dem = tiles.map(function(img) {
        return img.select(candidate.band).updateMask(img.select('MSK').neq(1));
      }).mosaic().setDefaultProjection(nativeProjection);
    } else {
      dem = ee.Image(candidate.asset).select(candidate.band);
      nativeProjection = dem.projection();
      sourceIds = candidate.asset;
    }
    // Congelar cálculo en malla NATIVA; sampleRegions no debe recalcular pendiente a 10 m.
    dem = dem.toFloat().reproject(nativeProjection);
    var slope = ee.Terrain.slope(dem).reproject(nativeProjection);
    var image = dem.rename('gee_elevation_m')
      .addBands(slope.rename('gee_slope_deg'))
      .addBands(slope.multiply(Math.PI / 180).rename('gee_slope_rad'));
    var table = image.sampleRegions({collection:points, properties:copiedProperties,
      projection:targetProjection, geometries:false}).map(function(feature) {
      return feature.set({candidate:candidate.name, asset_id:candidate.asset,
        source_ids:sourceIds, native_crs:nativeProjection.crs(),
        native_transform:nativeProjection.transform(), native_scale_m:nativeProjection.nominalScale(),
        slope_grid:'native_4_neighbors', sampling:'nearest_to_original_UTM_TIFF_grid'});
    });
    tables.push(table);
  });
});

Map.centerObject(ee.Geometry.Rectangle(PROBES[0].bounds,PROBES[0].crs,false),12);
var samples = ee.FeatureCollection(tables).flatten();
print('Cuatro puntos juntos: hasta 300 filas, 100 posiciones únicas y tres fuentes.');
print('Unidades: elevación en metros; ambas hipótesis de B13 se comparan en RADIANES.');
print('AW3D30 es un contraste diagnóstico: no equivale al DEM auxiliar de PALSAR RTC.');
print('No se ajustan desplazamientos, escalas ni sesgos para forzar coincidencias.');

samples.evaluate(function(result,error) {
  if (error) { print('ERROR al consultar: copiar este mensaje para revisar.',error); return; }
  var rows = result.features.map(function(feature) { return feature.properties; });
  var summary = summarizeTopography(rows,PROBES,candidates);
  print('Filas recuperadas / esperadas',rows.length,300);
  print('Comparación por punto y fuente',summary);
  if (summary.some(function(item) { return !item.complete; })) {
    print('COBERTURA INCOMPLETA: revisar los grupos con menos de 25 píxeles. No asumir coincidencia.');
  }
  print('RMSE cero verifica solo los valores consultados; un error no identifica por sí solo la fuente.');
  print('Remuestreo, datum vertical y cálculo de pendiente pueden explicar diferencias.');
  samples.getDownloadURL('csv',exportFields,'roy_topography_probe',function(url,urlError) {
    if (urlError) print('ERROR al crear enlace CSV',urlError);
    else print('DESCARGAR CSV',url);
  });
});

function summarizeTopography(rows,probes,sources) {
  var summary = [];
  probes.forEach(function(probe) {
    sources.forEach(function(source) {
      var subset = rows.filter(function(row) {
        return row.sample_id === probe.id && row.candidate === source.name;
      });
      var n = subset.length, elevationSquare = 0, elevationBias = 0;
      var radiansSquare = 0, degreesSquare = 0, exact = 0;
      subset.forEach(function(row) {
        var dh = row.ref_B14 - row.gee_elevation_m;
        var dr = row.ref_B13 - row.gee_slope_rad;
        var dd = row.ref_B13 * Math.PI / 180 - row.gee_slope_rad;
        elevationSquare += dh * dh;
        elevationBias += dh;
        radiansSquare += dr * dr;
        degreesSquare += dd * dd;
        if (dh === 0) exact++;
      });
      summary.push({sample_id:probe.id,candidate:source.name,n_pixels:n,
        complete:n === probe.pixels.length,
        b14_elevation_rmse_m:n ? Math.sqrt(elevationSquare / n) : null,
        b14_elevation_bias_m:n ? elevationBias / n : null,
        b14_exact_count:exact,
        b13_as_radians_rmse_rad:n ? Math.sqrt(radiansSquare / n) : null,
        b13_as_degrees_rmse_rad:n ? Math.sqrt(degreesSquare / n) : null});
    });
  });
  return summary;
}
