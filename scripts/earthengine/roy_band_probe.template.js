// Auditoría de bandas de Roy: consultas de lectura, sin subir imágenes ni máscaras.
// Archivo generado con scripts/prepare_roy_ee_probe.py. Pegar COMPLETO en Code Editor.
// Empezar con índice 0. Después ejecutar 1, 2 y 3, de uno en uno.
var POINT_INDEX = 0;
var COLLECTION = 'COPERNICUS/S2_HARMONIZED'; // L1C; no usar S2_SR.
var MAX_SCENES = 200; // Si se excede, se detiene: nunca truncar silenciosamente.
var PROBES = __ROY_PROBES__;

var probe = PROBES[POINT_INDEX];
if (!probe) throw new Error('POINT_INDEX debe estar entre 0 y ' + (PROBES.length - 1));
var referenceBands = ['B2','B3','B4','B8','B9','B10'];
var requestedBands = ['B2','B3','B4','B8','B8A','B9','B10'];
var outputBands = requestedBands.map(function(b) { return 'gee_' + b; });
var projection = ee.Projection('EPSG:32718', probe.transform);
var roi = ee.Geometry.Rectangle(probe.bounds, 'EPSG:32718', false);
var points = ee.FeatureCollection(probe.pixels.map(function(p) {
  var props = {sample_id: probe.id, pixel_id: p.id, row: p.row, col: p.col};
  referenceBands.forEach(function(b) { props['ref_' + b] = p.values[b]; });
  return ee.Feature(ee.Geometry.Point([p.x,p.y], 'EPSG:32718'), props);
}));
var copiedProperties = ['sample_id','pixel_id','row','col'].concat(
  referenceBands.map(function(b) { return 'ref_' + b; }));
var exportFields = ['scene_id','product_id','date','baseline','mgrs_tile','cloud_percent',
  'collection','window_start','window_end_exclusive'].concat(copiedProperties,outputBands);

// No filtrar por nubosidad: la escena real no debe desaparecer por un supuesto.
var candidates = ee.ImageCollection(COLLECTION)
  .filterBounds(roi).filterDate(probe.start,probe.end_exclusive).sort('system:time_start');
Map.centerObject(roi,14);
Map.addLayer(roi, {color:'yellow'}, probe.id);
print('Punto y ventana consultados', {id:probe.id, grupo:probe.author_group,
  inicio:probe.start, fin_exclusivo:probe.end_exclusive, pixeles:probe.pixels.length});
print('Solo auditoría: no entrena, no modifica el dataset ni sus particiones.');

// Comprobar cobertura antes de construir la consulta de píxeles.
candidates.size().evaluate(function(total,error) {
  if (error) { print('ERROR de acceso/consulta',error); return; }
  print('Escenas candidatas', total);
  if (!total) { print('SIN ESCENAS. Copiar este resultado para revisar la ventana de fechas.'); return; }
  if (total > MAX_SCENES) { print('Demasiadas escenas: revisar ventana antes de continuar.',total); return; }
  var samples = ee.FeatureCollection(candidates.toList(MAX_SCENES).map(function(item) {
    var img = ee.Image(item);
    // Proyección completa: misma malla y centros que el TIFF, sin normalización.
    // EE usa vecino más próximo por defecto; no aplicar resample bilinear/bicubic.
    return img.select(requestedBands,outputBands).sampleRegions({
      collection:points, properties:copiedProperties, projection:projection, geometries:false
    }).map(function(feature) {
      return feature.set({scene_id:img.get('system:index'), product_id:img.get('PRODUCT_ID'),
        date:img.date().format('YYYY-MM-dd HH:mm:ss'), baseline:img.get('PROCESSING_BASELINE'),
        mgrs_tile:img.get('MGRS_TILE'), cloud_percent:img.get('CLOUDY_PIXEL_PERCENTAGE'),
        collection:COLLECTION, window_start:probe.start, window_end_exclusive:probe.end_exclusive});
    });
  })).flatten();

  // Resumen en el navegador: elegir escena SOLO con las bandas de control.
  samples.evaluate(function(result,failure) {
    if (failure) { print('ERROR al consultar píxeles',failure); return; }
    var rows = result.features.map(function(f) { return f.properties; });
    var ranking = rankRoyScenes(rows,probe.pixels.length);
    print('RESULTADO: candidatas ordenadas por RMSE de B2/B3/B4/B8', ranking.slice(0,5));
    print('Escenas con 25 puntos válidos', ranking.filter(function(r) { return r.complete; }).length);
    print('RMSE=0 indica igualdad en los puntos comparados. Una aproximación no certifica identidad.');
    print('B9_como_B9 y B10_como_B10 contrastan el script R; B9_como_B8A y B10_como_B9 contrastan IRIS.');
    print('Descargar el CSV para revisar los valores y la procedencia, además del resumen.');
    samples.getDownloadURL('csv',exportFields,'roy_band_probe_' + probe.id,function(url,urlError) {
      if (urlError) print('ERROR al crear enlace CSV',urlError);
      else print('DESCARGAR CSV',url);
    });
  });
});

function rankRoyScenes(rows,expected) {
  var groups = {};
  rows.forEach(function(r) { (groups[r.scene_id] || (groups[r.scene_id] = [])).push(r); });
  var comparisons = {B9_como_B9:['B9','B9'], B9_como_B8A:['B9','B8A'],
    B10_como_B10:['B10','B10'], B10_como_B9:['B10','B9']};
  var result = Object.keys(groups).map(function(id) {
    var points = groups[id], sum = 0, exact = 0;
    points.forEach(function(r) {
      ['B2','B3','B4','B8'].forEach(function(b) {
        var d = r['ref_' + b] - r['gee_' + b]; sum += d*d; if (d === 0) exact++;
      });
    });
    var out = {scene_id:id, product_id:points[0].product_id, date:points[0].date,
      baseline:points[0].baseline, n_pixels:points.length, complete:points.length === expected,
      control_rmse:Math.sqrt(sum/(4*points.length)), control_exact_fraction:exact/(4*points.length)};
    Object.keys(comparisons).forEach(function(name) {
      var pair = comparisons[name], squareSum = 0;
      points.forEach(function(r) { var d = r['ref_' + pair[0]] - r['gee_' + pair[1]]; squareSum += d*d; });
      out[name] = Math.sqrt(squareSum/points.length);
    });
    return out;
  });
  return result.sort(function(a,b) {
    if (a.complete !== b.complete) return a.complete ? -1 : 1;
    return a.control_rmse - b.control_rmse;
  });
}
