// Auditoría de B13/B14. Pegar COMPLETO en Earth Engine Code Editor y pulsar Run UNA VEZ.
// Cuatro puntos train juntos; sin máscaras, entrenamiento ni cambios al dataset.
// Generado con scripts/prepare_roy_topography_probe.py.
var PROBES = [{"id":"point_0021","author_group":"16","crs":"EPSG:32718","transform":[10,0,459730.0,0,-10,8741410.0],"bounds":[459730.0,8739910.0,461230.0,8741410.0],"pixels":[{"id":"r17_c17","row":17,"col":17,"x":459905.0,"y":8741235.0,"values":{"B13":0.7791714072227478,"B14":3090.0}},{"id":"r17_c43","row":17,"col":43,"x":460165.0,"y":8741235.0,"values":{"B13":0.5404195189476013,"B14":3160.0}},{"id":"r17_c75","row":17,"col":75,"x":460485.0,"y":8741235.0,"values":{"B13":0.4961351454257965,"B14":3197.0}},{"id":"r17_c107","row":17,"col":107,"x":460805.0,"y":8741235.0,"values":{"B13":0.39580038189888,"B14":3145.0}},{"id":"r17_c133","row":17,"col":133,"x":461065.0,"y":8741235.0,"values":{"B13":0.6220034956932068,"B14":3051.0}},{"id":"r43_c17","row":43,"col":17,"x":459905.0,"y":8740975.0,"values":{"B13":0.5325937271118164,"B14":3228.0}},{"id":"r43_c43","row":43,"col":43,"x":460165.0,"y":8740975.0,"values":{"B13":0.6884896755218506,"B14":3263.0}},{"id":"r43_c75","row":43,"col":75,"x":460485.0,"y":8740975.0,"values":{"B13":0.7150408029556274,"B14":3063.0}},{"id":"r43_c107","row":43,"col":107,"x":460805.0,"y":8740975.0,"values":{"B13":0.7232511043548584,"B14":2967.0}},{"id":"r43_c133","row":43,"col":133,"x":461065.0,"y":8740975.0,"values":{"B13":0.6188536286354065,"B14":2914.0}},{"id":"r75_c17","row":75,"col":17,"x":459905.0,"y":8740655.0,"values":{"B13":0.22294685244560242,"B14":3404.0}},{"id":"r75_c43","row":75,"col":43,"x":460165.0,"y":8740655.0,"values":{"B13":0.7399604320526123,"B14":3213.0}},{"id":"r75_c75","row":75,"col":75,"x":460485.0,"y":8740655.0,"values":{"B13":0.8304587006568909,"B14":2926.0}},{"id":"r75_c107","row":75,"col":107,"x":460805.0,"y":8740655.0,"values":{"B13":0.6043264269828796,"B14":2731.0}},{"id":"r75_c133","row":75,"col":133,"x":461065.0,"y":8740655.0,"values":{"B13":0.7339788675308228,"B14":2678.0}},{"id":"r107_c17","row":107,"col":17,"x":459905.0,"y":8740335.0,"values":{"B13":0.37528374791145325,"B14":3493.0}},{"id":"r107_c43","row":107,"col":43,"x":460165.0,"y":8740335.0,"values":{"B13":0.7282951474189758,"B14":3237.0}},{"id":"r107_c75","row":107,"col":75,"x":460485.0,"y":8740335.0,"values":{"B13":0.7006763815879822,"B14":3019.0}},{"id":"r107_c107","row":107,"col":107,"x":460805.0,"y":8740335.0,"values":{"B13":0.9116414189338684,"B14":2813.0}},{"id":"r107_c133","row":107,"col":133,"x":461065.0,"y":8740335.0,"values":{"B13":0.8283289074897766,"B14":2574.0}},{"id":"r133_c17","row":133,"col":17,"x":459905.0,"y":8740075.0,"values":{"B13":1.0886194705963135,"B14":3506.0}},{"id":"r133_c43","row":133,"col":43,"x":460165.0,"y":8740075.0,"values":{"B13":0.8309985399246216,"B14":3227.0}},{"id":"r133_c75","row":133,"col":75,"x":460485.0,"y":8740075.0,"values":{"B13":0.5286636352539062,"B14":3031.0}},{"id":"r133_c107","row":133,"col":107,"x":460805.0,"y":8740075.0,"values":{"B13":0.5052757859230042,"B14":2894.0}},{"id":"r133_c133","row":133,"col":133,"x":461065.0,"y":8740075.0,"values":{"B13":0.8003873229026794,"B14":2700.0}}]},{"id":"point_0002","author_group":"1","crs":"EPSG:32718","transform":[10,0,478980.0,0,-10,8739880.0],"bounds":[478980.0,8738380.0,480480.0,8739880.0],"pixels":[{"id":"r17_c17","row":17,"col":17,"x":479155.0,"y":8739705.0,"values":{"B13":0.599579930305481,"B14":2675.0}},{"id":"r17_c43","row":17,"col":43,"x":479415.0,"y":8739705.0,"values":{"B13":0.5153411626815796,"B14":2800.0}},{"id":"r17_c75","row":17,"col":75,"x":479735.0,"y":8739705.0,"values":{"B13":0.6661474704742432,"B14":2928.0}},{"id":"r17_c107","row":17,"col":107,"x":480055.0,"y":8739705.0,"values":{"B13":0.6486165523529053,"B14":2885.0}},{"id":"r17_c133","row":17,"col":133,"x":480315.0,"y":8739705.0,"values":{"B13":0.45717477798461914,"B14":2895.0}},{"id":"r43_c17","row":43,"col":17,"x":479155.0,"y":8739445.0,"values":{"B13":0.40224406123161316,"B14":2642.0}},{"id":"r43_c43","row":43,"col":43,"x":479415.0,"y":8739445.0,"values":{"B13":0.5375869870185852,"B14":2666.0}},{"id":"r43_c75","row":43,"col":75,"x":479735.0,"y":8739445.0,"values":{"B13":0.6306224465370178,"B14":2820.0}},{"id":"r43_c107","row":43,"col":107,"x":480055.0,"y":8739445.0,"values":{"B13":0.5559262633323669,"B14":2753.0}},{"id":"r43_c133","row":43,"col":133,"x":480315.0,"y":8739445.0,"values":{"B13":0.5781542062759399,"B14":2765.0}},{"id":"r75_c17","row":75,"col":17,"x":479155.0,"y":8739125.0,"values":{"B13":0.4368226230144501,"B14":2590.0}},{"id":"r75_c43","row":75,"col":43,"x":479415.0,"y":8739125.0,"values":{"B13":0.5001082420349121,"B14":2562.0}},{"id":"r75_c75","row":75,"col":75,"x":479735.0,"y":8739125.0,"values":{"B13":0.4556979238986969,"B14":2701.0}},{"id":"r75_c107","row":75,"col":107,"x":480055.0,"y":8739125.0,"values":{"B13":0.7112051844596863,"B14":2690.0}},{"id":"r75_c133","row":75,"col":133,"x":480315.0,"y":8739125.0,"values":{"B13":0.5129249095916748,"B14":2537.0}},{"id":"r107_c17","row":107,"col":17,"x":479155.0,"y":8738805.0,"values":{"B13":0.4278903901576996,"B14":2431.0}},{"id":"r107_c43","row":107,"col":43,"x":479415.0,"y":8738805.0,"values":{"B13":0.506243109703064,"B14":2488.0}},{"id":"r107_c75","row":107,"col":75,"x":479735.0,"y":8738805.0,"values":{"B13":0.6249362826347351,"B14":2632.0}},{"id":"r107_c107","row":107,"col":107,"x":480055.0,"y":8738805.0,"values":{"B13":0.8446585536003113,"B14":2612.0}},{"id":"r107_c133","row":107,"col":133,"x":480315.0,"y":8738805.0,"values":{"B13":0.766344428062439,"B14":2515.0}},{"id":"r133_c17","row":133,"col":17,"x":479155.0,"y":8738545.0,"values":{"B13":0.35733479261398315,"B14":2354.0}},{"id":"r133_c43","row":133,"col":43,"x":479415.0,"y":8738545.0,"values":{"B13":0.35756975412368774,"B14":2474.0}},{"id":"r133_c75","row":133,"col":75,"x":479735.0,"y":8738545.0,"values":{"B13":0.7757135629653931,"B14":2622.0}},{"id":"r133_c107","row":133,"col":107,"x":480055.0,"y":8738545.0,"values":{"B13":0.556738018989563,"B14":2506.0}},{"id":"r133_c133","row":133,"col":133,"x":480315.0,"y":8738545.0,"values":{"B13":0.4622039794921875,"B14":2402.0}}]},{"id":"point_0023","author_group":"13","crs":"EPSG:32718","transform":[10,0,461470.0,0,-10,8745340.0],"bounds":[461470.0,8743840.0,462970.0,8745340.0],"pixels":[{"id":"r17_c17","row":17,"col":17,"x":461645.0,"y":8745165.0,"values":{"B13":0.7846971750259399,"B14":2390.0}},{"id":"r17_c43","row":17,"col":43,"x":461905.0,"y":8745165.0,"values":{"B13":0.5848467946052551,"B14":2225.0}},{"id":"r17_c75","row":17,"col":75,"x":462225.0,"y":8745165.0,"values":{"B13":0.6414885520935059,"B14":2028.0}},{"id":"r17_c107","row":17,"col":107,"x":462545.0,"y":8745165.0,"values":{"B13":0.40931767225265503,"B14":1962.0}},{"id":"r17_c133","row":17,"col":133,"x":462805.0,"y":8745165.0,"values":{"B13":0.44153091311454773,"B14":1963.0}},{"id":"r43_c17","row":43,"col":17,"x":461645.0,"y":8744905.0,"values":{"B13":0.5867305397987366,"B14":2170.0}},{"id":"r43_c43","row":43,"col":43,"x":461905.0,"y":8744905.0,"values":{"B13":0.4302401840686798,"B14":2167.0}},{"id":"r43_c75","row":43,"col":75,"x":462225.0,"y":8744905.0,"values":{"B13":0.6137775182723999,"B14":1943.0}},{"id":"r43_c107","row":43,"col":107,"x":462545.0,"y":8744905.0,"values":{"B13":0.585056722164154,"B14":1839.0}},{"id":"r43_c133","row":43,"col":133,"x":462805.0,"y":8744905.0,"values":{"B13":0.2772737741470337,"B14":1837.0}},{"id":"r75_c17","row":75,"col":17,"x":461645.0,"y":8744585.0,"values":{"B13":0.699008584022522,"B14":2041.0}},{"id":"r75_c43","row":75,"col":43,"x":461905.0,"y":8744585.0,"values":{"B13":0.4884612262248993,"B14":1924.0}},{"id":"r75_c75","row":75,"col":75,"x":462225.0,"y":8744585.0,"values":{"B13":0.533599317073822,"B14":1866.0}},{"id":"r75_c107","row":75,"col":107,"x":462545.0,"y":8744585.0,"values":{"B13":0.3649802803993225,"B14":1766.0}},{"id":"r75_c133","row":75,"col":133,"x":462805.0,"y":8744585.0,"values":{"B13":0.2607579529285431,"B14":1803.0}},{"id":"r107_c17","row":107,"col":17,"x":461645.0,"y":8744265.0,"values":{"B13":0.7028841972351074,"B14":1970.0}},{"id":"r107_c43","row":107,"col":43,"x":461905.0,"y":8744265.0,"values":{"B13":0.5548791885375977,"B14":1792.0}},{"id":"r107_c75","row":107,"col":75,"x":462225.0,"y":8744265.0,"values":{"B13":0.13839352130889893,"B14":1697.0}},{"id":"r107_c107","row":107,"col":107,"x":462545.0,"y":8744265.0,"values":{"B13":0.22995319962501526,"B14":1675.0}},{"id":"r107_c133","row":107,"col":133,"x":462805.0,"y":8744265.0,"values":{"B13":0.27367010712623596,"B14":1618.0}},{"id":"r133_c17","row":133,"col":17,"x":461645.0,"y":8744005.0,"values":{"B13":0.9752224683761597,"B14":1811.0}},{"id":"r133_c43","row":133,"col":43,"x":461905.0,"y":8744005.0,"values":{"B13":0.8599978089332581,"B14":1822.0}},{"id":"r133_c75","row":133,"col":75,"x":462225.0,"y":8744005.0,"values":{"B13":0.4795192778110504,"B14":1875.0}},{"id":"r133_c107","row":133,"col":107,"x":462545.0,"y":8744005.0,"values":{"B13":0.3875266909599304,"B14":1808.0}},{"id":"r133_c133","row":133,"col":133,"x":462805.0,"y":8744005.0,"values":{"B13":0.41697126626968384,"B14":1768.0}}]},{"id":"point_0502","author_group":"8","crs":"EPSG:32718","transform":[10,0,449350.0,0,-10,8806280.0],"bounds":[449350.0,8804780.0,450850.0,8806280.0],"pixels":[{"id":"r17_c17","row":17,"col":17,"x":449525.0,"y":8806105.0,"values":{"B13":0.48387598991394043,"B14":2007.0}},{"id":"r17_c43","row":17,"col":43,"x":449785.0,"y":8806105.0,"values":{"B13":0.3849948048591614,"B14":1935.0}},{"id":"r17_c75","row":17,"col":75,"x":450105.0,"y":8806105.0,"values":{"B13":0.3274747431278229,"B14":1825.0}},{"id":"r17_c107","row":17,"col":107,"x":450425.0,"y":8806105.0,"values":{"B13":0.8934645652770996,"B14":1950.0}},{"id":"r17_c133","row":17,"col":133,"x":450685.0,"y":8806105.0,"values":{"B13":0.5223402976989746,"B14":1924.0}},{"id":"r43_c17","row":43,"col":17,"x":449525.0,"y":8805845.0,"values":{"B13":0.45421338081359863,"B14":1918.0}},{"id":"r43_c43","row":43,"col":43,"x":449785.0,"y":8805845.0,"values":{"B13":0.469820499420166,"B14":1847.0}},{"id":"r43_c75","row":43,"col":75,"x":450105.0,"y":8805845.0,"values":{"B13":0.5539452433586121,"B14":1906.0}},{"id":"r43_c107","row":43,"col":107,"x":450425.0,"y":8805845.0,"values":{"B13":0.37725409865379333,"B14":2023.0}},{"id":"r43_c133","row":43,"col":133,"x":450685.0,"y":8805845.0,"values":{"B13":0.34751543402671814,"B14":1959.0}},{"id":"r75_c17","row":75,"col":17,"x":449525.0,"y":8805525.0,"values":{"B13":0.4429110586643219,"B14":1942.0}},{"id":"r75_c43","row":75,"col":43,"x":449785.0,"y":8805525.0,"values":{"B13":0.425154447555542,"B14":1975.0}},{"id":"r75_c75","row":75,"col":75,"x":450105.0,"y":8805525.0,"values":{"B13":0.4394479990005493,"B14":2069.0}},{"id":"r75_c107","row":75,"col":107,"x":450425.0,"y":8805525.0,"values":{"B13":0.3608374297618866,"B14":2117.0}},{"id":"r75_c133","row":75,"col":133,"x":450685.0,"y":8805525.0,"values":{"B13":0.29206761717796326,"B14":1999.0}},{"id":"r107_c17","row":107,"col":17,"x":449525.0,"y":8805205.0,"values":{"B13":0.32535451650619507,"B14":2085.0}},{"id":"r107_c43","row":107,"col":43,"x":449785.0,"y":8805205.0,"values":{"B13":0.21784661710262299,"B14":2124.0}},{"id":"r107_c75","row":107,"col":75,"x":450105.0,"y":8805205.0,"values":{"B13":0.5045828223228455,"B14":2190.0}},{"id":"r107_c107","row":107,"col":107,"x":450425.0,"y":8805205.0,"values":{"B13":0.6462607383728027,"B14":2200.0}},{"id":"r107_c133","row":107,"col":133,"x":450685.0,"y":8805205.0,"values":{"B13":0.48934030532836914,"B14":2086.0}},{"id":"r133_c17","row":133,"col":17,"x":449525.0,"y":8804945.0,"values":{"B13":0.5045828223228455,"B14":2180.0}},{"id":"r133_c43","row":133,"col":43,"x":449785.0,"y":8804945.0,"values":{"B13":0.37725409865379333,"B14":2249.0}},{"id":"r133_c75","row":133,"col":75,"x":450105.0,"y":8804945.0,"values":{"B13":0.37703588604927063,"B14":2282.0}},{"id":"r133_c107","row":133,"col":107,"x":450425.0,"y":8804945.0,"values":{"B13":0.5464827418327332,"B14":2276.0}},{"id":"r133_c133","row":133,"col":133,"x":450685.0,"y":8804945.0,"values":{"B13":0.4982703924179077,"B14":2171.0}}]}];
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
