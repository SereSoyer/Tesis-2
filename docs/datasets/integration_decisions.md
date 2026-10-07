# Admisión e incidencias para integración — 29/09/2026

**Registro histórico O1. Actualización 03/10/2026:** se adoptó la reconstrucción
de 14 canales, obteniendo 806 muestras y apartando 32 con motivos documentados.
Se prepararon particiones por solapamiento y normalización común. Las decisiones
vigentes están en [l4s_roy_14_v1.md](l4s_roy_14_v1.md); la identidad física de algunas
bandas continúa pendiente. Las cifras y tratamientos siguientes describen O1,
antes de esa reconstrucción, y no deben usarse como estado actual de admisión.

Este registro expresa decisiones sustentadas y pendientes. No cambia originales, máscaras o etiquetas. El detalle por muestra se conserva en `metadata/integrated_provisional.csv`; los registros `accepted.csv`, `excluded.csv` y `pending_review.csv` bajo `results/audits/o1/` se refieren a admisión al experimento integrado, no a disponibilidad física.

## Compatibilidad

| Aspecto | Evidencia | Estado |
|---|---|---|
| Forma/tipo | Ambos HDF5 128×128×14 float64, comprobados | Compatible estructuralmente |
| Máscara procesada | 128×128 binaria, etiquetas verificadas | Convención numérica compatible; semántica pendiente |
| Bandas L4S | Fuente oficial declara B1..B12, pendiente, DEM | Documentado; trazabilidad de copia local conservar |
| Bandas Roy | R, IRIS y Python discrepan B9/B10 frente B8A/B9; topografía invertida entre flujos | Bloquea integración |
| Escalado/unidades | L4S distribuido y Roy procesado presentan rangos distintos; exportador final desconocido | Bloquea normalización común |
| Geografía | Roy tiene coordenadas; L4S local sin evento/escena verificable en atributos | No independencia certificada |
| Alineación | Igual forma no demuestra registro; recortes/transposición Roy documentados | Evaluación Roy bloqueada |
| Anotación | Final/label/valid del autor no son etiquetas equivalentes | Admisión Roy pendiente |

Fuente primaria L4S: https://github.com/iarai/Landslide4Sense-2022 (consultada 29/09/2026). Declara 12 bandas Sentinel-2 y 2 capas topográficas ALOS, malla aproximada de 10 m. No atribuir resolución nativa 10 m a todas las bandas. La documentación de la competición no explica por sí sola las máscaras de validación/test presentes localmente; debe conservarse y confirmar la procedencia de su distribución. Las discrepancias Roy se sustentan en `roy_l4s_pe.md` y en los archivos del autor ya auditados.

## Registro de tratamientos

| Incidencia | Efecto | Tratamiento implementado | Información necesaria para resolver |
|---|---|---|---|
| 16 imágenes Roy con NaN (índices 12/13) | Entrenamiento y evaluación | Registro por canal; lector falla explícitamente; ninguna exclusión/imputación | Exportador y fuente válida; determinar origen y política NoData |
| point_0523/0524: imagen igual, máscaras con 21/28 píxeles; point_0588/0598: 775/779 | Fuga; referencia espacial contradictoria | Hashes actuales de imagen/máscara, agrupación de incidencia; no elegir una máscara | Versión de anotación válida o revisión del autor |
| Coordenadas repetidas, incluidas point_0010/0423 y point_0014/0631 | Fuga geográfica potencial | Grupos de coordenadas exactas; no tratarlos como inventario completo de solapamientos | Huellas, escenas y definición de grupos/eventos |
| 827 PNG y HDF5 coinciden mediante recorte [1:129,1:129].T | Evaluación espacial | Transformación comprobada nuevamente; ninguna modificación | Anclaje del PNG sobre TIFF y exportador final |
| Hipótesis de desplazamiento de un píxel | Evaluación; clasificación no localiza píxeles | Se mantiene como hipótesis condicionada; no se corrige | Semántica de mask_area y recortes de imagen/máscara |
| 11 PNG ausentes, HDF5 positivos presentes | Trazabilidad de clasificación y evaluación | Mantener HDF5 e incidencia; no reconstruir anotaciones | Recuperar PNG o documentar fuente/versión de HDF5 |
| 30 negativos; 29 label/valid=no; point_0905 si/si, obs=agri | Entrenamiento y métricas | Conservar etiqueta derivada, marcar aceptación pendiente | Confirmar anotación completa y significado temporal de campos |
| 7 píxeles NoData del PNG point_0122 | Evaluación espacial | Registrar sin convertir fondo a válido | Validez espacial trasladada mediante transformación confirmada |
| Grupo/Fechas sin diccionario | Particiones | Conservar texto original; Roy sin split | Confirmar evento/escena/lote y fecha de adquisición |

Las incidencias pueden coexistir en una muestra; sus conteos no son sumables como número de imágenes distintas. Ninguna corrección científica resulta inequívoca solo por observar estos archivos. Por eso se automatizan detección, validación y trazabilidad, no cambios de contenido.

## Estado de los registros

- 5.682 filas con estructura y etiqueta comprobadas.
- 5.682 pendientes para el experimento integrado por condiciones globales de compatibilidad/geografía; esto no invalida las comprobaciones locales de L4S.
- 0 aceptadas definitivamente para integración y 0 excluidas automáticamente.
- 3.799/245/800 asignaciones originales L4S conservadas en manifiestos provisionales; 838 Roy sin asignar.
- No se encontraron duplicados exactos cruzados L4S/Roy en los arrays actuales; los dos pares duplicados están dentro de Roy. Esto no descarta proximidad ni solapamiento.

Una futura decisión de admisión debe registrar fecha, responsable, evidencia, identificador, estado y motivo; después regenerar manifiestos, hashes y estadísticas de train. No editar manualmente `training_eligible=true` sin esas verificaciones ni cambiar `dataset` para eludir el lector.
