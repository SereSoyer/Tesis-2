# Experimento conjunto Landslide4Sense + Roy: propuesta v1

**Antecedente sustituido:** el usuario eligió conservar 14 canales y reconstruir
Roy. La versión vigente, ya reconstruida, está en
[l4s_roy_14_v1.md](l4s_roy_14_v1.md). Este documento conserva la alternativa de diez
canales y su auditoría histórica; sus decisiones pendientes y conteos no describen
el estado actual.

Fecha: 2026-10-03. El usuario eligió entrenar con ambos datasets en un experimento
nuevo. **Pendientes de respuesta:** selección de diez canales/min–max y reconstrucción
de Roy con exclusión conservadora. El código de preparación está implementado, pero
todavía no se han generado datos reconstruidos ni particiones entrenables.

## Auditoría actual y decisión de admisión propuesta

Se volvieron a leer los 838 HDF5 de imagen y máscara de Roy, los diez TIFF espectrales
seleccionados por punto, sus georreferencias y los PNG disponibles. Las etiquetas y
áreas coinciden con el inventario anterior. Los diez canales coinciden exactamente
con TIFF `[10:138,10:138].T` en los 838 casos y no contienen NaN/Inf ni el centinela
NoData de los TIFF. Los fallos topográficos no justifican descartar muestras cuando
la topografía no es entrada del modelo.

| Motivo para apartar de esta versión | Muestras |
|---|---:|
| Falta PNG original para reconstruir/trazar anotación | 11 |
| Imágenes idénticas con máscaras contradictorias (se apartan ambos miembros) | 4 |
| PNG con NoData que la máscara binaria no conserva, point_0122 | 1 |
| Máscara negativa pero label original `si`, point_0905 | 1 |
| Total | 17 |

IDs: point_0024, point_0039, point_0075, point_0079, point_0122, point_0191,
point_0465, point_0523, point_0524, point_0588, point_0598, point_0605,
point_0631, point_0786, point_0847, point_0905, point_0966.

Quedan **821 candidatos: 792 positivos y 29 negativos**. No se elimina ninguna
muestra por tamaño del deslizamiento ni por dificultad del modelo. Estas exclusiones
son conservadoras y específicas de esta versión; los originales permanecen intactos.
`Final=si` delimita la entrega del autor. `label`/`valid` del Excel no sustituyen la
etiqueta binaria derivada de la máscara final, ni certifican validación experta.
Aceptar candidatos implica adoptar las anotaciones entregadas como referencia con
esa limitación; no inventar una certificación de su calidad.

Informes completos en `results/audits/roy_integration_review_v1/`:
`admission_review.csv`, `candidates.csv`, `excluded.csv`, `overlap_edges.csv`,
`summary.json`, `proposed_split_counts.json` y `alignment_review.png`.
Todos los candidatos conservan `status=pending_review`, `training_eligible=false`.
La marca `candidate` no habilita el entrenamiento.

## Canales y escala propuestos

En ambos datasets usar índices `[0,1,2,3,4,5,6,7,10,11]`: posiciones documentadas
como B1–B8, B11 y B12. No usar las posiciones 8/9 (B9/B10 frente B8A/B9 en Roy)
ni 12/13 (topografía con fuente y orden documental contradictorios).
La correspondencia es documental y por posición TIFF/HDF5, no una nueva medición
de respuesta espectral. L4S declara esas bandas en su
[repositorio oficial](https://github.com/iarai/Landslide4Sense-2022#data-description).

Mantener 128×128, sin resize ni crop aleatorio. Convertir a float32 CHW y aplicar
min–max **por banda y por imagen**, también en validación/test. No ajustar estadísticas
con val/test. Banda constante → cero; no finitos en un canal seleccionado → error.
La topografía almacenada pero no seleccionada no entra al tensor ni a la normalización.
Este escalado reduce diferencias numéricas, pero no convierte L1C/L2A, no corrige
atmósfera ni garantiza equivalencia radiométrica. También pierde escala absoluta por
parche: es una decisión experimental que debe declararse.

Los modelos nuevos reciben diez entradas y requieren entrenamiento nuevo. No cargar
los checkpoints anteriores de catorce canales. Para aislar posteriormente el efecto
de añadir Roy, se necesitaría un control L4S-only con estos mismos diez canales y
preprocesamiento; compararlo directamente con el baseline anterior mezcla factores.

## Geometría propuesta: reconstruir, sin desplazar máscaras por score

La configuración local `data/roy/landslide_ok/landslides.json` contiene
`mask_area=[10,10,140,140]`. El código de IRIS interpreta esos límites como slices
de filas y columnas en la imagen, y calcula el tamaño de máscara por diferencias.
Fuente inspeccionada: commit `787073c83eca9feded9d438d1e7407f1103d9fad`,
[predict_mask](https://github.com/ESA-PhiLab/iris/blob/787073c83eca9feded9d438d1e7407f1103d9fad/iris/segmentation/__init__.py#L325).
Esto respalda el origen 10 del PNG; **no se dispone de la versión modificada exacta
ni del exportador final de Roy**.

La propuesta adopta explícitamente esa convención:

1. PNG 130×130 representa TIFF `[10:140,10:140]`.
2. Conservar máscara PNG roja `[1:129,1:129].T`, idéntica a la HDF5 revisada.
3. Construir imagen desde TIFF `[11:139,11:139].T` en cada canal.
4. Guardar en un directorio derivado nuevo y registrar hashes de TIFF/PNG/originales.

No se estima desplazamiento maximizando IoU ni inspeccionando el resultado del modelo.
La galería ilustra la diferencia, pero no certifica la anotación ni prueba alineación
subpíxel. Esta reconstrucción queda condicionada a la decisión pendiente del usuario.

## Particiones propuestas y controles de fuga

Las particiones originales de L4S (3799/245/800) permanecen asignadas como estaban.
Para Roy se leen huellas completas 1500×1500 m, EPSG:32718, de los TIFF. Se conectan
pares que intersectan o se tocan, sin separar según fecha. Las componentes conexas se
calculan **antes de excluir** muestras; una muestra apartada no rompe una relación.
Se encontraron 418 enlaces; los 821 candidatos ocupan 527 grupos, de hasta 17 muestras.
21 grupos contienen negativos.

Propuesta Roy 70/15/15, seed42. Se ensayan 2000 asignaciones de grupos, eligiendo
por desviación de conteos por clase y partición, sin usar métricas/modelos/área de máscara.
Desempate: primera asignación. Resultado propuesto:

| Split | L4S | Roy positivos | Roy negativos | Total conjunto |
|---|---:|---:|---:|---:|
| Train | 3799 | 552 | 21 | 4372 |
| Val | 245 | 119 | 4 | 368 |
| Test | 800 | 121 | 4 | 925 |

Solo cuatro negativos Roy en val/test hacen muy inestables las métricas de falsas
alarmas y clasificación específicas de Perú. Añadir negativos de L4S no resuelve esa
escasez por dominio. No duplicar negativos entre splits ni fabricar negativos
recortando con las máscaras reales del test.

Las huellas controlan solapamientos observados de Roy, no todos los eventos/escenas.
`Grupo` y `Fechas` se preservan como metadatos del autor sin llamarlos eventos verificados.
La falta de geografía verificable por muestra L4S impide certificar independencia
geográfica cruzada L4S/Roy. El test L4S ya fue inspeccionado en el experimento anterior;
esta ejecución es una extensión, no un test L4S completamente nuevo e intacto.

## Implementación y ejecución

- `src/datasets/roy_review.py` y `scripts/prepare_roy_review.py`: auditoría sin alterar
  originales, selección de candidatos, conflictos y grupos espaciales.
- `src/datasets/integration.py` y `scripts/prepare_l4s_roy.py`: reconstrucción,
  particiones versionadas y configuraciones para cuatro clasificadores y pipeline.
- `src/datasets/protocol.py`: mezcla solo mediante protocolo explícito, conservación
  de procedencia y rechazo de filas pendientes/particiones incorrectas.
- Clasificador sigue sin abrir máscaras; segmentador solo aprende pseudomáscaras.
- Mapas guardan `classification_by_dataset.json`; evaluación espacial guarda
  `by_dataset`; comparación mixta genera además `comparison_by_dataset.csv`.
- Umbral común por rama seleccionado con IoU macro positiva del val combinado;
  segmentador seleccionado con el mismo criterio. Cada muestra pesa igual, no cada
  dataset. No se crean parámetros distintos por fuente de manera oculta.

Auditoría reproducible (usar salida nueva, nunca sobrescribir evidencia):

```bash
.venv/bin/python scripts/prepare_roy_review.py --root . --output results/audits/roy_review_nueva
```

`configs/integration_l4s_roy_v1.json` conserva las decisiones como `null` y
`training_authorized=false` hasta recibir la respuesta. La preparación se detiene
antes de escribir datos si esas decisiones siguen pendientes. Después de aceptarlas:

```bash
.venv/bin/python scripts/prepare_l4s_roy.py --root . --policy configs/integration_l4s_roy_v1.json
```

La preparación está diseñada para escribir solo directorios nuevos. Si falla a mitad,
retener la salida parcial para diagnóstico y emplear una nueva versión después de
resolverlo. No confundir la existencia de carpetas con una preparación terminada;
`preparation.json` se escribe al final.

Verificación local: 28 pruebas pasan, incluyendo conservación del pipeline L4S,
flujo mixto sintético sin máscaras GT de train disponibles, geometría conocida,
agrupación transitiva, rechazo de test en calibración y bloqueo de decisiones pendientes.
La verificación de datos reconstruidos reales sigue pendiente de la decisión de geometría.
