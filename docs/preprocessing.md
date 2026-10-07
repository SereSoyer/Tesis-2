# Preprocesamiento O1 — versión 1

**Actualización 03/10/2026:** se reconstruyeron y verificaron 806 muestras Roy con
14 canales, sin imputación, y se preparó min–max por canal/imagen para ambas fuentes.
La identidad física de algunas bandas sigue pendiente. Detalle y protocolo vigente
del nuevo experimento: [l4s_roy_14_v1.md](datasets/l4s_roy_14_v1.md).
Las estadísticas y comprobaciones O1 siguientes conservan su carácter histórico.

## Flujo comprobado y límites

La operación común estructural es HDF5 `img` HWC 128×128×14 → selección posicional 0..13 → float32 → CHW 14×128×128. `Preprocess` conserva orientación y posiciones; no cambia máscaras, ni aplica resize, crop o aumento. En L4S se conserva el escalado distribuido. La entrada de 14 canales de Roy solo está comprobada a nivel técnico: no establece equivalencia física con L4S.

**Estado del flujo integrado: PARCIAL.** El usuario autorizó el experimento mixto y la reconstrucción de 14 canales. Se preparó min–max común; falta resolver el mapeo físico para habilitar el entrenamiento. Los archivos originales permanecen intactos.

## Estadísticas y selección

`results/audits/o1/channel_statistics.json` incluye por índice de canal media, desviación poblacional, mínimo, máximo, conteo finito y porcentaje no finito. Los momentos se combinan con una fórmula incremental estable (`src/preprocessing/statistics.py`) ponderada por píxeles finitos. La desviación es sqrt(M2/N), ddof=0. Valores no finitos se cuentan y no participan en momentos.

Poblaciones explícitas:

- `l4s_original_train_descriptive`: solo TrainData original; nunca ValidData o TestData.
- `roy_unsplit_descriptive_not_for_fit`: inventario Roy todavía sin partición. **No se puede usar para ajustar normalización.** Sus momentos sirven para diagnosticar heterogeneidad, no para elegir parámetros mediante un futuro test Roy.

Los canales se nombran por índice. Comparar números no demuestra que representen la misma variable. No se equiparan automáticamente reflectancias, pendiente o elevación.

Tras confirmar bandas, unidades y train definitivo, una opción es estandarizar por canal z=(x−μ_train)/σ_train. Las medias y desviaciones deben recalcularse únicamente en el train integrado aceptado, persistiendo IDs y hashes del ajuste. No se reutilizan los momentos Roy sin partición. σ=0 exige una decisión explícita sobre el canal; no se divide por cero. No se selecciona esta opción todavía porque estandarizar no corrige identidades o unidades incompatibles ni restaura información perdida por un escalado previo.

Min–max por imagen puede borrar diferencias absolutas entre parches, especialmente topografía. Es la política preparada para `l4s_roy_14_v1`, aplicada igual a ambas fuentes. El baseline L4S anterior conserva su escalado distribuido y requiere un control con el nuevo preprocesamiento para aislar el efecto de añadir Roy.

## NoData y valores no finitos

Política vigente: `nan_policy=error`. En la versión reconstruida de 14 canales se apartan los 16 recortes con NoData en cualquier canal, sin imputación; uno también carece de PNG y no se cuenta dos veces. Las incidencias y originales se conservan. Recuperar fuentes válidas permitiría una nueva versión documentada, sin modificar esta copia ni sus particiones en silencio.

El modo opcional `zero` existe en el módulo heredado, pero no está activado en este protocolo. Cero puede ser un valor válido; no se considera NoData sin evidencia. Las máscaras RGB originales tienen una clase NoData que las HDF5 binarias no preservan. No se usa esa zona como fondo confiable para evaluación espacial hasta resolver la transformación y validez.

## Transformaciones por partición

Train, validación y test usan la misma transformación determinista; solo train se baraja. No hay augmentation actual: evita introducir cambios sin justificación y simplifica la comprobación de correspondencia espacial. Si posteriormente se añade, será solo train, registrada por configuración. La evaluación deberá restaurar mapas a coordenadas originales; nunca reescalar la máscara con interpolación bilineal.

La ausencia de resize evita eliminar píxeles pequeños por remuestreo, pero no garantiza que el modelo los detecte. La conversión a float32 puede producir redondeo; la conservación se define contra el original convertido a float32, no contra igualdad exacta con float64.

## Comprobaciones y reproducción

Desde la raíz:

```bash
.venv/bin/python -m scripts.audit_o1
.venv/bin/python -m scripts.validate_o1
.venv/bin/python -m scripts.make_o1_evidence
```

La auditoría guarda momentos e inventario. Las pruebas sintéticas verifican reordenamiento sin cambios espaciales, políticas no finitas y momentos contra NumPy. La prueba real carga un positivo y un negativo finitos de L4S train y de Roy, por separado, y los pasa por los cuatro modelos en CPU sin gradientes ni optimizador. Comprueba forma, dtype y salidas finitas. Es una prueba estructural con `normalization=none`; no valida la semántica, aceptación de etiquetas Roy, alineación geográfica o rendimiento. No se accede a test para esa prueba.

Resultados efectivos: `results/audits/o1/validation.json` y `tests.txt`. Los informes históricos `l4s_preprocessing_{train,val,test}.json` documentan la comprobación exhaustiva de las transformaciones L4S. No se reejecuta esa misma prueba innecesariamente.
# Integración Roy de 14 canales

La versión reconstruida, las exclusiones y las configuraciones están en
[l4s_roy_14_v1.md](datasets/l4s_roy_14_v1.md). La alternativa de diez canales quedó
sustituida por decisión del usuario. No modifica el experimento L4S ya ejecutado.
