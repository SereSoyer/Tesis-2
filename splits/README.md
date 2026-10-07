# Particiones de Landslide4Sense

Los CSV l4s_train.csv (3799), l4s_val.csv (245) y l4s_test.csv (800) conservan exactamente los manifiestos auditados de TrainData, ValidData y TestData. No se mezclaron ni reasignaron muestras, por lo que no hay semilla de división. La semilla del entrenamiento se configura por separado.

Reproducir desde la raíz: `.venv/bin/python scripts/prepare_l4s_splits.py`.

El script valida cantidades, etiquetas, campos de auditoría, existencia de archivos y ausencia de IDs, rutas o hashes de imágenes repetidos. No sobrescribe particiones diferentes. Los hashes provienen de la auditoría previa, no de una nueva lectura de imágenes; conservar los datos sin cambios. La ausencia de duplicados exactos no establece independencia geográfica. Informe: results/audits/l4s_splits.json.

No se ejecutó entrenamiento ni una prueba de optimización con estos CSV. configs/train_l4s.json ya apunta a train/val; sus dos épocas y dispositivo CPU son valores de plantilla. Ajustar la configuración al servidor y al protocolo experimental antes de ejecutarla. Test no participa en entrenamiento ni selección de checkpoints.
