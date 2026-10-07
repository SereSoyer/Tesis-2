"""Render original O1 figures from current audit artifacts, never fit preprocessing."""
import csv
import json
import textwrap
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.datasets.manifests import read_manifest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/o1_evidence'
SOURCE='Fuente: elaboración propia con archivos locales y auditoría O1. 29/09/2026.'


def save(fig,name,caption=SOURCE):
    fig.text(.02,.015,caption,fontsize=8,color='#475569',wrap=True)
    fig.savefig(OUT/f'{name}.png',dpi=300,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)


def table(name,title,headers,rows,width=12):
    rows=[[textwrap.fill(str(value),38) for value in row] for row in rows]
    lengths=[max(len(line) for value in [headers[i]]+[row[i] for row in rows]
                 for line in value.splitlines())+4 for i in range(len(headers))]
    widths=np.asarray(lengths)/sum(lengths)
    lines=[max(value.count('\n')+1 for value in row) for row in rows]
    fig,ax=plt.subplots(figsize=(width,max(3,1.6+sum(lines)*.5)))
    ax.axis('off');ax.set_title(title,fontsize=16,pad=20)
    t=ax.table(cellText=rows,colLabels=headers,colWidths=widths,loc='center',cellLoc='left')
    t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,1.8)
    for (r,c),cell in t.get_celld().items():
        if r==0:cell.set_facecolor('#dbeafe');cell.set_text_props(weight='bold')
        else:cell.set_height(cell.get_height()*lines[r-1])
    save(fig,name)


def flow(name,title,labels,subtitle):
    fig,ax=plt.subplots(figsize=(14,3.6));ax.axis('off');ax.set_title(title,fontsize=17)
    xs=np.linspace(.10,.90,len(labels))
    for i,(x,label) in enumerate(zip(xs,labels)):
        ax.text(x,.55,label,ha='center',va='center',transform=ax.transAxes,fontsize=11,
                bbox=dict(boxstyle='round,pad=.7',facecolor='#dbeafe',edgecolor='#2563eb'))
        if i:
            ax.annotate('',xy=(x-.09,.55),xytext=(xs[i-1]+.09,.55),xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='->',color='#475569',lw=2))
    ax.text(.5,.15,subtitle,transform=ax.transAxes,ha='center',fontsize=11)
    save(fig,name)


def dataset_jpgs():
    """Export actual HDF5 samples as labeled scientific figures, not synthetic imagery."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(ROOT/'metadata/integrated_provisional.csv')
    samples = []
    for dataset, title, name in [('landslide4sense', 'Landslide4Sense (L4S)', '12_l4s_example'),
                                  ('roy_l4s_pe', 'L4S-PE (Roy)', '13_l4s_pe_example')]:
        row = next(r for r in rows if r['dataset'] == dataset and r['label'] == '1'
                   and int(r['nonfinite_values']) == 0
                   and (dataset != 'landslide4sense' or r['split'] == 'train'))
        with h5py.File(ROOT/row['image_path'], 'r') as stream:
            array = stream['img'][:, :, 0]
        samples.append((title, row, array))
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(array, cmap='gray', interpolation='nearest')
        ax.set_title(f"{title}\n{row['original_sample_id']} · canal posicional 0", fontsize=13)
        ax.axis('off')
        fig.text(.05, .035, 'Fuente: HDF5 local. 128 × 128 píxeles originales.\n'
                 'Grises con escalado visual; no es una composición RGB.', fontsize=9)
        fig.savefig(OUT/f'{name}.jpg', dpi=300, bbox_inches='tight', pil_kwargs={'quality':95})
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (title, row, array) in zip(axes, samples):
        ax.imshow(array, cmap='gray', interpolation='nearest')
        ax.set_title(f"{title}\n{row['original_sample_id']}", fontsize=13)
        ax.axis('off')
    fig.text(.05, .035, 'Fuente: archivos HDF5 locales. Canal posicional 0, escalado visual independiente.\n'
             'No es RGB ni una demostración de equivalencia semántica entre canales.', fontsize=10)
    fig.savefig(OUT/'14_both_datasets.jpg', dpi=300, bbox_inches='tight', pil_kwargs={'quality':95})
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dataset_jpgs()
    summary=json.loads((ROOT/'results/audits/o1/summary.json').read_text())
    validation=json.loads((ROOT/'results/audits/o1/validation.json').read_text())
    rows=read_manifest(ROOT/'metadata/integrated_provisional.csv')
    inv=summary['inventory']
    table('01_inventory','Inventario verificado; integración experimental pendiente',
          ['Colección','Imágenes / máscaras','Positivas','Negativas','Con NaN/Inf'],
          [['L4S' if r['dataset']=='landslide4sense' else 'L4S-PE',f"{r['samples']} / {r['masks']}",r['positive'],r['negative'],r['nonfinite_samples']] for r in inv])
    fig,ax=plt.subplots(figsize=(8,5));x=np.arange(2)
    for offset,key,color,label in [(-.18,'negative','#64748b','Ausencia (0)'),(.18,'positive','#2563eb','Presencia (1)')]:
        bars=ax.bar(x+offset,[r[key] for r in inv],.36,label=label,color=color)
        ax.bar_label(bars,padding=3)
    ax.set_xticks(x,['L4S','L4S-PE']);ax.set_ylabel('Imágenes');ax.set_ylim(0,3500)
    ax.set_title('Distribución de etiquetas derivadas de máscaras');ax.legend()
    save(fig,'02_classes')
    table('03_splits','Asignaciones conservadas; NO son splits integrados definitivos',
          ['Procedencia','Train','Validación','Test','Sin asignar'],
          [['L4S',3799,245,800,0],['L4S-PE',0,0,0,838]])
    fig,axes=plt.subplots(4,2,figsize=(9,12));fig.subplots_adjust(hspace=.5,bottom=.08,top=.94)
    fig.suptitle('Ejemplos reales: canal posicional 0 y máscara procesada',fontsize=15)
    for i,(dataset,label) in enumerate([(d,l) for d in ['landslide4sense','roy_l4s_pe'] for l in ['0','1']]):
        row=next(r for r in rows if r['dataset']==dataset and r['label']==label and int(r['nonfinite_values'])==0
                 and (dataset!='landslide4sense' or r['split']=='train'))
        with h5py.File(ROOT/row['image_path'],'r') as f: image=f['img'][:,:,0]
        with h5py.File(ROOT/row['mask_path'],'r') as f: mask=f['mask'][:]
        axes[i,0].imshow(image,cmap='gray',vmin=float(image.min()),vmax=float(image.max()))
        axes[i,1].imshow(mask,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        axes[i,0].set_title(f"{row['original_sample_id']} | y={label}",fontsize=10)
        axes[i,1].set_title(f"Máscara 0/1 | área={float(row['area_ratio'])*100:.2f}%",fontsize=10)
        for ax in axes[i]: ax.axis('off')
    save(fig,'04_examples',SOURCE+'\nGrises con min–max solo para visualización; no es RGB ni normalización de entrenamiento.\nRoy: anotación y alineación pendientes de aceptación. No son predicciones.')
    flow('05_labels','Generación de etiqueta binaria',
         ['Máscara original\n0/1','¿Existe algún\npíxel = 1?','Sí: y=1\nNo: y=0','CSV: etiqueta\ny area_ratio'],
         'La pérdida del clasificador recibe y; la máscara queda fuera del entrenamiento.')
    flow('06_preprocessing','Transformación estructural implementada',
         ['HDF5 img\n128 × 128 × 14','Canales 0…13\npor posición','float32\nHWC → CHW','Tensor\n14 × 128 × 128'],
         'Sin resize, crop o augmentation. Normalización integrada pendiente; NaN/Inf → error.')
    stats=json.loads((ROOT/'results/audits/o1/channel_statistics.json').read_text())['scopes']
    statrows=[]
    fig,axes=plt.subplots(2,1,figsize=(12,8));fig.subplots_adjust(hspace=.45,bottom=.13)
    for scope,values in stats.items():
        label='L4S train original' if scope.startswith('l4s') else 'Roy sin split (solo descriptivo)'
        for ax,key in zip(axes,['mean','std']):
            ax.plot(range(14),[r[key] for r in values],marker='o',label=label)
            ax.set_yscale('symlog',linthresh=1);ax.set_xticks(range(14));ax.set_xlabel('Índice posicional, no equivalencia semántica')
        for r in values: statrows.append(dict(scope=scope,**r))
    for ax,title in zip(axes,['Media por canal — escala symlog','Desviación por canal — escala symlog']):
        ax.set_title(title);ax.legend();ax.grid(alpha=.2)
    save(fig,'07_channel_statistics',SOURCE+'\nEstadísticas descriptivas. No hay ajuste de normalización; no se usan validación/test L4S.')
    with (OUT/'channel_statistics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=statrows[0]);w.writeheader();w.writerows(statrows)
    table('08_issues','Incidencias: decisiones conservadoras y trazables',
          ['Hallazgo','Alcance','Tratamiento actual'],
          [['Canales / radiometría Roy','Integración','Pendiente: exportador y unidades'],
           ['16 imágenes con NaN','Entrada del modelo','Error explícito; no imputar'],
           ['2 pares idénticos, máscaras distintas','Fuga / evaluación','Revisar; no fusionar máscaras'],
           ['11 PNG originales ausentes','Trazabilidad','Conservar HDF5; pedir procedencia'],
           ['30 negativos Roy','Etiquetado','Validación de anotación pendiente'],
           ['Posible desfase de recorte','Evaluación espacial','No desplazar sin evidencia']])
    table('09_status','Estado de los resultados esperados O1',
          ['Resultado','Estado','Evidencia / límite'],
          [['O1.R1','PARCIAL','Inventario y splits provisionales; admisión pendiente'],
           ['O1.R2','PARCIAL','Flujo estructural probado; armonización pendiente'],
           ['O1.R3','COMPLETO (marco)','Protocolo definido; atribución corresponde a O3']])
    table('10_validation','Verificación técnica; no mide rendimiento predictivo',
          ['Comprobación','Resultado'],
          [['Pruebas automatizadas',f"{validation['tests_passed']}/{validation['tests_run']}"],
           ['Forward real: 2 fuentes × 4 modelos',len(validation['real_checks'])],
           ['Muestras no finitas rechazadas explícitamente',len(validation['nonfinite_samples_explicitly_rejected'])],
           ['Entrenamiento definitivo','No ejecutado']])
    positive=[float(r['area_ratio'])*100 for r in rows if r['dataset']=='landslide4sense' and r['split']=='train' and r['label']=='1']
    fig,ax=plt.subplots(figsize=(9,5));ax.hist(positive,bins=40,color='#2563eb')
    ax.set_title('Área anotada en positivos de L4S train original');ax.set_xlabel('Área positiva / área del parche (%)');ax.set_ylabel('Imágenes')
    save(fig,'11_area_train')
    (OUT/'README.md').write_text('''# Evidencias originales de O1

Generadas por `python -m scripts.make_o1_evidence`, a partir de la auditoría local actual. Cada figura está disponible en PNG a 300 dpi y SVG. No son resultados predictivos ni material copiado del repositorio de Roy.

- 01_inventory: cantidades actuales; no inventario definitivo aceptado.
- 02_classes: etiquetas derivadas, aceptación Roy pendiente.
- 03_splits: particiones L4S originales y Roy sin asignación.
- 04_examples: canal posicional 0 en grises y máscara real; no RGB, ni prueba de alineación geográfica. Min–max solo de visualización.
- 05_labels: derivación de etiqueta; máscaras fuera de la pérdida.
- 06_preprocessing: transformación estructural comprobada.
- 07_channel_statistics y channel_statistics.csv: estadística descriptiva; no parámetros normalizadores.
- 08_issues: incidencias y decisiones pendientes.
- 09_status: matriz O1; R3 completo como marco, no como implementación de mapas.
- 10_validation: pruebas de código y entradas reales; no comparación de precisión.
- 11_area_train: distribución sobre positivos de train original; no fija umbrales small/medium/large.

- 12_l4s_example.jpg: ejemplo positivo L4S, `l4s_traindata_1`, procedente de `data/landslide4sense/TrainData/img/image_1.h5`.
- 13_l4s_pe_example.jpg: ejemplo positivo L4S-PE, `point_0001`, procedente de `data/roy/l4s/image_ls4pe/point_0001.h5`.
- 14_both_datasets.jpg: ambos ejemplos lado a lado. Los JPG muestran el canal posicional 0 en grises con escalado visual independiente, sin asumir equivalencia semántica ni composición RGB. Exportación a 300 dpi; la resolución original sigue siendo 128×128.

Las cantidades definitivas integradas no se grafican porque aún no existen. Se conservan originales y no se corrigieron máscaras.
''')
    print(f'Generated {len(list(OUT.glob("*.png")))} PNG and SVG pairs in {OUT.relative_to(ROOT)}')

if __name__=='__main__':main()
