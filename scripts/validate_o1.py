"""Run synthetic suite and inference-only real-data checks, with a saved report."""
import hashlib
import importlib.metadata
import io
import json
import platform
import unittest
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.datasets.classification import ClassificationDataset, make_loader
from src.datasets.manifests import read_manifest, validate_manifest, validate_partition_integrity
from src.models.classifiers import build_model, MODELS
from src.preprocessing.multispectral import Preprocess


def main():
    root=Path(__file__).resolve().parents[1]
    output=root/'results/audits/o1';output.mkdir(parents=True,exist_ok=True)
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(root/'tests'))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (output/'tests.txt').write_text(stream.getvalue())
    report=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                command='.venv/bin/python -m scripts.validate_o1',
                tests_run=result.testsRun, tests_passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
                failures=[(str(t),e) for t,e in result.failures], errors=[(str(t),e) for t,e in result.errors],
                skipped=[(str(t),e) for t,e in result.skipped],
                device='cpu',seed=42,real_training_performed=False,real_checks=[],
                versions={p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','h5py','timm','transformers','scikit-learn']},
                platform=platform.platform(),preprocessing='indices 0..13, float32 CHW, normalization none, nonfinite error; shape test only')
    if not result.wasSuccessful():
        (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        raise SystemExit('Synthetic suite failed; see tests.txt')
    torch.set_num_threads(2);torch.manual_seed(42)
    # Verify generated inventories and preserve the frozen L4S membership.
    inventory=read_manifest(root/'metadata/integrated_provisional.csv')
    validate_manifest(inventory,root)
    audit=json.loads((output/'summary.json').read_text())
    for source,digest in audit['source_manifest_sha256'].items():
        if hashlib.sha256((root/source).read_bytes()).hexdigest()!=digest:
            raise ValueError('Source CSV changed since audit; rerun audit_o1')
    partitions={s:read_manifest(root/f'splits/provisional/integrated_{s}.csv') for s in ['train','val','test','unassigned']}
    assigned={s:partitions[s] for s in ['train','val','test']}
    validate_partition_integrity(assigned)
    collected=[r for part in partitions.values() for r in part]
    if len(collected)!=len(inventory) or {r['sample_id'] for r in collected}!={r['sample_id'] for r in inventory}:
        raise ValueError('Provisional partitions do not cover inventory exactly once')
    for split,part in partitions.items():
        if any(r['split']!=split or r['status']!='pending_review' or r['training_eligible']!='false' for r in part):
            raise ValueError('Invalid provisional status or split')
        if split!='unassigned':
            original=read_manifest(root/f'splits/l4s_{split}.csv')
            if {r['sample_id'] for r in part}!={r['sample_id'] for r in original}:
                raise ValueError('Original L4S partition changed')
    for field in ['image_path','mask_path']:
        expected={root/r[field] for r in inventory}
        parents={p.parent for p in expected}
        actual={p for parent in parents for p in parent.glob('*.h5')}
        if actual!=expected:
            raise ValueError(f'Uninventoried or missing processed files in {field}')
    report['manifest_checks']=dict(rows=len(inventory),coverage_exact=True,
                                   original_l4s_membership_preserved=True,processed_file_inventory_exact=True,
                                   provisional_training_disabled=True)
    for manifest in ['splits/l4s_train.csv','metadata/roy.csv']:
        ds=ClassificationDataset(root/manifest,root,Preprocess(list(range(14))))
        # Two finite, opposing classes for a structural check only. Never uses test.
        indices=[next(i for i,r in enumerate(ds.rows) if r['label']==label and
                      int(r.get('nonfinite_values') or 0)==0 and int(r.get('nan') or 0)==0 and int(r.get('inf') or 0)==0)
                 for label in ['0','1']]
        batch=next(iter(make_loader(torch.utils.data.Subset(ds,indices),2,False,42)))
        assert tuple(batch['image'].shape)==(2,14,128,128)
        assert batch['image'].dtype==torch.float32 and torch.isfinite(batch['image']).all()
        for name in MODELS:
            torch.manual_seed(42)
            model=build_model(name,14,128,False).eval()
            with torch.no_grad():
                logits=model(batch['image'])
            if logits.shape != batch['label'].shape or not torch.isfinite(logits).all():
                raise ValueError(f'Invalid output {manifest}/{name}')
            report['real_checks'].append(dict(manifest=manifest,manifest_sha256=hashlib.sha256((root/manifest).read_bytes()).hexdigest(),
                model=name,sample_ids=batch['sample_id'],labels=batch['label'].tolist(),
                input_shape=list(batch['image'].shape),output_shape=list(logits.shape),finite=True,
                parameters=sum(p.numel() for p in model.parameters())))
            print(f'Forward OK: {manifest} / {name}',flush=True)
            del model
    # The known anomalous Roy samples must fail closed, not silently become zeros.
    ds=ClassificationDataset(root/'metadata/roy.csv',root,Preprocess(list(range(14))))
    rejected=[]
    for i,row in enumerate(ds.rows):
        if int(row['nonfinite_values']):
            try: ds[i]
            except ValueError as e:
                if 'Nonfinite' not in str(e): raise
                rejected.append(row['sample_id'])
            else: raise AssertionError('Nonfinite sample accepted silently')
    report['nonfinite_samples_explicitly_rejected']=rejected
    report['passed']=True
    (output/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(stream.getvalue())
    print(f'Real forward checks: {len(report["real_checks"])}; nonfinite rejected: {len(rejected)}')

if __name__=='__main__':main()
