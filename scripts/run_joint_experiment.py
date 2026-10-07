"""Run the admitted joint experiment; checkpoints resume and completed stages are retained."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.admit_l4s_roy14 import CONFIG_DIR, MODELS
from scripts.check_joint_experiment import check
from src.pipeline.artifacts import sha256
from src.training.train import save_checkpoint, write_json


def training_state(output, cfg, segmentation=False, pseudo=None):
    """Validate completed/partial checkpoints without trusting a stale summary.json."""
    if not output.exists():
        return 'new'
    if not (output / 'last.pt').is_file():
        raise ValueError(f'Output exists without last.pt: {output}. Preserve/rename it before retrying.')
    state = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
    if state['config'] != cfg:
        raise ValueError(f'Checkpoint configuration changed: {output}')
    if segmentation:
        expected = dict(train=cfg['manifest_sha256']['train'], val=cfg['manifest_sha256']['val'],
                        pseudo_manifest=sha256(pseudo / 'manifest.csv'), pseudo_metadata=sha256(pseudo / 'metadata.json'))
        if state.get('kind') != 'pseudo_segmentation' or state['provenance'] != expected:
            raise ValueError('Segmentation checkpoint provenance changed')
        epochs = cfg['segmentation']['epochs']
    else:
        if state['manifest_sha256'] != cfg['manifest_sha256']:
            raise ValueError('Classifier checkpoint population changed')
        epochs = cfg['epochs']
    if state['epoch'] > epochs:
        raise ValueError('Checkpoint exceeds configured training budget')
    if state['epoch'] < epochs:
        return 'resume'
    # Recover an interruption after last.pt but before best.pt was written at the final epoch.
    if state['best_epoch'] == state['epoch']:
        best_path = output / 'best.pt'
        best = torch.load(best_path, map_location='cpu', weights_only=False) if best_path.is_file() else None
        if best is None or best['epoch'] != state['best_epoch']:
            save_checkpoint(best_path, state)
    best = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
    if best['epoch'] != state['best_epoch'] or best['config'] != cfg:
        raise ValueError('Best checkpoint does not match completed run')
    return 'complete'


def run(root, config_dir, phase):
    root = Path(root).resolve()
    cfg_path = Path(config_dir) / 'pipeline.json'
    cfg = json.loads((root / cfg_path).read_text())
    check(root, config_dir, full=False)
    directory = root / cfg['run_directory']
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'runner.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError('This joint experiment is already running')
    with lock:
        plan = dict(configs={name: sha256(root / config_dir / name) for name in
                             [f'train_{m}.json' for m in MODELS] + ['pipeline.json']},
                    code={str(p.relative_to(root)): sha256(p) for p in sorted((root / 'src').rglob('*.py'))})
        plan_path = directory / 'plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError('Experiment configuration changed; create a new experiment instead')
        if not plan_path.exists():
            write_json(plan_path, plan)
            try:
                commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                                 text=True, stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError):
                commit = None
            write_json(directory / 'experiment.json', dict(experiment_id=cfg['experiment_id'],
                timestamp=datetime.now(timezone.utc).isoformat(), git_commit=commit,
                runner_sha256=sha256(Path(__file__)), data_admission=cfg['data_admission']))
        markers = directory / 'completed'
        markers.mkdir(exist_ok=True)
        if phase == 'test':
            marker = markers / 'comparison_val.json'
            comparison = directory / 'comparison_val/comparison.json'
            if not marker.is_file() or not comparison.is_file():
                raise ValueError('Complete all four training/validation branches before final test')
            if json.loads(marker.read_text())['final_sha256'] != sha256(comparison):
                raise ValueError('Validation comparison changed')
        with (directory / 'run.log').open('a', buffering=1) as log:
            log.write(f'\nSession {datetime.now(timezone.utc).isoformat()} phase={phase}\n')
            def command(args):
                cmd = [sys.executable, '-u', '-m', *map(str, args)]
                line = '$ ' + shlex.join(cmd)
                print(line, flush=True); log.write(line + '\n')
                with subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end='', flush=True); log.write(line)
                    if process.wait():
                        raise subprocess.CalledProcessError(process.returncode, cmd)

            def train(args, output, configuration, segmentation=False, pseudo=None):
                state = training_state(output, configuration, segmentation, pseudo)
                if state == 'complete':
                    print(f'Entrenamiento ya completo: {output}', flush=True)
                    return
                print(f'{"Reanudando" if state == "resume" else "Iniciando"}: {output}', flush=True)
                command(args + (['--resume'] if state == 'resume' else []))

            def stage(key, args, output, final_file):
                marker = markers / f'{key}.json'
                if marker.exists():
                    saved = json.loads(marker.read_text())
                    if saved['command'] != list(map(str, args)) or saved['final_sha256'] != sha256(output / final_file):
                        raise ValueError(f'Completed stage changed: {key}')
                    print(f'Etapa ya completa: {key}', flush=True)
                    return
                if output.exists():
                    suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                    archived = output.with_name(output.name + '.interrupted_' + suffix)
                    output.rename(archived)
                    print(f'Salida sin marca de finalización conservada en {archived}', flush=True)
                command(args)
                write_json(marker, dict(command=list(map(str, args)), final_sha256=sha256(output / final_file)))

            runs = []
            for model in MODELS:
                classifier_path = Path(config_dir) / f'train_{model}.json'
                classifier = json.loads((root / classifier_path).read_text())
                classifier_output = (root / cfg['branches'][model]['checkpoint']).parent
                if phase != 'test':
                    train(['src.training.train', '--config', classifier_path, '--root', '.', '--output', classifier_output],
                          classifier_output, classifier)
                elif training_state(classifier_output, classifier) != 'complete':
                    raise ValueError('Classifier must complete its training budget before final test')
                branch = directory / model
                runs.append(branch)
                if phase == 'classifiers':
                    continue
                split = 'test' if phase == 'test' else 'val'
                calibration = branch / 'calibration'
                if phase == 'test' and not (calibration / 'threshold.json').is_file():
                    raise ValueError('Complete training/validation before requesting test')
                maps = branch / f'maps_{split}'
                stage(f'{model}_maps_{split}', ['src.pipeline.maps', '--config', cfg_path, '--branch', model,
                      '--split', split, '--root', '.', '--output', maps], maps, 'metadata.json')
                if phase != 'test':
                    stage(f'{model}_calibration', ['src.pipeline.pseudo', 'calibrate', '--maps', maps,
                          '--root', '.', '--output', calibration], calibration, 'threshold.json')
                pseudo = branch / f'pseudo_{split}'
                stage(f'{model}_pseudo_{split}', ['src.pipeline.pseudo', 'generate', '--maps', maps,
                      '--calibration', calibration / 'threshold.json', '--output', pseudo], pseudo, 'metadata.json')
                localization = branch / f'localization_{split}'
                stage(f'{model}_localization_{split}', ['src.pipeline.pseudo', 'evaluate', '--pseudo', pseudo,
                      '--root', '.', '--output', localization], localization, 'summary.json')
                segmenter = branch / 'segmenter'
                if phase == 'test' and training_state(segmenter, cfg, True, branch / 'pseudo_train') != 'complete':
                    raise ValueError('Segmenter must complete its training budget before final test')
                if phase != 'test':
                    train_maps, train_pseudo = branch / 'maps_train', branch / 'pseudo_train'
                    stage(f'{model}_maps_train', ['src.pipeline.maps', '--config', cfg_path, '--branch', model,
                          '--split', 'train', '--root', '.', '--output', train_maps], train_maps, 'metadata.json')
                    stage(f'{model}_pseudo_train', ['src.pipeline.pseudo', 'generate', '--maps', train_maps,
                          '--calibration', calibration / 'threshold.json', '--output', train_pseudo], train_pseudo, 'metadata.json')
                    train(['src.segmentation.train', 'train', '--config', cfg_path, '--pseudo', train_pseudo,
                           '--root', '.', '--output', segmenter], segmenter, cfg, True, train_pseudo)
                evaluation = branch / f'segmentation_{split}'
                stage(f'{model}_segmentation_{split}', ['src.segmentation.train', 'evaluate', '--checkpoint',
                      segmenter / 'best.pt', '--split', split, '--root', '.', '--output', evaluation], evaluation, 'summary.json')
            if phase != 'classifiers':
                comparison = directory / f'comparison_{split}'
                stage(f'comparison_{split}', ['src.pipeline.compare', '--runs', *runs,
                      '--split', split, '--output', comparison], comparison, 'comparison.json')
            print(f'Completado: {phase}. Resultados: {directory}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--configs', default=CONFIG_DIR)
    parser.add_argument('--phase', choices=['classifiers', 'train-val', 'test'], default='train-val')
    args = parser.parse_args()
    run(args.root, args.configs, args.phase)
