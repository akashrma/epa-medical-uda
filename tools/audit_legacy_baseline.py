"""Capture legacy configs, data integrity, and checkpoint provenance."""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mmcv
import numpy as np
import torch

from mmseg.utils import collect_env


def git(*args):
    return subprocess.check_output(['git', *args], cwd=str(ROOT),
                                   text=True).strip()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def relative(path):
    return str(path.resolve().relative_to(ROOT))


def describe_array(array, labels=False):
    info = dict(
        shape=list(array.shape),
        dtype=str(array.dtype),
        finite=bool(np.isfinite(array).all()),
        minimum=float(array.min()),
        maximum=float(array.max()))
    if labels:
        values, counts = np.unique(array, return_counts=True)
        info['label_counts'] = {
            str(float(value)): int(count)
            for value, count in zip(values, counts)
        }
        info['valid_labels'] = bool(np.isin(array, [0, 1, 2, 3, 4, 255]).all())
    return info


def audit_data(out_dir):
    lists = {}
    paths = {}
    issues = []
    for list_path in sorted((ROOT / 'data/datalist').glob('*.txt')):
        entries = [
            line.strip() for line in list_path.read_text().splitlines()
            if line.strip()
        ]
        resolved = [(list_path.parent / entry).resolve() for entry in entries]
        lists[list_path.name] = dict(
            path=relative(list_path),
            sha256=sha256(list_path),
            count=len(entries),
            duplicate_paths=len(resolved) - len(set(resolved)),
            files=[relative(path) for path in resolved])
        for path in resolved:
            paths.setdefault(path, []).append(list_path.name)

    records = {}
    with (out_dir / 'dataset_files.jsonl').open('w') as handle:
        for index, (path, memberships) in enumerate(sorted(paths.items())):
            name = relative(path)
            record = dict(path=name, lists=memberships)
            if not path.is_file():
                issues.append(dict(path=name, error='Missing file'))
                record['error'] = 'Missing file'
            else:
                record.update(bytes=path.stat().st_size, sha256=sha256(path))
                try:
                    if path.suffix == '.npz':
                        with np.load(str(path), allow_pickle=False) as volume:
                            record['image'] = describe_array(volume['arr_0'])
                            record['labels'] = describe_array(
                                volume['arr_1'], labels=True)
                        if record['image']['shape'] != record['labels'][
                                'shape']:
                            raise ValueError(
                                'Volume image/label shapes differ')
                    else:
                        array = np.load(str(path), allow_pickle=False)
                        record['array'] = describe_array(
                            array,
                            labels=any('_gt.' in m for m in memberships))
                    arrays = [
                        record[key] for key in ('array', 'image', 'labels')
                        if key in record
                    ]
                    if any(not a['finite'] or not a.get('valid_labels', True)
                           for a in arrays):
                        issues.append(
                            dict(path=name, error='Invalid array values'))
                except Exception as error:
                    record['error'] = str(error)
                    issues.append(dict(path=name, error=str(error)))
            records[name] = record
            handle.write(json.dumps(record, sort_keys=True) + '\n')
            if (index + 1) % 5000 == 0:
                print(
                    'Audited {} / {} data files'.format(index + 1, len(paths)),
                    flush=True)

    pairs = {}
    for split in ('train_ct', 'train_mr', 'val_ct', 'val_mr'):
        images = lists[split + '.txt']['files']
        labels = lists[split + '_gt.txt']['files']
        mismatches = []
        for image, label in itertools.zip_longest(images, labels):
            if image is None or label is None:
                mismatches.append([image, label])
            elif records[image].get('array', {}).get('shape') != \
                    records[label].get('array', {}).get('shape'):
                mismatches.append([image, label])
            elif Path(image).stem + '_gt' != Path(label).stem:
                mismatches.append([image, label])
        pairs[split] = dict(
            images=len(images), labels=len(labels), mismatches=mismatches)
        if mismatches:
            issues.append(
                dict(split=split, error='Image/label pairing mismatch'))

    overlap = {}
    for domain in ('ct', 'mr'):
        for first, second in itertools.combinations(('train', 'val', 'test'),
                                                    2):
            a = set(lists[first + '_' + domain + '.txt']['files'])
            b = set(lists[second + '_' + domain + '.txt']['files'])
            a_hashes = {records[p].get('sha256') for p in a}
            b_hashes = {records[p].get('sha256') for p in b}
            overlap[domain + ':' + first + '/' + second] = dict(
                shared_paths=sorted(a & b),
                shared_file_hash_count=len((a_hashes & b_hashes) - {None}))

    summary = dict(
        lists=lists,
        pairs=pairs,
        overlap=overlap,
        issues=issues,
        unique_files=len(records),
        total_bytes=sum(r.get('bytes', 0) for r in records.values()),
        patient_overlap=
        'Not established: slice filenames do not encode patient IDs',
        manifest_sha256=sha256(out_dir / 'dataset_files.jsonl'))
    write_json(out_dir / 'data_audit.json', summary)
    return summary


def audit_checkpoints(out_dir):
    records = []
    checkpoint_dir = out_dir / 'checkpoints'
    checkpoint_dir.mkdir()
    for path in sorted((ROOT / 'pretrained').glob('*.pth')):
        print('Reading checkpoint: {}'.format(path.name), flush=True)
        checkpoint = torch.load(str(path), map_location='cpu')
        meta = deepcopy(checkpoint.get('meta', {}))
        config = meta.pop('config', None)
        if config:
            (checkpoint_dir / (path.stem + '.py')).write_text(config)
        state = checkpoint.get('state_dict', checkpoint)
        inventory = {
            name: dict(shape=list(value.shape), dtype=str(value.dtype))
            for name, value in state.items() if torch.is_tensor(value)
        }
        write_json(checkpoint_dir / (path.stem + '_tensors.json'), inventory)
        optimizer = checkpoint.get('optimizer', {})
        groups = [{k: v
                   for k, v in group.items() if k != 'params'}
                  for group in optimizer.get('param_groups', [])]
        records.append(
            dict(
                path=relative(path),
                bytes=path.stat().st_size,
                sha256=sha256(path),
                keys=list(checkpoint),
                meta=meta,
                tensor_count=len(inventory),
                state_prefixes=dict(
                    Counter(k.split('.')[0] for k in inventory)),
                optimizer_group_settings=groups,
                provenance=
                'Historical metadata; score not yet independently reproduced'))
        del checkpoint, state, optimizer
    write_json(out_dir / 'checkpoints.json', records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', action='append', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    if git('branch', '--show-current') != 'mmcv2':
        parser.error('All migration edits and progress must occur on mmcv2')
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.relative_to(ROOT / 'work_dirs')
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    packages = sorted('{}=={}'.format(d.metadata['Name'], d.version)
                      for d in importlib.metadata.distributions())
    (out_dir / 'packages.txt').write_text('\n'.join(packages) + '\n')
    env = collect_env()
    (out_dir /
     'environment.txt').write_text('\n'.join('{}: {}'.format(k, v)
                                             for k, v in env.items()) + '\n')
    gpu = subprocess.run([
        'nvidia-smi',
        '--query-gpu=name,driver_version,memory.total,memory.free',
        '--format=csv,noheader'
    ],
                         capture_output=True,
                         text=True)
    environment = dict(
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        torch=torch.__version__,
        mmcv=mmcv.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        cuda_available=torch.cuda.is_available(),
        gpu_query=gpu.stdout.strip(),
        gpu_error=gpu.stderr.strip(),
        collect_env=env)
    write_json(out_dir / 'environment.json', environment)
    (out_dir / 'source_status.txt').write_text(git('status', '--short') + '\n')
    source_inventory = {
        name: sha256(ROOT / name)
        for name in git('ls-files').splitlines() if (ROOT / name).is_file()
    }
    source_inventory[relative(Path(__file__))] = sha256(Path(__file__))
    write_json(out_dir / 'source_files.json', source_inventory)
    prototypes = {
        relative(path): sha256(path)
        for path in sorted((ROOT / 'etf_prototypes').glob('*'))
        if path.is_file()
    }
    write_json(out_dir / 'prototypes.json', prototypes)
    configs = []
    config_dir = out_dir / 'configs'
    config_dir.mkdir()
    for config_path in args.config:
        cfg = mmcv.Config.fromfile(str(ROOT / config_path))
        cfg.work_dir = 'work_dirs/migration_baseline/' + Path(config_path).stem
        cfg.model.train_cfg.work_dir = cfg.work_dir
        cfg.model.train_cfg.log_config = cfg.log_config
        cfg.gpu_ids = range(1)
        cfg.uda['model'] = deepcopy(cfg.model)
        cfg.uda['max_iters'] = cfg.runner.max_iters
        resolved = config_dir / Path(config_path).name
        cfg.dump(str(resolved))
        configs.append(
            dict(
                source=config_path,
                source_sha256=sha256(ROOT / config_path),
                resolved=relative(resolved),
                resolved_sha256=sha256(resolved),
                seed=cfg.seed,
                gpu_count=1,
                max_iters=cfg.runner.max_iters,
                evaluation=dict(cfg.evaluation),
                selection='Candidate; awaiting user choice'))
    data = audit_data(out_dir)
    checkpoints = audit_checkpoints(out_dir)
    manifest = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        branch='mmcv2',
        revision=git('rev-parse', 'HEAD'),
        main_revision=git('rev-parse', 'main'),
        configs=configs,
        environment=relative(out_dir / 'environment.json'),
        dataset=relative(out_dir / 'data_audit.json'),
        checkpoint_metadata=relative(out_dir / 'checkpoints.json'),
        checkpoints=[record['path'] for record in checkpoints],
        data_integrity_passed=not data['issues'],
        pending=[
            'Choose exact experiment and overrides',
            'Reproduce compatible checkpoint metrics',
            'Full unchanged legacy training / final-checkpoint evidence',
            'Capture fixed batches, outputs and optimizer traces'
        ])
    write_json(out_dir / 'manifest.json', manifest)
    print(
        json.dumps(
            dict(
                output=str(out_dir),
                data_files=data['unique_files'],
                integrity_issues=len(data['issues']),
                cuda_available=environment['cuda_available']),
            indent=2))


if __name__ == '__main__':
    main()
