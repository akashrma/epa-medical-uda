"""Calibrate legacy repeatability before inspecting modern numerical probes."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'work_dirs/migration_baseline'
OUT = BASE / 'step8_numerics_20261011'
PYTHONS = dict(
    legacy='/home/akash/miniconda3/envs/epa/bin/python',
    modern='/home/akash/miniconda3/envs/epa-mmcv2/bin/python')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def execute(stack,
            direction,
            output,
            mode='probe',
            replay=None,
            declared_prototypes=False,
            fixed_predictions=None):
    command = [
        PYTHONS[stack], '-u',
        str(ROOT / 'tools/capture_migration_numerics.py'), '--stack', stack,
        '--direction', direction, '--out',
        str(output), '--mode', mode
    ]
    if replay:
        command.extend(['--replay', str(replay)])
    if declared_prototypes:
        command.append('--declared-prototypes')
    if fixed_predictions:
        command.extend(['--fixed-predictions', str(fixed_predictions)])
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2',
        MPLCONFIGDIR=str(OUT / 'matplotlib'))
    with (output / 'console.log').open('w') as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError((output / 'console.log').read_text()[-7000:])
    print(f'Completed {stack} {direction} {mode}: {output.name}', flush=True)


def difference(a, b):
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    error = np.abs(a - b)
    scale = max(float(np.max(np.abs(a))), 1e-12)
    rms = max(float(np.sqrt(np.mean(a * a))), 1e-12)
    return dict(
        max_absolute=float(error.max()),
        max_relative=float((error / np.maximum(np.abs(a), 1e-12)).max()),
        normalized_max=float(error.max()) / scale,
        normalized_rms=float(np.sqrt(np.mean(error * error))) / rms)


def category(name):
    if name == 'fda_input':
        return 'fda'
    if 'logits_' in name:
        return 'logits'
    if name == 'pseudo_label':
        return 'labels'
    if name == 'pseudo_weight':
        return 'pseudo_weight'
    if name.startswith(('gradient/', 'exp_avg/', 'exp_avg_sq/')):
        return 'gradients'
    if name.startswith('delta/'):
        return 'updates'
    if name == 'etf_prototypes':
        return 'structural'
    return 'state'


def compare_probe(reference, candidate):
    import numpy as np
    import torch
    a_spec = json.loads((reference / 'probe.json').read_text())
    b_spec = json.loads((candidate / 'probe.json').read_text())
    for key in ('seed', 'horizon', 'checkpoint_sha256', 'fixed_batch_sha256'):
        assert a_spec[key] == b_spec[key], key
    reports = {}
    for domain in ('target', 'source'):
        old = torch.load(
            reference / (domain + '_fixed.pt'), map_location='cpu')
        new = torch.load(
            candidate / (domain + '_fixed.pt'), map_location='cpu')
        assert len(old) == len(new) == 3
        for a, b in zip(old, new):
            assert a['index'] == b['index']
            assert torch.equal(a['data']['img'][0], b['data']['img'][0])
            assert_metadata_equal(a['data']['img_metas'],
                                  b['data']['img_metas'])
            value = difference(a['logits'].numpy(), b['logits'].numpy())
            value['prediction_disagreement'] = float(
                np.mean(a['prediction'] != b['prediction']))
            reports[f'eval/{domain}/{a["index"]}'] = dict(
                category='logits', **value)
    for case, iterations in (('initial', (0, 1)), ('boundary', (1999, 2000,
                                                                2001))):
        old_trace = json.loads((reference / case / 'trace.json').read_text())
        new_trace = json.loads((candidate / case / 'trace.json').read_text())
        assert len(old_trace) == len(new_trace) == len(iterations)
        for a, b in zip(old_trace, new_trace):
            assert a['iteration'] == b['iteration']
            assert a['rates'] == b['rates']
            assert a['backwards'] == b['backwards'] and a['completed'] == b[
                'completed']
            assert a['stochastic_calls'] == b['stochastic_calls']
            assert set(a['losses']) == set(b['losses'])
            for key in a['losses']:
                reports[f'{case}/{a["iteration"]}/loss/{key}'] = dict(
                    category='loss',
                    **difference([a['losses'][key]], [b['losses'][key]]))
        for iteration in iterations:
            old = torch.load(
                reference / case / f'{iteration}.pt', map_location='cpu')
            new = torch.load(
                candidate / case / f'{iteration}.pt', map_location='cpu')
            assert old.keys() == new.keys()
            for name in old:
                kind = category(name)
                value = difference(old[name].numpy(), new[name].numpy())
                if kind == 'labels':
                    value['prediction_disagreement'] = float(
                        np.mean(old[name].numpy() != new[name].numpy()))
                reports[f'{case}/{iteration}/{name}'] = dict(
                    category=kind, **value)
            a_rng = torch.load(
                reference / case / f'{iteration}_rng.pt', map_location='cpu')
            b_rng = torch.load(
                candidate / case / f'{iteration}_rng.pt', map_location='cpu')
            assert_metadata_equal(a_rng['numpy'], b_rng['numpy'])
            assert a_rng['python'] == b_rng['python']
            assert torch.equal(a_rng['cpu'], b_rng['cpu'])
            # Torch 1.7 serializes 800 extra legacy RNG bytes before Philox.
            assert torch.equal(a_rng['cuda'][-16:], b_rng['cuda'][-16:])
            reports[f'{case}/{iteration}/rng'] = dict(
                category='rng',
                python_numpy_cpu_equal=True,
                philox_seed_offset_equal=True,
                cuda_serialized_bytes=[
                    a_rng['cuda'].numel(), b_rng['cuda'].numel()
                ])
        masks_old = torch.load(
            reference / case / 'stochastic.pt', map_location='cpu')
        masks_new = torch.load(
            candidate / case / 'stochastic.pt', map_location='cpu')
        if masks_new:
            assert len(masks_old) == len(masks_new)
            assert all((a['name'], a['kind'], a['p']) == (b['name'], b['kind'],
                                                          b['p'])
                       for a, b in zip(masks_old, masks_new))
            mismatches = sum(
                int((a['mask'] != b['mask']).sum())
                for a, b in zip(masks_old, masks_new))
            count = sum(a['mask'].numel() for a in masks_old)
            reports[case + '/stochastic_masks'] = dict(
                category='stochastic',
                disagreement=mismatches / count,
                elements=count)
    return reports


def assert_metadata_equal(a, b):
    import numpy as np
    if isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            if key in ('filename', 'ori_filename'):
                assert Path(a[key]).resolve() == Path(b[key]).resolve()
            else:
                assert_metadata_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        # Capture transport unwraps legacy metadata tuples into lists. The
        # real dataloader type contract is checked separately in step 5.
        assert isinstance(b, (tuple, list)) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_metadata_equal(x, y)
    else:
        assert a == b


def assess_probe(reports, policy):
    failures = []
    maxima = {}
    for name, value in reports.items():
        kind = value['category']
        maximum = maxima.setdefault(kind, {})
        for metric in ('normalized_rms', 'normalized_max',
                       'prediction_disagreement', 'disagreement'):
            if metric in value:
                maximum[metric] = max(maximum.get(metric, 0.), value[metric])
        if kind in policy['budgets']:
            for metric, limit in policy['budgets'][kind].items():
                if value[metric] > limit:
                    failures.append(
                        dict(
                            name=name,
                            metric=metric,
                            value=value[metric],
                            limit=limit))
        if value.get('prediction_disagreement',
                     0.) > policy['label_disagreement']:
            failures.append(
                dict(
                    name=name,
                    metric='prediction_disagreement',
                    value=value['prediction_disagreement'],
                    limit=policy['label_disagreement']))
        if kind == 'stochastic' and value['disagreement']:
            failures.append(
                dict(
                    name=name,
                    metric='stochastic_disagreement',
                    value=value['disagreement'],
                    limit=0.))
    return dict(passed=not failures, failures=failures, maxima=maxima)


def compare_full(direction, policy, reference_dir=None):
    import numpy as np
    reference = (reference_dir or OUT) / ('legacy_' + direction + '_full')
    candidate = OUT / ('modern_' + direction + '_full')
    a = json.loads((reference / 'full.json').read_text())
    b = json.loads((candidate / 'full.json').read_text())
    assert a['checkpoint_sha256'] == b['checkpoint_sha256']
    assert a['slices'] == b['slices']
    old = np.load(reference / 'predictions.npy', mmap_mode='r')
    new = np.load(candidate / 'predictions.npy', mmap_mode='r')
    assert old.shape == new.shape
    disagreements = sum(
        int(np.count_nonzero(x != y)) for x, y in zip(old, new))
    fraction = disagreements / old.size
    fixed = b['fixed_prediction_metrics']
    assert fixed and set(fixed) == set(a['metrics'])
    metric_errors = {}
    for name in fixed:
        x, y = np.asarray(a['metrics'][name]), np.asarray(fixed[name])
        assert x.shape == y.shape and np.array_equal(np.isnan(x), np.isnan(y))
        error = np.abs(x - y)
        metric_errors[name] = float(
            np.nanmax(error)) if not np.isnan(error).all() else 0.
    maximum = max(metric_errors.values())
    dice_delta = abs(a['metrics']['mDice'] - b['metrics']['mDice'])
    passed = (
        maximum <= policy['fixed_prediction_metrics_absolute']
        and dice_delta <= policy['full_split_mDice_absolute']
        and fraction <= policy['full_split_prediction_disagreement'])
    return dict(
        passed=passed,
        slices=a['slices'],
        checkpoint=a['checkpoint'],
        checkpoint_sha256=a['checkpoint_sha256'],
        legacy_metrics=a['metrics'],
        modern_metrics=b['metrics'],
        mDice_absolute_delta=dice_delta,
        changed_pixels=disagreements,
        prediction_disagreement=fraction,
        fixed_prediction_metric_errors=metric_errors)


def verify_cpu(policy):
    import tempfile
    import torch
    sys.path.insert(0, str(ROOT))
    from mmseg.models.utils.etf_prototypes import (_find_repo_root,
                                                   get_default_etf_dir)
    assert Path(get_default_etf_dir()) == ROOT / 'etf_prototypes'
    frozen = BASE / 'legacy_source'
    assert not (frozen / 'AGENTS.md').exists()
    assert Path(_find_repo_root(frozen / 'mmseg/models/utils')) == frozen
    with tempfile.TemporaryDirectory(dir=OUT) as temporary:
        checkout = Path(temporary)
        (checkout / 'README.md').touch()
        (checkout / 'configs/_base_').mkdir(parents=True)
        nested = checkout / 'mmseg/models/utils'
        nested.mkdir(parents=True)
        assert Path(_find_repo_root(nested)) == checkout
    for stack in ('legacy', 'modern'):
        execute(stack, 'ct2mr', OUT / (stack + '_fft'), mode='fft')
    old = torch.load(OUT / 'legacy_fft/fft.pt', map_location='cpu')
    new = torch.load(OUT / 'modern_fft/fft.pt', map_location='cpu')
    old_rates = torch.load(OUT / 'legacy_fft/rates.pt', map_location='cpu')
    new_rates = torch.load(OUT / 'modern_fft/rates.pt', map_location='cpu')
    assert old_rates.keys() == new_rates.keys()
    for key in old_rates:
        assert torch.equal(old_rates[key], new_rates[key]), key
    assert old.keys() == new.keys()
    comparisons = {}
    exploratory = {}
    for name in old:
        assert torch.equal(old[name]['source'], new[name]['source'])
        assert torch.equal(old[name]['target'], new[name]['target'])
        value = difference(old[name]['output'].numpy(),
                           new[name]['output'].numpy())
        passed = all(value[metric] <= limit
                     for metric, limit in policy['budgets']['fda'].items())
        value['passed'] = passed
        if name in ('(8, 8, 0.0)', '(256, 256, 0.01)'):
            assert passed, 'Configured FDA or round-trip CPU control failed'
            comparisons[name] = value
        else:
            exploratory[name] = value
    result = dict(
        fft_cpu=comparisons,
        legacy_drop_path_rates_exact=list(old_rates),
        root_without_optional_agents=True,
        frozen_root_resolution=True,
        exploratory_fft_outside_baseline=exploratory,
        scope='Configured 256x256/L=0.01 FDA and no-mutation round trip. '
        'Small/odd L=0.2 CPU legacy transforms differ and are not qualified.')
    write_json(OUT / 'cpu_checks.json', result)
    print(json.dumps(result, indent=2), flush=True)


def verify_guards():

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = json.loads((BASE / 'legacy_source_manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert digest(BASE / 'legacy_source' / name) == expected, name
    audit = BASE / 'step1_audit_20261011'
    configs = {
        name: expected
        for name, expected in json.loads((
            audit / 'source_files.json').read_text()).items()
        if name.startswith('configs/')
    }
    for name, expected in configs.items():
        assert digest(ROOT / name) == expected, name
    prototypes = json.loads((audit / 'prototypes.json').read_text())
    for name, expected in prototypes.items():
        assert digest(ROOT / name) == expected, name
    main = subprocess.check_output(['git', 'rev-parse', 'main'],
                                   text=True).strip()
    assert main == '4a589d11e637c4e61442a69fe6ac8e6f57a883e7'
    packages = json.loads(
        (BASE /
         'step6_training_20261011/environment_packages.json').read_text())
    counts = {}
    for stack, python in PYTHONS.items():
        name = 'epa' if stack == 'legacy' else 'epa-mmcv2'
        actual = json.loads(
            subprocess.check_output(
                [python, '-m', 'pip', 'list', '--format=json'], text=True))

        def normalize(rows):
            return {row['name'].lower(): row['version'] for row in rows}

        assert normalize(actual) == normalize(packages[name]), name
        counts[name] = len(actual)
    return dict(
        branch='mmcv2',
        main_revision=main,
        frozen_source_files_unchanged=len(manifest['sha256']),
        original_configs_unchanged=len(configs),
        prototypes_unchanged=len(prototypes),
        environment_packages_unchanged=counts)


def verify_source_smoke():
    # Earlier sandbox attempts cannot share worker tensors over Unix sockets.
    # Stop only those exact diagnostic commands before retrying on the host.
    rows = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    for row in rows.splitlines()[1:]:
        if ('tools/test.py' in row and '--device cpu' in row
                and '_source_cpu_metrics.json' in row and str(
                    OUT.relative_to(ROOT) / 'native_training_final') in row):
            os.kill(int(row.split(None, 1)[0]), signal.SIGTERM)
    reports = {}
    for direction, reverse in (('ct2mr', 'mr2ct'), ('mr2ct', 'ct2mr')):
        run = OUT / 'native_training_final' / (direction + '_initial')
        checkpoint = run / 'iter_2.pth'
        metrics = OUT / (direction + '_source_cpu_metrics.json')
        fixture = OUT / 'native_training_final/validation_lists'
        command = [
            sys.executable, 'tools/test.py', 'configs/medical/' + direction +
            '_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py',
            str(checkpoint), '--device', 'cpu', '--eval', 'mDice',
            '--metrics-out',
            str(metrics), '--options',
            f'data.test.img_list={fixture / (reverse + "_img_list.txt")}',
            f'data.test.ann_list={fixture / (reverse + "_ann_list.txt")}'
        ]
        environment = dict(
            os.environ,
            OMP_NUM_THREADS='2',
            MKL_NUM_THREADS='2',
            MPLCONFIGDIR=str(OUT / 'matplotlib'))
        with (OUT / (direction + '_source_cpu_host.log')).open('w') as stream:
            subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True)
        observation = json.loads((run / 'observations.json').read_text())
        reports[direction] = dict(
            seed=0,
            checkpoint=str(checkpoint),
            slices_per_domain=2,
            source_cpu=json.loads(metrics.read_text()),
            target_cuda=observation['records'][-1]['target_validation'])
    write_json(OUT / 'source_target_smoke.json', reports)
    print(json.dumps(reports, indent=2), flush=True)


def freeze_tolerances():
    # Floors are declared before any modern probe. TF32 products have 10
    # fraction bits; the larger gradient/update budgets allow cancellation.
    # Budgets apply to fixed-noise floating execution; RNG/mask differences
    # are diagnosed separately rather than hidden by these tolerances.
    floors = dict(
        fda=(1e-5, 1e-4),
        logits=(5 / 1024, 20 / 1024),
        loss=(5 / 1024, 5 / 1024),
        gradients=(20 / 1024, 100 / 1024),
        updates=(50 / 1024, 100 / 1024),
        state=(5 / 1024, 20 / 1024),
        pseudo_weight=(1e-3, 1e-3),
        structural=(0., 0.))
    aggregate = {
        key: dict(normalized_rms=0., normalized_max=0.)
        for key in floors
    }
    disagreements = []
    repeat_reports = {}
    for direction in ('ct2mr', 'mr2ct'):
        reference = OUT / ('legacy_' + direction + '_0')
        for repeat in (1, 2):
            report = compare_probe(reference,
                                   OUT / f'legacy_{direction}_{repeat}')
            repeat_reports[f'{direction}_{repeat}'] = report
            for value in report.values():
                kind = value['category']
                if kind in aggregate:
                    for key in aggregate[kind]:
                        aggregate[kind][key] = max(aggregate[kind][key],
                                                   value[key])
                if 'prediction_disagreement' in value:
                    disagreements.append(value['prediction_disagreement'])
                if kind == 'stochastic':
                    assert value[
                        'disagreement'] == 0, 'Legacy seed masks differ'
    budgets = {}
    for kind, (rms, maximum) in floors.items():
        budgets[kind] = dict(
            normalized_rms=max(rms, 5 * aggregate[kind]['normalized_rms']),
            normalized_max=max(maximum, 5 * aggregate[kind]['normalized_max']))
    policy = dict(
        version=1,
        branch='mmcv2',
        legacy_repeats_per_direction=3,
        frozen_before_modern=True,
        legacy_noise=aggregate,
        budgets=budgets,
        label_disagreement=max(5 / 1024, 5 * max(disagreements)),
        full_split_mDice_absolute=1e-3,
        full_split_prediction_disagreement=max(5 / 1024,
                                               5 * max(disagreements)),
        fixed_prediction_metrics_absolute=1e-10,
        method=('Five times observed legacy repeat spread or the predeclared '
                'TF32/FP32 floor; normalized RMS/max use global reference '
                'scale, not near-zero elementwise ratios. Dropout replay is '
                'a separately labeled control; default-mask differences '
                'are reported.'))
    write_json(OUT / 'legacy_repeat_comparisons.json', repeat_reports)
    write_json(OUT / 'tolerances.json', policy)
    print(json.dumps(policy, indent=2), flush=True)


def qualify_precision_candidate():
    """Reuse immutable legacy budgets/oracles in a separate candidate audit."""
    reference = BASE / 'step8_numerics_20261011'
    assert OUT != reference, 'Preserve the original numerical qualification'
    freeze = reference / 'tolerances.json'
    policy = json.loads(freeze.read_text())
    copied = OUT / 'tolerances.json'
    assert not copied.exists() or copied.read_bytes() == freeze.read_bytes()
    copied.write_bytes(freeze.read_bytes())
    assessments = {}
    full = {}
    for direction in ('ct2mr', 'mr2ct'):
        output = OUT / ('modern_' + direction)
        assert not (output / 'probe.json').exists(), 'Do not overwrite probes'
        execute('modern', direction, output)
        spec = json.loads((output / 'probe.json').read_text())
        assert not spec['cudnn_allow_tf32'] and spec['matmul_allow_tf32']
        comparison = compare_probe(
            reference / ('legacy_' + direction + '_declared'), output)
        write_json(OUT / (direction + '_probe_comparison.json'), comparison)
        assessments[direction] = assess_probe(comparison, policy)
        # Save the assessment before doing the more expensive split checks.
        write_json(OUT / 'probe_assessments.json', assessments)
        assert assessments[direction]['passed'], (
            direction + ' candidate exceeds immutable numerical budgets')
        output = OUT / ('modern_' + direction + '_full')
        assert not (output / 'full.json').exists(), 'Do not overwrite splits'
        execute(
            'modern',
            direction,
            output,
            mode='full',
            fixed_predictions=reference /
            ('legacy_' + direction + '_full/predictions.npy'))
        full[direction] = compare_full(direction, policy, reference)
        write_json(OUT / 'full_split_comparisons.json', full)
        assert full[direction]['passed'], 'Candidate split/metric gate failed'
    # Verify the public runner hook and translator actually use this policy.
    sys.path.insert(0, str(ROOT))
    from mmengine.config import Config
    from mmseg.engine.config import translate_legacy_config
    from mmseg.engine.hooks import LegacyPrecisionHook, backend_flags
    LegacyPrecisionHook().before_run(None)
    flags = backend_flags()
    assert flags == dict(
        cudnn_benchmark=True,
        cudnn_deterministic=False,
        cudnn_allow_tf32=False,
        matmul_allow_tf32=True,
        deterministic_algorithms=False)
    for direction in ('ct2mr', 'mr2ct'):
        cfg = Config.fromfile(
            str(ROOT / 'configs/medical' /
                (direction +
                 '_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py')))
        native = translate_legacy_config(
            cfg, work_dir=str(OUT / (direction + '_config')), stop_after=5000)
        assert native.train_cfg.max_iters == 50000
        assert native.train_cfg.stop_after == 5000
    report = dict(
        candidate='fp32-conv',
        reference_directory=str(reference),
        tolerance_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest(),
        probes=assessments,
        full_split=full,
        backend=flags,
        guards=verify_guards(),
        scope='Original fixed-input, optimizer/EMA/target-boundary probes '
        'and full target-split same-checkpoint/identical-prediction metrics. '
        'Short numerical qualification does not establish trained accuracy.',
        passed=all(x['passed']
                   for x in list(assessments.values()) + list(full.values())))
    write_json(OUT / 'qualification.json', report)
    print('FP32-convolution numerical qualification passed.', flush=True)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', help='Fresh or existing audit directory')
    parser.add_argument(
        '--phase',
        choices=[
            'legacy', 'oracle', 'modern', 'full', 'cpu', 'source-smoke',
            'compare', 'precision-candidate'
        ],
        required=True)
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    if args.out_dir:
        OUT = Path(args.out_dir).resolve()
        OUT.relative_to(BASE)
    os.chdir(ROOT)
    assert subprocess.check_output(['git', 'branch', '--show-current'],
                                   text=True).strip() == 'mmcv2'
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == 'precision-candidate':
        assert args.out_dir, 'Candidate checks require a separate output dir'
        qualify_precision_candidate()
    elif args.phase == 'legacy':
        assert not (OUT / 'tolerances.json').exists(), (
            'Frozen tolerances are immutable; use a fresh audit directory')
        assert not list(
            OUT.glob('modern*/probe.json')), 'Calibrate before modern probes'
        for direction in ('ct2mr', 'mr2ct'):
            for repeat in range(3):
                output = OUT / f'legacy_{direction}_{repeat}'
                if not args.reuse or not (output / 'probe.json').is_file():
                    execute('legacy', direction, output)
        freeze_tolerances()
    elif args.phase == 'oracle':
        assert (OUT / 'tolerances.json').is_file()
        policy = json.loads((OUT / 'tolerances.json').read_text())
        comparisons = {}
        for direction in ('ct2mr', 'mr2ct'):
            reference = OUT / ('legacy_' + direction + '_declared')
            for repeat in range(3):
                output = reference if repeat == 0 else OUT / (
                    reference.name + f'_{repeat}')
                if not args.reuse or not (output / 'probe.json').is_file():
                    execute(
                        'legacy', direction, output, declared_prototypes=True)
                if repeat:
                    report = compare_probe(reference, output)
                    comparisons[f'{direction}_{repeat}'] = dict(
                        comparison=report,
                        assessment=assess_probe(report, policy))
        write_json(OUT / 'declared_repeat_comparisons.json', comparisons)
        assert all(value['assessment']['passed']
                   for value in comparisons.values()), (
                       'Declared repeats exceed frozen budgets; '
                       'do not refreeze')
        print(
            'Declared-asset legacy repeats fit the unchanged frozen budgets.',
            flush=True)
    elif args.phase == 'modern':
        assert (OUT /
                'tolerances.json').is_file(), 'Freeze legacy tolerances first'
        for direction in ('ct2mr', 'mr2ct'):
            output = OUT / ('modern_' + direction)
            if not args.reuse or not (output / 'probe.json').is_file():
                execute('modern', direction, output)
        comparisons = {
            direction:
            compare_probe(OUT / ('legacy_' + direction + '_declared'),
                          OUT / ('modern_' + direction))
            for direction in ('ct2mr', 'mr2ct')
        }
        write_json(OUT / 'modern_default_comparisons.json', comparisons)
        print(
            'Default modern probes captured; comparisons written.', flush=True)
    elif args.phase == 'source-smoke':
        verify_source_smoke()
    elif args.phase == 'cpu':
        verify_cpu(json.loads((OUT / 'tolerances.json').read_text()))
    elif args.phase == 'full':
        assert (OUT / 'tolerances.json').is_file()
        for stack in ('legacy', 'modern'):
            for direction in ('ct2mr', 'mr2ct'):
                output = OUT / f'{stack}_{direction}_full'
                if not args.reuse or not (output / 'full.json').is_file():
                    fixed = (
                        OUT / f'legacy_{direction}_full/predictions.npy'
                        if stack == 'modern' else None)
                    execute(
                        stack,
                        direction,
                        output,
                        mode='full',
                        fixed_predictions=fixed)
    else:
        freeze = OUT / 'tolerances.json'
        policy = json.loads(freeze.read_text())
        probes = {
            direction:
            compare_probe(OUT / ('legacy_' + direction + '_declared'),
                          OUT / ('modern_' + direction))
            for direction in ('ct2mr', 'mr2ct')
        }
        write_json(OUT / 'modern_default_comparisons.json', probes)
        assessments = {
            key: assess_probe(value, policy)
            for key, value in probes.items()
        }
        full = {
            direction: compare_full(direction, policy)
            for direction in ('ct2mr', 'mr2ct')
        }
        cpu = json.loads((OUT / 'cpu_checks.json').read_text())
        repeats = json.loads(
            (OUT / 'declared_repeat_comparisons.json').read_text())
        native = json.loads(
            (OUT / 'native_training_final/verification.json').read_text())
        assert native['deterministic_cuda_resume']['floating_tensors_exact']
        assert native['boundary']['resume'][
            'nonfloating_optimizer_scheduler_uda_rng_data_exact']
        assert cpu['root_without_optional_agents']
        assert all(x['assessment']['passed'] for x in repeats.values())
        report = dict(
            tolerance_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest(),
            probes=assessments,
            full_split=full,
            cpu_controls=cpu,
            native_training=dict(
                artifact_directory=str(OUT / 'native_training_final'),
                early_resume=native['early_resume'],
                boundary=native['boundary'],
                deterministic_resume=native['deterministic_cuda_resume']),
            source_target_smoke=json.loads(
                (OUT / 'source_target_smoke.json').read_text()) if
            (OUT / 'source_target_smoke.json').exists() else None,
            declared_legacy_repeats_passed=True,
            guards=verify_guards(),
            scope='Both selected seed-0, single-GPU EPA/FDA medical configs; '
            '256x256/L=0.01. Modern 5k accuracy runs remain step 9.',
            passed=all(x['passed'] for x in list(assessments.values()) +
                       list(full.values())))
        write_json(OUT / 'qualification.json', report)
        print(json.dumps(report, indent=2), flush=True)
        if not report['passed']:
            raise SystemExit(
                'Numerical gate failed; do not start matched modern training')


if __name__ == '__main__':
    main()
