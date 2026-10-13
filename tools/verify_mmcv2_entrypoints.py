"""Qualify original-config translation and public medical entrypoints."""

import argparse
import gc
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# isort:skip_file

import numpy as np
import torch
from mmengine.config import Config

from mmseg.apis import set_random_seed
from mmseg.engine.config import translate_legacy_config
from mmseg.registry import DATASETS, MODELS

BASE = ROOT / 'work_dirs/migration_baseline'


def config_path(direction):
    return ROOT / 'configs/medical' / (
        direction + '_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py')


def execute(arguments, output):
    with output.open('w') as stream:
        result = subprocess.run(
            [sys.executable, '-u'] + arguments,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(output.read_text()[-6000:])


def experiment_view(native):
    """Exclude run identity/path metadata from experiment comparison."""
    native = deepcopy(native.to_dict())
    for key in ('work_dir', 'experiment_name', 'migration'):
        native.pop(key)
    native['model']['model']['train_cfg'].pop('work_dir')
    return native


def verify_translation(direction, output):
    source = Config.fromfile(str(config_path(direction)))
    frozen = Config.fromfile(
        str(BASE / 'legacy_source/configs/medical' /
            config_path(direction).name))
    assert source.to_dict() == frozen.to_dict()
    original = deepcopy(source.to_dict())
    native = translate_legacy_config(source, work_dir=str(output / direction))
    assert source.to_dict(
    ) == original, 'Translation mutated its specification'
    assert native.migration.legacy_config == original
    model = deepcopy(native.model.to_dict())
    student = model.pop('model')
    model.pop('max_iters')
    student['train_cfg'].pop('work_dir')
    student['train_cfg'].pop('log_config')
    assert student == original['model'] and model == original['uda']
    assert native.train_dataloader.dataset == original['data']['train']
    assert native.train_dataloader.batch_size == original['data'][
        'samples_per_gpu']
    assert native.train_dataloader.num_workers == original['data'][
        'workers_per_gpu']
    assert native.train_cfg.max_iters == native.model.max_iters == 50000
    assert native.train_cfg.stop_after == 5000
    assert native.train_cfg.val_interval == original['evaluation']['interval']
    assert native.model.model.train_cfg.log_config == original['log_config']
    assert native.optim_wrapper.optimizer == {
        k: v
        for k, v in original['optimizer'].items() if k != 'paramwise_cfg'
    }
    assert native.optim_wrapper.paramwise_cfg == original['optimizer'][
        'paramwise_cfg']
    for split in ('val', 'test'):
        dataset = deepcopy(native[split + '_dataloader'].dataset.to_dict())
        dataset.pop('test_mode')
        expected = deepcopy(original['data'][split])
        expected.pop('test_mode', None)
        assert dataset == expected
    # Exercise every CLI import and the actual generated-child inheritance.
    direct, generated = output / ('direct_' + direction), output / (
        'generated_' + direction)
    execute([
        'tools/train.py',
        str(config_path(direction)), '--dry-run', '--work-dir',
        str(direct)
    ], output / (direction + '_direct.log'))
    execute([
        'run_experiments.py', '--config',
        str(config_path(direction)), '--dry-run', '--work-dir',
        str(generated)
    ], output / (direction + '_generated.log'))
    resolved = Config.fromfile(str(direct / 'native_config.py'))
    child = Config.fromfile(str(generated / 'native_config.py'))
    assert experiment_view(resolved) == experiment_view(child)
    assert experiment_view(native) == experiment_view(resolved)
    rejected = []
    unsupported = (('unknown_runtime', dict(unsupported_runtime=True)),
                   ('amp',
                    dict(optimizer_config=dict(type='Fp16OptimizerHook'))),
                   ('epoch_lr',
                    dict(
                        lr_config=dict(original['lr_config'], by_epoch=True))),
                   ('alternate_metric', dict(evaluation=dict(metric='mIoU'))))
    for label, changes in unsupported:
        bad = Config(dict(original, **changes))
        try:
            translate_legacy_config(bad)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError('Unsupported settings silently accepted: ' +
                                 label)
    for bad_cap in (0, -1, 50001):
        for cfg in (source, native):
            try:
                translate_legacy_config(cfg, stop_after=bad_cap)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid stop accepted')
    dataset = DATASETS.build(native.train_dataloader.dataset)
    assert len(dataset) == 80640000
    assert dataset.metainfo['classes'] == dataset.source.CLASSES
    # Native model construction, independent of the old build_train_model path.
    set_random_seed(0)
    model = MODELS.build(native.model)
    model.init_weights()
    legacy = torch.load(
        BASE / 'step1_audit_20261011' / ('final_trace_' + direction + '_epa') /
        'initialized_state_dict.pt',
        map_location='cpu')
    current = model.state_dict()
    assert list(current) == list(legacy)
    for name, tensor in current.items():
        assert torch.equal(tensor, legacy[name]), name
    count = len(current)
    del model, current, legacy, dataset
    gc.collect()
    return dict(
        config=str(config_path(direction)),
        initialization_tensors=count,
        frozen_resolved_config_equal=True,
        native_dataset_length=80640000,
        direct_generated_experiment_equal=True,
        rejected=rejected,
        max_iters=50000,
        stop_after=5000,
        seed=0)


def verify_cuda(output, reuse_smoke=False):
    assert torch.cuda.is_available(), 'Run --cuda on the GPU host'
    free, total = torch.cuda.mem_get_info()
    assert free > 9 * 1024**3, 'Insufficient free GPU memory for qualification'
    reports = {}
    for direction in ('ct2mr', 'mr2ct'):
        run = output / ('smoke_' + direction)
        if direction == 'ct2mr':
            command = ['tools/train.py', str(config_path(direction))]
        else:
            command = [
                'run_experiments.py', '--config',
                str(config_path(direction))
            ]
        if not reuse_smoke or not (run / 'iter_2.pth').is_file():
            execute(
                command +
                ['--work-dir',
                 str(run), '--stop-after', '2', '--no-validate'],
                output / (direction + '_train.log'))
        stopped = json.loads((run / 'early_stop.json').read_text())
        assert stopped['completed_iterations'] == 2
        assert stopped['uda_local_iter'] == 2
        assert stopped['configured_max_iters'] == 50000
        checkpoint = run / 'iter_2.pth'
        assert checkpoint.is_file()
        # Evaluate exactly two actual target/source slices as diagnostics.
        results = {}
        for domain in ('target', 'source'):
            medical = 'mr' if (direction == 'ct2mr') == (domain
                                                         == 'target') else 'ct'
            lists = {}
            for key, suffix in (('img_list', ''), ('ann_list', '_gt')):
                source = ROOT / f'data/datalist/val_{medical}{suffix}.txt'
                destination = output / f'{medical}_{key}_two.txt'
                destination.write_text('\n'.join(
                    str((source.parent / row).resolve())
                    for row in source.read_text().splitlines()[:2]) + '\n')
                lists[key] = str(destination)
            options = [
                f'data.test.{key}={value}' for key, value in lists.items()
            ]
            metric_path = output / f'{direction}_{domain}_metrics.json'
            execute([
                'tools/test.py',
                str(config_path(direction)),
                str(checkpoint), '--eval', 'mDice', '--metrics-out',
                str(metric_path), '--options'
            ] + options, output / f'{direction}_{domain}_test.log')
            results[domain] = json.loads(metric_path.read_text())
            assert np.isfinite(results[domain]['mDice'])
            # Exercise native evaluation configs on the same split.
            native_options = [
                f'test_dataloader.dataset.{key}={value}'
                for key, value in lists.items()
            ]
            native_path = output / f'{direction}_{domain}_native_metrics.json'
            execute([
                'tools/test.py',
                str(run / 'native_config.py'),
                str(checkpoint), '--eval', 'mDice', '--metrics-out',
                str(native_path), '--options'
            ] + native_options,
                    output / f'{direction}_{domain}_native_test.log')
            native_metrics = json.loads(native_path.read_text())
            deltas = {
                key: native_metrics[key] - results[domain][key]
                for key in results[domain]
            }
            # GPU benchmark policy is intentionally nondeterministic. CPU is
            # the strict schema-equivalence control; fixed-prediction metric
            # equivalence was separately qualified in Step 5.
            cpu_legacy = output / f'{direction}_{domain}_cpu_metrics.json'
            cpu_native = output / (
                f'{direction}_{domain}_native_cpu_metrics.json')
            for config, overrides, metrics in ((config_path(direction),
                                                options, cpu_legacy),
                                               (run / 'native_config.py',
                                                native_options, cpu_native)):
                execute([
                    'tools/test.py',
                    str(config),
                    str(checkpoint), '--device', 'cpu', '--eval', 'mDice',
                    '--metrics-out',
                    str(metrics), '--options'
                ] + overrides, metrics.with_suffix('.log'))
            cpu_reference = json.loads(cpu_legacy.read_text())
            cpu_translated = json.loads(cpu_native.read_text())
            assert list(cpu_reference) == list(cpu_translated)
            np.testing.assert_allclose(
                list(cpu_reference.values()),
                list(cpu_translated.values()),
                rtol=0,
                atol=0,
                equal_nan=True)
            results[domain] = dict(
                gpu_original=results[domain],
                gpu_native=native_metrics,
                gpu_schema_deltas=deltas,
                cpu_schema_equal=True,
                cpu_metrics=json.loads(cpu_native.read_text()))
        # Public resume CLI restores state without further updates.
        # Step 6 separately qualifies uninterrupted/resumed update equivalence.
        execute([
            'tools/train.py',
            str(run / 'native_config.py'), '--resume-from',
            str(checkpoint), '--stop-after', '2', '--no-validate'
        ], output / (direction + '_resume.log'))
        resumed = json.loads((run / 'early_stop.json').read_text())
        assert resumed == stopped
        reports[direction] = dict(
            stop=stopped,
            checkpoint=str(checkpoint),
            diagnostics_two_slices=results,
            resume_passed=True)
    return reports


def verify_utilities(output):
    from mmseg.apis import inference_segmentor, init_segmentor
    from mmseg.datasets import build_dataset
    from mmseg.datasets.collate import uda_collate
    from tools.tsne_prelogit import (build_model, forward_model,
                                     register_prelogit_hook)

    torch.set_num_threads(2)
    source = Config.fromfile(str(config_path('ct2mr')))
    native = translate_legacy_config(source)
    before = deepcopy(source.to_dict())
    checkpoint = output / 'smoke_ct2mr/iter_2.pth'
    model = init_segmentor(source, str(checkpoint), device='cpu')
    assert source.to_dict() == before
    assert len(model.CLASSES) == 5 and len(model.PALETTE) == 5
    dataset = build_dataset(source.data.test)
    image = dataset.img_infos[0]['filename']
    original_prediction = inference_segmentor(model, image)
    model.cfg = native
    native_prediction = inference_segmentor(model, image)
    assert np.array_equal(original_prediction[0], native_prediction[0])
    missing = dict(model.state_dict())
    missing.pop(next(iter(missing)))
    malformed = output / 'missing_student_key.pth'
    torch.save(dict(state_dict=missing), malformed)
    from mmseg.apis.checkpoint import load_student_checkpoint
    try:
        load_student_checkpoint(model, str(malformed))
    except RuntimeError:
        pass
    else:
        raise AssertionError('Missing student weight was accepted')
    del model, missing
    malformed.unlink()
    gc.collect()
    model = build_model(native, str(checkpoint), dataset, device='cpu')
    holder, handle = register_prelogit_hook(model)
    forward_model(model, uda_collate([dataset[0]]), device='cpu')
    features = holder['features']
    assert features.shape[0:2] == (1, 768)
    assert torch.isfinite(features).all()
    feature_shape = list(features.shape)
    handle.remove()
    del model, features, holder, dataset
    gc.collect()
    execute(
        ['tools/get_param_count.py', '--config',
         str(config_path('ct2mr'))], output / 'parameter_count.log')
    execute([
        'tools/print_config.py',
        str(config_path('ct2mr')), '--native', '--out',
        str(output / 'printed_native.py')
    ], output / 'print_native.log')
    printed = Config.fromfile(str(output / 'printed_native.py'))
    assert experiment_view(printed) == experiment_view(native)
    return dict(
        npy_inference_schema_predictions_equal=True,
        input_config_not_mutated=True,
        strict_missing_key_rejected=True,
        prelogit_shape=feature_shape,
        parameter_count_cli=True,
        native_print_cli=True)


def verify_validation(output):
    """Exercise native validation/best selection via the original CLI."""
    run = output / 'validation_ct2mr'
    overrides = []
    for key in ('img_list', 'ann_list'):
        filename = output / f'mr_{key}_two.txt'
        overrides.append(f'data.val.{key}={filename}')
    execute([
        'tools/train.py',
        str(config_path('ct2mr')), '--work-dir',
        str(run), '--stop-after', '1', '--options'
    ] + overrides, output / 'validation_ct2mr.log')
    stopped = json.loads((run / 'early_stop.json').read_text())
    assert stopped['configured_max_iters'] == 50000
    assert stopped['uda_local_iter'] == stopped['completed_iterations'] == 1
    best = list(run.glob('best_target_mDice_iter_1.pth'))
    assert len(best) == 1 and (run / 'iter_1.pth').is_file()
    # Repeat identical original-schema evaluation on the MR->CT checkpoint.
    checkpoint = output / 'smoke_mr2ct/iter_2.pth'
    result_path = output / 'mr2ct_target_repeat_metrics.json'
    execute([
        'tools/test.py',
        str(config_path('mr2ct')),
        str(checkpoint), '--eval', 'mDice', '--metrics-out',
        str(result_path), '--options',
        f'data.test.img_list={output / "ct_img_list_two.txt"}',
        f'data.test.ann_list={output / "ct_ann_list_two.txt"}'
    ], output / 'mr2ct_target_repeat.log')
    repeated = json.loads(result_path.read_text())
    prior = json.loads(
        (output / 'cuda.json'
         ).read_text())['cuda']['mr2ct']['diagnostics_two_slices']['target']
    return dict(
        stop=stopped,
        best_checkpoint=str(best[0]),
        gpu_repeat_same_original_config=repeated,
        prior_original_mDice=prior['gpu_original']['mDice'],
        prior_native_mDice=prior['gpu_native']['mDice'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--cuda-only', action='store_true')
    parser.add_argument('--reuse-smoke', action='store_true')
    parser.add_argument('--utilities-only', action='store_true')
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument(
        '--output', default=str(BASE / 'step7_entrypoints_20261011'))
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    assert subprocess.check_output(['git', 'branch', '--show-current'],
                                   text=True).strip() == 'mmcv2'
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    if args.validation_only:
        assert torch.cuda.is_available(), 'Validation requires the GPU host'
        result['validation'] = verify_validation(output)
    elif args.utilities_only:
        result['utilities'] = verify_utilities(output)
    elif not args.cuda_only:
        for entrypoint in ('tools/train.py', 'tools/test.py',
                           'run_experiments.py', 'tools/print_config.py',
                           'tools/get_param_count.py',
                           'tools/tsne_prelogit.py'):
            execute([entrypoint, '--help'],
                    output / (Path(entrypoint).stem + '_help.log'))
        result['translation'] = {
            direction: verify_translation(direction, output)
            for direction in ('ct2mr', 'mr2ct')
        }
    if args.cuda or args.cuda_only:
        result['cuda'] = verify_cuda(output, args.reuse_smoke)
    filename = ('validation.json' if args.validation_only else
                'utilities.json' if args.utilities_only else
                'cuda.json' if args.cuda_only else 'verification.json')
    (output / filename).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
