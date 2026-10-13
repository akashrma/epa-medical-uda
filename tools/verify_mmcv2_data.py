"""Qualify Step 5 against frozen legacy preprocessing and predictions.

Run --legacy with the epa interpreter first; it imports only the verified
frozen source. Run --device cpu (or cuda) with epa-mmcv2 to check the port.
Artifacts stay under work_dirs; every invocation requires branch mmcv2.
"""

import argparse
import json
import numpy as np
import random
import subprocess
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'work_dirs/migration_baseline'
OUT = BASE / 'step5_data_20261011'
AUDIT = BASE / 'step1_audit_20261011'
NORM = dict(
    mean=[104.00698793, 116.66876762, 122.67891434],
    std=[1., 1., 1.],
    to_rgb=False)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def restore_tuples(value):
    if isinstance(value, dict):
        return {
            k:
            tuple(v) if k in ('img_scale', 'crop_size', 'size')
            and isinstance(v, list) else restore_tuples(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [restore_tuples(v) for v in value]
    return value


def fixtures():
    directory = OUT / 'fixtures'
    directory.mkdir(parents=True, exist_ok=True)
    for i, shape in enumerate(((9, 11), (5, 7), (13, 15))):
        image = np.linspace(-1, 1, np.prod(shape), dtype=np.float32)
        image = image.reshape(shape)
        label = np.indices(shape).sum(axis=0).astype(np.uint8) % 5
        label[0, 0] = 255
        np.save(str(directory / f'image{i}.npy'), image)
        np.save(str(directory / f'label{i}.npy'), label)
    (directory / 'images.txt').write_text(''.join(f'image{i}.npy\n'
                                                  for i in range(3)))
    (directory / 'labels.txt').write_text(''.join(f'label{i}.npy\n'
                                                  for i in range(3)))
    for i, shape in enumerate(((5, 7, 4), (6, 5, 3))):
        image = np.linspace(-1, 1, np.prod(shape), dtype=np.float32)
        label = np.zeros(shape, dtype=np.uint8)
        if i == 0:
            label[1:4, 2:6, 1:3] = 1
            label[2:4, 3:5, 1:3] = 2
            label[0, 0, 0] = 3
        np.savez(
            str(directory / f'volume{i}.npz'), image.reshape(shape), label)
    (directory / 'volumes.txt').write_text('volume0.npz\nvolume1.npz\n')
    pipeline = [
        dict(type='LoadNpyImageFromFile', input_range='-1_1', to_bgr=True),
        dict(type='LoadNpyAnnotations'),
        dict(type='RandomCrop', crop_size=(7, 8)),
        dict(type='RandomFlip', prob=0.5),
        dict(type='Normalize', **NORM),
        dict(type='Pad', size=(8, 9), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ]
    return directory, dict(
        type='MedNpyListDataset',
        img_list=str(directory / 'images.txt'),
        ann_list=str(directory / 'labels.txt'),
        pipeline=pipeline)


def volume_cfg(directory, axis, flips):
    return dict(
        type='MedNpzVolumeDataset',
        img_list=str(directory / 'volumes.txt'),
        slice_axis=axis,
        flip_axes=flips,
        pipeline=[
            dict(
                type='LoadNpzSliceFromFile',
                input_range='-1_1',
                flip_axes=flips,
                to_bgr=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(8, 9),
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **NORM),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ])


def predictions_for_volume(dataset):
    predictions = []
    for volume_idx, slice_idx in dataset._slice_map:
        with np.load(dataset._volume_paths[volume_idx]) as data:
            label = dataset._apply_flip(data['arr_1'])
        prediction = np.take(label, slice_idx, axis=dataset.slice_axis).copy()
        if volume_idx == 0:
            prediction[0, 0] = 4  # false positive, absent class and shape
            prediction[prediction == 2] = 1  # missing class
        predictions.append(prediction)
    return predictions


def unwrap_legacy(value):
    from mmcv.parallel import DataContainer
    if isinstance(value, DataContainer):
        return value.data[0]
    if isinstance(value, dict):
        return {k: unwrap_legacy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unwrap_legacy(v) for v in value]
    return value


def equal(left, right, path='batch'):
    if torch.is_tensor(left):
        assert torch.is_tensor(right) and left.dtype == right.dtype, path
        assert torch.equal(left, right), path
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, dict):
        assert set(left) == set(right), path
        for key in left:
            equal(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right)):
            equal(a, b, f'{path}[{i}]')
    else:
        assert left == right, (path, left, right)


def compare_metrics(actual, expected):
    assert set(actual) == set(expected), (actual.keys(), expected.keys())
    np.testing.assert_allclose(
        list(actual[k] for k in expected),
        list(expected.values()),
        rtol=0,
        atol=1e-10,
        equal_nan=True)
    return max((abs(actual[k] - expected[k])
                for k in expected if np.isfinite(expected[k])),
               default=0.)


def legacy_capture():
    frozen = BASE / 'legacy_source'
    manifest = json.loads((BASE / 'legacy_source_manifest.json').read_text())
    # Source integrity is also guarded by the isolated supervisor. Verify
    # package provenance here without changing the snapshot/manifest.
    sys.path.insert(0, str(frozen))
    import mmseg
    Path(mmseg.__file__).relative_to(frozen)
    import mmcv
    from mmcv.parallel import collate

    from mmseg.datasets import build_dataset
    assert mmcv.__version__ == '1.3.7' and manifest
    directory, cfg = fixtures()
    oracle = OUT / 'legacy'
    oracle.mkdir(exist_ok=True)
    report = dict(source=str(frozen), version=mmcv.__version__, cases={})
    from torch.utils.data import DataLoader, TensorDataset
    sampler_oracle = {}
    for size in (10, 127, 1000):
        for seed in (0, 7):
            seed_all(seed)
            loader = DataLoader(
                TensorDataset(torch.arange(size)),
                batch_size=4,
                shuffle=True,
                drop_last=True)
            order = torch.cat([batch[0] for batch in loader])
            sampler_oracle[f'{size}_{seed}'] = dict(
                order=order, rng=torch.get_rng_state())
    torch.save(sampler_oracle, str(oracle / 'sampler.pt'))
    for seed in (0, 1, 7):
        dataset = build_dataset(cfg)
        seed_all(seed)
        packed = unwrap_legacy(
            collate([dataset[i] for i in range(3)], samples_per_gpu=3))
        torch.save(packed, str(oracle / f'crop_flip_seed{seed}.pt'))
    variable_cfg = dict(
        cfg, pipeline=[t for t in cfg['pipeline'] if t['type'] != 'Pad'])
    seed_all(0)
    dataset = build_dataset(variable_cfg)
    torch.save(
        unwrap_legacy(
            collate([dataset[i] for i in range(3)], samples_per_gpu=3)),
        str(oracle / 'variable_shapes.pt'))
    # Empty classes, ignored pixels, false positives, and absent foreground.
    dataset = build_dataset(dict(cfg, test_mode=True))
    for case in ('mixed', 'background'):
        predictions = [
            np.load(info['ann']['seg_map']).copy()
            for info in dataset.img_infos
        ]
        for pred in predictions:
            if case == 'background':
                pred.fill(0)
            else:
                pred[pred == 2] = 1
                pred[1, 1] = 4
        np.savez(
            str(oracle / f'npy_{case}.npz'), **{
                f'pred{i}': p
                for i, p in enumerate(predictions)
            })
        report['cases'][f'npy_{case}'] = dict(
            dataset.evaluate(predictions, metric='mDice', logger='silent'))
    # A distinct all-background ground truth retains legacy Dice NaNs.
    bg_dir = directory / 'background'
    bg_dir.mkdir(exist_ok=True)
    np.save(str(bg_dir / 'label.npy'), np.zeros((5, 7), dtype=np.uint8))
    (bg_dir / 'labels.txt').write_text('label.npy\n')
    (bg_dir / 'images.txt').write_text('../image1.npy\n')
    bg_cfg = dict(
        cfg,
        ann_list=str(bg_dir / 'labels.txt'),
        img_list=str(bg_dir / 'images.txt'),
        test_mode=True)
    report['background_cfg'] = bg_cfg
    report['cases']['npy_all_empty'] = dict(
        build_dataset(bg_cfg).evaluate([np.zeros((5, 7), dtype=np.uint8)],
                                       logger='silent'))
    for axis in range(3):
        for flips in (None, (0, 1)):
            name = f'volume_axis{axis}_flip{flips is not None}'
            dataset = build_dataset(
                volume_cfg(directory, axis, flips), dict(test_mode=True))
            packed = unwrap_legacy(
                collate([dataset[i] for i in range(2)], samples_per_gpu=2))
            torch.save(packed, str(oracle / f'{name}.pt'))
            report['cases'][name] = dict(
                dataset.evaluate(
                    predictions_for_volume(dataset),
                    metric=['mDice', 'mASSD'],
                    logger='silent'))
    write_json(oracle / 'verification.json', report)
    print(
        'Frozen legacy crop/flip, NPY and six volume oracles saved',
        flush=True)


def verify(device, preprocessing_only=False):
    sys.path.insert(0, str(ROOT))
    from mmengine.config import Config
    from mmengine.evaluator import Evaluator
    from torch.utils.data import Subset, TensorDataset

    from mmseg.datasets import build_dataloader, build_dataset
    from mmseg.datasets.collate import uda_collate
    from mmseg.evaluation import MedicalMetric
    from mmseg.models.data_preprocessor import UDADataPreprocessor
    oracle = OUT / 'legacy'
    report = dict(branch='mmcv2', device=device, preprocessing={}, metrics={})
    directory, cfg = fixtures()
    legacy = json.loads((oracle / 'verification.json').read_text())
    for name, expected in torch.load(str(oracle / 'sampler.pt')).items():
        size, seed = (int(n) for n in name.split('_'))
        seed_all(seed)
        loader = build_dataloader(
            TensorDataset(torch.arange(size)),
            samples_per_gpu=4,
            workers_per_gpu=0,
            dist=False,
            shuffle=True,
            drop_last=True)
        order = torch.cat([batch[0] for batch in loader])
        equal(order, expected['order'], name + '.sampler')
        equal(torch.get_rng_state(), expected['rng'], name + '.rng')
    report['preprocessing']['sampler'] = 'Six shuffle orders/RNG states exact'
    for reference in ('reference_ct2mr', 'reference_mr2ct_current_slices'):
        ref = AUDIT / reference
        config = Config.fromfile(str(ref / 'evaluation_config.py'))
        fixed = json.loads((ref / 'fixed_batch.json').read_text())
        dataset = build_dataset(config.data.train)
        seed_all(fixed['seed'])
        batch = uda_collate([dataset[i] for i in fixed['indices']])
        expected = torch.load(
            str(ref / 'fixed_training_batch.pt'), map_location='cpu')
        equal(batch, expected)
        moved = UDADataPreprocessor().to(device)(batch, training=True)
        assert moved['img_metas'] is batch['img_metas']
        assert moved['target_img_metas'] is batch['target_img_metas']
        for key in ('img', 'gt_semantic_seg', 'target_img'):
            assert moved[key].device.type == device
            equal(moved[key].cpu(), expected[key], key)
        report['preprocessing'][reference] = 'All tensors/metadata exact'
        del moved, batch, expected, dataset
    for seed in (0, 1, 7):
        dataset = build_dataset(cfg)
        seed_all(seed)
        batch = uda_collate([dataset[i] for i in range(3)])
        equal(batch, torch.load(str(oracle / f'crop_flip_seed{seed}.pt')))
        assert (batch['gt_semantic_seg'] == 255).any()
        report['preprocessing'][f'crop_flip_seed{seed}'] = 'Exact'
    variable_cfg = dict(
        cfg, pipeline=[t for t in cfg['pipeline'] if t['type'] != 'Pad'])
    seed_all(0)
    dataset = build_dataset(variable_cfg)
    equal(
        uda_collate([dataset[i] for i in range(3)]),
        torch.load(str(oracle / 'variable_shapes.pt')))
    report['preprocessing']['variable_shape_padding'] = 'Exact'
    # Multiprocess loading preserves the sample order and per-image metadata;
    # Compare with direct indexing using that worker's NumPy/Python seed.
    seed_all(0)
    loader = build_dataloader(
        Subset(build_dataset(cfg), [0, 1, 2]),
        samples_per_gpu=2,
        workers_per_gpu=2,
        dist=False,
        shuffle=False,
        seed=0)
    assert not loader.persistent_workers
    batches = list(loader)
    seed_all(0)
    equal(batches[0], uda_collate([build_dataset(cfg)[i] for i in (0, 1)]))
    seed_all(1)
    equal(batches[1], uda_collate([build_dataset(cfg)[2]]))
    report['preprocessing']['workers'] = 'Both worker seeds/batches exact'
    # Paired optional mask and synchronized crop maintain ordinary tensors.
    pair_cfg = dict(
        type='UDADataset', source=cfg, target=cfg, sync_crop_size=(4, 5))
    pair = build_dataset(pair_cfg)
    seed_all(0)
    batch = uda_collate([pair[0], pair[4]])
    assert batch['img'].shape == (2, 3, 4, 5)
    assert batch['target_img'].shape == (2, 3, 4, 5)
    mask = torch.ones(1, 4, 5, dtype=torch.uint8)
    packed = uda_collate([dict(valid_pseudo_mask=mask)] * 2)
    assert packed['valid_pseudo_mask'].shape == (2, 1, 4, 5)
    report['preprocessing']['optional_keys_sync_crop'] = 'Passed'
    if preprocessing_only:
        write_json(OUT / f'preprocessing_{device}.json', report)
        print('Step 5 preprocessing and device checks passed', flush=True)
        return
    for case in ('mixed', 'background'):
        with np.load(oracle / f'npy_{case}.npz') as arrays:
            predictions = [arrays[k] for k in arrays.files]
        measured = build_dataset(dict(cfg, test_mode=True)).evaluate(
            predictions, logger='silent')
        report['metrics'][f'npy_{case}'] = compare_metrics(
            measured, legacy['cases'][f'npy_{case}'])
    measured = build_dataset(restore_tuples(
        legacy['background_cfg'])).evaluate([np.zeros((5, 7), dtype=np.uint8)],
                                            logger='silent')
    report['metrics']['npy_all_empty'] = compare_metrics(
        measured, legacy['cases']['npy_all_empty'])
    for axis in range(3):
        for flips in (None, (0, 1)):
            name = f'volume_axis{axis}_flip{flips is not None}'
            dataset = build_dataset(
                volume_cfg(directory, axis, flips), dict(test_mode=True))
            equal(
                uda_collate([dataset[i] for i in range(2)]),
                torch.load(str(oracle / f'{name}.pt')))
            evaluator = Evaluator(MedicalMetric(dataset, ['mDice', 'mASD']))
            evaluator.dataset_meta = dataset.metainfo
            evaluator.process(predictions_for_volume(dataset))
            measured = evaluator.evaluate(len(dataset))
            report['metrics'][name] = compare_metrics(
                {
                    k.split('/', 1)[1]: v
                    for k, v in measured.items()
                }, legacy['cases'][name])
            assert not evaluator.metrics[0].results
            report['preprocessing'][name] = 'TTA tensors and metadata exact'
    for reference in ('reference_ct2mr', 'reference_mr2ct',
                      'reference_mr2ct_current_slices'):
        for domain in ('source', 'target'):
            if reference == 'reference_mr2ct_current_slices' and \
                    domain == 'source':
                continue  # Identical to historical MR source, already checked.
            ref = AUDIT / reference
            saved = json.loads((ref / f'{domain}_metrics.json').read_text())
            dataset = build_dataset(
                restore_tuples(saved['dataset']), dict(test_mode=True))
            metrics = ['mDice', 'mASSD'] if saved['dataset_type'] == \
                'MedNpzVolumeDataset' else ['mDice']
            evaluator = Evaluator(
                MedicalMetric(dataset, metrics, prefix=domain))
            evaluator.dataset_meta = dataset.metainfo
            with np.load(ref / f'{domain}_predictions.npz') as arrays:
                predictions = arrays['predictions']
            for start in range(0, len(predictions), 32):
                evaluator.process(list(predictions[start:start + 32]))
            measured = evaluator.evaluate(len(dataset))
            actual = {k.split('/', 1)[1]: v for k, v in measured.items()}
            report['metrics'][f'{reference}/{domain}'] = dict(
                max_absolute_delta=compare_metrics(actual, saved['metrics']),
                slices=len(dataset),
                values=actual)
            print(
                f'{reference}/{domain}: identical fixed-prediction metrics',
                flush=True)
            del predictions, evaluator, dataset
    write_json(OUT / f'verification_{device}.json', report)
    print('Step 5 fixed-input and fixed-prediction gates passed', flush=True)


def inference_smoke():
    sys.path.insert(0, str(ROOT))
    from mmengine.config import Config
    from torch.utils.data import Subset

    from mmseg.apis import inference_segmentor, single_gpu_test
    from mmseg.datasets import build_dataloader, build_dataset
    from mmseg.datasets.collate import uda_collate
    from mmseg.models import build_segmentor
    cfg = Config.fromfile(str(AUDIT / 'reference_ct2mr/evaluation_config.py'))
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model).eval()
    checkpoint = torch.load(
        str(ROOT / 'pretrained/CT2MR_best_mDice.pth'), map_location='cpu')
    student = {
        k[len('model.'):]: v
        for k, v in checkpoint['state_dict'].items() if k.startswith('model.')
    }
    model.load_state_dict(student, strict=True)
    del checkpoint, student
    model.cfg = cfg
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    batch = uda_collate([dataset[0]])
    with torch.no_grad():
        expected = model(return_loss=False, rescale=True, **batch)
    loader = build_dataloader(
        Subset(dataset, [0]),
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False)
    actual = single_gpu_test(model, loader)
    equal(actual, expected, 'single_gpu_test')
    image = np.load(dataset.img_infos[0]['filename'])
    image = np.repeat(image[:, :, None], 3, axis=2)
    image = np.clip((image + 1.) * 127.5, 0, 255).astype(np.float32)
    actual = inference_segmentor(model, image)
    equal(actual, expected, 'inference_segmentor')
    write_json(
        OUT / 'inference_smoke.json',
        dict(
            config='reference_ct2mr/evaluation_config.py',
            checkpoint='pretrained/CT2MR_best_mDice.pth',
            strict_student_load=True,
            device='cpu',
            single_gpu_test='Exact prediction equality with direct student',
            inference_segmentor='Exact prediction equality with NPY loader',
            prediction_shape=list(actual[0].shape)))
    print('Real historical student passed both inference APIs', flush=True)


def verify_saved_reference(directory):
    """Recheck a newly captured baseline without repeating other gates."""
    sys.path.insert(0, str(ROOT))
    from mmengine.evaluator import Evaluator

    from mmseg.datasets import build_dataset
    from mmseg.evaluation import MedicalMetric
    reference = (ROOT / directory).resolve()
    reference.relative_to(BASE)
    report = dict(reference=str(reference.relative_to(ROOT)), metrics={})
    for domain in ('source', 'target'):
        saved = json.loads((reference / f'{domain}_metrics.json').read_text())
        dataset = build_dataset(
            restore_tuples(saved['dataset']), dict(test_mode=True))
        metrics = ['mDice', 'mASSD'] if saved['dataset_type'] == \
            'MedNpzVolumeDataset' else ['mDice']
        evaluator = Evaluator(MedicalMetric(dataset, metrics, prefix=domain))
        evaluator.dataset_meta = dataset.metainfo
        with np.load(reference / f'{domain}_predictions.npz') as arrays:
            predictions = arrays['predictions']
        for start in range(0, len(predictions), 32):
            evaluator.process(list(predictions[start:start + 32]))
        measured = evaluator.evaluate(len(dataset))
        actual = {k.split('/', 1)[1]: v for k, v in measured.items()}
        report['metrics'][domain] = dict(
            max_absolute_delta=compare_metrics(actual, saved['metrics']),
            slices=len(dataset),
            values=actual)
        print(
            f'{directory}/{domain}: fixed-prediction metrics match',
            flush=True)
    name = f'{reference.parent.name}_{reference.name}_metrics.json'
    write_json(OUT / name, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--inference-smoke', action='store_true')
    parser.add_argument('--reference-dir')
    parser.add_argument('--preprocessing-only', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    branch = subprocess.check_output(['git', 'branch', '--show-current'],
                                     cwd=ROOT,
                                     text=True).strip()
    if branch != 'mmcv2':
        parser.error('All migration edits and progress must occur on mmcv2')
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if args.legacy:
        legacy_capture()
    elif args.inference_smoke:
        inference_smoke()
    elif args.reference_dir:
        verify_saved_reference(args.reference_dir)
    else:
        verify(args.device, args.preprocessing_only)


if __name__ == '__main__':
    main()
