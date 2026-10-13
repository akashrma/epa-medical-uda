"""Reproduce a supplied legacy checkpoint and save fixed parity inputs."""

import argparse
import json
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, collate

from mmseg.apis import set_random_seed
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from tools.audit_legacy_baseline import sha256, write_json


def localize_lists(value, changes):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('img_list', 'ann_list'):
                path = ROOT / item
                if not path.is_file():
                    replacement = ROOT / 'data/datalist' / Path(item).name
                    if not replacement.is_file():
                        raise FileNotFoundError(item)
                    value[key] = str(replacement.relative_to(ROOT))
                    changes.append(dict(old=item, new=value[key]))
            else:
                localize_lists(item, changes)
    elif isinstance(value, (list, tuple)):
        for item in value:
            localize_lists(item, changes)


def capture_inputs(cfg, out_dir):
    set_random_seed(cfg.seed, deterministic=False)
    dataset = build_dataset(cfg.data.train)
    n_target = len(dataset.target)
    index = (len(dataset.source) // 2) * n_target + n_target // 2
    indices = list(range(index, index + cfg.data.samples_per_gpu))
    samples = [dataset[i] for i in indices]
    packed = collate(samples, samples_per_gpu=len(samples))
    batch = {key: value.data[0] for key, value in packed.items()}
    torch.save(batch, str(out_dir / 'fixed_training_batch.pt'))
    raw = {}
    for i, sample in enumerate(samples):
        for domain, key in (('source', 'img_metas'), ('target',
                                                      'target_img_metas')):
            path = Path(sample[key].data['filename'])
            raw['{}_{}'.format(domain, i)] = np.load(str(path))
        path = dataset.source.img_infos[indices[i] //
                                        n_target]['ann']['seg_map']
        raw['source_labels_{}'.format(i)] = np.load(path)
    np.savez_compressed(str(out_dir / 'fixed_raw_inputs.npz'), **raw)
    write_json(
        out_dir / 'fixed_batch.json',
        dict(
            seed=cfg.seed,
            indices=indices,
            generation=
            'Direct dataset indexing then legacy collation, not loader RNG',
            total_uda_pairs=len(dataset),
            source_filenames=[m['filename'] for m in batch['img_metas']],
            target_filenames=[
                m['filename'] for m in batch['target_img_metas']
            ],
            tensor_shapes={
                k: list(v.shape)
                for k, v in batch.items() if torch.is_tensor(v)
            },
            batch_sha256=sha256(out_dir / 'fixed_training_batch.pt')))
    return batch


def evaluate(model, dataset_cfg, out_dir, name):
    dataset = build_dataset(dataset_cfg, dict(test_mode=True))
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        num_gpus=1,
        dist=False,
        shuffle=False)
    results = []
    started = time.monotonic()
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            results.extend(model(return_loss=False, rescale=True, **batch))
            if (i + 1) % 200 == 0 or i + 1 == len(loader):
                print(
                    '{}: {} / {} slices'.format(name, i + 1, len(loader)),
                    flush=True)
    elapsed = time.monotonic() - started
    metrics = ['mDice', 'mASSD'] if dataset_cfg.type == 'MedNpzVolumeDataset' \
        else ['mDice']
    measured = dataset.evaluate(results, metric=metrics)
    predictions = out_dir / (name + '_predictions.npz')
    np.savez_compressed(
        str(predictions), predictions=np.stack(results).astype(np.uint8))
    write_json(
        out_dir / (name + '_metrics.json'),
        dict(
            metrics=dict(measured),
            dataset_type=dataset_cfg.type,
            dataset=dict(dataset_cfg),
            slices=len(dataset),
            inference_seconds=elapsed,
            prediction_sha256=sha256(predictions),
            prediction_file=str(predictions.relative_to(ROOT))))
    return dict(measured)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--source-domain', choices=['ct', 'mr'], required=True)
    args = parser.parse_args()
    branch = subprocess.check_output(['git', 'branch', '--show-current'],
                                     cwd=str(ROOT),
                                     text=True).strip()
    if branch != 'mmcv2':
        parser.error('All migration edits and progress must occur on mmcv2')
    if not torch.cuda.is_available():
        parser.error('CUDA unavailable; run with host GPU access')
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.relative_to(ROOT / 'work_dirs')
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cfg = mmcv.Config.fromfile(args.config)
    relocations = []
    localize_lists(cfg._cfg_dict, relocations)
    cfg.model.pretrained = None
    if 'model' in cfg.uda:
        cfg.uda.model.pretrained = None
    cfg.dump(str(out_dir / 'evaluation_config.py'))
    set_random_seed(cfg.seed, deterministic=False)
    torch.backends.cudnn.benchmark = bool(cfg.get('cudnn_benchmark', False))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = checkpoint['state_dict']
    student = {
        key[len('model.'):]: value
        for key, value in state.items() if key.startswith('model.')
    }
    model_cfg = deepcopy(cfg.model)
    model_cfg.train_cfg = None
    model = build_segmentor(model_cfg)
    model.load_state_dict(student, strict=True)
    historical = checkpoint.get('meta', {}).get('hook_msgs',
                                                {}).get('best_score')
    del checkpoint, state, student
    model = model.cuda().eval()
    batch = capture_inputs(cfg, out_dir)
    with torch.no_grad():
        logits = model.encode_decode(batch['img'].cuda(), batch['img_metas'])
        np.save(str(out_dir / 'fixed_source_logits.npy'), logits.cpu().numpy())
    wrapped = MMDataParallel(model, device_ids=[0])
    target = evaluate(wrapped, cfg.data.test, out_dir, 'target')
    source_cfg = deepcopy(cfg.data.test)
    source_cfg.type = 'MedNpyListDataset'
    source_cfg.img_list = 'data/datalist/val_{}.txt'.format(args.source_domain)
    source_cfg.ann_list = 'data/datalist/val_{}_gt.txt'.format(
        args.source_domain)
    for key in ('slice_axis', 'flip_axes', 'eval_name'):
        source_cfg.pop(key, None)
    # Source reporting uses the legacy NPY validation preprocessing, even when
    # the historical target uses NPZ volumes. This is a diagnostic split.
    source_cfg.pipeline = [
        dict(type='LoadNpyImageFromFile', input_range='-1_1', to_bgr=True),
        dict(
            type='MultiScaleFlipAug',
            img_scale=cfg.crop_size,
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=False),
                dict(type='Normalize', **dict(cfg.img_norm_cfg)),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ]
    source = evaluate(wrapped, source_cfg, out_dir, 'source')
    write_json(
        out_dir / 'summary.json',
        dict(
            config=args.config,
            checkpoint=args.checkpoint,
            checkpoint_sha256=sha256(Path(args.checkpoint)),
            historical_target_mDice=historical,
            reproduced_target=target,
            diagnostic_source=source,
            delta_from_metadata=None
            if historical is None else target['mDice'] - historical,
            relocations=relocations,
            strict_student_load=True,
            seed=cfg.seed,
            device=torch.cuda.get_device_name(0),
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            caveat=
            'Existing best checkpoint evaluation; no new training performed'))
    print('Reference capture complete: {}'.format(out_dir), flush=True)


if __name__ == '__main__':
    main()
