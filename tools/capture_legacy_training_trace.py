"""Capture initialized legacy UDA state and real updates on a saved batch."""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mmcv
import numpy as np
import torch
from mmcv.runner import build_optimizer
from mmcv.runner.hooks.lr_updater import PolyLrUpdaterHook

from mmseg.apis import set_random_seed
from mmseg.models.builder import build_train_model
from mmseg.utils import get_root_logger
from tools.audit_legacy_baseline import sha256, write_json
from tools.train import _register_linear_pred_hook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('batch')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--iterations', type=int, default=2)
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
    logger = get_root_logger(log_file=str(out_dir / 'trace.log'))
    logger.setLevel(logging.WARNING)
    cfg = mmcv.Config.fromfile(args.config)
    cfg.work_dir = str(out_dir.relative_to(ROOT))
    cfg.model.train_cfg.work_dir = cfg.work_dir
    cfg.model.train_cfg.log_config = cfg.log_config
    cfg.gpu_ids = range(1)
    cfg.dump(str(out_dir / 'trace_config.py'))
    write_json(
        out_dir / 'capture_provenance.json',
        dict(
            script_sha256=sha256(Path(__file__)),
            config_sha256=sha256(Path(args.config)),
            batch_sha256=sha256(Path(args.batch))))
    torch.backends.cudnn.benchmark = bool(cfg.get('cudnn_benchmark', False))
    set_random_seed(
        cfg.seed, deterministic=bool(cfg.get('deterministic', False)))
    model = build_train_model(cfg)
    model.init_weights()
    _register_linear_pred_hook(model)
    torch.save(model.state_dict(), str(out_dir / 'initialized_state_dict.pt'))
    model = model.cuda().train()
    optimizer = build_optimizer(model, cfg.optimizer)
    names = {
        id(parameter): name
        for name, parameter in model.named_parameters()
    }
    groups = []
    for group in optimizer.param_groups:
        settings = {
            key: value
            for key, value in group.items() if key != 'params'
        }
        settings['parameters'] = [names[id(p)] for p in group['params']]
        groups.append(settings)
    write_json(out_dir / 'optimizer_groups.json', groups)
    scheduler_cfg = dict(cfg.lr_config)
    if scheduler_cfg.pop('policy') != 'poly':
        parser.error('This capture utility supports the legacy poly schedule')
    scheduler = PolyLrUpdaterHook(**scheduler_cfg)
    runner = SimpleNamespace(
        optimizer=optimizer, max_iters=cfg.runner.max_iters, iter=0, epoch=0)
    scheduler.before_run(runner)
    # Generate the entire LR curve through the actual legacy hook, using two
    # representative optimizer groups to avoid a large repeated trace.
    lr_optimizer = torch.optim.AdamW([
        dict(params=[torch.nn.Parameter(torch.zeros(1))], lr=cfg.optimizer.lr),
        dict(
            params=[torch.nn.Parameter(torch.zeros(1))],
            lr=cfg.optimizer.lr * 10)
    ])
    lr_runner = SimpleNamespace(
        optimizer=lr_optimizer,
        max_iters=cfg.runner.max_iters,
        iter=0,
        epoch=0)
    lr_hook = PolyLrUpdaterHook(**scheduler_cfg)
    lr_hook.before_run(lr_runner)
    curve = []
    for iteration in range(cfg.runner.max_iters):
        lr_runner.iter = iteration
        lr_hook.before_train_iter(lr_runner)
        curve.append([group['lr'] for group in lr_optimizer.param_groups])
    np.save(str(out_dir / 'lr_curve.npy'), np.array(curve))
    batch = torch.load(args.batch, map_location='cpu')
    batch = {
        key: value.cuda() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    selected = [
        'model.backbone.patch_embed1.proj.weight',
        'model.backbone.block1.0.attn.q.weight',
        'model.decode_head.linear_fuse.conv.weight',
        'model.decode_head.linear_pred.weight'
    ]
    selected += [name.replace('model.', 'ema_model.', 1) for name in selected]
    selected += [
        '{}.decode_head.linear_fuse.bn.{}'.format(prefix, name)
        for prefix in ('model', 'ema_model')
        for name in ('running_mean', 'running_var', 'num_batches_tracked')
    ]
    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    traces = []
    for iteration in range(args.iterations):
        runner.iter = iteration
        scheduler.before_train_iter(runner)
        before = {
            name: params[name].detach().cpu().clone()
            for name in selected
        }
        torch.cuda.synchronize()
        started = time.monotonic()
        result = model.train_step(batch, optimizer)
        torch.cuda.synchronize()
        # Torch 1.7 replaces BatchNorm's batch counter during forward.
        params.update(dict(model.named_buffers()))
        snapshots = {}
        for name in selected:
            snapshots[name] = dict(
                before=before[name],
                after=params[name].detach().cpu().clone(),
                gradient=None
                if not isinstance(params[name], torch.nn.Parameter)
                or params[name].grad is None else
                params[name].grad.detach().cpu().clone())
        snapshot_path = out_dir / 'iteration_{}.pt'.format(iteration)
        torch.save(snapshots, str(snapshot_path))
        entry = dict(
            iter_zero_based=iteration,
            uda_local_iter=model.local_iter,
            seconds=time.monotonic() - started,
            log_vars=result['log_vars'],
            num_samples=result['num_samples'],
            learning_rates=sorted(
                {group['lr']
                 for group in optimizer.param_groups}),
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            snapshot_sha256=sha256(snapshot_path))
        traces.append(entry)
        write_json(out_dir / 'trace.json', traces)
        print(entry, flush=True)
    write_json(
        out_dir / 'summary.json',
        dict(
            config=args.config,
            config_sha256=sha256(Path(args.config)),
            seed=cfg.seed,
            batch=args.batch,
            batch_sha256=sha256(Path(args.batch)),
            initialized_state_sha256=sha256(out_dir /
                                            'initialized_state_dict.pt'),
            iterations=args.iterations,
            max_iters=cfg.runner.max_iters,
            cuda_device=torch.cuda.get_device_name(0),
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            caveat='Fixed-batch updates; full training accuracy not measured; '
            'target losses start later and are not exercised here'))


if __name__ == '__main__':
    main()
