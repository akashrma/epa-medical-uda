"""Build MMEngine runtime settings from medical experiment configurations."""

from copy import deepcopy
from pathlib import Path

from mmengine.config import Config


def _check_keys(value, allowed, where):
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f'Unsupported {where} settings: {sorted(unknown)}')


def student_config(cfg):
    """Extract the unchanged student architecture from either config schema."""
    model = cfg['model']
    return deepcopy(model['model'] if model.get('type') == 'DACS' else model)


def evaluation_dataset_config(cfg, split='test'):
    if 'data' in cfg:
        return deepcopy(cfg['data'][split])
    return deepcopy(cfg[split + '_dataloader']['dataset'])


def _medical_dataset(config):
    if not isinstance(config, dict):
        raise ValueError('Medical translation needs a dataset config dict')
    if config['type'] == 'UDADataset':
        _check_keys(config, ('type', 'source', 'target', 'sync_crop_size',
                             'rare_class_sampling'), 'UDA dataset')
        _medical_dataset(config['source'])
        _medical_dataset(config['target'])
    elif config['type'] not in ('MedNpyListDataset', 'MedNpzVolumeDataset'):
        raise ValueError('Expected a supported medical dataset; '
                         f'got {config["type"]}')


def _loader(dataset, batch_size, workers, seed, training=False):
    dataset = deepcopy(dataset)
    _medical_dataset(dataset)
    if not training:
        dataset['test_mode'] = True
    return dict(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=workers,
        drop_last=training,
        pin_memory=False,
        persistent_workers=False,
        sampler=dict(type='RandomSliceSampler') if training else dict(
            type='DefaultSampler', shuffle=False),
        collate_fn=dict(type='uda_collate'),
        worker_init_fn=dict(
            type='seed_data_worker', num_workers=workers, rank=0, seed=seed))


def build_runtime_config(cfg,
                         work_dir=None,
                         stop_after=None,
                         validate=True,
                         launcher='none'):
    """Build the runtime configuration without mutating the experiment.

    Native configs pass through with runtime path/cap adjustments. Unsupported
    settings raise instead of silently losing their behavior. Training uses
    the configured horizon unless an explicit stop cap is supplied.
    """
    cfg = deepcopy(cfg.to_dict() if isinstance(cfg, Config) else dict(cfg))
    if launcher != 'none':
        raise NotImplementedError('Medical UDA training supports one GPU')
    if 'train_dataloader' in cfg:
        native = Config(cfg)
        if work_dir is not None:
            native.work_dir = str(work_dir)
        if stop_after is not None:
            native.train_cfg.stop_after = stop_after
        else:
            native.train_cfg.setdefault('stop_after',
                                        native.train_cfg.max_iters)
        if not 0 < native.train_cfg.stop_after <= native.train_cfg.max_iters:
            raise ValueError('Stop cap must be positive and within max_iters')
        model = native.model.model if native.model.type == 'DACS' else \
            native.model
        if model.get('train_cfg') is not None:
            model.train_cfg.work_dir = native.work_dir
        if not validate:
            for key in ('val_dataloader', 'val_cfg', 'val_evaluator'):
                native[key] = None
            native.default_hooks.checkpoint.save_best = None
        return native

    allowed = ('model', 'uda', 'data', 'runner', 'evaluation',
               'checkpoint_config', 'optimizer', 'optimizer_config',
               'lr_config', 'log_config', 'log_level', 'dist_params',
               'workflow', 'cudnn_benchmark', 'load_from', 'resume_from',
               'seed', 'deterministic', 'work_dir', 'norm_cfg',
               'find_unused_parameters', 'use_ddp_wrapper', 'dataset_type',
               'img_norm_cfg', 'crop_size', 'source_train_pipeline',
               'target_train_pipeline', 'train_pipeline', 'test_pipeline',
               'name', 'exp', 'git_rev', 'gpu_ids')
    _check_keys(cfg,
                set(allowed) | {k
                                for k in cfg if k.startswith('name_')},
                'top-level')
    if cfg.get('workflow', [('train', 1)]) != [('train', 1)]:
        raise ValueError('Training requires an iteration-based workflow')
    if cfg.get('deterministic', False):
        raise ValueError('The training precision policy is nondeterministic')
    if not cfg.get('cudnn_benchmark', False):
        raise ValueError('Training requires cudnn_benchmark=True')
    if cfg.get('optimizer_config') not in (None, {}):
        raise ValueError('Optimizer hooks, clipping and AMP are unsupported')
    if len(cfg.get('gpu_ids', [0])) != 1:
        raise ValueError('Training supports one GPU')
    runner = cfg['runner']
    _check_keys(runner, ('type', 'max_iters'), 'runner')
    if runner['type'] != 'IterBasedRunner':
        raise ValueError('Expected IterBasedRunner')
    horizon = int(runner['max_iters'])
    cap = horizon if stop_after is None else int(stop_after)
    if not 0 < cap <= horizon:
        raise ValueError('Stop cap must be positive and within max_iters')
    lr = cfg['lr_config']
    _check_keys(lr, ('policy', 'warmup', 'warmup_iters', 'warmup_ratio',
                     'power', 'min_lr', 'by_epoch'), 'learning-rate')
    if lr['policy'].lower() != 'poly' or lr.get('warmup') != 'linear' or \
            lr.get('by_epoch', False):
        raise ValueError('Expected iteration poly LR with linear warmup')
    evaluation = cfg.get('evaluation', {})
    _check_keys(evaluation, ('interval', 'metric', 'save_best', 'rule'),
                'evaluation')
    if evaluation.get('metric', 'mDice') != 'mDice' or \
            evaluation.get('save_best', 'mDice') not in (None, 'mDice') or \
            evaluation.get('rule', 'greater') != 'greater':
        raise ValueError('Medical checkpoint selection requires mDice/greater')
    checkpoint = cfg.get('checkpoint_config')
    if checkpoint is None:
        raise ValueError('Full stopped checkpoints are required')
    _check_keys(checkpoint, ('by_epoch', 'interval', 'max_keep_ckpts',
                             'save_last', 'save_optimizer', 'meta'),
                'checkpoint')
    if checkpoint.get('by_epoch', False) or \
            not checkpoint.get('save_optimizer', True):
        raise ValueError('Full iteration checkpoints are required')
    log = cfg['log_config']
    _check_keys(log, ('interval', 'img_interval', 'hooks'), 'logging')
    backends = [dict(type='LocalVisBackend')]
    for hook in log['hooks']:
        _check_keys(hook, ('type', 'by_epoch'), 'logger')
        if hook.get('by_epoch', False):
            raise ValueError('Logging must be iteration based')
        if hook['type'] == 'TensorboardLoggerHook':
            backends.append(dict(type='TensorboardVisBackend'))
        elif hook['type'] not in ('TextLoggerHook', 'PrecisionLoggerHook'):
            raise ValueError('Unsupported logger: ' + hook['type'])
    data = cfg['data']
    _check_keys(data,
                ('train', 'val', 'test', 'samples_per_gpu', 'workers_per_gpu'),
                'data')
    seed = cfg.get('seed', 0)
    if seed is None:
        raise ValueError('An explicit seed is required')
    workers = data['workers_per_gpu']
    work = str(
        work_dir or cfg.get('work_dir')
        or Path('work_dirs') / cfg.get('name', 'medical'))
    model = deepcopy(cfg['model'])
    if model.get('train_cfg') is None:
        model['train_cfg'] = {}
    model['train_cfg'].update(work_dir=work, log_config=deepcopy(log))
    if 'uda' in cfg:
        wrapper = deepcopy(cfg['uda'])
        if wrapper['type'] != 'DACS':
            raise ValueError('Expected the DACS UDA model')
        wrapper.update(model=model, max_iters=horizon)
        model = wrapper
    optimizer = deepcopy(cfg['optimizer'])
    if optimizer['type'] != 'AdamW':
        raise ValueError('Training supports the AdamW optimizer')
    paramwise = optimizer.pop('paramwise_cfg', {})
    val = _loader(data['val'], 1, workers, seed)
    test = _loader(data['test'], 1, workers, seed)
    native = Config(
        dict(
            default_scope='mmseg',
            model=model,
            work_dir=work,
            experiment_name=cfg.get('name', 'medical'),
            launcher=launcher,
            train_dataloader=_loader(
                data['train'],
                data['samples_per_gpu'],
                workers,
                seed,
                training=True),
            train_cfg=dict(
                type='EPAIterBasedTrainLoop',
                max_iters=horizon,
                stop_after=cap,
                val_interval=evaluation.get('interval', 500)),
            val_dataloader=val if validate else None,
            val_cfg=dict(type='ValLoop') if validate else None,
            val_evaluator=dict(
                type='MedicalMetric', dataset=val['dataset'], prefix='target')
            if validate else None,
            test_dataloader=test,
            test_cfg=dict(type='TestLoop'),
            test_evaluator=dict(
                type='MedicalMetric', dataset=test['dataset'],
                prefix='target'),
            optim_wrapper=dict(
                type='UDAOptimWrapper',
                constructor='EPAOptimWrapperConstructor',
                optimizer=optimizer,
                paramwise_cfg=paramwise),
            param_scheduler=dict(
                type='PolyWarmupLR',
                total_iters=horizon,
                power=lr.get('power', 1.),
                min_lr=lr.get('min_lr', 0.),
                warmup_iters=lr.get('warmup_iters', 1500),
                warmup_ratio=lr.get('warmup_ratio', 1e-6),
                by_epoch=False),
            default_hooks=dict(
                param_scheduler=dict(type='PolyWarmupSchedulerHook'),
                checkpoint=dict(
                    type='EPACheckpointHook',
                    by_epoch=False,
                    interval=checkpoint.get('interval', -1),
                    max_keep_ckpts=checkpoint.get('max_keep_ckpts', -1),
                    save_last=checkpoint.get('save_last', True),
                    meta=deepcopy(checkpoint.get('meta')),
                    save_best='target/mDice'
                    if validate and evaluation.get('save_best') else None,
                    rule='greater'),
                logger=dict(
                    type='PrecisionLoggerHook',
                    interval=log['interval'],
                    log_metric_by_epoch=False)),
            custom_hooks=[
                dict(type='PrecisionPolicyHook'),
                dict(type='EPATrainingStateHook')
            ],
            randomness=dict(seed=seed, deterministic=False),
            env_cfg=dict(
                cudnn_benchmark=True,
                dist_cfg=cfg.get('dist_params', {}),
                mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)),
            log_level=cfg.get('log_level', 'INFO'),
            log_processor=dict(by_epoch=False, window_size=log['interval']),
            visualizer=dict(
                type='Visualizer', _scope_='mmengine', vis_backends=backends),
            load_from=cfg.get('resume_from') or cfg.get('load_from'),
            resume=bool(cfg.get('resume_from')),
            experiment_spec=cfg))
    return native
