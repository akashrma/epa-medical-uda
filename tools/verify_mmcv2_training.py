"""Qualify native iteration training and resume on branch mmcv2.

Control checks use synthetic RNG-sensitive inputs. CUDA checks use actual
four-pair medical batches and the unchanged EPA/FDA model settings. Boundary
checks deliberately initialize local_iter=1999; they test control flow, not
the accuracy of a model trained for 1999 updates.
"""

import argparse
import gc
import json
import logging
import numpy as np
import os
import random
import subprocess
import sys
import torch
from copy import deepcopy
from functools import partial
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.model import BaseModel
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmseg.datasets.builder import worker_init_fn
from mmseg.datasets.samplers import LegacyRandomSampler
from mmseg.engine import EPARunner, LegacyPolyLR, UDAOptimWrapper
from mmseg.engine.state import (ReplayDataIterator, capture_rng_state,
                                restore_rng_state)
from mmseg.registry import DATASETS, HOOKS, MODELS
from mmseg.utils import get_root_logger

BASE = ROOT / 'work_dirs/migration_baseline'
AUDIT = BASE / 'step1_audit_20261011'
OUT = BASE / 'step6_training_20261011'
SELECTED = ('backbone.patch_embed1.proj.weight',
            'backbone.block1.0.attn.q.weight',
            'decode_head.linear_fuse.conv.weight',
            'decode_head.linear_pred.weight')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_nested_equal(a, b, name='state'):
    if torch.is_tensor(a):
        assert torch.equal(a.cpu(), b.cpu()), name
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), name
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), name
        for key in a:
            assert_nested_equal(a[key], b[key], name + '/' + str(key))
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), name
        for i, (x, y) in enumerate(zip(a, b)):
            assert_nested_equal(x, y, name + '/' + str(i))
    else:
        assert a == b, (name, a, b)


@DATASETS.register_module()
class RandomDataset(Dataset):
    """Transform draws expose parent versus worker RNG mistakes."""

    def __len__(self):
        return 28

    def __getitem__(self, index):
        return dict(
            index=index,
            python=random.random(),
            numpy=np.random.rand(),
            torch_draws=torch.rand(2))


def verify_loader(workers):
    checks = []
    for consumed in (2, 7, 9):
        seed_all()
        dataset = RandomDataset()
        loader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=workers,
            pin_memory=False,
            sampler=LegacyRandomSampler(dataset),
            worker_init_fn=partial(
                worker_init_fn, num_workers=workers, rank=0, seed=0))
        driver = ReplayDataIterator(loader)
        for _ in range(consumed):
            next(driver)
            # Model-side stochastic operations occur between batches.
            random.random()
            np.random.rand()
            torch.rand(3)
        state = driver.state_dict()
        parent = capture_rng_state()
        expected = []
        for _ in range(9):
            expected.append(next(driver))
            random.random()
            np.random.rand()
            torch.rand(3)
        expected_rng = capture_rng_state()
        driver.close()
        restore_rng_state(parent)
        resumed = ReplayDataIterator(loader, state)
        actual = []
        for _ in range(9):
            actual.append(next(resumed))
            random.random()
            np.random.rand()
            torch.rand(3)
        assert_nested_equal(actual, expected, 'batches')
        assert_nested_equal(capture_rng_state(), expected_rng, 'parent_rng')
        resumed.close()
        checks.append(
            dict(
                consumed=consumed,
                replayed_batches=9,
                epoch_at_checkpoint=state['epoch'],
                exact=True))
    return dict(num_workers=workers, cases=checks)


def verify_schedule():
    parameters = [torch.nn.Parameter(torch.zeros(1)) for _ in range(2)]
    wrapper = UDAOptimWrapper(
        torch.optim.AdamW([
            dict(params=[parameters[0]], lr=6e-5),
            dict(params=[parameters[1]], lr=6e-5 * 10)
        ]))
    scheduler = LegacyPolyLR(wrapper)
    values = []
    for iteration in range(50000):
        scheduler.set_iteration(iteration)
        values.append(scheduler.get_last_value())
    actual = np.array(values)
    assert actual.shape == (50000, 2)
    for direction in ('ct2mr', 'mr2ct'):
        expected = np.load(AUDIT / ('final_trace_' + direction + '_epa') /
                           'lr_curve.npy')
        assert np.array_equal(actual, expected)
    np.save(OUT / 'lr_curve.npy', actual)
    return dict(
        iterations=50000,
        both_legacy_curves_exact=True,
        selected={
            str(i): values[i]
            for i in (0, 1, 1499, 1500, 1999, 2000, 4999)
        })


@MODELS.register_module()
class MigrationRandomModel(BaseModel):
    """Dropout/BN and an intermittently unused head test resume semantics."""

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(4)
        self.dropout = torch.nn.Dropout(.3)
        self.linear = torch.nn.Linear(4, 1)
        self.auxiliary = torch.nn.Linear(4, 1)
        self.local_iter = 0

    def forward(self, index, python, numpy, torch_draws, mode='loss'):
        inputs = torch.cat([torch_draws, python[:, None], numpy[:, None]],
                           dim=1).float()
        inputs = self.dropout(self.norm(inputs))
        target = index[:, None].float() / 28
        losses = dict(loss_main=(self.linear(inputs) - target).square().mean())
        if self.local_iter % 2 == 0:
            losses['loss_auxiliary'] = (self.auxiliary(inputs) -
                                        target).square().mean()
        self.local_iter += 1
        return losses

    def get_training_state(self):
        return dict(local_iter=self.local_iter)

    def load_training_state(self, state):
        self.local_iter = state['local_iter']


class MigrationCPURunner(EPARunner):
    """Keep this synthetic control model on CPU even on a CUDA host."""

    def wrap_model(self, model_wrapper_cfg, model):
        model.init_weights()
        return model


def verify_cpu_runner(workers):

    def run(name, stop, resume=None):
        work = OUT / f'cpu_workers{workers}_{name}'
        cfg = Config(
            dict(
                model=dict(type='MigrationRandomModel'),
                work_dir=str(work),
                experiment_name=f'cpu_workers{workers}_{name}',
                default_scope='mmseg',
                randomness=dict(seed=0),
                train_dataloader=dict(
                    batch_size=4,
                    num_workers=workers,
                    persistent_workers=False,
                    pin_memory=False,
                    dataset=dict(type='RandomDataset'),
                    sampler=dict(type='LegacyRandomSampler'),
                    collate_fn=dict(type='default_collate'),
                    worker_init_fn=dict(
                        type='legacy_worker_init',
                        num_workers=workers,
                        rank=0,
                        seed=0)),
                train_cfg=dict(
                    type='EPAIterBasedTrainLoop',
                    max_iters=50000,
                    stop_after=stop),
                optim_wrapper=dict(
                    type='UDAOptimWrapper',
                    constructor='LegacyOptimWrapperConstructor',
                    optimizer=dict(type='AdamW', lr=6e-5, weight_decay=.01)),
                param_scheduler=dict(
                    type='LegacyPolyLR', total_iters=50000, by_epoch=False),
                default_hooks=dict(
                    logger=dict(type='PrecisionLoggerHook', interval=1),
                    param_scheduler=dict(type='LegacyParamSchedulerHook'),
                    checkpoint=dict(
                        type='EPACheckpointHook',
                        by_epoch=False,
                        interval=2,
                        save_last=False)),
                custom_hooks=[
                    dict(type='LegacyPrecisionHook'),
                    dict(type='EPATrainingStateHook')
                ],
                log_processor=dict(by_epoch=False, window_size=1),
                log_level='WARNING',
                resume=resume is not None,
                load_from=str(resume) if resume else None))
        runner = MigrationCPURunner.from_cfg(cfg)
        runner.train()
        assert runner.iter == runner.model.local_iter == stop
        assert runner.optim_wrapper._inner_count == stop
        return work / f'iter_{stop}.pth'

    uninterrupted = run('uninterrupted', 10)
    run('stopped', 2)
    resumed = run('resumed', 10,
                  OUT / f'cpu_workers{workers}_stopped/iter_2.pth')
    return compare_checkpoints(uninterrupted, resumed)


@DATASETS.register_module()
class MigrationSavedBatch(Dataset):
    """Real fixed crops; duplicated pairs make a small diagnostic epoch."""

    def __init__(self, batch_filename):
        self.batch_filename = batch_filename
        self.batch = torch.load(batch_filename, map_location='cpu')

    def __len__(self):
        return 12

    def __getitem__(self, index):
        return {key: value[index % 4] for key, value in self.batch.items()}


@HOOKS.register_module()
class MigrationTrainingObserver(Hook):
    """Assertions observe existing tensors without drawing extra RNG."""

    priority = 'NORMAL'

    def __init__(self,
                 direction,
                 report,
                 artificial_start=0,
                 deterministic_control=False):
        self.direction = direction
        self.report = Path(report)
        self.artificial_start = artificial_start
        self.deterministic_control = deterministic_control
        self.records = []

    def before_run(self, runner):
        if self.deterministic_control:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
        model = runner.model
        legacy = torch.load(
            AUDIT / ('final_trace_' + self.direction + '_epa') /
            'initialized_state_dict.pt',
            map_location='cpu')
        current = model.state_dict()
        assert list(legacy) == list(current) and len(current) == 2143
        for name in legacy:
            assert_nested_equal(legacy[name], current[name], name)
        names = {id(p): name for name, p in model.named_parameters()}
        old = json.loads((AUDIT / ('final_trace_' + self.direction + '_epa') /
                          'optimizer_groups.json').read_text())
        groups = runner.optim_wrapper.optimizer.param_groups
        assert len(groups) == len(old) == 2134
        for group, expected in zip(groups, old):
            assert [names[id(p)] for p in group['params']] == \
                expected['parameters']
            assert group['initial_lr'] == expected['lr']
            for key in ('weight_decay', 'eps', 'amsgrad'):
                assert group[key] == expected[key]
            assert list(group['betas']) == expected['betas']
            assert group['foreach'] is False and group['fused'] is False
        self.events = []
        wrapper = runner.optim_wrapper
        clear = wrapper.zero_grad
        backward = wrapper.backward
        step = wrapper.step

        def zero_grad(**kwargs):
            assert kwargs == dict(set_to_none=False)
            self.events.append('zero_grad')
            return clear(**kwargs)

        def pass_backward(loss, **kwargs):
            assert loss.isfinite()
            self.events.append('backward')
            return backward(loss, **kwargs)

        def pass_step(**kwargs):
            self.events.append('step')
            assert all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters())
            return step(**kwargs)

        wrapper.zero_grad = zero_grad
        wrapper.backward = pass_backward
        wrapper.step = pass_step
        # The wrappers hold closures over the runner model; release them
        # after training so sequential checks do not retain CUDA models.
        self.original_methods = (clear, backward, step)

    def before_train(self, runner):
        if runner._resume:
            saved = torch.load(runner._load_from, map_location='cpu')
            assert_nested_equal(saved['state_dict'], runner.model.state_dict(),
                                'loaded_model')
            assert_nested_equal(saved['optimizer'],
                                runner.optim_wrapper.state_dict(),
                                'loaded_optimizer')
            assert_nested_equal(saved['epa_training']['uda'],
                                runner.model.get_training_state(),
                                'loaded_uda')
            self.loaded_state_exact = True
            del saved
        if self.artificial_start and not runner._resume:
            model = runner.model
            model._init_ema_weights()
            model.local_iter = self.artificial_start
            loop = runner.train_loop
            loop._iter = self.artificial_start
            runner.optim_wrapper._inner_count = self.artificial_start
            driver = ReplayDataIterator(loop.dataloader)
            loop.resume_data_state = driver.state_dict()
            loop.resume_rng_state = capture_rng_state()
            driver.close()

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        model = runner.model
        assert model.local_iter == runner.iter == batch_idx
        assert data_batch['img'].shape == (4, 3, 256, 256)
        assert data_batch['target_img'].shape == (4, 3, 256, 256)
        self.events.clear()
        student = dict(model.model.named_parameters())
        teacher = dict(model.ema_model.named_parameters())
        alpha = min(1 - 1 / (model.local_iter + 1), model.alpha)
        self.expected_ema = {
            name:
            (student[name].detach().clone()
             if model.local_iter == 0 else alpha * teacher[name].detach() +
             (1 - alpha) * student[name].detach())
            for name in SELECTED
        }
        self.teacher_bn = int(
            model.ema_model.decode_head.linear_fuse.bn.num_batches_tracked)

    def after_train_iter(self,
                         runner,
                         batch_idx,
                         data_batch=None,
                         outputs=None):
        model = runner.model
        backwards = 2 if batch_idx < 2000 else 4
        assert self.events == ['zero_grad'
                               ] + ['backward'] * backwards + ['step']
        assert model.local_iter == batch_idx + 1
        assert all(p.grad is None for p in model.ema_model.parameters())
        assert all(not p.requires_grad for p in model.ema_model.parameters())
        teacher = dict(model.ema_model.named_parameters())
        for name in SELECTED:
            assert torch.equal(teacher[name], self.expected_ema[name]), name
        bn = model.ema_model.decode_head.linear_fuse.bn
        assert bn.training and int(
            bn.num_batches_tracked) == self.teacher_bn + 1
        for module in model.ema_model.modules():
            if isinstance(module, torch.nn.modules.dropout._DropoutNd) or \
                    type(module).__name__ == 'DropPath':
                assert not module.training
        assert model.etf_prototypes.shape == (5, 768)
        assert model.etf_prototypes.isfinite().all()
        assert all(np.isfinite(v) for v in outputs.values())
        target = [
            name for name in outputs if name.startswith(('tgt.', 'mix.'))
        ]
        assert bool(target) == (batch_idx >= 2000)
        self.records.append(
            dict(
                iteration=batch_idx,
                completed=batch_idx + 1,
                events=list(self.events),
                losses=outputs,
                backbone_lr=runner.optim_wrapper.optimizer.param_groups[0]
                ['lr'],
                target_loss_keys=target,
                teacher_bn_count=int(bn.num_batches_tracked),
                ema_exact=True,
                gradients_finite=True))
        self.expected_ema.clear()

    def after_val_epoch(self, runner, metrics):
        assert 'target/mDice' in metrics
        self.records[-1]['target_validation'] = metrics

    def after_train(self, runner):
        write_json(
            self.report,
            dict(
                config=self.direction,
                seed=0,
                initialized_tensors_exact=2143,
                optimizer_groups_exact=2134,
                configured_horizon=runner.max_iters,
                artificial_start=self.artificial_start,
                records=self.records,
                loaded_state_exact=getattr(self, 'loaded_state_exact', None),
                deterministic_control=self.deterministic_control,
                completed=runner.iter,
                uda_local_iter=runner.model.local_iter,
                optimizer_update_counter=runner.optim_wrapper._inner_count,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
        wrapper = runner.optim_wrapper
        clear, backward, step = self.original_methods
        wrapper.zero_grad = clear
        wrapper.backward = backward
        wrapper.step = step
        self.original_methods = None


def diagnostic_val_cfg(cfg, direction):
    fixture = OUT / 'validation_lists'
    fixture.mkdir(exist_ok=True)
    val = deepcopy(dict(cfg.data.val))
    for field in ('img_list', 'ann_list'):
        original = ROOT / val[field]
        lines = original.read_text().splitlines()[:2]
        # Resolve as the original loader does before changing the list path.
        paths = [(original.parent / line).resolve() for line in lines]
        assert all(path.exists() for path in paths), (field, paths)
        destination = fixture / f'{direction}_{field}.txt'
        destination.write_text(''.join(str(path) + '\n' for path in paths))
        val[field] = str(destination)
    return val


def native_config(direction,
                  name,
                  stop_after,
                  resume=None,
                  artificial_start=0,
                  deterministic_control=False):
    original = Config.fromfile(
        str(ROOT / 'configs/medical' / (direction + '_selftrain_nomix_etf_tgt_'
                                        'dicece_segformer_mitb5_fda.py')))
    work = OUT / name
    model = deepcopy(dict(original.uda))
    model.update(model=deepcopy(dict(original.model)), max_iters=50000)
    model['model']['train_cfg'].update(
        work_dir=str(work), log_config=deepcopy(original.log_config))
    optimizer = deepcopy(dict(original.optimizer))
    paramwise = optimizer.pop('paramwise_cfg')
    reference = 'reference_ct2mr' if direction == 'ct2mr' else \
        'reference_mr2ct_current_slices'
    val = diagnostic_val_cfg(original, direction)
    return Config(
        dict(
            model=model,
            work_dir=str(work),
            default_scope='mmseg',
            experiment_name=name,
            randomness=dict(seed=0, deterministic=False),
            env_cfg=dict(
                cudnn_benchmark=True,
                mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)),
            train_dataloader=dict(
                batch_size=4,
                num_workers=4,
                persistent_workers=False,
                pin_memory=False,
                drop_last=False,
                dataset=dict(
                    type='MigrationSavedBatch',
                    batch_filename=str(AUDIT / reference /
                                       'fixed_training_batch.pt')),
                sampler=dict(type='LegacyRandomSampler'),
                collate_fn=dict(type='uda_collate'),
                worker_init_fn=dict(
                    type='legacy_worker_init', num_workers=4, rank=0, seed=0)),
            train_cfg=dict(
                type='EPAIterBasedTrainLoop',
                max_iters=50000,
                stop_after=stop_after,
                val_interval=1),
            val_dataloader=dict(
                batch_size=1,
                num_workers=4,
                persistent_workers=False,
                pin_memory=False,
                dataset=val,
                sampler=dict(type='DefaultSampler', shuffle=False),
                collate_fn=dict(type='uda_collate')),
            val_cfg=dict(type='ValLoop'),
            val_evaluator=dict(
                type='MedicalMetric', dataset=val, prefix='target'),
            optim_wrapper=dict(
                type='UDAOptimWrapper',
                constructor='LegacyOptimWrapperConstructor',
                optimizer=optimizer,
                paramwise_cfg=paramwise),
            param_scheduler=dict(
                type='LegacyPolyLR',
                total_iters=50000,
                by_epoch=False,
                warmup_iters=1500,
                warmup_ratio=1e-6),
            default_hooks=dict(
                param_scheduler=dict(type='LegacyParamSchedulerHook'),
                checkpoint=dict(
                    type='EPACheckpointHook',
                    by_epoch=False,
                    interval=1,
                    save_last=False,
                    save_best='target/mDice',
                    rule='greater'),
                logger=dict(type='PrecisionLoggerHook', interval=1)),
            custom_hooks=[
                dict(type='LegacyPrecisionHook'),
                dict(type='EPATrainingStateHook'),
                dict(
                    type='MigrationTrainingObserver',
                    direction=direction,
                    report=str(work / 'observations.json'),
                    artificial_start=artificial_start,
                    deterministic_control=deterministic_control)
            ],
            log_processor=dict(by_epoch=False, window_size=1),
            log_level='WARNING',
            visualizer=dict(type='mmengine.Visualizer'),
            resume=resume is not None,
            load_from=str(resume) if resume else None))


def run_native(direction,
               name,
               stop_after,
               resume=None,
               artificial_start=0,
               deterministic_control=False):
    print(f'RUN {name}: stop={stop_after}, resume={resume}', flush=True)
    torch.cuda.reset_peak_memory_stats()
    runner = EPARunner.from_cfg(
        native_config(direction, name, stop_after, resume, artificial_start,
                      deterministic_control))
    runner.train()
    checkpoint = Path(runner.work_dir) / f'iter_{stop_after}.pth'
    observation = json.loads(
        (Path(runner.work_dir) / 'observations.json').read_text())
    assert runner.iter == runner.model.local_iter == stop_after
    assert runner.max_iters == runner.model.max_iters == 50000
    assert runner.optim_wrapper._inner_count == stop_after
    # Break observer and native scheduler closures before clearing CUDA.
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    return checkpoint, observation


def compare_checkpoints(expected_path, actual_path, cuda_tolerance=False):
    expected = torch.load(expected_path, map_location='cpu')
    actual = torch.load(actual_path, map_location='cpu')
    differences = []
    within_tolerance = []

    def compare(a, b, name):
        if cuda_tolerance and torch.is_tensor(a) and a.is_floating_point():
            maximum = float((a - b).abs().max()) if a.numel() else 0.
            differences.append((maximum, name))
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
            within_tolerance.append(torch.allclose(a, b, atol=1e-5, rtol=1e-4))
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), name
            for key in a:
                compare(a[key], b[key], name + '/' + str(key))
        elif isinstance(a, (list, tuple)):
            assert type(a) is type(b) and len(a) == len(b), name
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, name + '/' + str(i))
        else:
            assert_nested_equal(a, b, name)

    for key in ('state_dict', 'optimizer', 'param_schedulers', 'epa_training'):
        compare(expected[key], actual[key], key)
    assert expected['meta']['iter'] == actual['meta']['iter']
    return dict(
        uninterrupted=str(expected_path.relative_to(ROOT)),
        resumed=str(actual_path.relative_to(ROOT)),
        nonfloating_optimizer_scheduler_uda_rng_data_exact=True,
        floating_tensors_exact=not cuda_tolerance,
        floating_continuation_is_diagnostic=cuda_tolerance,
        diagnostic_allclose_passed=all(within_tolerance),
        floating_tolerance=dict(atol=1e-5, rtol=1e-4)
        if cuda_tolerance else None,
        max_differences=sorted(differences, reverse=True)[:10],
        model_tensors=len(actual['state_dict']),
        completed=actual['meta']['iter'])


def verify_rejections(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    from types import SimpleNamespace

    from mmseg.engine.hooks import EPATrainingStateHook
    hook = EPATrainingStateHook()
    result = {}
    for name, saved in (('legacy_weights',
                         dict(state_dict=checkpoint['state_dict'])),
                        ('native_best_weights',
                         dict(
                             epa_training=checkpoint['epa_training'],
                             state_dict=checkpoint['state_dict']))):
        try:
            hook.after_load_checkpoint(SimpleNamespace(_resume=True), saved)
        except ValueError as error:
            result[name] = str(error)
        else:
            raise AssertionError('Incomplete resume unexpectedly accepted')
    return result


def verify_api():
    """Exercise the public API's prebuilt-model/config serialization path."""
    from mmseg.apis import train_segmentor
    files = sorted((OUT / 'cpu_workers0_uninterrupted').glob('*.py'))
    cfg = Config.fromfile(str(files[-1]))
    cfg.work_dir = str(OUT / 'public_api')
    cfg.experiment_name = 'public_api'
    cfg.train_cfg.stop_after = 2
    cfg.resume = False
    cfg.load_from = None
    seed_all()
    model = MigrationRandomModel()
    model.init_weights()
    expected_rng = capture_rng_state()
    runner = train_segmentor(model, None, cfg, validate=False)
    assert runner.model is model
    assert runner.iter == model.local_iter == 2
    assert isinstance(runner.cfg.model, dict)
    assert not torch.equal(expected_rng['torch'], torch.get_rng_state())
    result = dict(
        prebuilt_model_identity_preserved=True,
        completed=2,
        serialized_model_config=True,
        native_api_training=True)
    write_json(OUT / 'public_api.json', result)
    return result


def verify_evaluation_smoke():
    """Report source/target diagnostics on two real validation slices each."""
    reports = {}
    for direction in ('ct2mr', 'mr2ct'):
        checkpoint = OUT / (direction + '_initial') / 'iter_2.pth'
        reports[direction] = dict(
            checkpoint=str(checkpoint.relative_to(ROOT)),
            seed=0,
            slices_per_domain=2,
            scope='Diagnostic subsets; no accuracy gate')
        for domain in ('target', 'source'):
            cfg = native_config(direction, f'{direction}_{domain}_evaluation',
                                2)
            if domain == 'source':
                reverse = 'mr2ct' if direction == 'ct2mr' else 'ct2mr'
                reverse_cfg = Config.fromfile(
                    str(ROOT / 'configs/medical' /
                        (reverse + '_selftrain_nomix_etf_tgt_dicece_'
                         'segformer_mitb5_fda.py')))
                dataset = diagnostic_val_cfg(reverse_cfg, reverse)
                cfg.val_dataloader.dataset = dataset
                cfg.val_evaluator.dataset = dataset
                cfg.val_evaluator.prefix = domain
            cfg.custom_hooks = [
                dict(type='LegacyPrecisionHook'),
                dict(type='EPATrainingStateHook')
            ]
            cfg.default_hooks.checkpoint = None
            cfg.load_from = str(checkpoint)
            if domain == 'source':
                cfg.test_dataloader = cfg.val_dataloader
                cfg.test_evaluator = cfg.val_evaluator
                cfg.test_cfg = dict(type='TestLoop')
            runner = EPARunner.from_cfg(cfg)
            reports[direction][domain] = runner.val() if domain == 'target' \
                else runner.test()
            del runner
            gc.collect()
            torch.cuda.empty_cache()
    write_json(OUT / 'evaluation_smoke.json', reports)
    return reports


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out-dir', help='Separate diagnostic artifact directory')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--control-only', action='store_true')
    parser.add_argument(
        '--reuse-initial',
        action='store_true',
        help='Reuse successful initial GPU smoke artifacts')
    parser.add_argument('--api-only', action='store_true')
    parser.add_argument('--evaluation-smoke-only', action='store_true')
    args = parser.parse_args()
    if args.out_dir:
        OUT = Path(args.out_dir).resolve()
        OUT.relative_to(BASE)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    branch = subprocess.check_output(['git', 'branch', '--show-current'],
                                     cwd=ROOT,
                                     text=True).strip()
    assert branch == 'mmcv2'
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    get_root_logger(log_level=logging.WARNING)
    if args.api_only:
        verify_api()
        return
    if args.evaluation_smoke_only:
        verify_evaluation_smoke()
        return
    result = dict(branch=branch, schedule=verify_schedule())
    result['loader_resume'] = [verify_loader(w) for w in (0, 4)]
    result['cpu_runner_resume'] = [verify_cpu_runner(w) for w in (0, 4)]
    write_json(OUT / 'control.json', result)
    print(
        'CONTROL PASSED: exact LR, loaders, and native CPU resume', flush=True)
    if args.control_only:
        return
    assert args.cuda and torch.cuda.is_available(
    ), 'Use --cuda on training host'
    free, total = torch.cuda.mem_get_info()
    assert free > 9 * 1024**3, 'Run full UDA checks when GPU is free'
    torch.cuda.set_per_process_memory_fraction(min(.90, .90 * free / total))
    result['gpu'] = dict(name=torch.cuda.get_device_name(), free_bytes=free)
    checkpoints = {}
    observations = {}
    for direction in ('ct2mr', 'mr2ct'):
        if args.reuse_initial:
            work = OUT / (direction + '_initial')
            checkpoints[direction] = work / 'iter_2.pth'
            observations[direction] = json.loads(
                (work / 'observations.json').read_text())
            assert observations[direction]['completed'] == 2
        else:
            checkpoints[direction], observations[direction] = run_native(
                direction, direction + '_initial', 2)
    resumed, observations['ct2mr_resume'] = run_native(
        'ct2mr', 'ct2mr_resume', 2, OUT / 'ct2mr_initial/iter_1.pth')
    result['early_resume'] = compare_checkpoints(
        checkpoints['ct2mr'], resumed, cuda_tolerance=True)
    result['boundary'] = {}
    for direction in ('ct2mr', 'mr2ct'):
        boundary, observations[direction + '_boundary'] = run_native(
            direction, direction + '_boundary', 2002, artificial_start=1999)
        if direction == 'ct2mr':
            restarted, observations['boundary_resume'] = run_native(
                direction,
                'ct2mr_boundary_resume',
                2002,
                OUT / 'ct2mr_boundary/iter_2000.pth',
                artificial_start=1999)
            result['boundary']['resume'] = compare_checkpoints(
                boundary, restarted, cuda_tolerance=True)
    deterministic, observations['deterministic_boundary'] = run_native(
        'ct2mr',
        'ct2mr_deterministic_boundary',
        2002,
        artificial_start=1999,
        deterministic_control=True)
    deterministic_resume, observations['deterministic_boundary_resume'] = \
        run_native('ct2mr', 'ct2mr_deterministic_boundary_resume', 2002,
                   OUT / 'ct2mr_deterministic_boundary/iter_2000.pth',
                   artificial_start=1999, deterministic_control=True)
    result['deterministic_cuda_resume'] = compare_checkpoints(
        deterministic, deterministic_resume)
    result['resume_rejections'] = verify_rejections(checkpoints['ct2mr'])
    result['runs'] = observations
    result['scope'] = (
        'Real fixed four-pair crops, four workers, tiny target '
        'validation after each update. Artificial iteration '
        '1999 checks the threshold only. Accuracy parity and '
        'the full-size data/config launcher remain later gates.')
    write_json(OUT / 'verification.json', result)
    print('STEP 6 CUDA QUALIFICATION PASSED', flush=True)


if __name__ == '__main__':
    main()
