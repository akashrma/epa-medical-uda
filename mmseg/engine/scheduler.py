# Adapted from MMCV v1.3.7 mmcv/runner/hooks/lr_updater.py.
# Copyright (c) OpenMMLab. All rights reserved.
"""Native scheduler with the exact frozen polynomial/warmup arithmetic."""

from mmengine.hooks import Hook
from mmengine.optim.scheduler import _ParamScheduler

from mmseg.registry import HOOKS, PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module()
class LegacyPolyLR(_ParamScheduler):

    def __init__(self,
                 optimizer,
                 total_iters=50000,
                 power=1.,
                 min_lr=0.,
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 by_epoch=False,
                 last_step=-1,
                 begin=0,
                 end=None):
        if by_epoch or total_iters <= 0 or warmup_iters < 0:
            raise ValueError('Expected a positive iteration-based horizon')
        if begin != 0 or end not in (None, total_iters, total_iters + 1):
            raise ValueError('Scheduler boundaries must retain the horizon')
        self.total_iters = total_iters
        self.power = power
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        super().__init__(
            optimizer,
            'lr',
            end=total_iters + 1,
            last_step=last_step,
            by_epoch=False)

    def _get_value(self):
        coefficient = (1 - self.last_step / self.total_iters)**self.power
        values = [(base - self.min_lr) * coefficient + self.min_lr
                  for base in self.base_values]
        if self.last_step < self.warmup_iters:
            k = (1 - self.last_step / self.warmup_iters) * (1 -
                                                            self.warmup_ratio)
            values = [value * (1 - k) for value in values]
        return values

    def set_iteration(self, iteration):
        if not 0 <= iteration <= self.total_iters:
            raise ValueError('Iteration outside configured schedule horizon')
        self.last_step = self._global_step = iteration
        self._last_value = self._get_value()
        for group, value in zip(self.optimizer.param_groups, self._last_value):
            group['lr'] = value


@HOOKS.register_module()
class LegacyParamSchedulerHook(Hook):
    priority = 'HIGHEST'

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        # Keep the last *used* LR in checkpoints, like the legacy hook.
        # RuntimeInfoHook then records this iteration's current LR.
        for scheduler in runner.param_schedulers or []:
            if not isinstance(scheduler, LegacyPolyLR):
                raise TypeError('Expected the qualified LegacyPolyLR')
            scheduler.set_iteration(runner.iter)
