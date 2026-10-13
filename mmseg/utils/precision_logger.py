# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Copyright (c) OpenMMLab. All rights reserved.
# Modifications: MMEngine hooks; full-precision JSON and correct step counters.

import json
from mmengine.hooks import LoggerHook
from pathlib import Path

from mmseg.registry import HOOKS


@HOOKS.register_module()
class PrecisionLoggerHook(LoggerHook):
    """Native text/visualizer logging plus unrounded JSON training metrics."""

    def before_run(self, runner):
        super().before_run(runner)
        self.precision_json_path = Path(runner.work_dir,
                                        runner.timestamp + '.log.json')

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self.current_lr = runner.optim_wrapper.optimizer.param_groups[0]['lr']

    def after_train_iter(self,
                         runner,
                         batch_idx,
                         data_batch=None,
                         outputs=None):
        super().after_train_iter(runner, batch_idx, data_batch, outputs)
        if runner.rank != 0 or (batch_idx + 1) % self.interval:
            return
        record = dict(mode='train', iter=batch_idx + 1, lr=self.current_lr)
        for key in outputs or {}:
            record[key] = float(
                runner.message_hub.get_scalar('train/' + key).mean(
                    self.interval))
        self._append(record)

    def after_val_epoch(self, runner, metrics=None):
        super().after_val_epoch(runner, metrics)
        if runner.rank == 0:
            self._append(dict(mode='val', iter=runner.iter, **(metrics or {})))

    def _append(self, record):
        with self.precision_json_path.open('a') as stream:
            stream.write(json.dumps(record) + '\n')
