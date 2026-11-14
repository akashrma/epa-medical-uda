# Adapted from MMEngine v0.10.7 mmengine/runner/loops.py.
# Copyright (c) OpenMMLab. All rights reserved.
"""Iteration training with a separate stop cap and resumable data stream."""

import json
from mmengine.runner import BaseLoop, IterBasedTrainLoop
from pathlib import Path

from mmseg.registry import LOOPS
from .state import ReplayDataIterator, restore_rng_state


@LOOPS.register_module()
class EPAIterBasedTrainLoop(IterBasedTrainLoop):

    def __init__(self,
                 runner,
                 dataloader,
                 max_iters=50000,
                 stop_after=None,
                 val_begin=1,
                 val_interval=500):
        # Native IterBasedTrainLoop eagerly starts loader workers. Defer that
        # until weights/optimizer/resume are ready to preserve initialization
        # and sampling RNG order and avoid an extra iterator during resume.
        BaseLoop.__init__(self, runner, dataloader)
        self._max_iters = int(max_iters)
        self._max_epochs = 1
        self._epoch = self._iter = 0
        if stop_after is None:
            stop_after = self._max_iters
        if not 0 < stop_after <= self._max_iters:
            raise ValueError('Stop cap must be within the configured horizon')
        self.stop_after = int(stop_after)
        self.val_begin = val_begin
        self.val_interval = val_interval
        self.stop_training = False
        self.dataloader_iterator = None
        self.resume_data_state = None
        self.resume_rng_state = None
        self.resume_sampler_type = None
        if hasattr(self.dataloader.dataset, 'metainfo'):
            runner.visualizer.dataset_meta = self.dataloader.dataset.metainfo

    def run(self):
        self.runner.call_hook('before_train')
        self.runner.call_hook('before_train_epoch')
        if self._iter and self.resume_data_state is None:
            raise ValueError(
                'Nonzero training iteration requires resume state')
        if self.resume_rng_state is not None:
            restore_rng_state(self.resume_rng_state)
        self.dataloader_iterator = ReplayDataIterator(self.dataloader,
                                                      self.resume_data_state,
                                                      self.resume_sampler_type)
        try:
            while self._iter < self.stop_after and not self.stop_training:
                self.runner.model.train()
                self.run_iter(next(self.dataloader_iterator))
                if self.runner.val_loop is not None and \
                        self._iter >= self.val_begin and \
                        (self._iter % self.val_interval == 0 or
                         self._iter == self.stop_after):
                    self.runner.val_loop.run()
            if self._iter == self.stop_after:
                # Save after final validation and best-checkpoint selection;
                # explicitly pass completed counters to avoid +1 ambiguity.
                self.runner.save_checkpoint(
                    self.runner.work_dir,
                    f'iter_{self._iter}.pth',
                    meta=dict(iter=self._iter, epoch=self._epoch),
                    by_epoch=False)
                Path(self.runner.work_dir, 'last_checkpoint').write_text(
                    str(Path(self.runner.work_dir, f'iter_{self._iter}.pth')))
                module = getattr(self.runner.model, 'module',
                                 self.runner.model)
                Path(self.runner.work_dir, 'early_stop.json').write_text(
                    json.dumps(
                        dict(
                            completed_iterations=self._iter,
                            configured_max_iters=self._max_iters,
                            uda_local_iter=getattr(module, 'local_iter', None),
                            learning_rates=sorted({
                                group['lr']
                                for group in self.runner.optim_wrapper.
                                optimizer.param_groups
                            }),
                            stop_reason='Fixed iteration cap'),
                        indent=2) + '\n')
            self.runner.call_hook('after_train_epoch')
            self.runner.call_hook('after_train')
        finally:
            self.dataloader_iterator.close()
        return self.runner.model
