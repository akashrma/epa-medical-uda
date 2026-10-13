"""Stop a legacy baseline without shortening its learning-rate schedule."""

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmcv.runner import Hook

from tools import train


class BaselineStopped(Exception):
    """The requested number of optimizer updates has completed."""


class StopBaselineHook(Hook):
    """Save after validation/logging, then finish at an exact update count."""

    def __init__(self, stop_after):
        self.stop_after = stop_after

    def before_run(self, runner):
        if not 0 < self.stop_after <= runner.max_iters:
            raise ValueError('Stop count must be within the configured run')
        if runner.meta is None:
            runner.meta = {}
        runner.meta['baseline_protocol'] = dict(
            configured_max_iters=runner.max_iters,
            stop_after=self.stop_after,
            preserve_lr_horizon=True)

    def after_train_iter(self, runner):
        completed = runner.iter + 1
        if completed < self.stop_after:
            return
        runner.save_checkpoint(runner.work_dir)
        model = getattr(runner.model, 'module', runner.model)
        evidence = dict(
            completed_iterations=completed,
            configured_max_iters=runner.max_iters,
            learning_rates=runner.current_lr(),
            uda_local_iter=getattr(model, 'local_iter', None),
            checkpoint='iter_{}.pth'.format(completed),
            stop_reason='User-requested fixed iteration cap')
        path = Path(runner.work_dir) / 'early_stop.json'
        path.write_text(json.dumps(evidence, indent=2) + '\n')
        runner.logger.info(
            'Stopping baseline after %d updates; schedule horizon stays %d.',
            completed, runner.max_iters)
        raise BaselineStopped


def finish_stopped_runner(runner):
    # IterBasedRunner increments these counters after after_train_iter. The
    # stop hook exits that call, so complete bookkeeping before closing hooks.
    runner._iter += 1
    runner._inner_iter += 1
    runner.call_hook('after_epoch')
    runner.call_hook('after_run')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stop-after', type=int, default=5000)
    args, train_args = parser.parse_known_args()
    branch = subprocess.check_output(['git', 'branch', '--show-current'],
                                     cwd=str(ROOT),
                                     text=True).strip()
    if branch != 'mmcv2':
        parser.error('All migration work and progress must occur on mmcv2')
    api = importlib.import_module('mmseg.apis.train')
    original_build_runner = api.build_runner
    runners = []

    def build_runner(*positional, **keyword):
        runner = original_build_runner(*positional, **keyword)
        # LOWEST runs after evaluation, best-checkpoint selection and loggers.
        runner.register_hook(StopBaselineHook(args.stop_after), 'LOWEST')
        runners.append(runner)
        return runner

    api.build_runner = build_runner
    try:
        train.main(train_args)
    except BaselineStopped:
        finish_stopped_runner(runners[-1])
    finally:
        api.build_runner = original_build_runner


if __name__ == '__main__':
    main()
