# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Add startup test
# - Save configs as python file instead of json file
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import argparse
import os
import uuid
from datetime import datetime
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.utils import get_git_hash

from experiments import generate_experiment_cfgs
from tools import train


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Generate and launch experiments')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--exp', type=int)
    group.add_argument('--config')
    parser.add_argument('--machine', choices=['local'], default='local')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--startup-test',
        action='store_true',
        help='Stop after two updates; retain schedule horizon')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--work-dir')
    parser.add_argument('--resume-from')
    parser.add_argument('--no-validate', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument(
        '--options', '--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args(argv)
    configs = Path(__file__).resolve().parent / 'configs'
    if args.config:
        original = Config.fromfile(args.config)
        specs = [
            dict(
                _base_=str(Path(args.config).resolve()),
                name=original.get('name',
                                  Path(args.config).stem),
                exp=original.get('exp', 'medical'))
        ]
    else:
        specs = generate_experiment_cfgs(args.exp)
    if len(specs) != 1 and (args.work_dir or args.resume_from):
        raise ValueError('Explicit work-dir/resume requires one experiment')
    generated = []
    for spec in specs:
        spec = dict(spec)
        if not args.config:
            spec['_base_'] = [str(configs / base) for base in spec['_base_']]
        if args.debug:
            spec.setdefault('log_config', {}).update(
                interval=10, img_interval=10)
            spec.setdefault('evaluation', {})['interval'] = 200
            spec.setdefault('uda', {})['debug_img_interval'] = 10
        name = (f'{datetime.now():%y%m%d_%H%M%S}_'
                f'{spec["name"]}_{uuid.uuid4().hex[:5]}')
        experiment = f'{args.machine}-{spec.get("exp", args.exp)}'
        spec.update(
            name=name,
            git_rev=get_git_hash(),
            work_dir=args.work_dir
            or os.path.join('work_dirs', experiment, name))
        filename = configs / 'generated' / experiment / (name + '.py')
        filename.parent.mkdir(parents=True, exist_ok=True)
        Config(spec).dump(str(filename))
        launch = [str(filename)]
        cap = args.stop_after if args.stop_after is not None else \
            (2 if args.startup_test else None)
        if cap is not None:
            launch.extend(['--stop-after', str(cap)])
        for flag, value in (('--resume-from', args.resume_from), ('--seed',
                                                                  args.seed)):
            if value is not None:
                launch.extend([flag, str(value)])
        if args.no_validate or args.startup_test:
            launch.append('--no-validate')
        if args.dry_run:
            launch.append('--dry-run')
        # Merge structured overrides into the generated child before launch.
        if args.options:
            child = Config.fromfile(str(filename))
            child.merge_from_dict(args.options)
            child.dump(str(filename))
        print(f'Run job {name}')
        train.main(launch)
        generated.append(str(filename))
    return generated


if __name__ == '__main__':
    main()
