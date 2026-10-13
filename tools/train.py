# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Provide args as argument to main()
# - Snapshot source code
# - Build UDA model instead of regular one
# - Add deterministic config flag

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# isort:skip_file

import torch
from mmengine.config import Config, DictAction
from mmengine.utils import get_git_hash

from mmseg.engine import EPARunner
from mmseg.engine.config import translate_legacy_config
from mmseg.utils.collect_env import gen_code_archive


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Train the medical EPA model')
    parser.add_argument('config')
    parser.add_argument('--work-dir')
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument('--load-from', help='Initialize weights only')
    weights.add_argument(
        '--resume-from',
        '--resume',
        nargs='?',
        const='auto',
        help='Resume full native state; omit path for auto')
    parser.add_argument(
        '--stop-after',
        type=int,
        help='Update cap; default 5000, horizon unchanged')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Write resolved configs without building a model')
    parser.add_argument('--no-validate', action='store_true')
    gpus = parser.add_mutually_exclusive_group()
    gpus.add_argument('--gpus', type=int)
    gpus.add_argument('--gpu-ids', type=int, nargs='+')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument(
        '--options', '--cfg-options', nargs='+', action=DictAction)
    parser.add_argument(
        '--launcher',
        default='none',
        choices=['none', 'pytorch', 'slurm', 'mpi'])
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)
    input_sha256 = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    original = Config.fromfile(args.config)
    if args.options:
        original.merge_from_dict(args.options)
    native_input = 'train_dataloader' in original
    if args.seed is not None:
        if native_input:
            original.randomness.seed = args.seed
            for split in ('train', 'val', 'test'):
                loader = original.get(split + '_dataloader')
                if loader and loader.get('worker_init_fn', {}).get('type') \
                        == 'legacy_worker_init':
                    loader.worker_init_fn.seed = args.seed
            if original.get('migration'):
                original.migration.legacy_config.seed = args.seed
        else:
            original.seed = args.seed
    if args.deterministic:
        raise ValueError('Matched legacy training uses nondeterministic CUDA')
    gpu_ids = args.gpu_ids or list(range(args.gpus or 1))
    if len(gpu_ids) != 1 or args.gpus == 0:
        raise ValueError('Matched training requires one GPU')
    if not native_input:
        original.gpu_ids = gpu_ids
    work_dir = args.work_dir or original.get('work_dir') or \
        str(Path('work_dirs') / Path(args.config).stem)
    cfg = translate_legacy_config(
        original,
        work_dir=work_dir,
        stop_after=args.stop_after,
        validate=not args.no_validate,
        launcher=args.launcher)
    if args.load_from:
        cfg.load_from, cfg.resume = args.load_from, False
    if args.resume_from:
        cfg.load_from = (None
                         if args.resume_from == 'auto' else args.resume_from)
        cfg.resume = True
    if cfg.resume and cfg.load_from and not Path(cfg.load_from).is_file():
        raise FileNotFoundError(cfg.load_from)
    output = Path(cfg.work_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Distinct names retain the original specification and the native runtime.
    specification = Config(original.migration.legacy_config) \
        if native_input and original.get('migration') else original
    specification.dump(str(output / 'experiment_spec.py'))
    cfg.dump(str(output / 'native_config.py'))
    manifest = dict(
        config=str(Path(args.config).resolve()),
        branch=subprocess.check_output(['git', 'branch', '--show-current'],
                                       text=True).strip(),
        git_rev=get_git_hash(),
        config_sha256=input_sha256,
        native_config_sha256=hashlib.sha256(
            (output / 'native_config.py').read_bytes()).hexdigest(),
        seed=cfg.randomness.seed,
        max_iters=cfg.train_cfg.max_iters,
        stop_after=cfg.train_cfg.stop_after,
        resume=cfg.resume,
        load_from=cfg.load_from,
        gpu_ids=gpu_ids,
        dry_run=args.dry_run)
    (output / 'launch_manifest.json').write_text(
        json.dumps(manifest, indent=2))
    if args.dry_run:
        print(f'Resolved native config: {output / "native_config.py"}')
        return cfg
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    if not torch.cuda.is_available():
        raise RuntimeError('Matched training requires a visible CUDA GPU')
    torch.cuda.set_device(gpu_ids[0])
    gen_code_archive(str(output))
    runner = EPARunner.from_cfg(cfg)
    runner.train()
    return cfg


if __name__ == '__main__':
    main()
