"""Capture long-run sampling and initialization evidence without retuning."""

import argparse
import contextlib
import hashlib
import json
import logging
import os
import pickle
import random
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'work_dirs/migration_baseline'
OUT = BASE / 'step9_diagnosis_20261012'


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def tensor_hash(tensor):
    raw = memoryview(tensor.cpu().contiguous().numpy()).cast('B')
    digest = hashlib.sha256()
    for offset in range(0, len(raw), 8 * 1024 * 1024):
        digest.update(raw[offset:offset + 8 * 1024 * 1024])
    return digest.hexdigest()


def runtime_probe(args, source):
    import numpy as np
    import torch
    directory = OUT / args.tag
    directory.mkdir(exist_ok=False)
    cfg = ('configs/medical/' + args.direction +
           '_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py')
    events = []
    selected = (
        'backbone.patch_embed1.proj.weight', 'decode_head.linear_pred.weight',
        'decode_head.linear_pred.bias', 'decode_head.linear_fuse.bn.weight',
        'decode_head.linear_fuse.bn.bias',
        'decode_head.linear_fuse.bn.running_mean',
        'decode_head.linear_fuse.bn.running_var',
        'decode_head.linear_fuse.bn.num_batches_tracked') + tuple(
            'decode_head.linear_c.' + str(index) + '.proj.bias'
            for index in range(4))

    def rng():
        numpy_state = np.random.get_state()
        numpy_digest = hashlib.sha256(numpy_state[1].tobytes())
        numpy_digest.update(
            json.dumps([numpy_state[0]] + list(numpy_state[2:])).encode())
        return dict(
            cpu=tensor_hash(torch.get_rng_state()),
            cuda=tensor_hash(torch.cuda.get_rng_state()[-16:]),
            numpy=numpy_digest.hexdigest(),
            python=hashlib.sha256(pickle.dumps(random.getstate(),
                                               protocol=4)).hexdigest())

    def unpack(batch):
        if args.stack == 'legacy':
            return {key: value.data[0] for key, value in batch.items()}
        return batch

    def record_batch(batch):
        digest = hashlib.sha256()
        for key, value in sorted(unpack(batch).items()):
            digest.update(key.encode())
            if torch.is_tensor(value):
                digest.update(tensor_hash(value).encode())
            elif key.endswith('img_metas'):
                filenames = [
                    str(Path(item['filename']).resolve()) for item in value
                ]
                digest.update(json.dumps(filenames).encode())
        return digest.hexdigest()

    if args.stack == 'legacy':
        from mmcv.runner import Hook
    else:
        from mmengine.hooks import Hook

    class TraceHook(Hook):

        def before_run(self, runner):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = args.cudnn_tf32 == 'on'
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch, 'use_deterministic_algorithms'):
                torch.use_deterministic_algorithms(False)
            write(directory / 'initial_rng.json', rng())

        def before_train_iter(self, runner, batch_idx=None, data_batch=None):
            if data_batch is not None:
                self.batch_hash = record_batch(data_batch)

        def after_train_iter(self,
                             runner,
                             batch_idx=None,
                             data_batch=None,
                             outputs=None):
            completed = runner.iter + 1
            result = runner.outputs['log_vars'] if args.stack == 'legacy' \
                else outputs
            events.append(
                dict(
                    iter=completed,
                    batch=self.batch_hash,
                    rng=rng(),
                    losses=dict(result)))
            if completed % 50 == 0 or completed in (1, 501):
                model = getattr(runner.model, 'module', runner.model)
                tensors = model.state_dict()
                snapshot = {
                    prefix + '.' + name:
                    tensors[prefix + '.' + name].detach().cpu().clone()
                    for prefix in ('model', 'ema_model')
                    for name in selected
                }
                optimizer = runner.optimizer if args.stack == 'legacy' \
                    else runner.optim_wrapper.optimizer
                for name, parameter in model.named_parameters():
                    if name not in snapshot:
                        continue
                    if parameter.grad is not None:
                        snapshot['gradient/' + name] = \
                            parameter.grad.detach().cpu().clone()
                    state = optimizer.state.get(parameter, {})
                    for moment in ('exp_avg', 'exp_avg_sq'):
                        if moment in state:
                            snapshot[moment + '/' + name] = \
                                state[moment].detach().cpu().clone()
                torch.save(
                    snapshot,
                    str(directory / ('state_' + str(completed) + '.pt')))
                write(directory / 'trace.json', events)

    hook = TraceHook()
    if args.stack == 'legacy':
        import mmseg.apis.train as api
        import mmseg.models.uda.dacs as dacs
        dacs.get_default_etf_dir = lambda: str(source / 'etf_prototypes')
        original = api.build_runner

        def build(*items, **kwargs):
            runner = original(*items, **kwargs)
            original_step = runner.model.train_step

            def step(batch, *positional, **keywords):
                hook.batch_hash = record_batch(batch)
                return original_step(batch, *positional, **keywords)

            runner.model.train_step = step
            runner.register_hook(hook, 'NORMAL')
            return runner

        api.build_runner = build
        sys.argv = [
            'train_legacy_baseline.py', '--stop-after',
            str(args.updates), cfg, '--work-dir',
            str(directory)
        ]
        runpy.run_path(
            str(source / 'tools/train_legacy_baseline.py'),
            run_name='__main__')
    else:
        from mmseg.engine import EPARunner
        original = EPARunner.__init__

        def initialize(runner, *items, **kwargs):
            original(runner, *items, **kwargs)
            runner.register_hook(hook, 'NORMAL')

        EPARunner.__init__ = initialize
        sys.argv = [
            'train.py', cfg, '--stop-after',
            str(args.updates), '--work-dir',
            str(directory)
        ]
        runpy.run_path(str(source / 'tools/train.py'), run_name='__main__')
    write(directory / 'trace.json', events)
    write(
        directory / 'complete.json',
        dict(
            stack=args.stack,
            direction=args.direction,
            iterations=len(events),
            horizon=50000,
            diagnostic_only=True,
            configuration_overrides=False,
            diagnostic_precision_override=args.cudnn_tf32 == 'off',
            cudnn_tf32=args.cudnn_tf32,
            convolution_api=args.conv_api,
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stack', choices=('legacy', 'modern'), required=True)
    parser.add_argument(
        '--mode',
        choices=('sampler', 'initial', 'runtime', 'reference'),
        required=True)
    parser.add_argument('--tag')
    parser.add_argument('--checkpoint')
    parser.add_argument('--updates', type=int, default=700)
    parser.add_argument(
        '--direction', choices=('ct2mr', 'mr2ct'), default='ct2mr')
    parser.add_argument('--cudnn-tf32', choices=('on', 'off'), default='on')
    parser.add_argument(
        '--conv-api', choices=('default', 'v7'), default='default')
    args = parser.parse_args()
    assert subprocess.check_output(['git', 'branch', '--show-current'],
                                   cwd=ROOT,
                                   text=True).strip() == 'mmcv2'
    OUT.mkdir(exist_ok=True)
    source = BASE / 'legacy_source' if args.stack == 'legacy' else ROOT
    sys.path.insert(0, str(source))
    os.chdir(source)
    if args.conv_api == 'v7':
        os.environ['TORCH_CUDNN_V8_API_DISABLED'] = '1'
    import torch
    torch.set_num_threads(1)
    if args.mode == 'reference':
        assert args.stack == 'legacy' and args.tag and args.checkpoint
        cfg = ('configs/medical/' + args.direction +
               '_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py')
        with (OUT / (args.tag + '.log')).open('x') as stream:
            with contextlib.redirect_stdout(stream), \
                    contextlib.redirect_stderr(stream):
                namespace = runpy.run_path(
                    str(source / 'tools/capture_legacy_reference.py'))
                namespace['main'].__globals__['ROOT'] = ROOT
                sys.argv = [
                    'capture_legacy_reference.py', cfg, args.checkpoint,
                    '--out-dir',
                    str(OUT / args.tag), '--source-domain',
                    'ct' if args.direction == 'ct2mr' else 'mr'
                ]
                namespace['main']()
        print('Legacy checkpoint evaluation complete:', args.tag, flush=True)
        return
    if args.mode == 'runtime':
        assert args.tag and torch.cuda.is_available()
        with (OUT / (args.tag + '.log')).open('x') as stream:
            with contextlib.redirect_stdout(stream), \
                    contextlib.redirect_stderr(stream):
                runtime_probe(args, source)
        print('Diagnostic complete:', args.tag, flush=True)
        return
    if args.mode == 'sampler':
        torch.manual_seed(0)
        # DataLoader's base seed, then RandomSampler's unused seed.
        seeds = [
            torch.empty((), dtype=torch.int64).random_().item()
            for _ in range(2)
        ]
        before = torch.get_rng_state()
        permutation = torch.randperm(80640000)
        report = dict(
            stack=args.stack,
            torch=torch.__version__,
            size=len(permutation),
            seeds=seeds,
            before_rng_sha256=tensor_hash(before),
            after_rng_sha256=tensor_hash(torch.get_rng_state()),
            full_permutation_sha256=tensor_hash(permutation))
        torch.save(permutation[:20032].clone(),
                   str(OUT / (args.stack + '_first_20032_pairs.pt')))
        write(OUT / (args.stack + '_sampler.json'), report)
        print(json.dumps(report, indent=2), flush=True)
        return
    from mmseg.apis import set_random_seed
    if args.stack == 'legacy':
        from mmcv import Config
        from mmseg.models.builder import build_train_model
        from tools.train import _register_linear_pred_hook
    else:
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmseg.engine.config import translate_legacy_config
        from mmseg.registry import MODELS
        init_default_scope('mmseg')
    results = {}
    for direction in ('ct2mr', 'mr2ct'):
        cfg = Config.fromfile('configs/medical/' + direction +
                              '_selftrain_nomix_etf_tgt_dicece_' +
                              'segformer_mitb5_fda.py')
        set_random_seed(0)
        if args.stack == 'legacy':
            cfg.model.train_cfg = dict(
                work_dir=str(OUT), log_config=cfg.log_config)
            cfg.uda.model = cfg.model
            cfg.uda.max_iters = cfg.runner.max_iters
            model = build_train_model(cfg)
            model.init_weights()
            _register_linear_pred_hook(model)
        else:
            native = translate_legacy_config(cfg, work_dir=str(OUT))
            model = MODELS.build(native.model)
            logging.disable(logging.INFO)
            model.init_weights()
        state = torch.get_rng_state()
        torch.save(
            state,
            str(OUT /
                (args.stack + '_' + direction + '_after_initial_rng.pt')))
        results[direction] = dict(
            cpu_rng_sha256=tensor_hash(state),
            model_state_tensors=len(model.state_dict()))
        del model
    write(OUT / (args.stack + '_initial.json'), results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
