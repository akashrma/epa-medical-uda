"""Observe legacy/native numerical behavior without changing model settings.

Legacy imports use the frozen source. Outputs always go to the mmcv2 worktree.
Dropout replay is an explicitly labeled diagnostic control, not training.
"""

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'work_dirs/migration_baseline'
AUDIT = BASE / 'step1_audit_20261011'
SELECTED = ('backbone.patch_embed1.proj.weight',
            'backbone.block1.0.attn.q.weight',
            'decode_head.linear_fuse.conv.weight',
            'decode_head.linear_pred.weight')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stack', choices=['legacy', 'modern'], required=True)
    parser.add_argument(
        '--direction', choices=['ct2mr', 'mr2ct'], required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument(
        '--mode', choices=['probe', 'full', 'fft'], default='probe')
    parser.add_argument(
        '--replay', help='Legacy probe directory; stochastic control')
    parser.add_argument(
        '--fixed-predictions',
        help='Evaluate identical saved predictions for metric parity')
    parser.add_argument(
        '--declared-prototypes',
        action='store_true',
        help='Pin the frozen declared assets for the oracle')
    args = parser.parse_args()
    source = BASE / 'legacy_source' if args.stack == 'legacy' else ROOT
    sys.path.insert(0, str(source))
    os.chdir(source)
    import numpy as np
    import torch
    from mmseg.apis import set_random_seed
    from mmseg.datasets import build_dataset
    from mmseg.models import build_segmentor
    if args.stack == 'legacy':
        import mmcv
        from mmcv.parallel import DataContainer, collate
        from mmcv.runner import build_optimizer
        from mmcv.runner.hooks.lr_updater import PolyLrUpdaterHook
        from mmseg.models.builder import build_train_model
        from tools.train import _register_linear_pred_hook
        Config = mmcv.Config
    else:
        from mmengine.config import Config
        from mmengine.optim import build_optim_wrapper
        from mmengine.registry import init_default_scope
        from mmseg.datasets.collate import uda_collate
        from mmseg.engine import LegacyPolyLR
        from mmseg.engine.config import translate_legacy_config
        from mmseg.registry import MODELS
        init_default_scope('mmseg')
    assert str(source) in __import__('mmseg').__file__
    out = Path(args.out).resolve()
    out.relative_to(BASE)
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'fft':
        from mmseg.models.utils.dacs_transforms import fda_source_to_target
        torch.set_num_threads(2)
        generator = np.random.RandomState(7)
        cases = {}
        for height, width, strength in ((8, 8, 0.), (8, 8, .2), (9, 11, .2),
                                        (256, 256, .01)):
            shape = (2, 3, height, width)
            a, b = [
                torch.from_numpy(
                    generator.normal(size=shape).astype(np.float32))
                for _ in range(2)
            ]
            result = fda_source_to_target(a, b, L=strength)
            assert result.shape == a.shape and torch.isfinite(result).all()
            cases[str((height, width, strength))] = dict(
                source=a, target=b, output=result)
        torch.save(cases, str(out / 'fft.pt'))
        if args.stack == 'modern':
            from mmseg.models.backbones.mix_transformer import (
                legacy_drop_path_rates)
        rates = {}
        for steps in (1, 8, 14, 16, 28, 52, 54):
            for rate in (.1, .3):
                if args.stack == 'legacy':
                    values = torch.linspace(0, rate, steps)
                else:
                    values = torch.tensor(legacy_drop_path_rates(rate, steps))
                rates[str((steps, rate))] = values
        torch.save(rates, str(out / 'rates.pt'))
        return
    assert torch.cuda.is_available(), 'Use the GPU host'
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # Keep the frozen legacy oracle's defaults; qualify the modern candidate.
    torch.backends.cudnn.allow_tf32 = args.stack == 'legacy'
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(False)
    free, total = (
        torch.cuda.mem_get_info() if hasattr(torch.cuda, 'mem_get_info') else
        (None, None))
    if free is not None:
        assert free > 9 * 1024**3
        torch.cuda.set_per_process_memory_fraction(.9 * free / total)
    filename = (
        args.direction +
        '_selftrain_nomix_etf_tgt_dicece_segformer_mitb5_fda.py')
    cfg = Config.fromfile(str(source / 'configs/medical' / filename))
    assert cfg.runner.max_iters == 50000 and cfg.seed == 0
    checkpoint = BASE / (args.direction +
                         '_epa_fda_seed0_5k') / 'iter_5000.pth'

    def unpack(value):
        if args.stack == 'legacy' and isinstance(value, DataContainer):
            return unpack(value.data[0])
        if isinstance(value, dict):
            return {key: unpack(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [unpack(item) for item in value]
        return value

    def to_cuda(value):
        if torch.is_tensor(value):
            return value.cuda()
        if isinstance(value, dict):
            return {key: to_cuda(item) for key, item in value.items()}
        if isinstance(value, list):
            return [to_cuda(item) for item in value]
        return value

    def test_batch(dataset, index):
        sample = dataset[index]
        if args.stack == 'legacy':
            return unpack(collate([sample], samples_per_gpu=1))
        return uda_collate([sample])

    # Checkpoint inference precedes training; each training case resets seed.
    set_random_seed(0, deterministic=False)
    student = deepcopy(cfg.model)
    student.pretrained = student.train_cfg = None
    model = build_segmentor(student)
    saved = torch.load(str(checkpoint), map_location='cpu')
    state = {
        key[len('model.'):]: tensor
        for key, tensor in saved['state_dict'].items()
        if key.startswith('model.')
    }
    model.load_state_dict(state, strict=True)
    del state, saved
    model.cuda().eval()
    model.automatic_debug = False
    model.debug = False
    target_cfg = deepcopy(cfg.data.test)
    target_cfg.test_mode = True
    target = build_dataset(target_cfg)
    evaluation = {}
    if args.mode == 'full':
        # Sequential evaluation uses the actual packing and prediction API.
        predictions = np.lib.format.open_memmap(
            str(out / 'predictions.npy'),
            mode='w+',
            dtype=np.uint8,
            shape=(len(target), 256, 256))
        started = time.monotonic()
        with torch.no_grad():
            for i in range(len(target)):
                data = to_cuda(test_batch(target, i))
                prediction = model(return_loss=False, rescale=True, **data)[0]
                assert prediction.shape == (256, 256)
                predictions[i] = prediction
                if (i + 1) % 200 == 0 or i + 1 == len(target):
                    print(
                        f'{args.stack} {args.direction} full: '
                        f'{i + 1}/{len(target)}',
                        flush=True)
        predictions.flush()
        metrics = target.evaluate(list(predictions), metric=['mDice'])
        fixed_metrics = None
        if args.fixed_predictions:
            fixed = np.load(args.fixed_predictions, mmap_mode='r')
            assert fixed.shape == predictions.shape
            fixed_metrics = dict(
                target.evaluate(list(fixed), metric=['mDice']))
        write_json(
            out / 'full.json',
            dict(
                metrics=dict(metrics),
                slices=len(target),
                fixed_prediction_metrics=fixed_metrics,
                fixed_predictions=args.fixed_predictions,
                inference_seconds=time.monotonic() - started,
                checkpoint=str(checkpoint),
                checkpoint_sha256=sha256(checkpoint),
                config=str(source / 'configs/medical' / filename),
                seed=0,
                stack=args.stack,
                data=target_cfg.to_dict()))
        return
    for domain in ('target', 'source'):
        dataset_cfg = deepcopy(target_cfg)
        if domain == 'source':
            medical = 'ct' if args.direction == 'ct2mr' else 'mr'
            dataset_cfg.img_list = str(ROOT /
                                       f'data/datalist/val_{medical}.txt')
            dataset_cfg.ann_list = str(ROOT /
                                       f'data/datalist/val_{medical}_gt.txt')
        dataset = build_dataset(dataset_cfg)
        indices = [0, len(dataset) // 2, len(dataset) - 1]
        records = []
        with torch.no_grad():
            for index in indices:
                data = test_batch(dataset, index)
                gpu = to_cuda(data)
                logits = model.whole_inference(gpu['img'][0],
                                               gpu['img_metas'][0], True)
                probabilities = torch.softmax(logits, dim=1)
                prediction = model(return_loss=False, rescale=True, **gpu)[0]
                assert np.array_equal(
                    probabilities.argmax(1).cpu().numpy()[0], prediction)
                records.append(
                    dict(
                        index=index,
                        data=data,
                        logits=logits.cpu(),
                        prediction=prediction))
        torch.save(records, str(out / (domain + '_fixed.pt')))
        evaluation[domain] = dict(indices=indices, slices=len(dataset))
    del model, target, dataset, records, data, gpu, logits, probabilities
    gc.collect()
    torch.cuda.empty_cache()
    reference = 'reference_ct2mr' if args.direction == 'ct2mr' \
        else 'reference_mr2ct_current_slices'
    batch_path = AUDIT / reference / 'fixed_training_batch.pt'
    batch = torch.load(str(batch_path), map_location='cpu')
    gpu_batch = to_cuda(batch)
    traces = {}
    for case, start, count in (('initial', 0, 2), ('boundary', 1999, 3)):
        case_out = out / case
        case_out.mkdir(exist_ok=True)
        set_random_seed(0, deterministic=False)
        spec = deepcopy(cfg)
        spec.model.train_cfg.work_dir = str(case_out)
        spec.model.train_cfg.log_config = deepcopy(spec.log_config)
        if args.stack == 'legacy':
            model = build_train_model(spec)
            model.init_weights()
            _register_linear_pred_hook(model)
            model.cuda().train()
            optimizer = build_optimizer(model, spec.optimizer)
            scheduler_cfg = dict(spec.lr_config)
            scheduler_cfg.pop('policy')
            scheduler = PolyLrUpdaterHook(**scheduler_cfg)
            runner = SimpleNamespace(
                optimizer=optimizer, max_iters=50000, iter=0, epoch=0)
            scheduler.before_run(runner)
        else:
            native = translate_legacy_config(spec, work_dir=str(case_out))
            model = MODELS.build(native.model)
            model.init_weights()
            model.cuda().train()
            wrapper = build_optim_wrapper(model, native.optim_wrapper)
            optimizer = wrapper.optimizer
            scheduler = LegacyPolyLR(
                wrapper, **{
                    key: value
                    for key, value in native.param_scheduler.items()
                    if key not in ('type', 'by_epoch')
                })
        if start:
            model._init_ema_weights()
            model.local_iter = start
        if args.declared_prototypes:
            model.etf_prototype_dir = str(source / 'etf_prototypes')
            assert (Path(model.etf_prototype_dir) /
                    'etf_prototypes_K5_D768.pt').is_file()
        named = dict(model.named_parameters())
        selected = [
            prefix + '.' + name for prefix in ('model', 'ema_model')
            for name in SELECTED
        ]
        selected += [
            f'{prefix}.decode_head.linear_fuse.bn.{name}'
            for prefix in ('model', 'ema_model')
            for name in ('running_mean', 'running_var', 'num_batches_tracked')
        ]
        observations = {}
        stochastic = []
        replay = None
        if args.replay:
            replay = torch.load(
                str(Path(args.replay) / case / 'stochastic.pt'),
                map_location='cpu')
        stochastic_index = [0]
        originals = []
        for module_name, module in model.named_modules():
            kind = type(module).__name__
            probability = (
                getattr(module, 'drop_prob', None)
                if kind == 'DropPath' else getattr(module, 'p', None))
            if kind not in ('DropPath', 'Dropout',
                            'Dropout2d') or not probability:
                continue
            old_forward = module.forward

            def observe_dropout(inputs,
                                *extra,
                                _module=module,
                                _old=old_forward,
                                _name=module_name,
                                _kind=kind,
                                _p=probability,
                                **kwargs):
                if not _module.training:
                    return _old(inputs, *extra, **kwargs)
                if replay is None:
                    result = _old(inputs, *extra, **kwargs)
                    if _kind == 'DropPath':
                        mask = result.detach().reshape(result.shape[0],
                                                       -1).abs().sum(1).ne(0)
                    elif _kind == 'Dropout2d':
                        output_nonzero = result.detach().flatten(2).abs().sum(
                            2).ne(0)
                        input_zero = inputs.detach().flatten(2).abs().sum(
                            2).eq(0)
                        mask = output_nonzero | input_zero
                    else:
                        mask = result.detach().ne(0) | inputs.detach().eq(0)
                    stochastic.append(
                        dict(name=_name, kind=_kind, p=_p, mask=mask.cpu()))
                    return result
                record = replay[stochastic_index[0]]
                stochastic_index[0] += 1
                assert record['name'] == _name and record[
                    'kind'] == _kind and record['p'] == _p
                mask = record['mask'].to(inputs.device)
                if _kind == 'DropPath':
                    mask = mask.reshape((-1, ) + (1, ) * (inputs.ndim - 1))
                elif _kind == 'Dropout2d':
                    mask = mask[:, :, None, None]
                return inputs / (1 - _p) * mask

            module.forward = observe_dropout
            originals.append((module, old_forward))
        heads = []
        for role, head in (('student', model.get_model().decode_head),
                           ('teacher', model.get_ema_model().decode_head)):

            def capture_head(module, inputs, output, _role=role):
                key = _role + '_logits_' + str(
                    sum(
                        k.startswith(_role + '_logits_')
                        for k in observations))
                observations[key] = output.detach().cpu().clone()

            heads.append(head.register_forward_hook(capture_head))
        old_fda = model._maybe_apply_fda
        old_pseudo = model.get_pseudo_label_and_weight

        def capture_fda(*items, _forward=old_fda, **kwargs):
            result = _forward(*items, **kwargs)
            observations['fda_input'] = result.detach().cpu().clone()
            return result

        def capture_pseudo(*items, _forward=old_pseudo, **kwargs):
            label, weight = _forward(*items, **kwargs)
            observations['pseudo_label'] = label.detach().cpu().clone()
            observations['pseudo_weight'] = weight.detach().cpu().clone()
            return label, weight

        model._maybe_apply_fda = capture_fda
        model.get_pseudo_label_and_weight = capture_pseudo
        events = []
        backward = torch.Tensor.backward

        def record_backward(tensor, *items, **kwargs):
            assert torch.isfinite(tensor).all()
            events.append('backward')
            return backward(tensor, *items, **kwargs)

        torch.Tensor.backward = record_backward
        entries = []
        for iteration in range(start, start + count):
            observations.clear()
            events.clear()
            if args.stack == 'legacy':
                runner.iter = iteration
                scheduler.before_train_iter(runner)
            else:
                scheduler.set_iteration(iteration)
            named.update(dict(model.named_buffers()))
            before = {
                name: named[name].detach().cpu().clone()
                for name in selected
            }
            if args.stack == 'legacy':
                result = model.train_step(gpu_batch, optimizer)['log_vars']
            else:
                result = model.train_step(gpu_batch, wrapper)
            named.update(dict(model.named_buffers()))
            assert len(events) == (2 if iteration < 2000 else 4)
            assert model.local_iter == iteration + 1
            tensors = dict(observations)
            for name in selected:
                parameter = named[name]
                tensors['before/' + name] = before[name]
                tensors['after/' + name] = parameter.detach().cpu().clone()
                tensors['delta/' +
                        name] = tensors['after/' + name] - before[name]
                if parameter.grad is not None:
                    tensors['gradient/' +
                            name] = parameter.grad.detach().cpu().clone()
                state = optimizer.state.get(parameter, {})
                for key in ('exp_avg', 'exp_avg_sq'):
                    if key in state:
                        tensors[key + '/' +
                                name] = state[key].detach().cpu().clone()
            tensors['etf_prototypes'] = model.etf_prototypes.detach().cpu(
            ).clone()
            tensors['source_class_means'] = model.source_class_means.detach(
            ).cpu().clone()
            for name, value in tensors.items():
                assert torch.isfinite(value).all(), name
            torch.save(tensors, str(case_out / f'{iteration}.pt'))
            torch.save(
                dict(
                    cpu=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state(),
                    numpy=np.random.get_state(),
                    python=random.getstate()),
                str(case_out / f'{iteration}_rng.pt'))
            entry = dict(
                iteration=iteration,
                completed=model.local_iter,
                losses=dict(result),
                backwards=len(events),
                rates=sorted({g['lr']
                              for g in optimizer.param_groups}),
                pseudo_fraction=float(observations['pseudo_weight'].mean()),
                stochastic_calls=len(stochastic)
                if replay is None else stochastic_index[0],
                peak_gpu_bytes=torch.cuda.max_memory_allocated())
            entries.append(entry)
            print(
                f'{args.stack} {args.direction} {case} {iteration}: {result}',
                flush=True)
        if replay is not None:
            assert stochastic_index[0] == len(replay)
        torch.save(stochastic, str(case_out / 'stochastic.pt'))
        write_json(case_out / 'trace.json', entries)
        traces[case] = entries
        torch.Tensor.backward = backward
        for module, function in originals:
            module.forward = function
        for handle in heads:
            handle.remove()
        del model, optimizer, named, before, tensors, parameter, state
        del old_fda, old_pseudo, originals
        if args.stack == 'modern':
            del wrapper
        else:
            del runner
        del scheduler
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        out / 'probe.json',
        dict(
            stack=args.stack,
            direction=args.direction,
            seed=0,
            horizon=50000,
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha256(checkpoint),
            fixed_batch=str(batch_path),
            fixed_batch_sha256=sha256(batch_path),
            evaluation=evaluation,
            traces=traces,
            replay_control=args.replay,
            declared_prototypes=args.declared_prototypes,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            torch=torch.__version__,
            source=str(source)))


if __name__ == '__main__':
    main()
