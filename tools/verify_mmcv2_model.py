"""Verify construction, initialization and source-model smoke checks."""

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'work_dirs/migration_baseline/step1_audit_20261011'
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_direction(direction, out, device):
    import numpy as np
    import torch
    from mmengine.config import Config
    from mmengine.optim import build_optim_wrapper

    from mmseg.models.builder import build_train_model

    filename = (
        direction + '_selftrain_nomix_etf_tgt_dicece_'
        'segformer_mitb5_fda.py')
    cfg = Config.fromfile(str(ROOT / 'configs/medical' / filename))
    cfg.model.train_cfg.work_dir = str(out)
    cfg.model.train_cfg.log_config = cfg.log_config
    assert cfg.runner.max_iters == 50000 and cfg.seed == 0
    seed_all(cfg.seed)
    model = build_train_model(cfg)
    model.init_weights()
    assert type(model).__name__ == 'DACS'
    trace = AUDIT / ('final_trace_' + direction + '_epa')
    legacy = torch.load(
        trace / 'initialized_state_dict.pt', map_location='cpu')
    current = model.state_dict()
    assert list(current) == list(legacy), 'State key/order mismatch'
    for name, value in current.items():
        assert value.dtype == legacy[name].dtype, name
        assert value.shape == legacy[name].shape, name
        assert torch.equal(value,
                           legacy[name]), 'Initialization mismatch: ' + name
        if value.is_floating_point():
            assert value.dtype == torch.float32, name
    result = dict(
        config=str(Path('configs/medical') / filename),
        seed=cfg.seed,
        initialized_tensors=len(current),
        initialized_state_exact=True,
        state_key_order_exact=True,
        configured_max_iters=cfg.runner.max_iters,
        uda_max_iters=model.max_iters,
        initialization_float_dtype='float32')
    assert model.max_iters == 50000
    result['pretrained_loads'] = {}
    for name in ('model', 'ema_model'):
        info = getattr(model, name).backbone.pretrained_load_info
        assert not info['missing_keys'] and not info['unexpected_keys']
        result['pretrained_loads'][name] = info
    bn = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            assert type(module) is torch.nn.BatchNorm2d
            assert module.eps == 1e-5 and module.momentum == .1
            assert module.affine and module.track_running_stats
            bn[name] = dict(
                type=type(module).__name__,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats)
    assert len(bn) == 2
    result['batch_norm'] = bn
    result['named_parameters'] = len(list(model.named_parameters()))
    del legacy, current

    model = model.to(device)
    optim_cfg = deepcopy(dict(cfg.optimizer))
    paramwise = optim_cfg.pop('paramwise_cfg')
    optim_cfg.update(foreach=False, fused=False)
    wrapper = build_optim_wrapper(
        model,
        dict(
            type='OptimWrapper', optimizer=optim_cfg, paramwise_cfg=paramwise))
    names = {
        id(parameter): name
        for name, parameter in model.named_parameters()
    }
    old_groups = json.loads((trace / 'optimizer_groups.json').read_text())
    groups = wrapper.optimizer.param_groups
    assert len(groups) == len(old_groups) == 2134
    for group, old in zip(groups, old_groups):
        assert [names[id(p)] for p in group['params']] == old['parameters']
        for key in ('lr', 'weight_decay', 'eps', 'amsgrad'):
            assert group[key] == old[key], (old['parameters'], key)
        assert list(group['betas']) == old['betas']
        assert group['foreach'] is False and group['fused'] is False
    result['optimizer_group_inventory'] = dict(
        groups=len(groups),
        legacy_names_settings_order_exact=True,
        foreach=False,
        fused=False)

    reference = AUDIT / ('reference_ct2mr' if direction == 'ct2mr' else
                         'reference_mr2ct_current_slices')
    batch = torch.load(
        reference / 'fixed_training_batch.pt', map_location='cpu')
    student = model.get_model()
    student.automatic_debug = False
    student.train()
    img = batch['img'][:1].to(device)
    labels = batch['gt_semantic_seg'][:1].to(device)
    metas = batch['img_metas'][:1]
    wrapper.zero_grad(set_to_none=False)
    losses = student.forward_train(img, metas, labels)
    loss, logs = model._parse_losses(losses)
    wrapper.backward(loss)
    gradients = [p.grad for p in student.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert all(p.grad is None for p in model.ema_model.parameters())
    assert torch.isfinite(loss)
    wrapper.step()
    result['source_training_smoke'] = dict(
        batch_size=1,
        optimizer_updates=1,
        losses=logs,
        finite_gradients=True,
        gradient_tensors=len(gradients),
        scope='Source segmentation only; full EPA/FDA update is step 6/8')
    del wrapper, groups, gradients, losses, loss

    checkpoint_path = ROOT / 'pretrained' / ('CT2MR_best_mDice.pth'
                                             if direction == 'ct2mr' else
                                             'MR2CT_best_mDice.pth')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = {
        name[len('model.'):]: value
        for name, value in checkpoint['state_dict'].items()
        if name.startswith('model.')
    }
    assert set(state) == set(student.state_dict())
    for name, value in state.items():
        assert value.shape == student.state_dict()[name].shape, name
    incompatible = student.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    del checkpoint, state
    student.eval()
    with torch.no_grad():
        logits = student.encode_decode(img, metas)
    assert logits.shape == (1, 5, 256, 256)
    assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
    actual = logits.cpu().numpy()
    expected = np.load(reference / 'fixed_source_logits.npy')[:1]
    difference = np.abs(actual - expected)
    result['historical_checkpoint_probe'] = dict(
        checkpoint=str(checkpoint_path.relative_to(ROOT)),
        strict_student_load=True,
        missing_keys=[],
        unexpected_keys=[],
        logit_shape=list(actual.shape),
        logits_finite_fp32=True,
        reference=str(reference.relative_to(ROOT)),
        max_absolute_logit_difference=float(difference.max()),
        mean_absolute_logit_difference=float(difference.mean()),
        pixel_argmax_agreement=float(
            np.mean(actual.argmax(axis=1) == expected.argmax(axis=1))),
        scope=('Fixed source slice diagnostic; '
               'numerical thresholds set in step 8'))
    np.save(out / (direction + '_checkpoint_source_logits.npy'), actual)
    if device == 'cuda':
        torch.cuda.synchronize()
        result['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
    del model, student, logits, img, labels
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    write_json(out / (direction + '.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    branch = subprocess.check_output(['git', 'branch', '--show-current'],
                                     cwd=ROOT,
                                     text=True).strip()
    if branch != 'mmcv2':
        parser.error('All migration edits and progress must occur on mmcv2')
    out = (ROOT / args.out_dir).resolve()
    out.relative_to(ROOT / 'work_dirs')
    out.mkdir(parents=True, exist_ok=False)
    os.environ['MPLCONFIGDIR'] = str(out / 'matplotlib_cache')
    import torch
    from mmengine.registry import DefaultScope

    import mmseg
    import mmseg.models
    from mmseg.registry import (DATASETS, HOOKS, METRICS, MODELS, TASK_UTILS,
                                TRANSFORMS)
    from mmseg.utils import get_root_logger

    torch.set_num_threads(2)
    if args.device == 'cuda':
        assert torch.cuda.is_available(), 'Host GPU access is required'
        torch.cuda.set_per_process_memory_fraction(.25)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    get_root_logger(
        log_file=str(out / 'initialization.log'), log_level=logging.WARNING)
    DefaultScope.get_instance('model_verification', scope_name='mmseg')
    for registry in (DATASETS, HOOKS, METRICS, MODELS, TASK_UTILS, TRANSFORMS):
        assert registry.scope == 'mmseg'
    assert Path(mmseg.__file__).resolve() == ROOT / 'mmseg/__init__.py'
    assert MODELS.get('mmseg.DACS') is MODELS.get('DACS')
    report = dict(
        branch=branch,
        python=sys.version,
        local_package=mmseg.__file__,
        fork_version=mmseg.__version__,
        device=args.device,
        registered_models=sorted(MODELS.module_dict),
        pretrained_sha256=hashlib.sha256(
            (ROOT / 'pretrained/mit_b5_mmseg.pth').read_bytes()).hexdigest(),
        tf32_enabled=True,
        cudnn_benchmark=True,
        directions={})
    try:
        for direction in ('ct2mr', 'mr2ct'):
            print('Verifying ' + direction, flush=True)
            report['directions'][direction] = verify_direction(
                direction, out, args.device)
        report['passed'] = True
    except Exception as error:
        report.update(passed=False, error=repr(error))
        raise
    finally:
        write_json(out / 'verification.json', report)
    print(
        'Both model construction and initialization gates passed.', flush=True)


if __name__ == '__main__':
    main()
