"""Qualify the isolated modern environment without importing legacy models."""

import argparse
import ast
import importlib
import importlib.metadata as metadata
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def dependency_checks(out_dir, report):
    import cv2
    import kornia
    import numpy as np
    import torch
    from medpy.metric.binary import asd, assd
    from mmengine.config import Config
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_
    from timm.models.layers.drop import drop_path
    from timm.models.layers.weight_init import _no_grad_trunc_normal_

    versions = {}
    for line in (ROOT / 'requirements-mmcv2.txt').read_text().splitlines():
        if '==' not in line or line.startswith('#'):
            continue
        name, expected = line.split('==')
        versions[name] = metadata.version(name)
        assert versions[name] == expected, (name, versions[name], expected)
    report['versions'] = versions
    for name in ('mmcv-full', 'mmcv-lite', 'mmsegmentation'):
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        raise AssertionError('Unexpected conflicting distribution: ' + name)
    assert sys.version_info[:2] == (3, 11)
    spec = importlib.util.find_spec('mmseg')
    assert Path(spec.origin).resolve() == ROOT / 'mmseg/__init__.py'
    report['local_mmseg_origin'] = spec.origin
    # Report the source port status separately from dependency qualification.
    try:
        importlib.import_module('mmseg')
    except AssertionError as error:
        assert 'MMCV' in str(error)
        report['local_mmseg_import'] = dict(
            status='Pending source port', reason=str(error))
    else:
        report['local_mmseg_import'] = dict(status='Import succeeded')

    for direction in ('ct2mr', 'mr2ct'):
        name = (
            direction + '_selftrain_nomix_etf_tgt_dicece_'
            'segformer_mitb5_fda.py')
        cfg = Config.fromfile(str(ROOT / 'configs/medical' / name))
        assert cfg.runner.max_iters == 50000 and cfg.seed == 0
        cfg.dump(str(out_dir / name))
    report['mmengine_config_load_dump'] = 'Both selected configs passed'

    probes = ROOT / 'work_dirs/migration_baseline/step3_environment_20261011'
    expected = json.loads(
        (probes / 'legacy_dependency_probes.json').read_text())
    arrays = np.load(str(probes / 'legacy_dependency_probes.npz'))
    measured = {}
    for name, reference in expected['metrics'].items():
        pred, gt = arrays[name + '_pred'], arrays[name + '_gt']
        try:
            # MedPy 0.4 averages directional means; 0.5 pools surfaces.
            # The dataset port must explicitly retain this legacy formula.
            value = float(np.mean((asd(pred, gt), asd(gt, pred))))
            upstream_value = float(assd(pred, gt))
            fallback = False
        except Exception:
            value, upstream_value, fallback = 1.0, 1.0, True
        assert fallback == reference['exception_fallback']
        assert abs(value - reference['assd']) <= 1e-10
        measured[name] = dict(
            legacy_compatible_assd=value,
            upstream_assd=upstream_value,
            exception_fallback=fallback)
    report['medical_dependency_probes'] = dict(
        cases=measured,
        required_dataset_port='Mean of directional ASDs, empty fallback 1.0',
        legacy_formula_matches=True)

    def function_ast(source):
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.body and isinstance(
                    node.body[0], ast.Expr) and isinstance(
                        node.body[0].value, ast.Constant) and isinstance(
                            node.body[0].value.value, str):
                node.body.pop(0)
        return ast.dump(tree, include_attributes=False)

    assert function_ast(inspect.getsource(drop_path)) == function_ast(
        expected['timm_drop_path_source'])
    assert function_ast(
        inspect.getsource(_no_grad_trunc_normal_)) == function_ast(
            expected['timm_trunc_normal_source'])
    torch.manual_seed(expected['seed_drop_path'])
    actual = DropPath(.5).train()(torch.ones(16, 2, 3)).numpy()
    np.testing.assert_array_equal(actual, arrays['drop_path'])
    torch.manual_seed(expected['seed_trunc_normal'])
    weight = torch.empty(32, 16)
    trunc_normal_(weight, std=.02)
    np.testing.assert_allclose(
        weight.numpy(), arrays['trunc_normal'], rtol=0, atol=1e-8)
    assert to_2tuple(16) == (16, 16)
    report['timm_layer_probes'] = dict(
        source_ast_matches=True,
        drop_path_exact=True,
        trunc_normal_max_absolute_error=float(
            np.max(np.abs(weight.numpy() - arrays['trunc_normal']))))
    image = torch.linspace(0, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    image.requires_grad_(True)
    augmented = kornia.augmentation.ColorJitter(
        brightness=.2, contrast=.2, saturation=.2, hue=.2)(
            image)
    blurred = kornia.filters.GaussianBlur2d((3, 3), (.5, .5))(augmented)
    blurred.sum().backward()
    assert torch.isfinite(blurred).all() and torch.isfinite(image.grad).all()
    report['kornia_augmentation_forward_backward'] = 'Passed'
    resized = cv2.resize(np.ones((16, 16), np.float32), (32, 32))
    assert resized.shape == (32, 32) and resized.dtype == np.float32
    report['opencv_resize'] = 'Passed'
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(out_dir / 'tensorboard_probe')) as writer:
        writer.add_scalar('environment/check', 1, 0)
    for name in ('matplotlib', 'pandas', 'seaborn', 'sklearn', 'SimpleITK',
                 'cityscapesscripts.helpers.labels'):
        importlib.import_module(name)
    report['tooling_imports_tensorboard_write'] = 'Passed'


def gpu_checks(report):
    import torch
    from mmcv.cnn import ConvModule
    from mmcv.ops import (get_compiler_version, get_compiling_cuda_version,
                          nms, roi_align)

    assert torch.cuda.is_available(), 'Host GPU access is required'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.cuda.set_per_process_memory_fraction(.1)
    torch.manual_seed(0)
    model = ConvModule(3, 8, 3, padding=1, norm_cfg=dict(type='BN')).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=6e-5, foreach=False, fused=False)
    optimizer.zero_grad(set_to_none=False)
    inputs = torch.randn(2, 3, 32, 32, device='cuda')
    loss = model(inputs).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.],
                          [20., 20., 30., 30.]])
    scores = torch.tensor([.9, .8, .7])
    _, cpu_keep = nms(boxes, scores, .5)
    _, cuda_keep = nms(boxes.cuda(), scores.cuda(), .5)
    assert cpu_keep.tolist() == cuda_keep.cpu().tolist() == [0, 2]
    feature = torch.arange(
        256, device='cuda',
        dtype=torch.float32).reshape(1, 1, 16, 16).requires_grad_(True)
    rois = torch.tensor([[0., 1., 1., 10., 10.]], device='cuda')
    aligned = roi_align(feature, rois, (3, 3), 1.)
    aligned.sum().backward()
    assert aligned.shape == (1, 1, 3, 3)
    assert torch.isfinite(aligned).all() and torch.isfinite(feature.grad).all()
    torch.testing.assert_close(
        torch.fft.ifft2(torch.fft.fft2(inputs)).real,
        inputs,
        rtol=1e-5,
        atol=1e-5)
    torch.cuda.synchronize()
    report['gpu'] = dict(
        name=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        cuda_runtime=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        compiler=get_compiler_version(),
        mmcv_build_cuda=get_compiling_cuda_version(),
        conv_bn_adamw_forward_backward='Passed',
        cpu_cuda_nms='Passed',
        cuda_roi_align_forward_backward='Passed',
        fft_roundtrip='Passed',
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        test_allocator_fraction=.1,
        tf32_enabled=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    branch = subprocess.check_output(['git', 'branch', '--show-current'],
                                     cwd=str(ROOT),
                                     text=True).strip()
    if branch != 'mmcv2':
        parser.error('All migration work must occur on mmcv2')
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.relative_to(ROOT / 'work_dirs')
    out_dir.mkdir(parents=True, exist_ok=False)
    os.environ['MPLCONFIGDIR'] = str(out_dir / 'matplotlib_cache')
    report = dict(python=sys.version, executable=sys.executable, branch=branch)
    try:
        dependency_checks(out_dir, report)
        gpu_checks(report)
        report['passed'] = True
    except Exception as error:
        report.update(passed=False, error=repr(error))
        raise
    finally:
        (out_dir /
         'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
