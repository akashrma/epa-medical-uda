# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Modification of config and checkpoint to support legacy models
# - Add inference mode and HRDA output flag

import argparse
import os
import os.path as osp
import sys

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


def _sync_dataset_meta(dataset, classes, palette):
    """Propagate resolved class/palette metadata to wrapper and children."""
    if hasattr(dataset, 'CLASSES'):
        dataset.CLASSES = classes
    if hasattr(dataset, 'PALETTE'):
        dataset.PALETTE = palette
    if hasattr(dataset, 'datasets'):
        for ds in dataset.datasets:
            _sync_dataset_meta(ds, classes, palette)


def _iter_dataset_cfgs(cfg_node):
    """Yield leaf dataset configs from dict/list wrappers."""
    if isinstance(cfg_node, (list, tuple)):
        for item in cfg_node:
            yield from _iter_dataset_cfgs(item)
        return

    if not isinstance(cfg_node, dict):
        return

    if 'datasets' in cfg_node and isinstance(cfg_node['datasets'],
                                             (list, tuple)):
        for item in cfg_node['datasets']:
            yield from _iter_dataset_cfgs(item)
        return

    if 'dataset' in cfg_node and isinstance(cfg_node['dataset'], dict):
        yield from _iter_dataset_cfgs(cfg_node['dataset'])
        return

    yield cfg_node


def _fix_img_scale_tuple(dataset_cfg):
    pipeline = dataset_cfg.get('pipeline', None)
    if not isinstance(pipeline, (list, tuple)) or len(pipeline) < 2:
        return
    second = pipeline[1]
    if isinstance(second, dict) and 'img_scale' in second:
        img_scale = second['img_scale']
        if isinstance(img_scale, list):
            second['img_scale'] = tuple(img_scale)


def _set_test_mode(cfg_node, test_mode=True):
    for dataset_cfg in _iter_dataset_cfgs(cfg_node):
        dataset_cfg['test_mode'] = test_mode


def _replace_val_with_test(cfg_node):
    for dataset_cfg in _iter_dataset_cfgs(cfg_node):
        for k, v in list(dataset_cfg.items()):
            if isinstance(v, str):
                dataset_cfg[k] = v.replace('val', 'test')


def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    for dataset_cfg in _iter_dataset_cfgs(cfg.data.get('test', [])):
        _fix_img_scale_tuple(dataset_cfg)
    for dataset_cfg in _iter_dataset_cfgs(cfg.data.get('val', [])):
        _fix_img_scale_tuple(dataset_cfg)
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    if cfg.model.type == 'MultiResEncoderDecoder':
        cfg.model.type = 'HRDAEncoderDecoder'
    if cfg.model.decode_head.type == 'MultiResAttentionWrapper':
        cfg.model.decode_head.type = 'HRDAHead'
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--inference-mode',
        choices=[
            'same',
            'whole',
            'slide',
        ],
        default='same',
        help='Inference mode.')
    parser.add_argument(
        '--test-set',
        action='store_true',
        help='Run inference on the test set')
    parser.add_argument(
        '--hrda-out',
        choices=['', 'LR', 'HR', 'ATT'],
        default='',
        help='Extract LR and HR predictions from HRDA architecture.')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gt-dir',
        help='directory where ground-truth label images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir or args.gt_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show", "--show-dir" or "--gt-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg = update_legacy_cfg(cfg)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        for dataset_cfg in _iter_dataset_cfgs(cfg.data.test):
            pipeline = dataset_cfg.get('pipeline', None)
            if isinstance(pipeline, (list, tuple)) and len(pipeline) > 1 and \
                    isinstance(pipeline[1], dict):
                pipeline[1]['img_ratios'] = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
                pipeline[1]['flip'] = True
    cfg.model.pretrained = None
    _set_test_mode(cfg.data.test, test_mode=True)
    if args.inference_mode == 'same':
        # Use pre-defined inference mode
        pass
    elif args.inference_mode == 'whole':
        print('Force whole inference.')
        cfg.model.test_cfg.mode = 'whole'
    elif args.inference_mode == 'slide':
        print('Force slide inference.')
        cfg.model.test_cfg.mode = 'slide'
        crsize = cfg.data.train.get('sync_crop_size', cfg.crop_size)
        cfg.model.test_cfg.crop_size = crsize
        cfg.model.test_cfg.stride = [int(e / 2) for e in crsize]
        cfg.model.test_cfg.batched_slide = True
    else:
        raise NotImplementedError(args.inference_mode)

    if args.hrda_out == 'LR':
        cfg['model']['decode_head']['fixed_attention'] = 0.0
    elif args.hrda_out == 'HR':
        cfg['model']['decode_head']['fixed_attention'] = 1.0
    elif args.hrda_out == 'ATT':
        cfg['model']['decode_head']['debug_output_attention'] = True
    elif args.hrda_out == '':
        pass
    else:
        raise NotImplementedError(args.hrda_out)

    if args.test_set:
        _replace_val_with_test(cfg.data.test)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), ('model.', '')])
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    # Keep visualization palette/classes consistent with resolved model meta.
    _sync_dataset_meta(dataset, model.CLASSES, model.PALETTE)

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  args.gt_dir, efficient_test, args.opacity)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            res = dataset.evaluate(outputs, args.eval, **kwargs)
            print([k for k, v in res.items() if 'IoU' in k])
            print([round(v * 100, 1) for k, v in res.items() if 'IoU' in k])


if __name__ == '__main__':
    main()
