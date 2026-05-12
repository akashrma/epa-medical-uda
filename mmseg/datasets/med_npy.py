# Custom dataset for MR/CT .npy segmentation lists.

import os
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable

from mmseg.utils import get_root_logger
from mmseg.core import eval_metrics
from .builder import DATASETS
from .custom import CustomDataset
from collections import OrderedDict

try:
    from medpy.metric.binary import assd as medpy_assd
except ImportError:  # pragma: no cover - optional dependency
    medpy_assd = None


@DATASETS.register_module()
class MedNpyListDataset(CustomDataset):
    """Dataset that reads paired image/label .npy paths from text files."""

    CLASSES = ('background', 'Myo', 'LAC', 'LVC', 'AA')
    PALETTE = [
        [0, 0, 0],
        [255, 248, 220],
        [100, 149, 237],
        [102, 205, 170],
        [205, 133, 63],
    ]

    def __init__(self,
                 img_list,
                 ann_list=None,
                 data_root=None,
                 **kwargs):
        self._data_root = data_root
        self.img_list = self._resolve_path(img_list)
        self.ann_list = self._resolve_path(ann_list) if ann_list else None
        super().__init__(
            img_dir='',
            ann_dir='' if ann_list else None,
            img_suffix='',
            seg_map_suffix='',
            data_root=None,
            **kwargs)

    def _resolve_path(self, path):
        if path is None:
            return None
        if osp.isabs(path):
            return path

        candidates = []
        if self._data_root:
            candidates.append(osp.join(self._data_root, path))

        file_dir = osp.dirname(__file__)  # .../MIC/seg/mmseg/datasets
        seg_root = osp.abspath(osp.join(file_dir, '..', '..'))  # .../MIC/seg
        mic_root = osp.abspath(osp.join(file_dir, '..', '..', '..'))  # .../MIC
        repo_root = osp.abspath(osp.join(file_dir, '..', '..', '..', '..'))  # .../medUDA
        launch_root = os.getcwd()
        candidates.extend([
            osp.join(launch_root, path),
            osp.join(seg_root, path),
            osp.join(mic_root, path),
            osp.join(repo_root, path),
        ])

        stripped = path
        while stripped.startswith('../'):
            stripped = stripped[3:]
            candidates.extend([
                osp.join(seg_root, stripped),
                osp.join(mic_root, stripped),
                osp.join(repo_root, stripped),
            ])

        for cand in candidates:
            cand = osp.abspath(cand)
            if osp.exists(cand):
                return cand

        return path

    def _read_list(self, list_path):
        base_dir = osp.dirname(list_path)
        paths = []
        for p in mmcv.list_from_file(list_path):
            if not p:
                continue
            if osp.isabs(p):
                paths.append(p)
            else:
                candidate = osp.abspath(osp.join(base_dir, p))
                paths.append(candidate if osp.exists(candidate) else self._resolve_path(p))
        return paths

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        img_paths = self._read_list(self.img_list)
        ann_paths = None
        if self.ann_list is not None:
            ann_paths = self._read_list(self.ann_list)
            if len(img_paths) != len(ann_paths):
                raise ValueError(
                    f'Image and annotation list lengths differ: '
                    f'{len(img_paths)} vs {len(ann_paths)}')
        elif not self.test_mode:
            raise ValueError('ann_list must be set when test_mode=False')

        img_infos = []
        for i, img_path in enumerate(img_paths):
            img_path = self._resolve_path(img_path)
            img_info = dict(filename=img_path)
            if ann_paths is not None:
                seg_map = self._resolve_path(ann_paths[i])
                img_info['ann'] = dict(seg_map=seg_map)
            img_infos.append(img_info)

        print_log(
            f'Loaded {len(img_infos)} images from {self.img_list}',
            logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = None
        results['seg_prefix'] = None
        if self.custom_classes:
            results['label_map'] = self.label_map

    def get_gt_seg_maps(self, efficient_test=False):
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = img_info['ann']['seg_map']
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = np.load(seg_map)
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def evaluate(self,
                 results,
                 metric='mDice',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        if medpy_assd is None:
            raise ImportError('medpy is required to compute ASSD metrics')

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metrics=['mDice'],
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)

        class_names = self.CLASSES
        dice_all = ret_metrics['Dice']
        dice_no_bg = dice_all[1:]
        class_names_no_bg = class_names[1:]

        assd_list = [[] for _ in class_names_no_bg]
        for pred, gt in zip(results, gt_seg_maps):
            if isinstance(pred, str):
                if pred.endswith('.npy'):
                    pred_arr = np.load(pred)
                else:
                    pred_arr = mmcv.imread(pred, flag='unchanged', backend='pillow')
            else:
                pred_arr = pred
            if isinstance(gt, str):
                gt_arr = np.load(gt)
            else:
                gt_arr = gt

            for idx, c in enumerate(range(1, num_classes)):
                pred_c = (pred_arr == c).astype(np.uint8)
                gt_c = (gt_arr == c).astype(np.uint8)
                try:
                    assd_val = float(medpy_assd(pred_c, gt_c))
                except Exception:
                    assd_val = 1.0
                assd_list[idx].append(assd_val)

        assd_arr = np.array([np.mean(v) if v else np.nan for v in assd_list])

        eval_results = OrderedDict()
        eval_results['mDice'] = float(np.nanmean(dice_no_bg))
        eval_results['mASSD'] = float(np.nanmean(assd_arr))

        for idx, name in enumerate(class_names_no_bg):
            eval_results[f'Dice.{name}'] = float(dice_no_bg[idx])
            eval_results[f'ASSD.{name}'] = float(assd_arr[idx])

        # Logging
        dice_table = PrettyTable()
        dice_table.add_column('Class', class_names_no_bg)
        dice_table.add_column('Dice', np.round(dice_no_bg * 100, 2))
        assd_table = PrettyTable()
        assd_table.add_column('Class', class_names_no_bg)
        assd_table.add_column('ASSD', np.round(assd_arr, 4))
        print_log('Per-class Dice (exclude background):', logger)
        print_log('\n' + dice_table.get_string(), logger=logger)
        print_log('Per-class ASSD (exclude background):', logger)
        print_log('\n' + assd_table.get_string(), logger=logger)

        summary_table = PrettyTable()
        summary_table.add_column('mDice', [np.round(eval_results['mDice'] * 100, 2)])
        summary_table.add_column('mASSD', [np.round(eval_results['mASSD'], 4)])
        print_log('Summary:', logger)
        print_log('\n' + summary_table.get_string(), logger=logger)

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)

        return eval_results
