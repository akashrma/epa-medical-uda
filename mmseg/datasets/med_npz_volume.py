# Dataset for .npz volumes (arr_0=image, arr_1=label) with slice-wise inference
# and volume-level evaluation to match SE_ASA/domain_adaptation/eval_UDA.py.

import os
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable

from mmseg.utils import get_root_logger
from .builder import DATASETS
from .custom import CustomDataset
from collections import OrderedDict

try:
    from medpy.metric.binary import assd as medpy_assd
    from medpy.metric.binary import dc as medpy_dc
except ImportError:  # pragma: no cover - optional dependency
    medpy_assd = None
    medpy_dc = None


@DATASETS.register_module()
class MedNpzVolumeDataset(CustomDataset):
    """Dataset that reads .npz volumes and exposes slices for inference.

    Each .npz must contain:
      - arr_0: image volume (H, W, D)
      - arr_1: label volume (H, W, D)
    """

    CLASSES = ('background', 'Myo', 'LAC', 'LVC', 'AA')
    PALETTE = [
        [0, 0, 0],
        [255, 64, 64],
        [64, 255, 64],
        [64, 128, 255],
        [255, 220, 64],
    ]

    def __init__(self,
                 img_list,
                 data_root=None,
                 slice_axis=2,
                 flip_axes=(0, 1),
                 eval_name=None,
                 **kwargs):
        self._data_root = data_root
        self.img_list = self._resolve_path(img_list)
        self.slice_axis = slice_axis
        self.flip_axes = tuple(flip_axes) if flip_axes is not None else None
        self.eval_name = eval_name

        self._volume_paths = []
        self._volume_shapes = []
        self._slice_map = []
        super().__init__(
            img_dir='',
            ann_dir='',
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

        # Try common launch roots so config-relative paths are robust.
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

        # Also try stripping leading "../" segments for cross-root configs.
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

        # Keep previous behavior if no candidate exists.
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
        volume_paths = self._read_list(self.img_list)
        img_infos = []
        self._volume_paths = []
        self._volume_shapes = []
        self._slice_map = []

        for vol_idx, vol_path in enumerate(volume_paths):
            vol_path = self._resolve_path(vol_path)
            with np.load(vol_path) as npz_file:
                data = npz_file['arr_0']

            if data.ndim != 3:
                raise ValueError(f'Expected 3D volume, got shape {data.shape}')

            self._volume_paths.append(vol_path)
            self._volume_shapes.append(data.shape)

            num_slices = data.shape[self.slice_axis]
            for slice_idx in range(num_slices):
                img_info = dict(
                    filename=vol_path,
                    slice_idx=slice_idx,
                    slice_axis=self.slice_axis,
                    volume_idx=vol_idx,
                )
                img_infos.append(img_info)
                self._slice_map.append((vol_idx, slice_idx))

        print_log(
            f'Loaded {len(self._volume_paths)} volumes with '
            f'{len(img_infos)} slices from {self.img_list}',
            logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = None
        results['seg_prefix'] = None
        if self.custom_classes:
            results['label_map'] = self.label_map

    def _apply_flip(self, arr):
        if not self.flip_axes:
            return arr
        for axis in self.flip_axes:
            arr = np.flip(arr, axis=axis)
        return arr

    def _compute_metric(self, pred, target, num_classes, compute_assd=False):
        pred = pred.astype(int)
        target = target.astype(int)
        dice_list = []
        assd_list = []

        for c in range(1, num_classes):
            test_pred = pred.copy()
            test_pred[test_pred != c] = 0

            test_gt = target.copy()
            test_gt[test_gt != c] = 0

            dice = medpy_dc(test_pred, test_gt)

            dice_list.append(dice)
            if compute_assd:
                try:
                    assd_metric = medpy_assd(test_pred, test_gt)
                except Exception:
                    assd_metric = 1
                assd_list.append(assd_metric)

        if compute_assd:
            assd_arr = np.array(assd_list)
        else:
            assd_arr = None
        return np.array(dice_list), assd_arr

    def evaluate(self,
                 results,
                 metric='mDice',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        if medpy_dc is None:
            raise ImportError('medpy is required to compute Dice metrics')

        if isinstance(metric, str):
            metric = [metric]
        # Accept both spellings; keep internal handling on mASSD.
        metric = ['mASSD' if m == 'mASD' else m for m in metric]
        allowed_metrics = ['mDice', 'mASSD']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        compute_assd = 'mASSD' in metric
        if compute_assd and medpy_assd is None:
            raise ImportError('medpy is required to compute ASSD metrics')

        num_classes = len(self.CLASSES)
        class_names_no_bg = self.CLASSES[1:]

        volume_preds = [
            np.zeros(shape, dtype=np.int16) for shape in self._volume_shapes
        ]

        for idx, pred in enumerate(results):
            if isinstance(pred, str):
                if pred.endswith('.npy'):
                    pred_arr = np.load(pred)
                else:
                    pred_arr = mmcv.imread(pred, flag='unchanged', backend='pillow')
            else:
                pred_arr = pred

            pred_arr = np.squeeze(pred_arr)

            vol_idx, slice_idx = self._slice_map[idx]
            if self.slice_axis == 0:
                volume_preds[vol_idx][slice_idx, :, :] = pred_arr
            elif self.slice_axis == 1:
                volume_preds[vol_idx][:, slice_idx, :] = pred_arr
            elif self.slice_axis == 2:
                volume_preds[vol_idx][:, :, slice_idx] = pred_arr
            else:
                raise ValueError(f'Unsupported slice_axis: {self.slice_axis}')

        dice_list = []
        assd_list = []
        for vol_idx, vol_path in enumerate(self._volume_paths):
            with np.load(vol_path) as npz_file:
                gt = npz_file['arr_1']
            gt = self._apply_flip(gt)
            pred_vol = volume_preds[vol_idx]
            dice, assd = self._compute_metric(
                pred_vol, gt, num_classes, compute_assd=compute_assd)
            dice_list.append(dice)
            if compute_assd:
                assd_list.append(assd)

        dice_arr = np.vstack(dice_list)  # N_vol * N_class

        dice_mean = np.mean(dice_arr, axis=0)
        dice_std = np.std(dice_arr, axis=0)
        if compute_assd:
            assd_arr = np.vstack(assd_list)  # N_vol * N_class
            assd_mean = np.mean(assd_arr, axis=0)
            assd_std = np.std(assd_arr, axis=0)

        eval_results = OrderedDict()
        eval_results['mDice'] = float(np.mean(dice_mean))

        for idx, name in enumerate(class_names_no_bg):
            eval_results[f'Dice.{name}'] = float(dice_mean[idx])
            if compute_assd:
                eval_results[f'ASSD.{name}'] = float(assd_mean[idx])
        if compute_assd:
            eval_results['mASSD'] = float(np.mean(assd_mean))

        # Logging (match eval_UDA style with mean/std)
        dice_table = PrettyTable()
        dice_table.add_column('Class', class_names_no_bg)
        dice_table.add_column(
            'Dice (mean±std)',
            [f'{m * 100:.1f}({s * 100:.1f})' for m, s in zip(dice_mean, dice_std)]
        )
        print_log('Per-class Dice (exclude background):', logger)
        print_log('\n' + dice_table.get_string(), logger=logger)
        if compute_assd:
            assd_table = PrettyTable()
            assd_table.add_column('Class', class_names_no_bg)
            assd_table.add_column(
                'ASSD (mean±std)',
                [f'{m:.1f}({s:.1f})' for m, s in zip(assd_mean, assd_std)]
            )
            print_log('Per-class ASSD (exclude background):', logger)
            print_log('\n' + assd_table.get_string(), logger=logger)

        summary_table = PrettyTable()
        summary_table.add_column('mDice', [np.round(eval_results['mDice'] * 100, 2)])
        if compute_assd:
            summary_table.add_column('mASSD', [np.round(eval_results['mASSD'], 4)])
        print_log('Summary:', logger)
        print_log('\n' + summary_table.get_string(), logger=logger)

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)

        return eval_results
