# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os.path as osp
import io

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadNpyImageFromFile(object):
    """Load a .npy image from file.

    This loader expects a single-channel image stored as a numpy array.
    It expands to 3 channels, scales values, and converts to BGR to align
    with existing preprocessing in SE_ASA.

    Args:
        to_float32 (bool): Whether to convert the loaded image to float32.
            Defaults to True.
        input_range (str): Expected range of input values. Supported:
            '-1_1', '0_1', or '0_255'. Defaults to '-1_1'.
        to_bgr (bool): Whether to convert RGB to BGR by channel reversal.
            Defaults to True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=True,
                 input_range='-1_1',
                 to_bgr=True,
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.input_range = input_range
        self.to_bgr = to_bgr
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _scale_image(self, img):
        if self.input_range == '-1_1':
            return (img + 1.0) * 127.5
        if self.input_range == '0_1':
            return img * 255.0
        if self.input_range == '0_255':
            return img
        raise ValueError(f'Unsupported input_range: {self.input_range}')

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = np.load(io.BytesIO(img_bytes))

        if img.ndim == 2:
            img = img[:, :, None]
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        img = self._scale_image(img)

        if self.to_bgr:
            img = img[:, :, ::-1].copy()

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f" input_range='{self.input_range}',"
        repr_str += f' to_bgr={self.to_bgr})'
        return repr_str


@PIPELINES.register_module()
class LoadNpzSliceFromFile(object):
    """Load a 2D slice from a .npz volume for test-time inference.

    Expected ``results['img_info']`` fields:
      - filename: path to .npz file containing arr_0 (image volume)
      - slice_idx: index of the target slice
      - slice_axis: axis along which slices are enumerated
    """

    def __init__(self,
                 to_float32=True,
                 input_range='-1_1',
                 modality=None,
                 flip_axes=(0, 1),
                 to_bgr=True,
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.input_range = input_range
        self.modality = modality
        self.flip_axes = tuple(flip_axes) if flip_axes is not None else None
        self.to_bgr = to_bgr
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _scale_image(self, img):
        if self.input_range == '-1_1':
            return (img + 1.0) * 127.5
        if self.input_range == '0_1':
            return img * 255.0
        if self.input_range == '0_255':
            return img
        raise ValueError(f'Unsupported input_range: {self.input_range}')

    def _normalize_modality(self, img):
        if self.modality is None:
            return img
        modality = str(self.modality).upper()
        if modality == 'CT':
            return ((img - (-2.8)) / (3.2 - (-2.8))) * 2.0 - 1.0
        if modality == 'MR':
            return ((img - (-1.8)) / (4.4 - (-1.8))) * 2.0 - 1.0
        return img

    def _extract_slice(self, volume, slice_axis, slice_idx):
        if slice_axis == 0:
            return volume[slice_idx, :, :]
        if slice_axis == 1:
            return volume[:, slice_idx, :]
        if slice_axis == 2:
            return volume[:, :, slice_idx]
        raise ValueError(f'Unsupported slice_axis: {slice_axis}')

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        vol_bytes = self.file_client.get(filename)
        with np.load(io.BytesIO(vol_bytes)) as npz_file:
            volume = npz_file['arr_0']

        if self.flip_axes:
            for axis in self.flip_axes:
                volume = np.flip(volume, axis=axis)

        slice_axis = int(results['img_info'].get('slice_axis', 2))
        slice_idx = int(results['img_info']['slice_idx'])
        img = self._extract_slice(volume, slice_axis, slice_idx)
        img = self._normalize_modality(img.astype(np.float32))

        if img.ndim == 2:
            img = img[:, :, None]
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        img = self._scale_image(img)
        img = np.clip(img, 0, 255)

        if self.to_bgr:
            img = img[:, :, ::-1].copy()

        if self.to_float32:
            img = img.astype(np.float32)

        base_name = osp.basename(filename)
        if base_name.endswith('.npz'):
            base_name = base_name[:-4]
        ori_filename = f'{base_name}_slice_{slice_idx:04d}.png'

        results['filename'] = filename
        results['ori_filename'] = ori_filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f" input_range='{self.input_range}',"
        repr_str += f" modality='{self.modality}',"
        repr_str += f' flip_axes={self.flip_axes},'
        repr_str += f' to_bgr={self.to_bgr})'
        return repr_str


@PIPELINES.register_module()
class LoadNpyAnnotations(object):
    """Load .npy annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
        file_client_args (dict): Arguments to instantiate a FileClient.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk')):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']

        seg_bytes = self.file_client.get(filename)
        gt_semantic_seg = np.load(io.BytesIO(seg_bytes))

        if gt_semantic_seg.ndim > 2:
            gt_semantic_seg = np.squeeze(gt_semantic_seg)

        gt_semantic_seg = gt_semantic_seg.astype(np.uint8)

        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id

        if self.reduce_zero_label:
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255

        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label})'
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
