"""Tensor batches for the local UDA forward contract (no DataContainer)."""

import torch
import torch.nn.functional as F
from collections.abc import Mapping, Sequence
from mmengine.registry import FUNCTIONS
from torch.utils.data import default_collate

METADATA_KEYS = frozenset(('img_metas', 'target_img_metas'))
SPATIAL_KEYS = frozenset(
    ('img', 'target_img', 'gt_semantic_seg', 'valid_pseudo_mask'))


def _collate(values, key=None):
    first = values[0]
    if key in METADATA_KEYS and isinstance(first, Mapping):
        # Keep normalization arrays and geometry as per-image CPU metadata.
        return list(values)
    if torch.is_tensor(first):
        if key in SPATIAL_KEYS and first.ndim == 3:
            if any(v.shape[:-2] != first.shape[:-2] for v in values):
                raise ValueError(f'Inconsistent channel dimensions for {key}')
            height = max(v.shape[-2] for v in values)
            width = max(v.shape[-1] for v in values)
            # Match the old container's padding_value=0. Pipeline Pad has
            # already applied image=0/label=255 padding before this step.
            values = [
                F.pad(v, (0, width - v.shape[-1], 0, height - v.shape[-2]))
                for v in values
            ]
        return torch.stack(values)
    if isinstance(first, Mapping):
        if any(set(v) != set(first) for v in values):
            raise ValueError('Samples must have identical batch keys')
        return {k: _collate([v[k] for v in values], k) for k in first}
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        if any(len(v) != len(first) for v in values):
            raise ValueError('Samples must have the same augmentation count')
        return [_collate(list(v), key) for v in zip(*values)]
    return default_collate(values)


@FUNCTIONS.register_module()
def uda_collate(batch):
    """Stack source/target tensors; preserve metadata and TTA nesting.

    Training yields NCHW tensors and lists of metadata dictionaries. Testing
    keeps an outer augmentation list around each tensor/metadata batch.
    Normalization, resizing and pipeline padding are never repeated here.
    """
    if not batch:
        raise ValueError('Cannot collate an empty batch')
    return _collate(batch)
