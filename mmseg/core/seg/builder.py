# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

from mmseg.registry import TASK_UTILS

PIXEL_SAMPLERS = TASK_UTILS


def build_pixel_sampler(cfg, **default_args):
    """Build pixel sampler for segmentation map."""
    return PIXEL_SAMPLERS.build(cfg, default_args=default_args)
