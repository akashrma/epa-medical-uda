# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: MMEngine training; original config translation is separate.

from copy import deepcopy
from mmengine.config import Config
from mmengine.runner import set_random_seed

from mmseg.engine import EPARunner
from mmseg.utils import get_root_logger


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Train an initialized local model with a native MMEngine config.

    Build data, runner and optimizer settings with build_runtime_config.
    Refuse experiment-only settings rather than silently changing them.
    The retained dataset argument is superseded by train_dataloader.dataset.
    """
    if 'train_dataloader' not in cfg or 'optim_wrapper' not in cfg:
        raise ValueError('Translate the experiment to MMEngine config first')
    native = deepcopy(cfg)
    if not isinstance(native, Config):
        native = Config(native)
    native.launcher = 'pytorch' if distributed else 'none'
    if not validate:
        native.val_dataloader = native.val_cfg = native.val_evaluator = None
    runner = EPARunner.from_cfg_with_model(native, model)
    runner.train()
    return runner


__all__ = ['train_segmentor', 'set_random_seed', 'get_root_logger']
