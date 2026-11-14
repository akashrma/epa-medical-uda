# Obtained from: https://github.com/open-mmlab/mmgeneration
# Copyright (c) OpenMMLab. All rights reserved.
# Modifications: MMEngine wrapping with a single-GPU UDA constraint.

from mmengine.model import MMDistributedDataParallel
from mmengine.registry import MODEL_WRAPPERS


@MODEL_WRAPPERS.register_module()
class DistributedDataParallelWrapper(MMDistributedDataParallel):
    """Native wrapper for ordinary segmentors.

    DACS performs backward inside its train_step. The generic DDP train_step
    would perform an additional backward/update. Multi-rank UDA needs a
    dedicated synchronization strategy and is rejected here.
    """

    def __init__(self, module, **kwargs):
        if hasattr(module, 'get_training_state'):
            raise NotImplementedError('Distributed UDA synchronization is '
                                      'unsupported')
        super().__init__(module, **kwargs)
