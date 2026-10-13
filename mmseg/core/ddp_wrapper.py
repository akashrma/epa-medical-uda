# Obtained from: https://github.com/open-mmlab/mmgeneration
# Copyright (c) OpenMMLab. All rights reserved.
# Modifications: Native MMEngine wrapping with explicit UDA qualification gate.

from mmengine.model import MMDistributedDataParallel
from mmengine.registry import MODEL_WRAPPERS


@MODEL_WRAPPERS.register_module()
class DistributedDataParallelWrapper(MMDistributedDataParallel):
    """Native wrapper for ordinary segmentors.

    DACS performs backward inside its train_step. The generic DDP train_step
    would perform an additional backward/update. Multi-rank UDA needs a
    separately qualified synchronization strategy and is rejected here.
    The matched EPA/FDA baselines use one GPU without a DDP wrapper.
    """

    def __init__(self, module, **kwargs):
        if hasattr(module, 'get_training_state'):
            raise NotImplementedError('Distributed UDA synchronization is '
                                      'not qualified for the parity port')
        super().__init__(module, **kwargs)
