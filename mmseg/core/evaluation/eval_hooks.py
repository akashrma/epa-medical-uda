# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Validation/testing now use native loops and evaluators.

from mmengine.runner import TestLoop, ValLoop

__all__ = ['ValLoop', 'TestLoop']
