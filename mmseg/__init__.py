import mmcv
import mmengine
from mmengine.utils import digit_version

from .version import __version__, version_info

MMCV_MIN = '2.1.0'
MMCV_MAX = '2.1.0'
MMENGINE_VERSION = '0.10.7'

mmcv_min_version = digit_version(MMCV_MIN)
mmcv_max_version = digit_version(MMCV_MAX)
mmcv_version = digit_version(mmcv.__version__)

assert (mmcv_min_version <= mmcv_version <= mmcv_max_version), (
    f'MMCV=={mmcv.__version__} is used but incompatible. '
    f'This fork qualifies mmcv=={MMCV_MIN}; see README.md.')

assert digit_version(
    mmengine.__version__) == digit_version(MMENGINE_VERSION), (
        f'MMEngine=={mmengine.__version__} is unsupported. '
        f'Please install mmengine=={MMENGINE_VERSION}.')

__all__ = ['__version__', 'version_info']
