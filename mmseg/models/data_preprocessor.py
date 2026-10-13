"""Device transfer for images already normalized by the dataset pipeline."""

from mmengine.model import BaseDataPreprocessor

from mmseg.datasets.collate import METADATA_KEYS
from mmseg.registry import MODELS


@MODELS.register_module()
class UDADataPreprocessor(BaseDataPreprocessor):
    """Keep the fork's forward keys and per-image CPU metadata intact.

    This explicitly implements the local model input contract rather than
    requiring upstream SegDataSample/inputs packing. No image processing or
    random draws occur at this stage. Use model.data_preprocessor in native
    training/validation steps, or the standalone adapter for inference.
    """

    def forward(self, data, training=False):
        return {
            key: value if key in METADATA_KEYS else self.cast_data(value)
            for key, value in data.items()
        }
