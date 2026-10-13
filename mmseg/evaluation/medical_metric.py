"""MMEngine evaluator retaining the local medical dataset semantics."""

import numpy as np
import torch
from copy import deepcopy
from mmengine.evaluator import BaseMetric

from mmseg.datasets import build_dataset
from mmseg.registry import METRICS


@METRICS.register_module()
class MedicalMetric(BaseMetric):
    """Collect ordered slice predictions and evaluate with the dataset.

    Use prefix='target' for checkpoint selection (target/mDice) and a
    separate prefix='source' evaluator for diagnostic source reporting.
    Sequential validation samplers are required, including the standard
    strided distributed sampler supported by BaseMetric's rank collection.
    """

    default_prefix = 'target'

    def __init__(self,
                 dataset,
                 metric='mDice',
                 prefix=None,
                 collect_device='cpu',
                 collect_dir=None):
        super().__init__(collect_device, prefix, collect_dir)
        self.dataset_cfg = deepcopy(dataset) if isinstance(dataset,
                                                           dict) else None
        self._dataset = None if self.dataset_cfg is not None else dataset
        self.metric = metric

    @property
    def dataset(self):
        if self._dataset is None:
            self._dataset = build_dataset(self.dataset_cfg,
                                          dict(test_mode=True))
        return self._dataset

    def process(self, data_batch, data_samples):
        for prediction in data_samples:
            if isinstance(prediction, dict):
                prediction = prediction['pred_sem_seg']
            if hasattr(prediction,
                       'data') and not isinstance(prediction,
                                                  (np.ndarray, torch.Tensor)):
                prediction = prediction.data
            if torch.is_tensor(prediction):
                prediction = prediction.detach().cpu().numpy()
            prediction = np.asarray(prediction)
            if prediction.ndim == 3 and prediction.shape[0] == 1:
                prediction = prediction[0]
            if prediction.ndim != 2:
                raise ValueError('Medical predictions must be 2D label maps')
            self.results.append(prediction.copy())

    def compute_metrics(self, results):
        if len(results) != len(self.dataset):
            raise ValueError('Prediction count must match dataset slice order')
        return dict(
            self.dataset.evaluate(
                results, metric=self.metric, logger='silent'))
