# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets, separate_eval=True):
        super(ConcatDataset, self).__init__(datasets)
        self.separate_eval = separate_eval
        self.CLASSES = datasets[0].CLASSES
        self.PALETTE = datasets[0].PALETTE

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Evaluate results for each sub-dataset separately."""
        assert len(results) == self.cumulative_sizes[-1], \
            'Dataset and results have different sizes'

        eval_results = {}
        start_idx = 0
        for idx, (dataset, end_idx) in enumerate(
                zip(self.datasets, self.cumulative_sizes)):
            dataset_results = results[start_idx:end_idx]
            start_idx = end_idx
            dataset_eval = dataset.evaluate(
                dataset_results, metric=metric, logger=logger, **kwargs)
            prefix = getattr(dataset, 'eval_name', f'dataset{idx}')
            for k, v in dataset_eval.items():
                eval_results[f'{prefix}.{k}'] = v
        return eval_results


@DATASETS.register_module()
class RepeatDataset(object):
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        self.PALETTE = dataset.PALETTE
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get item from original dataset."""
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """The length is multiplied by ``times``"""
        return self.times * self._ori_len
