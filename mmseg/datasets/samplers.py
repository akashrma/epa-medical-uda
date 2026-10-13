"""Preserve the selected PyTorch 1.7.1 shuffle/RNG behavior."""

import torch
from torch.utils.data import RandomSampler

from mmseg.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class LegacyRandomSampler(RandomSampler):
    """Without-replacement sampler with the frozen global RNG semantics.

    Torch 1.7.1 consumed an unused generator seed, then used the global
    generator for randperm. Torch 2 uses the newly seeded generator and
    makes an extra permutation at exhaustion, changing sample order/RNG.
    Reference: pytorch/pytorch v1.7.1 torch/utils/data/sampler.py.
    """

    def __init__(self, dataset, seed=None, generator=None):
        super().__init__(dataset, replacement=False, generator=generator)
        # Runner may pass a seed argument. The global RNG is seeded by its
        # randomness settings before loader iteration, as in the baseline.
        self.seed = seed

    def __iter__(self):
        if self.generator is None:
            torch.empty((), dtype=torch.int64).random_().item()
        yield from torch.randperm(
            len(self.data_source), generator=self.generator).tolist()

    def set_epoch(self, epoch):
        # Single-GPU RandomSampler did not reseed at epoch boundaries.
        pass
