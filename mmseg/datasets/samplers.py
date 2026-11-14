"""Single-GPU slice sampling with a replayable global RNG stream."""

import torch
from torch.utils.data import RandomSampler

from mmseg.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class RandomSliceSampler(RandomSampler):
    """Sample one permutation per epoch using the global Torch generator.

    Reserve an iterator seed before drawing the permutation so loader and
    sampler RNG consumption remain stable across training and resume.
    """

    def __init__(self, dataset, seed=None, generator=None):
        super().__init__(dataset, replacement=False, generator=generator)
        # Runner may pass a seed argument. The global RNG is seeded by its
        # randomness settings before loader iteration.
        self.seed = seed

    def __iter__(self):
        if self.generator is None:
            torch.empty((), dtype=torch.int64).random_().item()
        yield from torch.randperm(
            len(self.data_source), generator=self.generator).tolist()

    def set_epoch(self, epoch):
        # Continue the global RNG stream across epoch boundaries.
        pass
