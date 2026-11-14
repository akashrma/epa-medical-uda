"""Checkpointable random state and deterministic loader replay."""

import hashlib
import json
import numpy as np
import random
import torch
from pathlib import Path


def capture_rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() else [])


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Checkpoint CUDA device count differs')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def _dataset_signature(dataset):
    signature = dict(
        type=f'{type(dataset).__module__}.{type(dataset).__name__}',
        length=len(dataset))
    if hasattr(dataset, 'pipeline_cfg'):
        signature['pipeline'] = dataset.pipeline_cfg
    elif hasattr(dataset, 'pipeline'):
        signature['pipeline'] = repr(dataset.pipeline)
    if hasattr(dataset, 'resume_sampling_cfg'):
        signature['sampling'] = dataset.resume_sampling_cfg
    for field in ('img_list', 'ann_list', 'batch_filename'):
        value = getattr(dataset, field, None)
        if value is not None:
            path = Path(value).resolve()
            signature[field] = dict(
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for field in ('source', 'target', 'dataset'):
        if hasattr(dataset, field):
            signature[field] = _dataset_signature(getattr(dataset, field))
    if hasattr(dataset, 'indices'):
        signature['indices'] = list(dataset.indices)
    return signature


def loader_signature(loader, sampler_type=None):
    init = loader.worker_init_fn
    function = getattr(init, 'func', init)
    signature = dict(
        dataset=_dataset_signature(loader.dataset),
        batch_size=loader.batch_size,
        drop_last=loader.drop_last,
        num_workers=loader.num_workers,
        persistent_workers=loader.persistent_workers,
        pin_memory=loader.pin_memory,
        sampler=sampler_type or type(loader.sampler).__name__,
        worker_init=getattr(function, '__qualname__', None),
        worker_kwargs=getattr(init, 'keywords', None))
    return hashlib.sha256(
        json.dumps(signature, sort_keys=True,
                   default=str).encode()).hexdigest()


class ReplayDataIterator:
    """Resume without changing sample order, worker RNG, or model RNG.

    For workers, replay the current epoch from its starting RNG state to
    reconstruct each worker's transforms and prefetch position. For zero
    workers, advance indices only, avoiding additional dataset random draws.
    Then restore checkpoint-time parent RNG before fetching the next batch.
    Private iterator access is limited to pinned Torch 2.1's zero-worker
    sampler iterator and worker shutdown.
    """

    def __init__(self, loader, state=None, saved_sampler_type=None):
        if loader.persistent_workers or loader.generator is not None or \
                getattr(loader.sampler, 'generator', None) is not None:
            raise ValueError('Resume requires nonpersistent workers and '
                             'the global loader RNG')
        self.loader = loader
        self.signature = loader_signature(loader)
        self.epoch = 0
        self.consumed = 0
        self.iterator = None
        if state is None:
            self._start_epoch()
        else:
            # Class names can change without changing the sampling policy.
            # Compare every loader setting using the saved name as well.
            signatures = {self.signature}
            if saved_sampler_type:
                signatures.add(loader_signature(loader, saved_sampler_type))
            if state['signature'] not in signatures:
                raise ValueError('Checkpoint data-loader contract differs')
            parent_rng = capture_rng_state()
            self.epoch = state['epoch']
            self.epoch_rng = state['epoch_rng']
            restore_rng_state(self.epoch_rng)
            self.iterator = iter(loader)
            for _ in range(state['consumed']):
                if loader.num_workers:
                    next(self.iterator)
                else:
                    next(self.iterator._sampler_iter)
            self.consumed = state['consumed']
            restore_rng_state(parent_rng)

    def _start_epoch(self):
        self.epoch_rng = capture_rng_state()
        self.consumed = 0
        self.iterator = iter(self.loader)

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.close()
            self.epoch += 1
            if hasattr(self.loader.sampler, 'set_epoch'):
                self.loader.sampler.set_epoch(self.epoch)
            self._start_epoch()
            batch = next(self.iterator)
        self.consumed += 1
        return batch

    def state_dict(self):
        return dict(
            signature=self.signature,
            epoch=self.epoch,
            consumed=self.consumed,
            epoch_rng=self.epoch_rng)

    def close(self):
        if self.iterator is not None and self.loader.num_workers:
            self.iterator._shutdown_workers()
        self.iterator = None
