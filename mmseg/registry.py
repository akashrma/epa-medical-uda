# Copyright (c) OpenMMLab. All rights reserved.
"""Scoped registries for the local EPA segmentation fork."""

from mmengine.registry import DATA_SAMPLERS as ENGINE_SAMPLERS
from mmengine.registry import DATASETS as ENGINE_DATASETS
from mmengine.registry import HOOKS as ENGINE_HOOKS
from mmengine.registry import LOOPS as ENGINE_LOOPS
from mmengine.registry import METRICS as ENGINE_METRICS
from mmengine.registry import MODELS as ENGINE_MODELS
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS as ENGINE_CONSTRUCTORS
from mmengine.registry import OPTIM_WRAPPERS as ENGINE_OPTIM_WRAPPERS
from mmengine.registry import OPTIMIZERS as ENGINE_OPTIMIZERS
from mmengine.registry import PARAM_SCHEDULERS as ENGINE_SCHEDULERS
from mmengine.registry import TASK_UTILS as ENGINE_TASK_UTILS
from mmengine.registry import TRANSFORMS as ENGINE_TRANSFORMS
from mmengine.registry import WEIGHT_INITIALIZERS as ENGINE_INITIALIZERS
from mmengine.registry import Registry

MODELS = Registry(
    'model', parent=ENGINE_MODELS, scope='mmseg', locations=['mmseg.models'])
DATASETS = Registry(
    'dataset',
    parent=ENGINE_DATASETS,
    scope='mmseg',
    locations=['mmseg.datasets'])
DATA_SAMPLERS = Registry(
    'data sampler',
    parent=ENGINE_SAMPLERS,
    scope='mmseg',
    locations=['mmseg.datasets'])
TRANSFORMS = Registry(
    'transform',
    parent=ENGINE_TRANSFORMS,
    scope='mmseg',
    locations=['mmseg.datasets.pipelines'])
METRICS = Registry(
    'metric',
    parent=ENGINE_METRICS,
    scope='mmseg',
    locations=['mmseg.evaluation'])
HOOKS = Registry(
    'hook', parent=ENGINE_HOOKS, scope='mmseg', locations=['mmseg.engine'])
LOOPS = Registry(
    'loop', parent=ENGINE_LOOPS, scope='mmseg', locations=['mmseg.engine'])
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    parent=ENGINE_SCHEDULERS,
    scope='mmseg',
    locations=['mmseg.engine'])
TASK_UTILS = Registry(
    'task util',
    parent=ENGINE_TASK_UTILS,
    scope='mmseg',
    locations=['mmseg.core.seg'])
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=ENGINE_INITIALIZERS,
    scope='mmseg',
    locations=['mmseg.registry'])
OPTIMIZERS = Registry(
    'optimizer',
    parent=ENGINE_OPTIMIZERS,
    scope='mmseg',
    locations=['mmseg.registry'])
OPTIM_WRAPPERS = Registry(
    'optim wrapper',
    parent=ENGINE_OPTIM_WRAPPERS,
    scope='mmseg',
    locations=['mmseg.engine'])
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optim wrapper constructor',
    parent=ENGINE_CONSTRUCTORS,
    scope='mmseg',
    locations=['mmseg.engine'])

__all__ = [
    'MODELS', 'DATASETS', 'TRANSFORMS', 'METRICS', 'HOOKS', 'TASK_UTILS',
    'WEIGHT_INITIALIZERS', 'OPTIMIZERS', 'OPTIM_WRAPPERS',
    'OPTIM_WRAPPER_CONSTRUCTORS', 'LOOPS', 'PARAM_SCHEDULERS', 'DATA_SAMPLERS'
]
