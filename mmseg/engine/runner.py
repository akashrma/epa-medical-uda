"""MMEngine runner with seeded CPU initialization and full-state resume."""

import torch
from copy import deepcopy
from inspect import signature
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from .state import capture_rng_state, restore_rng_state


class EPARunner(Runner):

    @classmethod
    def from_cfg_with_model(cls, cfg, model):
        """Use an initialized model while retaining a serializable config."""
        config = deepcopy(cfg)
        accepted = signature(Runner.__init__).parameters
        options = {
            key: value
            for key, value in config.items()
            if key in accepted and key != 'cfg'
        }
        options.update(model=model, cfg=config)
        return cls(**options)

    def __init__(self, *args, model=None, **kwargs):
        # A caller may already have built/initialized the model after seeding.
        # Native Runner's seed reset must not rewind that model's RNG stream.
        self._provided_model_rng = capture_rng_state() \
            if isinstance(model, torch.nn.Module) else None
        super().__init__(*args, model=model, **kwargs)

    def wrap_model(self, model_wrapper_cfg, model):
        if self._provided_model_rng is not None:
            restore_rng_state(self._provided_model_rng)
            self._provided_model_rng = None
        if not is_model_wrapper(model) and not getattr(model, '_is_init',
                                                       False):
            # Initialize on CPU before device transfer so seeded weights do
            # not depend on the CUDA initialization RNG stream.
            model.init_weights()
        if self.distributed and hasattr(model, 'get_training_state'):
            raise NotImplementedError('Distributed UDA is unsupported')
        return super().wrap_model(model_wrapper_cfg, model)

    def _init_model_weights(self):
        model = getattr(self.model, 'module', self.model)
        if not getattr(model, '_is_init', False):
            super()._init_model_weights()

    def resume(self,
               filename,
               resume_optimizer=True,
               resume_param_scheduler=True,
               map_location='cpu'):
        if not resume_optimizer or not resume_param_scheduler:
            raise ValueError('Native resume requires optimizer and scheduler')
        self._resume = True
        return super().resume(
            filename,
            resume_optimizer=True,
            resume_param_scheduler=True,
            map_location='cpu')

    def load_checkpoint(self,
                        filename,
                        map_location='cpu',
                        strict=False,
                        revise_keys=None):
        if revise_keys is None:
            revise_keys = [(r'^module.', '')]
        return super().load_checkpoint(
            filename,
            map_location=map_location,
            strict=strict or self._resume,
            revise_keys=revise_keys)
