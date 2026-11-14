"""FP32 optimization preserving the fork's multiple-backward updates."""

from copy import deepcopy
from mmengine.optim import DefaultOptimWrapperConstructor, OptimWrapper

from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS


@OPTIM_WRAPPERS.register_module()
class UDAOptimWrapper(OptimWrapper):
    """Count optimizer updates, allowing several backwards per update."""

    def __init__(self, optimizer, accumulative_counts=1, **kwargs):
        if accumulative_counts != 1:
            raise ValueError('UDA requires one optimizer update per batch')
        super().__init__(optimizer, accumulative_counts=1, **kwargs)

    @property
    def param_groups(self):
        # Native wrappers add a synthetic group to track a default LR.
        # Schedule only the actual named-parameter groups.
        return self.optimizer.param_groups

    def get_lr(self):
        return dict(lr=[group['lr'] for group in self.optimizer.param_groups])

    def backward(self, loss, **kwargs):
        loss.backward(**kwargs)

    def step(self, **kwargs):
        super().step(**kwargs)
        self._inner_count += 1

    def update_params(self, loss, step_kwargs=None, zero_kwargs=None):
        clearing = dict(zero_kwargs or {})
        clearing.setdefault('set_to_none', False)
        self.zero_grad(**clearing)
        self.backward(loss)
        self.step(**(step_kwargs or {}))


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class EPAOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """Preserve every named parameter, including detached EMA parameters.

    AdamW custom keys
    use MMCV's longest-match/alphabetical tie order. No blanket bias or BN
    decay exclusions are introduced. The group order matches named_parameters.
    """

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        config = deepcopy(optim_wrapper_cfg)
        optimizer = config['optimizer']
        if optimizer['type'] != 'AdamW':
            raise ValueError('Expected the AdamW optimizer')
        optimizer.setdefault('eps', 1e-8)
        optimizer.setdefault('amsgrad', False)
        optimizer.update(foreach=False, fused=False)
        if set(paramwise_cfg or {}) - {'custom_keys'}:
            raise ValueError('Parameter grouping supports custom_keys only')
        super().__init__(config, paramwise_cfg)

    def add_params(self, params, module, prefix='', **kwargs):
        custom = self.paramwise_cfg.get('custom_keys', {})
        keys = sorted(sorted(custom), key=len, reverse=True)
        for name, parameter in module.named_parameters():
            group = dict(params=[parameter])
            full_name = f'{prefix}.{name}' if prefix else name
            for key in keys:
                if key in full_name:
                    group['lr'] = self.base_lr * custom[key].get('lr_mult', 1.)
                    group['weight_decay'] = self.base_wd * custom[key].get(
                        'decay_mult', 1.)
                    break
            params.append(group)
