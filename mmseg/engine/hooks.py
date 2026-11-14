"""Native precision and complete training-state checkpoint hooks."""

import torch
from mmengine.config import Config
from mmengine.hooks import CheckpointHook, Hook

from mmseg.registry import HOOKS
from .state import capture_rng_state


def backend_flags():
    return dict(
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


@HOOKS.register_module()
class PrecisionPolicyHook(Hook):
    priority = 'HIGHEST'

    def before_run(self, runner):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # Use FP32 convolutions to limit decoder gradient noise.
        # Matmul TF32 and cuDNN benchmark remain enabled.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.use_deterministic_algorithms(False)


@HOOKS.register_module()
class EPATrainingStateHook(Hook):
    priority = 'HIGHEST'

    def before_save_checkpoint(self, runner, checkpoint):
        model = getattr(runner.model, 'module', runner.model)
        driver = runner.train_loop.dataloader_iterator
        if driver is None:
            raise RuntimeError(
                'Cannot save resumable state before data starts')
        checkpoint['epa_training'] = dict(
            version=1,
            world_size=runner.world_size,
            total_iters=runner.max_iters,
            rng=capture_rng_state(),
            data=driver.state_dict(),
            backend=backend_flags(),
            gradient_presence=[
                name for name, p in model.named_parameters()
                if p.grad is not None
            ],
            uda=model.get_training_state()
            if hasattr(model, 'get_training_state') else None)

    def after_load_checkpoint(self, runner, checkpoint):
        if not runner._resume:
            return  # Explicit weights initialization.
        state = checkpoint.get('epa_training')
        if not state or state.get('version') != 1:
            raise ValueError('Weights-only checkpoints cannot resume '
                             'native training; use weights initialization')
        if not {'optimizer', 'param_schedulers'}.issubset(checkpoint):
            raise ValueError('Resume needs optimizer and scheduler state')
        if state['world_size'] != runner.world_size or \
                state['total_iters'] != runner.max_iters or \
                state['backend'] != backend_flags():
            raise ValueError('Checkpoint runtime/precision contract differs')
        model = getattr(runner.model, 'module', runner.model)
        if hasattr(model, 'load_training_state'):
            model.load_training_state(state['uda'])
            if model.local_iter != checkpoint['meta']['iter']:
                raise ValueError('Runner and UDA iteration counters differ')
        # Resize lazily allocated dynamic-prototype buffers before native
        # state_dict loading. Fixed EPA and EMA/BN tensor keys remain unchanged.
        for name in ('source_prototypes', 'source_proto_valid',
                     'source_proto_seen'):
            if name in checkpoint['state_dict']:
                saved = checkpoint['state_dict'][name]
                current = getattr(model, name)
                if saved.shape != current.shape:
                    setattr(model, name,
                            torch.empty_like(saved, device=current.device))
        names = set(state['gradient_presence'])
        parameters = dict(model.named_parameters())
        if not names.issubset(parameters):
            raise ValueError('Checkpoint gradient parameter names differ')
        for name, p in parameters.items():
            p.grad = torch.zeros_like(p) if name in names else None
        runner.train_loop.resume_data_state = state['data']
        runner.train_loop.resume_rng_state = state['rng']
        saved_config = checkpoint.get('meta', {}).get('cfg')
        if saved_config:
            saved_config = Config.fromstring(saved_config, '.py')
            sampler = saved_config.get('train_dataloader',
                                       {}).get('sampler', {})
            runner.train_loop.resume_sampler_type = sampler.get('type')


@HOOKS.register_module()
class EPACheckpointHook(CheckpointHook):
    """Capture periodic resume checkpoints after scheduled validation.

    Validation consumes loader RNG even if the best score does not improve.
    A pre-validation checkpoint would therefore restart with a different
    parent RNG stream. Best checkpoints keep native strict-greater selection.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.by_epoch:
            raise ValueError('EPA checkpointing is iteration based')
        self._pending_save = False

    def after_train_iter(self,
                         runner,
                         batch_idx,
                         data_batch=None,
                         outputs=None):
        completed = runner.iter + 1
        loop = runner.train_loop
        due = self.every_n_train_iters(runner, self.interval, self.save_begin)
        validate = (
            runner.val_loop is not None and completed >= loop.val_begin
            and (completed % loop.val_interval == 0
                 or completed == loop.stop_after))
        if due and validate:
            self._pending_save = True
        else:
            super().after_train_iter(runner, batch_idx, data_batch, outputs)

    def after_val_epoch(self, runner, metrics):
        super().after_val_epoch(runner, metrics)
        if self._pending_save:
            self._save_checkpoint_with_step(
                runner, runner.iter,
                dict(iter=runner.iter, epoch=runner.epoch))
            self._pending_save = False
