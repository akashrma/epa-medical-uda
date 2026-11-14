"""Strict student loading from segmentor and EPA checkpoints."""

import re

from mmengine.runner import CheckpointLoader


def load_student_checkpoint(model, filename, revise_keys=None):
    checkpoint = CheckpointLoader.load_checkpoint(filename, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    revisions = revise_keys if revise_keys is not None else [(r'^module\.',
                                                              '')]
    for pattern, replacement in revisions:
        state = {re.sub(pattern, replacement, k): v for k, v in state.items()}
    if any(k.startswith('model.') for k in state):
        state = {
            k[len('model.'):]: v
            for k, v in state.items() if k.startswith('model.')
        }
    model.load_state_dict(state, strict=True)
    return checkpoint


def set_checkpoint_metadata(model,
                            checkpoint,
                            dataset=None,
                            classes=None,
                            palette=None):
    meta = checkpoint.get('meta', {}) if checkpoint else {}
    dataset_meta = meta.get('dataset_meta', {})
    fallback_classes = meta.get('CLASSES', getattr(dataset, 'CLASSES', None))
    fallback_palette = meta.get('PALETTE', getattr(dataset, 'PALETTE', None))
    model.CLASSES = classes if classes is not None else dataset_meta.get(
        'classes', fallback_classes)
    model.PALETTE = palette if palette is not None else dataset_meta.get(
        'palette', fallback_palette)
