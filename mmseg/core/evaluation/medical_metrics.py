"""Medical surface-distance and volume Dice definitions."""

import numpy as np

try:
    from medpy.metric.binary import asd
except ImportError:
    asd = None


def medical_assd(result, reference, voxelspacing=None, connectivity=1):
    """Average the two directional surface-distance means with equal weight.
    Keep voxel units/default connectivity and let the dataset preserve its
    existing exception-to-1.0 behavior for empty/invalid masks.
    """
    if asd is None:
        raise ImportError('medpy is required to compute ASSD metrics')
    return np.mean((asd(result, reference, voxelspacing, connectivity),
                    asd(reference, result, voxelspacing, connectivity)))


def require_medpy():
    if asd is None:
        raise ImportError('medpy is required to compute ASSD metrics')


def volume_dice(result, reference):
    """Binary Dice with a both-empty value of zero."""
    result = np.atleast_1d(np.asarray(result).astype(np.bool_))
    reference = np.atleast_1d(np.asarray(reference).astype(np.bool_))
    denominator = np.count_nonzero(result) + np.count_nonzero(reference)
    if denominator == 0:
        return 0.0
    return 2.0 * np.count_nonzero(result & reference) / float(denominator)
