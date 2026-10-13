"""Medical metric definitions matching the frozen MedPy 0.4.0 baseline."""

import numpy as np

try:
    from medpy.metric.binary import asd
except ImportError:
    asd = None


def legacy_assd(result, reference, voxelspacing=None, connectivity=1):
    """Average the two directional means, as MedPy 0.4.0 did.

    MedPy 0.5 pools surface distances instead, weighting by surface size.
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


def legacy_dc(result, reference):
    """Binary Dice with MedPy 0.4.0's both-empty value of zero."""
    result = np.atleast_1d(np.asarray(result).astype(np.bool_))
    reference = np.atleast_1d(np.asarray(reference).astype(np.bool_))
    denominator = np.count_nonzero(result) + np.count_nonzero(reference)
    if denominator == 0:
        return 0.0
    return 2.0 * np.count_nonzero(result & reference) / float(denominator)
