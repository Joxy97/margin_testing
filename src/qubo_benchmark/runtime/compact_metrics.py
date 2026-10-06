"""Checkpoint statistics for independently scored minimization objectives.

TTS99 = t * log(0.01) / log(1-p). Its literal limits are infinity at
p=0 and zero at p=1; the latter is not replaced with one execution budget.
Unavailable energies are misses, and make the mean signed gap unavailable.
"""
import math

import numpy as np


CHECKPOINTS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
AGGREGATE_COLUMNS = ('checkpoint_seconds', 'gap_to_solution', 'hit_probability', 'TTS99')


def tts99(checkpoint_seconds, probability):
    """Return the requested restart formula, including its exact endpoint limits."""
    if not 0 <= probability <= 1:
        raise ValueError('Hit probability must be in [0, 1]')
    if probability == 0:
        return math.inf
    if probability == 1:
        return 0.0
    return float(checkpoint_seconds) * math.log(0.01) / math.log1p(-probability)


def aggregate_energies(energies, checkpoints, reference, reference_type='BKS',
                       tolerance=0.0, complete=True):
    """Aggregate all planned attempts; partial groups cannot publish final rates.

    NaN is the only missing-energy representation. Missing values count as
    misses in the planned-attempt denominator. At such checkpoints the gap is
    NaN instead of a conditional mean over the successful subset.
    """
    if not complete:
        raise ValueError('Incomplete group cannot publish aggregate statistics')
    matrix = np.asarray(energies, dtype=np.float64)
    times = np.asarray(checkpoints, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != len(times):
        raise ValueError('Energy matrix must have one row per planned seed and one column per checkpoint')
    if not np.isfinite(reference) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Reference and nonnegative tolerance must be finite')
    if times.ndim != 1 or not np.all(np.isfinite(times)) or np.any(times <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError('Checkpoint times must be finite, positive and strictly increasing')
    if np.any(np.isinf(matrix)):
        raise ValueError('Unavailable energy must be NaN, not infinity')
    if reference_type not in ('OPTIMUM', 'BKS'):
        raise ValueError('Reference type must be OPTIMUM or BKS')
    present = np.isfinite(matrix)
    if reference_type == 'OPTIMUM' and np.any(present & (matrix < reference - tolerance)):
        raise ValueError('Independently scored energy is below the proven optimum')
    hits = present & (matrix <= reference + tolerance)
    probabilities = hits.sum(axis=0) / matrix.shape[0]
    rows = []
    for column, seconds in enumerate(times):
        gap = float(np.mean(matrix[:, column] - reference)) if present[:, column].all() else math.nan
        probability = float(probabilities[column])
        rows.append(dict(zip(AGGREGATE_COLUMNS,
                             (float(seconds), gap, probability, tts99(seconds, probability)))))
    return rows
