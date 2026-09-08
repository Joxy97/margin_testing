"""Reference and aggregation utilities for the Torch MaxCut benchmark.

One timed observation is one solver invocation.  In particular, ``runs=4``
means that its success indicator describes the best of four trajectories; it
must not be interpreted as four independently timed observations.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QualityMetrics:
    """Quality of a cut relative to an independently established reference."""

    approximationRatio: float
    relativeGap: float
    success: bool


@dataclass(frozen=True)
class AggregateMetrics:
    """Aggregate repeated timed invocations of one solver/instance pair."""

    trials: int
    successes: int
    successProbability: float
    meanAmortizedSeconds: float
    medianAmortizedSeconds: float
    meanApproximationRatio: float
    minimumApproximationRatio: float
    bestCut: float
    timeToSolution99Seconds: float


def quality_metrics(
    cut: float,
    referenceCut: float,
    *,
    absoluteTolerance: float = 1e-9,
) -> QualityMetrics:
    """Return ratio, nonnegative relative gap, and reference-hit indicator.

    MaxCut instances in this benchmark have nonnegative edge weights, hence a
    positive reference is expected.  Rejecting a zero reference prevents a
    superficially convenient but uninformative ratio of one for empty graphs.
    """

    cut = _finite_float("cut", cut)
    referenceCut = _finite_float("referenceCut", referenceCut)
    absoluteTolerance = _finite_float("absoluteTolerance", absoluteTolerance)
    if referenceCut <= 0.0:
        raise ValueError("referenceCut must be positive")
    if cut < -absoluteTolerance:
        raise ValueError("cut must be nonnegative")
    if absoluteTolerance < 0.0:
        raise ValueError("absoluteTolerance must be nonnegative")
    ratio = cut / referenceCut
    return QualityMetrics(
        approximationRatio=ratio,
        relativeGap=max(0.0, (referenceCut - cut) / referenceCut),
        success=cut >= referenceCut - absoluteTolerance,
    )


def attempts_to_solution(
    successProbability: float,
    *,
    targetProbability: float = 0.99,
) -> int | float:
    """Return invocations needed to reach the requested cumulative success.

    The conventional independent-restart definition is
    ``ceil(log(1-target) / log(1-p))``.  A measured success probability of zero
    is reported as infinity instead of being hidden by smoothing.
    """

    successProbability = _finite_float("successProbability", successProbability)
    targetProbability = _finite_float("targetProbability", targetProbability)
    if not 0.0 <= successProbability <= 1.0:
        raise ValueError("successProbability must be between zero and one")
    if not 0.0 < targetProbability < 1.0:
        raise ValueError("targetProbability must be strictly between zero and one")
    if successProbability == 0.0:
        return math.inf
    if successProbability == 1.0:
        return 1
    return max(
        1,
        math.ceil(math.log1p(-targetProbability) / math.log1p(-successProbability)),
    )


def time_to_solution(
    successProbability: float,
    secondsPerInvocation: float,
    *,
    targetProbability: float = 0.99,
) -> float:
    """Estimate restart-based TTS using per-instance amortized wall time."""

    secondsPerInvocation = _finite_float("secondsPerInvocation", secondsPerInvocation)
    if secondsPerInvocation < 0.0:
        raise ValueError("secondsPerInvocation must be nonnegative")
    attempts = attempts_to_solution(
        successProbability, targetProbability=targetProbability
    )
    if math.isinf(attempts):
        return math.inf
    return secondsPerInvocation * attempts


def aggregate_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    targetProbability: float = 0.99,
    absoluteTolerance: float = 1e-9,
) -> AggregateMetrics:
    """Aggregate raw CSV-like rows for one instance and one solver.

    Required fields are ``cut``, ``reference_cut``, and
    ``amortized_seconds``.  All rows must use the same positive reference.
    ``success`` is deliberately recomputed so stale CSV values cannot affect
    TTS.
    """

    materialized = list(rows)
    if not materialized:
        raise ValueError("at least one benchmark row is required")
    references = [_finite_float("reference_cut", row["reference_cut"]) for row in materialized]
    reference = references[0]
    if any(not math.isclose(value, reference, rel_tol=0.0, abs_tol=absoluteTolerance)
           for value in references[1:]):
        raise ValueError("all rows must use the same reference_cut")

    quality = [
        quality_metrics(
            row["cut"], reference, absoluteTolerance=absoluteTolerance
        )
        for row in materialized
    ]
    times = [
        _finite_float("amortized_seconds", row["amortized_seconds"])
        for row in materialized
    ]
    if any(value < 0.0 for value in times):
        raise ValueError("amortized_seconds must be nonnegative")
    successes = sum(item.success for item in quality)
    probability = successes / len(materialized)
    mean_time = statistics.fmean(times)
    ratios = [item.approximationRatio for item in quality]
    return AggregateMetrics(
        trials=len(materialized),
        successes=successes,
        successProbability=probability,
        meanAmortizedSeconds=mean_time,
        medianAmortizedSeconds=statistics.median(times),
        meanApproximationRatio=statistics.fmean(ratios),
        minimumApproximationRatio=min(ratios),
        bestCut=max(_finite_float("cut", row["cut"]) for row in materialized),
        timeToSolution99Seconds=time_to_solution(
            probability, mean_time, targetProbability=targetProbability
        ),
    )


def cut_value(
    sample: Sequence[int], edges: Iterable[tuple[int, int, float]]
) -> float:
    """Evaluate a weighted cut independently of QUBO solver energies."""

    bits = tuple(int(value) for value in sample)
    if any(value not in (0, 1) for value in bits):
        raise ValueError("sample must contain only binary values")
    value = 0.0
    for head, tail, weight in edges:
        if head < 0 or tail < 0 or head >= len(bits) or tail >= len(bits):
            raise ValueError("edge endpoint is outside the sample")
        weight = _finite_float("edge weight", weight)
        if weight < 0.0:
            raise ValueError("edge weights must be nonnegative")
        if bits[head] != bits[tail]:
            value += weight
    return value


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
