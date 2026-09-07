"""Continuous empirical-body marginals with separate fitted EVT tails."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy


def _readonly(values: numpy.ndarray) -> numpy.ndarray:
    result = numpy.ascontiguousarray(values, dtype=numpy.float64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GeneralizedPareto:
    """A generalized Pareto exceedance distribution."""

    shape: float
    scale: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.shape) or self.shape >= 0.5:
            raise ValueError("GPD shape must be finite and less than one half")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("GPD scale must be positive and finite")

    def cdf(self, values: float | numpy.ndarray) -> float | numpy.ndarray:
        y = numpy.asarray(values, dtype=numpy.float64)
        if numpy.any(y < 0.0):
            raise ValueError("GPD exceedances must be nonnegative")
        support = 1.0 + self.shape * y / self.scale
        if numpy.any(support <= 0.0):
            raise ValueError("GPD value lies outside its finite support")
        if abs(self.shape) < 1e-8:
            result = -numpy.expm1(-y / self.scale)
        else:
            result = 1.0 - support ** (-1.0 / self.shape)
        return float(result) if result.ndim == 0 else result

    def quantile(
        self,
        probabilities: float | numpy.ndarray,
    ) -> float | numpy.ndarray:
        probability = numpy.asarray(probabilities, dtype=numpy.float64)
        if numpy.any(probability < 0.0) or numpy.any(probability >= 1.0):
            raise ValueError("GPD probabilities must lie in [0, 1)")
        if abs(self.shape) < 1e-8:
            result = -self.scale * numpy.log1p(-probability)
        else:
            result = self.scale / self.shape * (
                numpy.exp(-self.shape * numpy.log1p(-probability)) - 1.0
            )
        return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class SemiparametricMarginal:
    """Normalized empirical-body/GPD-tail quantile evaluator."""

    sortedValues: numpy.ndarray
    cumulativeWeights: numpy.ndarray
    lowerTailMass: float
    upperTailMass: float
    lowerThreshold: float
    upperThreshold: float
    lowerTail: GeneralizedPareto
    upperTail: GeneralizedPareto
    rawMean: float
    rawScale: float
    usedFallback: bool = False
    _empiricalProbabilities: numpy.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )
    _empiricalValues: numpy.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        values = _readonly(self.sortedValues)
        cumulative = _readonly(self.cumulativeWeights)
        if values.ndim != 1 or cumulative.shape != values.shape or not len(values):
            raise ValueError("empirical quantile knots must be equal nonempty vectors")
        if not numpy.isfinite(values).all() or numpy.any(numpy.diff(values) < 0.0):
            raise ValueError("empirical values must be finite and sorted")
        if (
            not numpy.isfinite(cumulative).all()
            or numpy.any(numpy.diff(cumulative) <= 0.0)
            or cumulative[-1] <= 0.0
        ):
            raise ValueError("empirical cumulative weights must increase")
        if not (
            0.0 < self.lowerTailMass < 0.5
            and 0.0 < self.upperTailMass < 0.5
            and self.lowerTailMass + self.upperTailMass < 1.0
        ):
            raise ValueError("tail masses must leave a nonempty empirical body")
        if self.lowerThreshold > self.upperThreshold:
            raise ValueError("lower threshold must not exceed upper threshold")
        if not math.isfinite(self.rawMean):
            raise ValueError("raw marginal mean must be finite")
        if not math.isfinite(self.rawScale) or self.rawScale <= 0.0:
            raise ValueError("raw marginal scale must be positive and finite")
        object.__setattr__(self, "sortedValues", values)
        object.__setattr__(self, "cumulativeWeights", cumulative)
        object.__setattr__(
            self,
            "_empiricalProbabilities",
            _readonly(numpy.concatenate((numpy.array([0.0]), cumulative))),
        )
        object.__setattr__(
            self,
            "_empiricalValues",
            _readonly(numpy.concatenate((numpy.array([values[0]]), values))),
        )

    def rawQuantile(
        self,
        probabilities: float | numpy.ndarray,
    ) -> float | numpy.ndarray:
        probability = numpy.asarray(probabilities, dtype=numpy.float64)
        if numpy.any(probability <= 0.0) or numpy.any(probability >= 1.0):
            raise ValueError("marginal probabilities must lie strictly inside (0, 1)")
        flat = probability.reshape(-1)
        result = numpy.empty_like(flat)
        lower = flat < self.lowerTailMass
        upper = flat > 1.0 - self.upperTailMass
        body = ~(lower | upper)
        if numpy.any(lower):
            conditional = 1.0 - flat[lower] / self.lowerTailMass
            result[lower] = self.lowerThreshold - self.lowerTail.quantile(
                conditional
            )
        if numpy.any(upper):
            conditional = 1.0 - (1.0 - flat[upper]) / self.upperTailMass
            result[upper] = self.upperThreshold + self.upperTail.quantile(
                conditional
            )
        if numpy.any(body):
            result[body] = numpy.interp(
                flat[body],
                self._empiricalProbabilities,
                self._empiricalValues,
            )
        reshaped = result.reshape(probability.shape)
        return float(reshaped) if reshaped.ndim == 0 else reshaped

    def quantile(
        self,
        probabilities: float | numpy.ndarray,
    ) -> float | numpy.ndarray:
        raw = numpy.asarray(self.rawQuantile(probabilities), dtype=numpy.float64)
        result = (raw - self.rawMean) / self.rawScale
        return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class _TailCandidate:
    mass: float
    threshold: float
    distribution: GeneralizedPareto
    effectiveCount: float
    usedFallback: bool


class EVTMarginalFitter:
    """Fit deterministic empirical-body/GPD-tail marginal quantiles."""

    def __init__(
        self,
        tailMassCandidates: tuple[float, ...] = (0.025, 0.05, 0.075, 0.10),
        minimumTailObservations: int = 20,
        shapeLowerBound: float = -0.45,
        shapeUpperBound: float = 0.45,
        quadraturePoints: int = 512,
        thresholdStabilityTolerance: float = 0.20,
    ) -> None:
        masses = tuple(float(item) for item in tailMassCandidates)
        if not masses or any(
            not math.isfinite(item) or not 0.0 < item < 0.5 for item in masses
        ):
            raise ValueError("tailMassCandidates must contain probabilities in (0, .5)")
        if any(left >= right for left, right in zip(masses, masses[1:])):
            raise ValueError("tailMassCandidates must be strictly increasing")
        if any(2.0 * item >= 1.0 for item in masses):
            raise ValueError("tail masses must leave a nonempty body")
        if (
            isinstance(minimumTailObservations, bool)
            or not isinstance(minimumTailObservations, int)
            or minimumTailObservations < 2
        ):
            raise ValueError(
                "minimumTailObservations must be an integer of at least two"
            )
        if (
            not math.isfinite(shapeLowerBound)
            or not math.isfinite(shapeUpperBound)
            or shapeLowerBound >= shapeUpperBound
            or shapeUpperBound >= 0.5
        ):
            raise ValueError("EVT shape bounds must be ordered below one half")
        if (
            isinstance(quadraturePoints, bool)
            or not isinstance(quadraturePoints, int)
            or quadraturePoints < 32
        ):
            raise ValueError("quadraturePoints must be an integer of at least 32")
        if (
            not math.isfinite(thresholdStabilityTolerance)
            or thresholdStabilityTolerance < 0.0
        ):
            raise ValueError("thresholdStabilityTolerance must be nonnegative")
        self.tailMassCandidates = masses
        self.minimumTailObservations = minimumTailObservations
        self.shapeLowerBound = float(shapeLowerBound)
        self.shapeUpperBound = float(shapeUpperBound)
        self.quadraturePoints = quadraturePoints
        self.thresholdStabilityTolerance = float(thresholdStabilityTolerance)

    def fit(
        self,
        residuals: numpy.ndarray,
        weights: numpy.ndarray,
    ) -> SemiparametricMarginal:
        """Fit one normalized marginal from synchronized weighted residuals."""
        values = numpy.asarray(residuals, dtype=numpy.float64)
        probability = numpy.asarray(weights, dtype=numpy.float64)
        if values.ndim != 1 or probability.shape != values.shape:
            raise ValueError(
                "residuals and weights must be equal one-dimensional arrays"
            )
        if (
            not numpy.isfinite(values).all()
            or not numpy.isfinite(probability).all()
            or numpy.any(probability <= 0.0)
        ):
            raise ValueError(
                "marginal residuals and weights must be finite and positive"
            )
        probability = probability / probability.sum()
        order = numpy.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_weights = probability[order]
        cumulative = numpy.cumsum(sorted_weights)
        cumulative[-1] = 1.0

        pairs: list[tuple[_TailCandidate, _TailCandidate]] = []
        for mass in self.tailMassCandidates:
            lower_threshold = self._empiricalQuantile(
                sorted_values, cumulative, mass
            )
            upper_threshold = self._empiricalQuantile(
                sorted_values, cumulative, 1.0 - mass
            )
            lower_mask = values < lower_threshold
            upper_mask = values > upper_threshold
            if (
                numpy.count_nonzero(lower_mask) < 2
                or numpy.count_nonzero(upper_mask) < 2
            ):
                continue
            lower = self._fitTail(
                lower_threshold - values[lower_mask],
                probability[lower_mask],
                mass,
                lower_threshold,
            )
            upper = self._fitTail(
                values[upper_mask] - upper_threshold,
                probability[upper_mask],
                mass,
                upper_threshold,
            )
            pairs.append((lower, upper))
        if not pairs:
            raise ValueError("no EVT threshold has at least two exceedances per tail")

        selected = self._selectStablePair(pairs)
        lower, upper = selected
        provisional = SemiparametricMarginal(
            sortedValues=sorted_values,
            cumulativeWeights=cumulative,
            lowerTailMass=lower.mass,
            upperTailMass=upper.mass,
            lowerThreshold=lower.threshold,
            upperThreshold=upper.threshold,
            lowerTail=lower.distribution,
            upperTail=upper.distribution,
            rawMean=0.0,
            rawScale=1.0,
            usedFallback=lower.usedFallback or upper.usedFallback,
        )
        nodes, node_weights = numpy.polynomial.legendre.leggauss(
            self.quadraturePoints
        )
        probabilities = 0.5 * (nodes + 1.0)
        integration_weights = 0.5 * node_weights
        raw = numpy.asarray(provisional.rawQuantile(probabilities))
        raw_mean = float(integration_weights @ raw)
        variance = float(integration_weights @ (raw - raw_mean) ** 2)
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError("spliced marginal must have positive finite variance")
        return SemiparametricMarginal(
            sortedValues=sorted_values,
            cumulativeWeights=cumulative,
            lowerTailMass=lower.mass,
            upperTailMass=upper.mass,
            lowerThreshold=lower.threshold,
            upperThreshold=upper.threshold,
            lowerTail=lower.distribution,
            upperTail=upper.distribution,
            rawMean=raw_mean,
            rawScale=math.sqrt(variance),
            usedFallback=lower.usedFallback or upper.usedFallback,
        )

    def _fitTail(
        self,
        exceedances: numpy.ndarray,
        weights: numpy.ndarray,
        mass: float,
        threshold: float,
    ) -> _TailCandidate:
        positive = numpy.asarray(exceedances, dtype=numpy.float64)
        tail_weights = numpy.asarray(weights, dtype=numpy.float64)
        tail_weights = tail_weights / tail_weights.sum()
        effective = float(1.0 / numpy.sum(tail_weights**2))
        fallback = (
            len(positive) < self.minimumTailObservations
            or effective < self.minimumTailObservations
        )
        fitted = None if fallback else self._fitGPD(positive, tail_weights)
        if fitted is None:
            scale = float(tail_weights @ positive)
            if not math.isfinite(scale) or scale <= 0.0:
                positive_values = positive[positive > 0.0]
                if not len(positive_values):
                    raise ValueError("EVT exceedances must include a positive value")
                scale = float(numpy.mean(positive_values))
            fitted = GeneralizedPareto(0.0, max(scale, numpy.finfo(float).eps))
            fallback = True
        return _TailCandidate(mass, threshold, fitted, effective, fallback)

    def _fitGPD(
        self,
        exceedances: numpy.ndarray,
        weights: numpy.ndarray,
    ) -> GeneralizedPareto | None:
        try:
            from scipy.optimize import minimize
        except ImportError as error:  # pragma: no cover - dependency contract
            raise RuntimeError("scipy is required for EVT fitting") from error
        scale_start = float(weights @ exceedances)
        if not math.isfinite(scale_start) or scale_start <= 0.0:
            return None

        best: tuple[float, int, object] | None = None
        starts = (-0.10, 0.0, 0.10)
        for start_id, shape in enumerate(starts):
            result = minimize(
                self._gpdNegativeLogLikelihood,
                numpy.array([shape, math.log(scale_start)]),
                args=(exceedances, weights),
                method="L-BFGS-B",
                bounds=(
                    (self.shapeLowerBound, self.shapeUpperBound),
                    (math.log(numpy.finfo(float).eps), None),
                ),
                options={"maxiter": 500, "ftol": 1e-12},
            )
            objective = float(result.fun)
            if result.success and math.isfinite(objective):
                candidate = (objective, start_id, result)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
        if best is None:
            return None
        best_result = best[2]
        parameters = numpy.asarray(
            getattr(best_result, "x"), dtype=numpy.float64
        )
        distribution = GeneralizedPareto(
            float(parameters[0]), float(math.exp(parameters[1]))
        )
        support = 1.0 + distribution.shape * exceedances / distribution.scale
        return distribution if numpy.all(support > 1e-10) else None

    @staticmethod
    def _gpdNegativeLogLikelihood(
        parameters: numpy.ndarray,
        exceedances: numpy.ndarray,
        weights: numpy.ndarray,
    ) -> float:
        shape = float(parameters[0])
        scale = math.exp(float(parameters[1]))
        support = 1.0 + shape * exceedances / scale
        if not numpy.all(numpy.isfinite(support)) or numpy.any(support <= 0.0):
            return math.inf
        if abs(shape) < 1e-7:
            terms = math.log(scale) + exceedances / scale
        else:
            terms = math.log(scale) + (1.0 + 1.0 / shape) * numpy.log(
                support
            )
        result = float(weights @ terms)
        return result if math.isfinite(result) else math.inf

    def _selectStablePair(
        self,
        pairs: list[tuple[_TailCandidate, _TailCandidate]],
    ) -> tuple[_TailCandidate, _TailCandidate]:
        adequately_informed = [
            pair
            for pair in pairs
            if pair[0].effectiveCount >= self.minimumTailObservations
            and pair[1].effectiveCount >= self.minimumTailObservations
            and not pair[0].usedFallback
            and not pair[1].usedFallback
        ]
        for first, second in zip(adequately_informed, adequately_informed[1:]):
            changes = (
                abs(first[0].distribution.shape - second[0].distribution.shape),
                abs(first[1].distribution.shape - second[1].distribution.shape),
            )
            if max(changes) <= self.thresholdStabilityTolerance:
                return first
        if adequately_informed:
            return adequately_informed[-1]
        return max(
            pairs,
            key=lambda pair: (
                min(pair[0].effectiveCount, pair[1].effectiveCount),
                pair[0].mass,
            ),
        )

    @staticmethod
    def _empiricalQuantile(
        sortedValues: numpy.ndarray,
        cumulativeWeights: numpy.ndarray,
        probability: float,
    ) -> float:
        return float(
            numpy.interp(
                probability,
                numpy.concatenate((numpy.array([0.0]), cumulativeWeights)),
                numpy.concatenate((numpy.array([sortedValues[0]]), sortedValues)),
            )
        )
