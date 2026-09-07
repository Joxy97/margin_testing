"""Per-factor conditional mean and GARCH-family volatility filtering."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy


def _readonly(values: numpy.ndarray) -> numpy.ndarray:
    result = numpy.ascontiguousarray(values, dtype=numpy.float64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ConditionalFilterResult:
    """Immutable fitted paths and next-day forecast for one factor."""

    mean: numpy.ndarray
    innovations: numpy.ndarray
    variances: numpy.ndarray
    standardizedResiduals: numpy.ndarray
    meanForecast: float
    varianceForecast: float
    volatilityForecast: float
    parameters: Mapping[str, float]
    modelFamily: str
    validStart: int
    usedFallback: bool = False

    def __post_init__(self) -> None:
        paths = tuple(
            _readonly(value)
            for value in (
                self.mean,
                self.innovations,
                self.variances,
                self.standardizedResiduals,
            )
        )
        if len({len(value) for value in paths}) != 1:
            raise ValueError("conditional-filter paths must have equal lengths")
        if not 0 <= self.validStart < len(paths[0]):
            raise ValueError("validStart must identify a filter-path row")
        retained = slice(self.validStart, None)
        if not all(numpy.isfinite(value[retained]).all() for value in paths):
            raise ValueError("retained conditional-filter paths must be finite")
        if numpy.any(paths[2][retained] <= 0.0):
            raise ValueError("conditional variances must be positive")
        if (
            not math.isfinite(self.meanForecast)
            or not math.isfinite(self.varianceForecast)
            or self.varianceForecast <= 0.0
            or not math.isfinite(self.volatilityForecast)
            or self.volatilityForecast <= 0.0
        ):
            raise ValueError("conditional forecasts must be finite and positive")
        object.__setattr__(self, "mean", paths[0])
        object.__setattr__(self, "innovations", paths[1])
        object.__setattr__(self, "variances", paths[2])
        object.__setattr__(self, "standardizedResiduals", paths[3])
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(
                {str(key): float(value) for key, value in self.parameters.items()}
            ),
        )


class ConditionalFilter:
    """Fit a parsimonious GJR-GARCH, GARCH, or EWMA filter.

    GARCH fitting uses deterministic starts and Gaussian quasi-likelihood.
    Failed primary fits use the configured EWMA fallback and report that fact.
    """

    _VARIANCE_MODELS = {"gjr_garch", "garch", "ewma"}
    _MEAN_MODELS = {"zero", "constant", "ar"}

    def __init__(
        self,
        meanModel: str = "constant",
        arOrder: int = 1,
        varianceModel: str = "gjr_garch",
        burnIn: int = 50,
        ewmaLambda: float = 0.94,
        persistenceBuffer: float = 0.005,
        optimizerMaxIterations: int = 500,
    ) -> None:
        if meanModel not in self._MEAN_MODELS:
            raise ValueError(f"Unknown conditional mean model: {meanModel!r}")
        if varianceModel not in self._VARIANCE_MODELS:
            raise ValueError(
                f"Unknown conditional variance model: {varianceModel!r}"
            )
        if isinstance(arOrder, bool) or not isinstance(arOrder, int) or arOrder < 0:
            raise ValueError("arOrder must be a nonnegative integer")
        if meanModel == "ar" and arOrder < 1:
            raise ValueError("arOrder must be positive for an AR mean")
        if isinstance(burnIn, bool) or not isinstance(burnIn, int) or burnIn < 0:
            raise ValueError("burnIn must be a nonnegative integer")
        if not math.isfinite(ewmaLambda) or not 0.0 < ewmaLambda < 1.0:
            raise ValueError("ewmaLambda must lie strictly between zero and one")
        if (
            not math.isfinite(persistenceBuffer)
            or not 0.0 < persistenceBuffer < 1.0
        ):
            raise ValueError("persistenceBuffer must lie strictly between 0 and 1")
        if (
            isinstance(optimizerMaxIterations, bool)
            or not isinstance(optimizerMaxIterations, int)
            or optimizerMaxIterations <= 0
        ):
            raise ValueError("optimizerMaxIterations must be a positive integer")
        self.meanModel = meanModel
        self.arOrder = arOrder if meanModel == "ar" else 0
        self.varianceModel = varianceModel
        self.burnIn = burnIn
        self.ewmaLambda = float(ewmaLambda)
        self.persistenceBuffer = float(persistenceBuffer)
        self.optimizerMaxIterations = optimizerMaxIterations

    def fit(self, values: numpy.ndarray) -> ConditionalFilterResult:
        """Fit one factor and return synchronized historical paths."""
        observed = numpy.asarray(values, dtype=numpy.float64)
        if observed.ndim != 1:
            raise ValueError("conditional-filter input must be one-dimensional")
        if not numpy.isfinite(observed).all():
            raise ValueError("conditional-filter input must be finite")
        minimum = self.arOrder + self.burnIn + 3
        if len(observed) < minimum:
            raise ValueError(
                f"conditional filter requires at least {minimum} observations"
            )

        scale = 100.0
        scaled = observed * scale
        initial_mean = self._fitInitialMean(scaled)
        if self.varianceModel == "ewma":
            return self._fitEWMA(scaled, initial_mean, scale, False)

        fitted = self._fitGARCH(scaled, initial_mean)
        if fitted is None:
            return self._fitEWMA(scaled, initial_mean, scale, True)
        parameters, family = fitted
        mu, innovations = self._meanAndInnovations(scaled, parameters)
        omega, alpha, gamma, beta = self._varianceParameters(parameters)
        variances = self._variancePath(
            innovations,
            self.arOrder,
            omega,
            alpha,
            gamma,
            beta,
        )
        valid_start = self.arOrder + self.burnIn
        residuals = innovations / numpy.sqrt(variances)
        mean_forecast = self._forecastMean(scaled, parameters)
        variance_forecast = (
            omega
            + alpha * innovations[-1] ** 2
            + gamma * float(innovations[-1] < 0.0) * innovations[-1] ** 2
            + beta * variances[-1]
        )
        stored_parameters = self._parameterRecord(parameters, scale)
        stored_parameters["initialVariance"] = float(
            variances[self.arOrder] / scale**2
        )
        return ConditionalFilterResult(
            mean=mu / scale,
            innovations=innovations / scale,
            variances=variances / scale**2,
            standardizedResiduals=residuals,
            meanForecast=float(mean_forecast / scale),
            varianceForecast=float(variance_forecast / scale**2),
            volatilityForecast=float(math.sqrt(variance_forecast) / scale),
            parameters=stored_parameters,
            modelFamily=family,
            validStart=valid_start,
        )

    def applyCalibrated(
        self,
        values: numpy.ndarray,
        calibration: ConditionalFilterResult,
    ) -> ConditionalFilterResult:
        """Apply frozen structural parameters to an updated return history."""
        observed = numpy.asarray(values, dtype=numpy.float64)
        if observed.ndim != 1 or not numpy.isfinite(observed).all():
            raise ValueError("conditional-filter input must be a finite vector")
        minimum = self.arOrder + self.burnIn + 3
        if len(observed) < minimum:
            raise ValueError(
                f"conditional filter requires at least {minimum} observations"
            )

        parameters = self._calibratedMeanParameters(calibration.parameters)
        mean, innovations = self._meanAndInnovations(observed, parameters)
        initial_variance = float(calibration.parameters["initialVariance"])
        if calibration.modelFamily == "ewma":
            decay = float(calibration.parameters["lambda"])
            variances = self._ewmaVariancePath(
                innovations,
                self.arOrder,
                decay,
                initial_variance,
            )
            variance_forecast = (
                decay * variances[-1]
                + (1.0 - decay) * innovations[-1] ** 2
            )
        else:
            omega = float(calibration.parameters["omega"])
            alpha = float(calibration.parameters["alpha"])
            gamma = float(calibration.parameters["gamma"])
            beta = float(calibration.parameters["beta"])
            variances = self._variancePath(
                innovations,
                self.arOrder,
                omega,
                alpha,
                gamma,
                beta,
                initial_variance,
            )
            variance_forecast = (
                omega
                + alpha * innovations[-1] ** 2
                + gamma
                * float(innovations[-1] < 0.0)
                * innovations[-1] ** 2
                + beta * variances[-1]
            )
        residuals = innovations / numpy.sqrt(variances)
        mean_forecast = self._forecastMean(observed, parameters)
        return ConditionalFilterResult(
            mean=mean,
            innovations=innovations,
            variances=variances,
            standardizedResiduals=residuals,
            meanForecast=mean_forecast,
            varianceForecast=float(variance_forecast),
            volatilityForecast=float(math.sqrt(variance_forecast)),
            parameters=calibration.parameters,
            modelFamily=calibration.modelFamily,
            validStart=self.arOrder + self.burnIn,
            usedFallback=calibration.usedFallback,
        )

    def _fitInitialMean(self, values: numpy.ndarray) -> numpy.ndarray:
        if self.meanModel == "zero":
            return numpy.empty(0, dtype=numpy.float64)
        if self.meanModel == "constant":
            return numpy.array([float(numpy.mean(values))])
        response = values[self.arOrder :]
        design = numpy.column_stack(
            (
                numpy.ones(len(response)),
                *(
                    values[self.arOrder - lag : -lag]
                    for lag in range(1, self.arOrder + 1)
                ),
            )
        )
        coefficients, _, _, _ = numpy.linalg.lstsq(design, response, rcond=None)
        return coefficients

    def _fitGARCH(
        self,
        values: numpy.ndarray,
        initialMean: numpy.ndarray,
    ) -> tuple[numpy.ndarray, str] | None:
        try:
            from scipy.optimize import minimize
        except ImportError as error:  # pragma: no cover - dependency contract
            raise RuntimeError("scipy is required for GARCH fitting") from error

        initial_mu, initial_innovations = self._meanAndInnovations(
            values,
            initialMean,
        )
        del initial_mu
        usable = initial_innovations[self.arOrder :]
        residual_variance = float(numpy.var(usable, ddof=1))
        if not math.isfinite(residual_variance) or residual_variance <= 0.0:
            return None

        asymmetric = self.varianceModel == "gjr_garch"
        variance_starts = (
            (0.05, 0.05 if asymmetric else 0.0, 0.85),
            (0.08, 0.02 if asymmetric else 0.0, 0.75),
            (0.03, 0.10 if asymmetric else 0.0, 0.88),
        )
        mean_count = len(initialMean)
        bounds: list[tuple[float | None, float | None]] = [
            (None, None)
        ] * mean_count
        bounds.extend([(1e-12, None), (0.0, 1.0)])
        if asymmetric:
            bounds.append((-1.0, 2.0))
        bounds.append((0.0, 1.0))

        candidates: list[tuple[float, int, Any]] = []
        for start_id, (alpha, gamma, beta) in enumerate(variance_starts):
            persistence = alpha + beta + 0.5 * gamma
            if persistence >= 1.0 - self.persistenceBuffer:
                beta = max(
                    0.0,
                    1.0 - self.persistenceBuffer - alpha - 0.5 * gamma - 0.01,
                )
                persistence = alpha + beta + 0.5 * gamma
            omega = residual_variance * max(1e-4, 1.0 - persistence)
            variance_values = [omega, alpha]
            if asymmetric:
                variance_values.append(gamma)
            variance_values.append(beta)
            start = numpy.concatenate((initialMean, variance_values))
            result = minimize(
                self._negativeLogLikelihood,
                start,
                args=(values,),
                method="SLSQP",
                bounds=bounds,
                constraints=(
                    {
                        "type": "ineq",
                        "fun": lambda theta: 1.0
                        - self.persistenceBuffer
                        - self._persistence(theta),
                    },
                    {
                        "type": "ineq",
                        "fun": lambda theta: self._shockFloor(theta),
                    },
                ),
                options={
                    "maxiter": self.optimizerMaxIterations,
                    "ftol": 1e-10,
                    "disp": False,
                },
            )
            objective = float(result.fun)
            if (
                result.success
                and math.isfinite(objective)
                and self._isFeasible(result.x)
            ):
                candidates.append((objective, start_id, result))
        if not candidates:
            return None
        result = min(candidates, key=lambda item: (item[0], item[1]))[2]
        return numpy.asarray(result.x, dtype=numpy.float64), self.varianceModel

    def _negativeLogLikelihood(
        self,
        parameters: numpy.ndarray,
        values: numpy.ndarray,
    ) -> float:
        if not self._isFeasible(parameters):
            return math.inf
        _, innovations = self._meanAndInnovations(values, parameters)
        omega, alpha, gamma, beta = self._varianceParameters(parameters)
        variances = self._variancePath(
            innovations,
            self.arOrder,
            omega,
            alpha,
            gamma,
            beta,
        )
        first = self.arOrder + max(1, self.burnIn)
        retained_h = variances[first:]
        retained_a = innovations[first:]
        if (
            not numpy.isfinite(retained_h).all()
            or numpy.any(retained_h <= 0.0)
        ):
            return math.inf
        return float(
            0.5 * numpy.sum(numpy.log(retained_h) + retained_a**2 / retained_h)
        )

    def _meanAndInnovations(
        self,
        values: numpy.ndarray,
        parameters: numpy.ndarray,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        mean = numpy.full(len(values), numpy.nan, dtype=numpy.float64)
        if self.meanModel == "zero":
            mean[:] = 0.0
        elif self.meanModel == "constant":
            mean[:] = parameters[0]
        else:
            coefficient_count = self.arOrder + 1
            coefficients = parameters[:coefficient_count]
            mean[self.arOrder :] = coefficients[0]
            for lag in range(1, self.arOrder + 1):
                mean[self.arOrder :] += (
                    coefficients[lag]
                    * values[self.arOrder - lag : -lag]
                )
        return mean, values - mean

    @staticmethod
    def _variancePath(
        innovations: numpy.ndarray,
        start: int,
        omega: float,
        alpha: float,
        gamma: float,
        beta: float,
        initialVariance: float | None = None,
    ) -> numpy.ndarray:
        """Run the chronological GJR-GARCH recursion."""
        result = numpy.full(len(innovations), numpy.nan, dtype=numpy.float64)
        usable = innovations[start:]
        initial = (
            float(numpy.var(usable, ddof=1))
            if initialVariance is None
            else float(initialVariance)
        )
        if not math.isfinite(initial) or initial <= 0.0:
            return result
        result[start] = initial
        previous = innovations[start:-1]
        inputs = (
            omega
            + alpha * previous**2
            + gamma * (previous < 0.0) * previous**2
        )
        if len(inputs):
            from scipy.signal import lfilter

            result[start + 1 :] = lfilter(
                [1.0],
                [1.0, -beta],
                inputs,
                zi=[beta * initial],
            )[0]
        if (
            not numpy.isfinite(result[start:]).all()
            or numpy.any(result[start:] <= 0.0)
        ):
            result[start:] = numpy.nan
        return result

    def _fitEWMA(
        self,
        values: numpy.ndarray,
        meanParameters: numpy.ndarray,
        scale: float,
        usedFallback: bool,
    ) -> ConditionalFilterResult:
        mean, innovations = self._meanAndInnovations(values, meanParameters)
        start = self.arOrder
        initial = float(numpy.var(innovations[start:], ddof=1))
        if not math.isfinite(initial) or initial <= 0.0:
            raise ValueError("EWMA requires positive innovation variance")
        variances = self._ewmaVariancePath(
            innovations,
            start,
            self.ewmaLambda,
            initial,
        )
        valid_start = start + self.burnIn
        residuals = innovations / numpy.sqrt(variances)
        mean_forecast = self._forecastMean(values, meanParameters)
        variance_forecast = (
            self.ewmaLambda * variances[-1]
            + (1.0 - self.ewmaLambda) * innovations[-1] ** 2
        )
        parameters = self._meanParameterRecord(meanParameters, scale)
        parameters["lambda"] = self.ewmaLambda
        parameters["initialVariance"] = initial / scale**2
        return ConditionalFilterResult(
            mean=mean / scale,
            innovations=innovations / scale,
            variances=variances / scale**2,
            standardizedResiduals=residuals,
            meanForecast=float(mean_forecast / scale),
            varianceForecast=float(variance_forecast / scale**2),
            volatilityForecast=float(math.sqrt(variance_forecast) / scale),
            parameters=parameters,
            modelFamily="ewma",
            validStart=valid_start,
            usedFallback=usedFallback,
        )

    def _calibratedMeanParameters(
        self,
        parameters: Mapping[str, float],
    ) -> numpy.ndarray:
        if self.meanModel == "zero":
            return numpy.empty(0, dtype=numpy.float64)
        values = [float(parameters["constant"])]
        if self.meanModel == "ar":
            values.extend(
                float(parameters[f"phi{lag}"])
                for lag in range(1, self.arOrder + 1)
            )
        return numpy.asarray(values, dtype=numpy.float64)

    @staticmethod
    def _ewmaVariancePath(
        innovations: numpy.ndarray,
        start: int,
        decay: float,
        initialVariance: float,
    ) -> numpy.ndarray:
        result = numpy.full(len(innovations), numpy.nan, dtype=numpy.float64)
        if not math.isfinite(initialVariance) or initialVariance <= 0.0:
            return result
        result[start] = initialVariance
        inputs = (1.0 - decay) * innovations[start:-1] ** 2
        if len(inputs):
            from scipy.signal import lfilter

            result[start + 1 :] = lfilter(
                [1.0],
                [1.0, -decay],
                inputs,
                zi=[decay * initialVariance],
            )[0]
        return result

    def _forecastMean(
        self,
        values: numpy.ndarray,
        parameters: numpy.ndarray,
    ) -> float:
        if self.meanModel == "zero":
            return 0.0
        if self.meanModel == "constant":
            return float(parameters[0])
        lags = values[-self.arOrder :][::-1]
        return float(parameters[0] + parameters[1 : self.arOrder + 1] @ lags)

    def _varianceParameters(
        self,
        parameters: numpy.ndarray,
    ) -> tuple[float, float, float, float]:
        index = self._meanParameterCount
        omega = float(parameters[index])
        alpha = float(parameters[index + 1])
        if self.varianceModel == "gjr_garch":
            gamma = float(parameters[index + 2])
            beta = float(parameters[index + 3])
        else:
            gamma = 0.0
            beta = float(parameters[index + 2])
        return omega, alpha, gamma, beta

    @property
    def _meanParameterCount(self) -> int:
        if self.meanModel == "zero":
            return 0
        return 1 if self.meanModel == "constant" else self.arOrder + 1

    def _persistence(self, parameters: numpy.ndarray) -> float:
        _, alpha, gamma, beta = self._varianceParameters(parameters)
        return alpha + beta + 0.5 * gamma

    def _shockFloor(self, parameters: numpy.ndarray) -> float:
        _, alpha, gamma, _ = self._varianceParameters(parameters)
        return alpha + gamma

    def _isFeasible(self, parameters: numpy.ndarray) -> bool:
        if not numpy.isfinite(parameters).all():
            return False
        omega, alpha, gamma, beta = self._varianceParameters(parameters)
        return (
            omega > 0.0
            and alpha >= 0.0
            and beta >= 0.0
            and alpha + gamma >= 0.0
            and self._persistence(parameters) <= 1.0 - self.persistenceBuffer
        )

    def _meanParameterRecord(
        self,
        parameters: numpy.ndarray,
        scale: float,
    ) -> dict[str, float]:
        if self.meanModel == "zero":
            return {}
        result = {"constant": float(parameters[0] / scale)}
        if self.meanModel == "ar":
            result.update(
                {
                    f"phi{lag}": float(parameters[lag])
                    for lag in range(1, self.arOrder + 1)
                }
            )
        return result

    def _parameterRecord(
        self,
        parameters: numpy.ndarray,
        scale: float,
    ) -> dict[str, float]:
        result = self._meanParameterRecord(parameters, scale)
        omega, alpha, gamma, beta = self._varianceParameters(parameters)
        result.update(
            {
                "omega": omega / scale**2,
                "alpha": alpha,
                "gamma": gamma,
                "beta": beta,
                "persistence": alpha + beta + 0.5 * gamma,
            }
        )
        return result
