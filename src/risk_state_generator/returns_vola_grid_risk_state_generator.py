"""Returns-volatility-grid risk-state generator."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import product
import math
from numbers import Integral

import numpy
from download_unit import DataRequest, Period
from portfolio import Portfolio

from .pca_grid import ReturnsPCAGrid
from .pca_grid_provider import PCAGridProvider
from .pca_key import ReturnsPCAKey
from .pca_scenario import ReturnsVolaGridPCAScenario
from .risk_state import ReturnsVolaGridRiskState
from .risk_state_generation_context import RiskStateGenerationContext
from .risk_state_generator import RiskStateGenerator


@dataclass(frozen=True)
class _RiskStateContext:
    """Intermediate values shared while constructing one risk state."""

    pcaMean: numpy.ndarray
    factorsRecent: numpy.ndarray
    residualsRecent: numpy.ndarray
    componentScale: numpy.ndarray
    maxAbsoluteZ: numpy.ndarray
    logReturnMean: numpy.ndarray
    logReturnScale: numpy.ndarray


@dataclass(frozen=True)
class _ConditionedRiskState:
    """Intermediate result reusable by specialized risk-state generators."""

    returnsVolaGrid: dict[str, numpy.ndarray]
    context: _RiskStateContext
    scenarioCenter: numpy.ndarray
    nearestResiduals: numpy.ndarray
    inflationFactor: float
    residualSigma: numpy.ndarray


class ReturnsVolaGridRiskStateGenerator(RiskStateGenerator):
    """Generate returns-volatility-grid risk states from PCA grids."""

    def __init__(
        self,
        pcaGridProvider: PCAGridProvider | None = None,
        ew_window: int = 30,
        ew_lambda: float = 0.94,
        components: int = 1,
        scenariosPerComponents: Iterable[int] = (),
        tailDensityGamma: float = 1.0,
        nZBins: int = 21,
        nNearest: int | None = None,
        residualSigmaRange: float = 5.0,
        allowEmptyBinFallback: bool = True,
        distanceInflationAlpha: float = 0.5,
        distanceInflationPower: float = 2.0,
        maxInflationFactor: float = 5.0,
    ) -> None:
        if (
            not math.isfinite(distanceInflationAlpha)
            or distanceInflationAlpha < 0
        ):
            raise ValueError(
                "distanceInflationAlpha must be finite and nonnegative"
            )
        if (
            not math.isfinite(distanceInflationPower)
            or distanceInflationPower <= 0
        ):
            raise ValueError(
                "distanceInflationPower must be positive and finite"
            )
        if not math.isfinite(maxInflationFactor) or maxInflationFactor < 1:
            raise ValueError("maxInflationFactor must be finite and at least 1")
        self.distanceInflationAlpha = float(distanceInflationAlpha)
        self.distanceInflationPower = float(distanceInflationPower)
        self.maxInflationFactor = float(maxInflationFactor)
        if (
            isinstance(ew_window, bool)
            or not isinstance(ew_window, Integral)
            or ew_window <= 0
        ):
            raise ValueError("ew_window must be a positive integer")
        if not numpy.isfinite(ew_lambda) or not 0.0 < ew_lambda <= 1.0:
            raise ValueError("ew_lambda must be greater than 0 and at most 1")
        if (
            isinstance(components, bool)
            or not isinstance(components, Integral)
            or components <= 0
        ):
            raise ValueError("components must be a positive integer")
        self.ew_window = int(ew_window)
        self.ew_lambda = float(ew_lambda)
        self.components = int(components)
        self.__pcaGridProvider = (
            pcaGridProvider
            if pcaGridProvider is not None
            else PCAGridProvider()
        )
        self.scenariosPerComponents = tuple(scenariosPerComponents)
        if any(
            isinstance(count, bool)
            or not isinstance(count, Integral)
            or count <= 0
            for count in self.scenariosPerComponents
        ):
            raise ValueError(
                "scenariosPerComponents must contain positive integers"
            )
        if not numpy.isfinite(tailDensityGamma) or tailDensityGamma <= 0.0:
            raise ValueError("tailDensityGamma must be positive and finite")
        self.tailDensityGamma = float(tailDensityGamma)
        if not isinstance(nZBins, Integral) or nZBins <= 0:
            raise ValueError("nZBins must be a positive integer")
        if nNearest is not None and (
            not isinstance(nNearest, Integral) or nNearest <= 0
        ):
            raise ValueError("nNearest must be a positive integer or None")
        if not numpy.isfinite(residualSigmaRange) or residualSigmaRange <= 0:
            raise ValueError("residualSigmaRange must be positive and finite")
        self.nZBins = int(nZBins)
        self.nNearest = None if nNearest is None else int(nNearest)
        self.residualSigmaRange = float(residualSigmaRange)
        self.allowEmptyBinFallback = bool(allowEmptyBinFallback)

    def createDataRequest(
        self,
        portfolio: Portfolio,
        marginDate: date,
    ) -> DataRequest:
        """Build the close-price request required by this generator."""
        return DataRequest(
            instruments=tuple(portfolio.weights),
            start_date=marginDate - timedelta(days=2 * self.ew_window),
            end_date=marginDate,
            data_type="closePrices",
            period=Period.ONE_DAY,
        )

    def getRiskStates(
        self,
        context: RiskStateGenerationContext,
    ) -> Iterator[ReturnsVolaGridRiskState]:
        """Lazily create returns-volatility-grid risk states."""
        pca_key = self._createPCAKey(context)
        pca_grid = self.__pcaGridProvider.getPCAGrid(pca_key)
        if pca_grid is None:
            pca_grid = self.__pcaGridProvider.createPCAGrid(
                pca_key,
                context.marketData,
            )
        for pca_scenario in self._generatePCAScenarios(
            pca_key,
            pca_grid,
            self.scenariosPerComponents,
            self.tailDensityGamma,
        ):
            yield self.getRiskState(pca_scenario, pca_grid, context)

    def _createPCAKey(
        self,
        context: RiskStateGenerationContext,
    ) -> ReturnsPCAKey:
        """Combine runtime context with the generator's PCA configuration."""
        return ReturnsPCAKey(
            instruments=context.dataRequest.instruments,
            ew_window=self.ew_window,
            start_date=context.marginDate,
            ew_lambda=self.ew_lambda,
            components=self.components,
        )

    def _generatePCAScenarios(
        self,
        pcaKey: ReturnsPCAKey,
        pcaGrid: ReturnsPCAGrid,
        scenariosPerComponents: tuple[int, ...],
        tailDensityGamma: float,
    ) -> Iterator[ReturnsVolaGridPCAScenario]:
        """Lazily generate the Cartesian factor grid for ``pcaGrid``."""
        lambdas = self._validateScenarioConfiguration(
            pcaGrid,
            scenariosPerComponents,
            tailDensityGamma,
        )
        component_grids = self._buildComponentGrids(
            lambdas,
            scenariosPerComponents,
            tailDensityGamma,
        )
        return self._createPCAScenarios(
            pcaKey,
            self._buildScenarioPoints(component_grids),
        )

    def _validateScenarioConfiguration(
        self,
        pcaGrid: ReturnsPCAGrid,
        scenariosPerComponents: tuple[int, ...],
        tailDensityGamma: float,
    ) -> numpy.ndarray:
        """Validate grid inputs and return the PCA eigenvalues."""
        if pcaGrid.lambdas is None:
            raise ValueError("pcaGrid must contain PCA eigenvalues")

        lambdas = numpy.asarray(pcaGrid.lambdas, dtype=float)
        if len(scenariosPerComponents) != self.components:
            raise ValueError(
                "scenariosPerComponents length must match components"
            )
        if len(lambdas) != self.components:
            raise ValueError(
                "pcaGrid eigenvalue count must match components"
            )
        if any(
            count <= 0 or count % 2 == 0
            for count in scenariosPerComponents
        ):
            raise ValueError(
                "scenariosPerComponents must contain positive odd integers"
            )
        if not numpy.isfinite(tailDensityGamma) or tailDensityGamma <= 0.0:
            raise ValueError("tailDensityGamma must be positive and finite")
        return lambdas

    @staticmethod
    def _buildComponentGrids(
        lambdas: numpy.ndarray,
        scenariosPerComponents: tuple[int, ...],
        tailDensityGamma: float,
    ) -> list[numpy.ndarray]:
        """Build the scenario axis for each principal component."""
        component_sigmas = numpy.sqrt(numpy.maximum(lambdas, 0.0))
        component_grids = []
        for component_sigma, scenario_count in zip(
            component_sigmas,
            scenariosPerComponents,
        ):
            half_width = scenario_count // 2
            positions = numpy.linspace(-1.0, 1.0, scenario_count)
            warped_multipliers = (
                half_width
                * numpy.sign(positions)
                * numpy.abs(positions) ** float(tailDensityGamma)
            )
            component_grids.append(warped_multipliers * component_sigma)
        return component_grids

    @staticmethod
    def _buildScenarioPoints(
        componentGrids: list[numpy.ndarray],
    ) -> Iterator[tuple[float, ...]]:
        """Yield the Cartesian product without materializing its mesh."""
        return product(*(grid.tolist() for grid in componentGrids))

    @staticmethod
    def _createPCAScenarios(
        pcaKey: ReturnsPCAKey,
        scenarioPoints: Iterable[tuple[float, ...]],
    ) -> Iterator[ReturnsVolaGridPCAScenario]:
        """Lazily associate generated factor points with their PCA key."""
        return (
            ReturnsVolaGridPCAScenario(
                pcaKey=pcaKey,
                point=tuple(float(value) for value in point),
            )
            for point in scenarioPoints
        )

    def getRiskState(
        self,
        pcaScenario: ReturnsVolaGridPCAScenario,
        pcaGrid: ReturnsPCAGrid,
        context: RiskStateGenerationContext,
    ) -> ReturnsVolaGridRiskState:
        """Create one risk state conditioned on ``pcaScenario``."""
        conditioned_state = self._buildConditionedRiskState(
            pcaScenario,
            pcaGrid,
        )
        return ReturnsVolaGridRiskState(conditioned_state.returnsVolaGrid)

    def _buildConditionedRiskState(
        self,
        pcaScenario: ReturnsVolaGridPCAScenario,
        pcaGrid: ReturnsPCAGrid,
    ) -> _ConditionedRiskState:
        """Build reusable scenario-conditioned state-space values."""
        context = self._buildRiskStateContext(pcaGrid)
        scenario_center = self._getScenarioCenter(
            pcaScenario,
            pcaGrid,
            context.pcaMean,
        )
        nearest_residuals, scenario_distance = self._getNearestResiduals(
            pcaScenario,
            pcaGrid,
            context,
        )
        inflation_factor = self._getInflationFactor(scenario_distance)
        residual_sigma = self._getInflatedResidualSigma(
            nearest_residuals,
            context.residualsRecent,
            inflation_factor,
        )
        returns_vola_grid = self._buildReturnsVolaGrid(
            pcaGrid,
            scenario_center,
            residual_sigma,
            context,
        )
        return _ConditionedRiskState(
            returnsVolaGrid=returns_vola_grid,
            context=context,
            scenarioCenter=scenario_center,
            nearestResiduals=nearest_residuals,
            inflationFactor=inflation_factor,
            residualSigma=residual_sigma,
        )

    @staticmethod
    def _requirePCAValues(pcaGrid: ReturnsPCAGrid) -> None:
        """Require all fitted PCA values needed for risk-state construction."""
        if (
            pcaGrid.loadings is None
            or pcaGrid.lambdas is None
            or pcaGrid.factors is None
            or pcaGrid.pcaMean is None
            or pcaGrid.residuals is None
            or pcaGrid.maxAbsoluteZ is None
            or pcaGrid.logReturnMean is None
            or pcaGrid.logReturnScale is None
        ):
            raise ValueError("pcaGrid must be fitted before building risk states")

    def _buildRiskStateContext(
        self,
        pcaGrid: ReturnsPCAGrid,
    ) -> _RiskStateContext:
        """Select recent fitted values needed for risk-state construction."""
        self._requirePCAValues(pcaGrid)
        recent_start = max(0, len(pcaGrid.factors) - pcaGrid.ew_window)
        return _RiskStateContext(
            pcaMean=pcaGrid.pcaMean,
            factorsRecent=pcaGrid.factors[recent_start:],
            residualsRecent=pcaGrid.residuals[recent_start:],
            componentScale=numpy.sqrt(
                numpy.maximum(pcaGrid.lambdas, 1e-12)
            ),
            maxAbsoluteZ=pcaGrid.maxAbsoluteZ,
            logReturnMean=pcaGrid.logReturnMean,
            logReturnScale=pcaGrid.logReturnScale,
        )

    @staticmethod
    def _getScenarioCenter(
        pcaScenario: ReturnsVolaGridPCAScenario,
        pcaGrid: ReturnsPCAGrid,
        pcaMean: numpy.ndarray,
    ) -> numpy.ndarray:
        """Map a PCA factor scenario back into standardized return space."""
        selected_scenario = numpy.asarray(pcaScenario.point, dtype=float)
        if len(selected_scenario) != pcaGrid.components:
            raise ValueError("pcaScenario point length must match components")
        return pcaMean + selected_scenario @ pcaGrid.loadings

    def _getNearestResiduals(
        self,
        pcaScenario: ReturnsVolaGridPCAScenario,
        pcaGrid: ReturnsPCAGrid,
        context: _RiskStateContext,
    ) -> tuple[numpy.ndarray, float]:
        """Find residual observations nearest to the PCA scenario."""
        selected_scenario = numpy.asarray(pcaScenario.point, dtype=float)
        scaled_differences = (
            context.factorsRecent - selected_scenario
        ) / context.componentScale
        distances = numpy.linalg.norm(scaled_differences, axis=1)
        nearest_count = self.nNearest or min(
            100,
            len(context.factorsRecent),
            pcaGrid.ew_window,
        )
        if nearest_count == len(distances):
            nearest_indices = numpy.arange(len(distances))
        else:
            nearest_indices = numpy.argpartition(
                distances,
                nearest_count - 1,
            )[:nearest_count]
        nearest_residuals = context.residualsRecent[nearest_indices]
        scenario_distance = float(distances[nearest_indices].mean())
        return nearest_residuals, scenario_distance

    def _getInflationFactor(self, scenarioDistance: float) -> float:
        """Calculate capped distance-based residual inflation."""
        inflation = 1.0 + self.distanceInflationAlpha * (
            scenarioDistance**self.distanceInflationPower
        )
        return float(min(inflation, self.maxInflationFactor))

    @staticmethod
    def _getInflatedResidualSigma(
        nearestResiduals: numpy.ndarray,
        fallbackResiduals: numpy.ndarray,
        inflationFactor: float,
    ) -> numpy.ndarray:
        """Estimate local residual volatility with stable fallbacks."""
        asset_count = fallbackResiduals.shape[1]
        local_sigma = (
            numpy.nanstd(nearestResiduals, axis=0, ddof=1)
            if len(nearestResiduals) > 1
            else numpy.full(asset_count, numpy.nan)
        )
        fallback_sigma = (
            numpy.nanstd(fallbackResiduals, axis=0, ddof=1)
            if len(fallbackResiduals) > 1
            else numpy.full(asset_count, numpy.nan)
        )
        valid_local = numpy.isfinite(local_sigma) & (local_sigma > 1e-12)
        residual_sigma = numpy.where(
            valid_local,
            local_sigma,
            fallback_sigma,
        )
        valid_fallback = numpy.isfinite(residual_sigma) & (
            residual_sigma > 1e-12
        )
        residual_sigma = numpy.where(valid_fallback, residual_sigma, 1.0)
        return residual_sigma * inflationFactor

    def _buildReturnsVolaGrid(
        self,
        pcaGrid: ReturnsPCAGrid,
        scenarioCenter: numpy.ndarray,
        residualSigma: numpy.ndarray,
        context: _RiskStateContext,
    ) -> dict[str, numpy.ndarray]:
        """Build a return-volatility state grid for every instrument."""
        return {
            instrument: self._buildInstrumentGrid(
                scenarioCenter[asset],
                residualSigma[asset],
                context.maxAbsoluteZ[asset],
                context.logReturnMean[asset],
                context.logReturnScale[asset],
                instrument,
            )
            for asset, instrument in enumerate(pcaGrid.instruments)
        }

    def _buildInstrumentGrid(
        self,
        scenarioCenter: float,
        residualSigma: float,
        maximumAbsoluteZ: float,
        logReturnMean: float,
        logReturnScale: float,
        instrument: str,
    ) -> numpy.ndarray:
        """Build all retained return-volatility states for one instrument."""
        full_range = (
            scenarioCenter - maximumAbsoluteZ,
            scenarioCenter + maximumAbsoluteZ,
        )
        bin_edges = numpy.linspace(*full_range, self.nZBins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        residual_range = (
            scenarioCenter - self.residualSigmaRange * residualSigma,
            scenarioCenter + self.residualSigmaRange * residualSigma,
        )
        keep_mask = (bin_edges[:-1] >= residual_range[0]) & (
            bin_edges[1:] <= residual_range[1]
        )
        if not numpy.any(keep_mask):
            if not self.allowEmptyBinFallback:
                raise ValueError(f"No valid return bins for {instrument}")
            nearest_center = numpy.argmin(
                numpy.abs(bin_centers - scenarioCenter)
            )
            keep_mask[int(nearest_center)] = True

        standardized_grid = bin_centers[keep_mask]
        log_return_grid = (
            logReturnMean + logReturnScale * standardized_grid
        )
        simple_return_grid = numpy.expm1(log_return_grid)
        conditional_volatility = logReturnScale * residualSigma
        return numpy.column_stack(
            (
                simple_return_grid,
                numpy.full(len(simple_return_grid), conditional_volatility),
            )
        )
