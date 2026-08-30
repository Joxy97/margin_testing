"""Typed configurations for concrete risk-state generators."""

from dataclasses import dataclass

from .correlated_returns_vola_grid_risk_state_generator import (
    CorrelatedReturnsVolaGridRiskStateGenerator,
)
from .pca_grid_provider import PCAGridProvider
from .returns_vola_grid_risk_state_generator import (
    ReturnsVolaGridRiskStateGenerator,
)
from .option_scenario_risk_state_generator import OptionScenarioRiskStateGenerator


@dataclass(frozen=True)
class ReturnsVolaGridRiskStateGeneratorConfig:
    """Configuration for returns-volatility-grid generation."""

    pcaGridProvider: PCAGridProvider | None = None
    ew_window: int = 30
    ew_lambda: float = 0.94
    components: int = 1
    scenariosPerComponents: tuple[int, ...] = ()
    tailDensityGamma: float = 1.0
    nZBins: int = 21
    nNearest: int | None = None
    residualSigmaRange: float = 5.0
    allowEmptyBinFallback: bool = True
    distanceInflationAlpha: float = 0.5
    distanceInflationPower: float = 2.0
    maxInflationFactor: float = 5.0

    def _parameters(self) -> dict[str, object]:
        return {
            name: value
            for name, value in vars(self).items()
            if name not in {"topKNeighbors", "correlationBlockBytes"}
        }

    def createRiskStateGenerator(self) -> ReturnsVolaGridRiskStateGenerator:
        return ReturnsVolaGridRiskStateGenerator(**self._parameters())


@dataclass(frozen=True)
class CorrelatedReturnsVolaGridRiskStateGeneratorConfig(
    ReturnsVolaGridRiskStateGeneratorConfig
):
    """Configuration for correlated returns-volatility-grid generation."""

    topKNeighbors: int = 5
    correlationBlockBytes: int = 128 * 1024 * 1024

    def createRiskStateGenerator(
        self,
    ) -> CorrelatedReturnsVolaGridRiskStateGenerator:
        return CorrelatedReturnsVolaGridRiskStateGenerator(
            **self._parameters(),
            topKNeighbors=self.topKNeighbors,
            correlationBlockBytes=self.correlationBlockBytes,
        )


@dataclass(frozen=True)
class OptionScenarioRiskStateGeneratorConfig:
    """Configuration for futures/equity option stress scenarios."""

    historyDays: int = 365
    riskFreeRate: float = 0.04
    dayCountBasis: float = 365.0
    tradingDaysPerYear: int = 252
    marginPeriodDays: int = 5
    projectionHorizonDays: int = 0
    confidenceLevel: float = 0.99
    stressMultiplier: float = 1.0
    priceScenarioSteps: int = 9
    minimumPriceShock: float = 0.03
    maximumPriceShock: float = 0.15
    volatilityShifts: tuple[float, ...] = (-0.03, 0.0, 0.03)
    minimumVolatility: float = 0.01
    maximumVolatility: float = 3.0
    maximumVolatilityShift: float = 0.10
    ewmaLambda: float = 0.94
    fallbackRho: float = -0.75
    fallbackVolOfVolatility: float = 0.90
    volShockMinimumObservations: int = 5
    maximumSmileExtrapolation: float = 0.15
    americanOptionSteps: int = 200

    def createRiskStateGenerator(self) -> OptionScenarioRiskStateGenerator:
        return OptionScenarioRiskStateGenerator(**vars(self))


RiskStateGeneratorConfig = (
    ReturnsVolaGridRiskStateGeneratorConfig
    | CorrelatedReturnsVolaGridRiskStateGeneratorConfig
    | OptionScenarioRiskStateGeneratorConfig
)
