"""Typed configurations for concrete risk-state generators."""

from dataclasses import dataclass

from .correlated_returns_vola_grid_risk_state_generator import (
    CorrelatedReturnsVolaGridRiskStateGenerator,
)
from .pca_grid_provider import PCAGridProvider
from .returns_vola_grid_risk_state_generator import (
    ReturnsVolaGridRiskStateGenerator,
)


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


RiskStateGeneratorConfig = (
    ReturnsVolaGridRiskStateGeneratorConfig
    | CorrelatedReturnsVolaGridRiskStateGeneratorConfig
)
