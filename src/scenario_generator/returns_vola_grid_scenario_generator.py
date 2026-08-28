"""Returns-volatility-grid scenario generator."""

from collections.abc import Iterable
from typing import Any

from .pca_grid_provider import PCAGridProvider
from .pca_key import PCAKey
from .pca_scenario import PCAScenario, ReturnsVolaGridPCAScenario
from .risk_state import ReturnsVolaGridRiskState
from .scenario_generator import ScenarioGenerator


class ReturnsVolaGridScenarioGenerator(ScenarioGenerator):
    """Generate returns-volatility-grid risk states from PCA grids."""

    def __init__(self, pcaGridProvider: PCAGridProvider | None = None) -> None:
        self.__pcaGridProvider = (
            pcaGridProvider
            if pcaGridProvider is not None
            else PCAGridProvider()
        )

    def getRiskStates(self, data: Any) -> list[ReturnsVolaGridRiskState]:
        """Create returns-volatility-grid risk states from ``data``."""
        pca_key = self.getPCAKey(data)
        pca_grid = self.__pcaGridProvider.getPCAGrid(pca_key)
        if pca_grid is None:
            pca_grid = self.__pcaGridProvider.createPCAGrid(pca_key, data)
        return [
            self.getRiskState(pca_scenario)
            for pca_scenario in self.generateScenarios(pca_key)
        ]

    def getPCAKey(self, data: Any) -> PCAKey:
        """Create the PCA key required for ``data``."""
        raise NotImplementedError

    def generateScenarios(
        self,
        pcaKey: PCAKey,
    ) -> Iterable[ReturnsVolaGridPCAScenario]:
        """Generate PCA scenarios for ``pcaKey``."""
        return ()

    def getRiskState(
        self,
        pcaScenario: PCAScenario,
    ) -> ReturnsVolaGridRiskState:
        """Create a risk state from ``pcaScenario``."""
        raise NotImplementedError
