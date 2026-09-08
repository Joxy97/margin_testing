"""Risk-state objects used to generate BQM models."""

from cache import Cache, CacheFactory, LRUCache

from .correlated_returns_vola_grid_risk_state_generator import (
    CorrelatedReturnsVolaGridRiskStateGenerator,
)
from .config import (
    CorrelatedReturnsVolaGridRiskStateGeneratorConfig,
    ReturnsVolaGridRiskStateGeneratorConfig,
    OptionScenarioRiskStateGeneratorConfig,
    RiskStateGeneratorConfig,
)
from .pca_grid import PCAGrid, ReturnsPCAGrid
from .pca_backend import PCABackend, PCABackendConfig, PCAFit, NumpyPCABackend, TorchPCABackend
from .pca_grid_factory import PCAGridFactory
from .pca_grid_provider import PCAGridProvider
from .pca_key import PCAKey, ReturnsPCAKey
from .pca_scenario import PCAScenario, ReturnsVolaGridPCAScenario
from .risk_state import (
    CorrelationFactors,
    CorrelatedReturnsVolaGridRiskState,
    DenseReturnsVolaGrid,
    RiskState,
    OptionScenarioRiskState,
    ReturnsVolaGridRiskState,
)
from .risk_state_generator import RiskStateGenerator
from .risk_state_generation_context import RiskStateGenerationContext
from .returns_vola_grid_risk_state_generator import (
    ReturnsVolaGridRiskStateGenerator,
)
from .option_scenario_risk_state_generator import OptionScenarioRiskStateGenerator

from .pca_grid_provider import PCAGridProviderConfig

__all__ = [
    "PCAGridProviderConfig",

    "Cache",
    "CacheFactory",
    "CorrelatedReturnsVolaGridRiskStateGenerator",
    "CorrelatedReturnsVolaGridRiskStateGeneratorConfig",
    "CorrelatedReturnsVolaGridRiskState",
    "CorrelationFactors",
    "DenseReturnsVolaGrid",
    "LRUCache",
    "PCAGrid",
    "PCABackend",
    "PCABackendConfig",
    "PCAFit",
    "NumpyPCABackend",
    "TorchPCABackend",
    "PCAGridFactory",
    "PCAGridProvider",
    "PCAKey",
    "PCAScenario",
    "PortfolioRiskStateBQMVisitor",
    "StructuralQUBOTemplateCache",
    "RiskState",
    "OptionScenarioRiskState",
    "OptionScenarioRiskStateGenerator",
    "OptionScenarioRiskStateGeneratorConfig",
    "ReturnsPCAGrid",
    "ReturnsPCAKey",
    "ReturnsVolaGridRiskState",
    "ReturnsVolaGridPCAScenario",
    "ReturnsVolaGridRiskStateGenerator",
    "ReturnsVolaGridRiskStateGeneratorConfig",
    "RiskStateGeneratorConfig",
    "RiskStateGenerator",
    "RiskStateGenerationContext",
]


def __getattr__(name):
    if name in {"PortfolioRiskStateBQMVisitor", "StructuralQUBOTemplateCache"}:
        from margin_calculator.optimization import portfolio_risk_state_bqm_visitor
        return getattr(portfolio_risk_state_bqm_visitor, name)
    raise AttributeError(name)
