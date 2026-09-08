"""Typed top-level margin-engine configuration."""

from dataclasses import dataclass, field
from .numerical_execution_config import TorchNumericalExecutionConfig

from data_manager import DataManagerConfig, DerivativeQuoteDataManagerConfig
from download_manager import DownloadManagerConfig
from margin_calculator import BQMMarginCalculatorConfig, MarginCalculatorConfig
from risk_state_generator import (
    ReturnsVolaGridRiskStateGeneratorConfig,
    RiskStateGeneratorConfig,
)


@dataclass(frozen=True)
class MarginEngineConfig:
    """Complete dependency configuration for one margin engine."""

    downloadManager: DownloadManagerConfig = field(
        default_factory=DownloadManagerConfig
    )
    dataManager: DataManagerConfig | DerivativeQuoteDataManagerConfig = field(
        default_factory=DataManagerConfig
    )
    riskStateGenerator: RiskStateGeneratorConfig = field(
        default_factory=ReturnsVolaGridRiskStateGeneratorConfig
    )
    marginCalculator: MarginCalculatorConfig = field(
        default_factory=BQMMarginCalculatorConfig
    )

    numericalExecution: TorchNumericalExecutionConfig | None = None

    def __post_init__(self):
        if self.numericalExecution is None:
            return
        from risk_state_generator.config import CorrelatedReturnsVolaGridRiskStateGeneratorConfig
        from margin_calculator import GreedyMarginCalculatorConfig, StateAwareGreedyMarginCalculatorConfig
        if type(self.riskStateGenerator) not in (ReturnsVolaGridRiskStateGeneratorConfig,
                                                 CorrelatedReturnsVolaGridRiskStateGeneratorConfig):
            raise ValueError("Torch numerical execution requires a returns-grid risk generator")
        if isinstance(self.marginCalculator, BQMMarginCalculatorConfig):
            from .numerical_execution_config import validateResidentCollaborators
            validateResidentCollaborators(self.marginCalculator.bqmVisitor, self.marginCalculator.executionPolicy)
            solver = self.marginCalculator.solver
            if solver.solverType not in {"torch_sbm", "adaptive_torch_sbm", "torch_svl"}:
                raise ValueError("Resident BQM execution requires a Torch solver")
            devices = solver.constructorParameters.get("devices", ())
            if len(devices) > 1:
                raise ValueError("Resident numerical execution requires one solver device")
        elif type(self.marginCalculator) not in (GreedyMarginCalculatorConfig, StateAwareGreedyMarginCalculatorConfig):
            raise ValueError("Unsupported calculator for resident numerical execution")
