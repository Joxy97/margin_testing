"""Typed configurations for margin calculators."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from risk_state_generator import PortfolioRiskStateBQMManager

from .bqm_margin_calculator import BQMMarginCalculator
from .greedy_margin_calculator import GreedyMarginCalculator
from .greedy_portfolio_risk_state_scenario import (
    GreedyPortfolioRiskStateScenario,
)
from .optimization.optimization_solver.bqm_solver.bqm_solver_config import (
    BQMSolverConfig,
)
from .optimization.optimization_solver.bqm_solver.bqm_execution_policy import (
    BQMExecutionPolicy,
    SequentialBQMExecutionPolicy,
)


@dataclass(frozen=True)
class BQMMarginCalculatorConfig:
    """Configuration for BQM construction, solving, and decoding."""

    solver: BQMSolverConfig = field(default_factory=BQMSolverConfig)
    modelParameters: Mapping[str, Any] = field(default_factory=dict)
    bqmManager: PortfolioRiskStateBQMManager | None = None
    executionPolicy: BQMExecutionPolicy = field(
        default_factory=SequentialBQMExecutionPolicy
    )

    def createMarginCalculator(self) -> BQMMarginCalculator:
        return BQMMarginCalculator(
            bqmSolver=self.solver.createBQMSolver(),
            modelParameters=self.modelParameters,
            solverParameters=self.solver.solverParameters,
            bqmManager=self.bqmManager,
            executionPolicy=self.executionPolicy,
        )


@dataclass(frozen=True)
class GreedyMarginCalculatorConfig:
    """Configuration for deterministic greedy margin calculation."""

    scenarioVisitor: GreedyPortfolioRiskStateScenario | None = None

    def createMarginCalculator(self) -> GreedyMarginCalculator:
        return GreedyMarginCalculator(self.scenarioVisitor)


MarginCalculatorConfig = BQMMarginCalculatorConfig | GreedyMarginCalculatorConfig
