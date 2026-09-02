"""Typed configurations for margin calculators."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from risk_state_generator import PortfolioRiskStateBQMVisitor

from .bqm_margin_calculator import BQMMarginCalculator
from .greedy_margin_calculator import GreedyMarginCalculator
from .state_aware_greedy_margin_calculator import (
    StateAwareGreedyMarginCalculator,
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
    bqmVisitor: PortfolioRiskStateBQMVisitor | None = None
    executionPolicy: BQMExecutionPolicy = field(
        default_factory=SequentialBQMExecutionPolicy
    )
    comparisonPnlAnchor: str | None = None

    def createMarginCalculator(self) -> BQMMarginCalculator:
        return BQMMarginCalculator(
            bqmSolver=self.solver.createBQMSolver(),
            modelParameters=self.modelParameters,
            solverParameters=self.solver.solverParameters,
            bqmVisitor=self.bqmVisitor,
            executionPolicy=self.executionPolicy,
            comparisonPnlAnchor=self.comparisonPnlAnchor,
        )


@dataclass(frozen=True)
class GreedyMarginCalculatorConfig:
    """Configuration for deterministic greedy margin calculation."""

    def createMarginCalculator(self) -> GreedyMarginCalculator:
        return GreedyMarginCalculator()


@dataclass(frozen=True)
class StateAwareGreedyMarginCalculatorConfig:
    """Configuration for risk-state-specific scenario margining."""

    pnlAnchor: str = "market"

    def createMarginCalculator(self) -> StateAwareGreedyMarginCalculator:
        return StateAwareGreedyMarginCalculator(pnlAnchor=self.pnlAnchor)


MarginCalculatorConfig = (
    BQMMarginCalculatorConfig
    | GreedyMarginCalculatorConfig
    | StateAwareGreedyMarginCalculatorConfig
)
