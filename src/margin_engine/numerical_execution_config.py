"""Opt-in numerical execution settings, separate from risk model parameters."""

from dataclasses import dataclass
from risk_state_generator.pca_backend import PCABackendConfig


@dataclass(frozen=True)
class TorchNumericalExecutionConfig:
    device: str = "auto"
    dtype: str = "auto"

    def __post_init__(self):
        PCABackendConfig(type="torch", device=self.device, dtype=self.dtype)

    def createExecution(self, generator, calculator):
        from .torch_returns_execution import TorchReturnsExecution
        return TorchReturnsExecution(self, generator, calculator)


def validateResidentCollaborators(visitor, policy, comparison=None):
    """The fused path only implements standard portfolio semantics and scheduling."""
    from margin_calculator.optimization.portfolio_risk_state_bqm_visitor import PortfolioRiskStateBQMVisitor
    from margin_calculator import BatchBQMExecutionPolicy, SequentialBQMExecutionPolicy
    from margin_calculator.state_aware_greedy_risk_state_visitor import StateAwareGreedyRiskStateVisitor
    if visitor is not None and type(visitor) is not PortfolioRiskStateBQMVisitor:
        raise ValueError("Resident execution does not support custom BQM visitors; use the host path")
    if type(policy) not in (BatchBQMExecutionPolicy, SequentialBQMExecutionPolicy):
        raise ValueError("Resident execution does not support custom execution policies; use the host path")
    if comparison is not None and type(comparison) is not StateAwareGreedyRiskStateVisitor:
        raise ValueError("Resident execution does not support custom comparison visitors; use the host path")
