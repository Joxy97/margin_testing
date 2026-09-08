"""Margin-calculation interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING
from .calculation_outcome import CalculationOutcome

if TYPE_CHECKING:
    from portfolio import Portfolio
    from risk_state_generator import RiskState


class MarginCalculator(ABC):
    """Calculate portfolio margin across a collection of risk states."""

    def calculateOutcome(self, riskStates: Iterable[RiskState], portfolio: Portfolio) -> CalculationOutcome:
        """Return results without retaining diagnostics on the calculator."""
        return CalculationOutcome(self.calculateMargin(riskStates, portfolio))

    @abstractmethod
    def calculateMargin(
        self,
        riskStates: Iterable[RiskState],
        portfolio: Portfolio,
    ) -> float:
        """Return the required margin for ``portfolio``."""
        raise NotImplementedError
