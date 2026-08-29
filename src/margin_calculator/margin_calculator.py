"""Margin-calculation interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from portfolio import Portfolio
    from risk_state_generator import RiskState


class MarginCalculator(ABC):
    """Calculate portfolio margin across a collection of risk states."""

    @abstractmethod
    def calculateMargin(
        self,
        riskStates: Iterable[RiskState],
        portfolio: Portfolio,
    ) -> float:
        """Return the required margin for ``portfolio``."""
        raise NotImplementedError
