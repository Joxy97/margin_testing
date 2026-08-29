"""Risk-state generation interface."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import date

from download_unit import DataRequest
from portfolio import Portfolio

from .risk_state import RiskState
from .risk_state_generation_context import RiskStateGenerationContext


class RiskStateGenerator(ABC):
    """Generate risk states from downloaded data."""

    @abstractmethod
    def createDataRequest(
        self,
        portfolio: Portfolio,
        marginDate: date,
    ) -> DataRequest:
        """Describe the market data needed for this calculation date."""
        raise NotImplementedError

    @abstractmethod
    def getRiskStates(
        self,
        context: RiskStateGenerationContext,
    ) -> Iterator[RiskState]:
        """Lazily create risk states from the supplied runtime context."""
        raise NotImplementedError
