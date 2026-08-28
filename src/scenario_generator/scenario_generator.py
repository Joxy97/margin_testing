"""Scenario generation interface."""

from abc import ABC, abstractmethod
from typing import Any

from download_unit import UnifiedFormatCommand

from .risk_state import RiskState


class ScenarioGenerator(ABC):
    """Generate risk states from downloaded data."""

    def dataRequirements(
        self,
        command: UnifiedFormatCommand,
    ) -> UnifiedFormatCommand:
        """Return the generator's data requirements without modification."""
        return command

    @abstractmethod
    def getRiskStates(self, data: Any) -> list[RiskState]:
        """Create risk states from downloaded data."""
        raise NotImplementedError
