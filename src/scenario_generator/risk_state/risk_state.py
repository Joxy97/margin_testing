"""Base interface for risk states used in BQM generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..bqm_model_generator import BQMModelGenerator

if TYPE_CHECKING:
    from portfolio import Portfolio


class RiskState(ABC):
    """Accept a BQM model generator for this risk state."""

    @abstractmethod
    def accept(
        self,
        bqmModelGenerator: BQMModelGenerator,
        portfolio: Portfolio,
    ) -> None:
        """Dispatch this risk state and portfolio to the model generator."""
        raise NotImplementedError
