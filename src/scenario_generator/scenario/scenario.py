"""Base interface for BQM generation scenarios."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..bqm_model_generator import BQMModelGenerator


class Scenario(ABC):
    """Accept a BQM model generator for this scenario."""

    @abstractmethod
    def accept(self, bqmModelGenerator: BQMModelGenerator) -> None:
        """Dispatch this scenario to ``bqmModelGenerator``."""
        raise NotImplementedError
