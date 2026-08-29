"""Base class for optimization-driven margin calculators."""

from abc import ABC
from typing import Any, Mapping

from .margin_calculator import MarginCalculator


class OptimizationMarginCalculator(MarginCalculator, ABC):
    """Store parameters shared by optimization-based calculations."""

    def __init__(
        self,
        solverParameters: Mapping[str, Any] | None = None,
    ) -> None:
        self.solverParameters: dict[str, Any] = dict(solverParameters or {})
