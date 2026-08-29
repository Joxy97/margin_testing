"""Margin calculation report."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MarginReport:
    """Contain the margin produced by one engine run."""

    margin: float
