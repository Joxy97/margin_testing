"""Greedy returns-grid margin calculation."""

from .state_aware_greedy_margin_calculator import StateAwareGreedyMarginCalculator


class GreedyMarginCalculator(StateAwareGreedyMarginCalculator):
    """Scenario calculator configured with the default greedy visitor."""
