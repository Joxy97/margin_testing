"""Returns-volatility-grid PCA scenario."""

from dataclasses import dataclass
from numbers import Real

from .pca_scenario import PCAScenario


@dataclass
class ReturnsVolaGridPCAScenario(PCAScenario):
    """Represent one real-valued point in a returns PCA grid."""

    point: tuple[Real, ...]
