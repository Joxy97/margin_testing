"""Base PCA-scenario type."""

from dataclasses import dataclass

from ..pca_key import PCAKey


@dataclass
class PCAScenario:
    """Represent a PCA scenario associated with a cached grid key."""

    pcaKey: PCAKey
