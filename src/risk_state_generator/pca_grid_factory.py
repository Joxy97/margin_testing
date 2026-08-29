"""Factory for PCA grids."""

from typing import Any

from .pca_grid import PCAGrid, ReturnsPCAGrid
from .pca_key import PCAKey, ReturnsPCAKey


class PCAGridFactory:
    """Create the PCA-grid implementation corresponding to a PCA key."""

    @staticmethod
    def createPCAGrid(key: PCAKey, data: Any) -> PCAGrid:
        """Create a PCA grid corresponding to ``key`` and ``data``."""
        if isinstance(key, ReturnsPCAKey):
            return ReturnsPCAGrid.construct(key, data)
        raise ValueError(f"Unsupported PCA key type: {type(key).__name__}")
