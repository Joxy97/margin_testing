"""PCA-grid provider interface."""

from __future__ import annotations

from collections.abc import Iterable
from threading import Lock
from typing import Any, ClassVar

from .pca_grid import PCAGrid
from .pca_grid_factory import PCAGridFactory
from .pca_key import PCAKey


class PCAGridProvider:
    """Maintain the shared collection of PCA grids."""

    _instance: ClassVar[PCAGridProvider | None] = None
    _instanceLock: ClassVar[Lock] = Lock()

    def __new__(cls) -> PCAGridProvider:
        if cls._instance is None:
            with cls._instanceLock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance.pcaGrids = {}
                    cls._instance = instance
        return cls._instance

    pcaGrids: dict[PCAKey, PCAGrid]

    def getPCAGrid(self, key: PCAKey) -> PCAGrid | None:
        """Return the PCA grid stored under ``key``, if one exists."""
        return self.pcaGrids.get(key)

    def createGridPCA(self, key: PCAKey, data: Any) -> PCAGrid:
        """Create, cache, and return a PCA grid for ``key`` and ``data``."""
        pca_grid = PCAGridFactory.createPCAGrid(key, data)
        self.pcaGrids[key] = pca_grid
        return pca_grid

    def createPCAGrid(self, key: PCAKey, data: Any) -> PCAGrid:
        """Create a PCA grid; alias for :meth:`createGridPCA`."""
        return self.createGridPCA(key, data)

    def removePCAGrid(self, key: PCAKey) -> None:
        """Remove the PCA grid stored under ``key``, if one exists."""
        self.pcaGrids.pop(key, None)

    def removePCAGrids(self, keys: Iterable[PCAKey]) -> None:
        """Remove the PCA grids stored under each supplied key."""
        for key in tuple(keys):
            self.removePCAGrid(key)

    def purgePCAGrids(self) -> None:
        """Remove every cached PCA grid."""
        self.pcaGrids.clear()
