"""Base key type for cached PCA grids."""

from abc import ABC, abstractmethod
from typing import Any


class PCAKey(ABC):
    """Identify a PCA grid by the parameters used to create it."""

    @abstractmethod
    def equals(self, pcaGrid: Any) -> bool:
        """Return whether this key describes ``pcaGrid``."""
        raise NotImplementedError
