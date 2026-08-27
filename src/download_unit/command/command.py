"""Base command data object."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    """Immutable base class for provider-specific command data."""

    pass
