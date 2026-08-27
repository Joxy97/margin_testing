"""Command data used by the yfinance provider."""

from dataclasses import dataclass
from typing import Any

from .command import Command


@dataclass(frozen=True)
class YfinanceCommand(Command):
    """Store immutable positional parameters for :func:`yfinance.download`."""

    parameters: tuple[Any, ...] = ()
