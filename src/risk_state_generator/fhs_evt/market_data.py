"""Synchronous close-price preparation for the FHS-EVT factor model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy
import pandas


def _readonly(values: numpy.ndarray) -> numpy.ndarray:
    result = numpy.ascontiguousarray(values, dtype=numpy.float64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LogReturnHistory:
    """Immutable synchronized log changes and their stable row identity."""

    instruments: tuple[str, ...]
    changes: numpy.ndarray
    dates: tuple[date, ...]
    rowIds: tuple[str, ...]

    def __post_init__(self) -> None:
        changes = _readonly(self.changes)
        if changes.ndim != 2 or changes.shape[1] != len(self.instruments):
            raise ValueError("changes must contain one column per instrument")
        if changes.shape[0] != len(self.dates) or len(self.rowIds) != len(self.dates):
            raise ValueError("change rows, dates, and row IDs must align")
        if not numpy.isfinite(changes).all():
            raise ValueError("synchronous log returns must be finite")
        if len(set(self.instruments)) != len(self.instruments):
            raise ValueError("history instruments must be unique")
        if len(set(self.rowIds)) != len(self.rowIds):
            raise ValueError("history row IDs must be unique")
        object.__setattr__(self, "instruments", tuple(self.instruments))
        object.__setattr__(self, "changes", changes)
        object.__setattr__(self, "dates", tuple(self.dates))
        object.__setattr__(self, "rowIds", tuple(self.rowIds))


def prepareLogReturnHistory(
    data: pandas.DataFrame,
    instruments: tuple[str, ...],
    forecastDate: date,
    minimumObservations: int,
) -> LogReturnHistory:
    """Create synchronous consecutive changes without forward filling.

    The forecast-date quote is excluded so a backtest cannot feed the realized
    return into the scenario calibration for that same date.
    """
    if not isinstance(data, pandas.DataFrame):
        raise TypeError("marketData must be a pandas DataFrame")
    missing = set(instruments).difference(data.columns)
    if missing:
        raise ValueError(f"marketData is missing instruments: {sorted(missing)}")
    prices = data.copy()
    if "date" in prices.columns:
        prices["date"] = pandas.to_datetime(prices["date"], errors="raise")
        prices = prices.set_index("date")
    else:
        prices.index = pandas.to_datetime(prices.index, errors="raise")
    if not prices.index.is_unique:
        raise ValueError("market-data dates must be unique")
    prices = prices.sort_index().loc[
        lambda frame: frame.index < pandas.Timestamp(forecastDate),
        list(instruments),
    ]
    if len(prices) < 2:
        raise ValueError("marketData must contain at least two prior price rows")
    try:
        price_values = prices.to_numpy(dtype=numpy.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("market prices must be numeric") from error
    finite = numpy.isfinite(price_values)
    if numpy.any(price_values[finite] <= 0.0):
        raise ValueError("log-return prices must be strictly positive")
    synchronous_pairs = numpy.all(finite[1:] & finite[:-1], axis=1)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        all_changes = numpy.log(price_values[1:] / price_values[:-1])
    changes = all_changes[synchronous_pairs]
    dates = tuple(
        timestamp.date() for timestamp in prices.index[1:][synchronous_pairs]
    )
    if len(changes) < minimumObservations:
        raise ValueError(
            "marketData contains "
            f"{len(changes)} synchronous changes; "
            f"minimumObservations is {minimumObservations}"
        )
    row_ids = tuple(
        f"{current.isoformat()}:{index:06d}"
        for index, current in enumerate(dates)
    )
    return LogReturnHistory(tuple(instruments), changes, dates, row_ids)
