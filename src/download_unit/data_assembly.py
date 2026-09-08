"""Assemble normalized observations without conflating price and quote schemas."""

from collections.abc import Iterable

from .command import DataRequest


def assembleData(command: DataRequest, frames: Iterable):
    """Return one requested table, joining fragments by observation identity."""
    import pandas

    data = pandas.concat(frames, ignore_index=True)
    data["date"] = pandas.to_datetime(data["date"], errors="raise")
    data = data.loc[data["date"].between(
        pandas.Timestamp(command.start_date), pandas.Timestamp(command.end_date)
    )]
    if command.data_type == "derivativeQuotes":
        data = data.loc[data["symbol"].astype(str).isin(command.instruments)].copy()
        identity = ["date", "symbol", "instrument_type", "expiration_date"]
        identity += [name for name in ("strike", "option_type", "exercise_style")
                     if name in data]
        if (data.groupby(identity, dropna=False).nunique(dropna=True) > 1).any().any():
            raise ValueError("Conflicting observations for the same derivative quote")
        return data.drop_duplicates(identity).sort_values(identity).reset_index(drop=True)
    if (data.groupby("date").nunique(dropna=True) > 1).any().any():
        raise ValueError("Conflicting observations for the same date and instrument")
    return (data.groupby("date", sort=True).first()
            .reindex(columns=command.instruments).reset_index())
