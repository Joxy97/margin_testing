"""Input readers for the Yahoo equities grouping job."""

import csv
from collections.abc import Iterator
from pathlib import Path


def read_universe(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["symbol"]: row["region"] for row in csv.DictReader(handle)}


def read_substantial_symbols(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["symbol"] for row in csv.DictReader(handle)]


def price_path(prices_dir: Path, symbol: str) -> Path:
    return prices_dir / f"{symbol}.csv"


def read_price_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.reader(handle))
    if not raw_rows:
        raise ValueError(f"{path} is empty")

    date_header_index = next(
        (index for index, row in enumerate(raw_rows) if row and row[0] == "Date"),
        None,
    )
    if date_header_index is None:
        raise ValueError(f"{path} does not contain a Date column")
    if date_header_index == 0:
        fieldnames = raw_rows[0]
        data_rows = raw_rows[1:]
    else:
        fieldnames = raw_rows[0]
        fieldnames[0] = "Date"
        data_rows = raw_rows[date_header_index + 1 :]
    width = len(fieldnames)
    rows = [dict(zip(fieldnames, row + [""] * (width - len(row)))) for row in data_rows]
    return fieldnames, rows


def read_dates(path: Path) -> list[str]:
    _, rows = read_price_rows(path)
    return sorted({row["Date"] for row in rows if row.get("Date")})


def all_dates(paths: Iterator[Path]) -> list[str]:
    dates: set[str] = set()
    for path in paths:
        dates.update(read_dates(path))
    return sorted(dates)
