"""Download historical FX rates and convert grouped prices to EUR."""

import argparse
import csv
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://api.frankfurter.dev/v2/rates"
START_DATE = "2025-01-01"
END_DATE = "2026-09-07"

# Yahoo reports these values in minor units rather than ISO 4217 major units.
CURRENCY_ALIASES = {
    "GBp": ("GBP", 100.0),
    "ZAc": ("ZAR", 100.0),
    "ILA": ("ILS", 100.0),
    "KWF": ("KWD", 1000.0),
}


def _currency_for_row(row: dict[str, str]) -> tuple[str, float]:
    currency = row["currency"].strip()
    if currency in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[currency]
    if currency:
        return currency, 1.0
    # These are the only blank currencies among substantial_tickers.csv.
    if row["exchange"] == "SAU":
        return "SAR", 1.0
    if row["exchange"] in {"SGO", "CCS"}:
        return "CLP", 1.0
    raise ValueError(f"Cannot infer currency for {row['symbol']}")


def _get_json(url: str, attempts: int = 6) -> list[dict[str, object]]:
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "margin-testing-fx-converter/1.0"})
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as error:
            retryable = error.code == 429 or 500 <= error.code < 600
            if not retryable or attempt == attempts - 1:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
        except URLError:
            if attempt == attempts - 1:
                raise
            delay = 2**attempt
        time.sleep(delay)
    raise RuntimeError("FX request retry loop terminated unexpectedly")


def _load_currency_map(source_dir: Path) -> dict[str, tuple[str, float]]:
    with (source_dir / "universe.csv").open(newline="", encoding="utf-8") as handle:
        universe = {row["symbol"]: row for row in csv.DictReader(handle)}
    with (source_dir / "substantial_tickers.csv").open(newline="", encoding="utf-8") as handle:
        symbols = [row["symbol"] for row in csv.DictReader(handle)]
    return {symbol: _currency_for_row(universe[symbol]) for symbol in symbols}


def _download_rates(
    currencies: set[str],
    output_path: Path,
    start_date: str,
    end_date: str,
) -> dict[tuple[str, str], float]:
    query = urlencode(
        {
            "from": start_date,
            "to": end_date,
            "base": "EUR",
            "quotes": ",".join(sorted(currencies)),
        }
    )
    rows = _get_json(f"{API_URL}?{query}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "base", "quote", "rate"])
        writer.writeheader()
        writer.writerows(rows)
    return {(str(row["date"]), str(row["quote"])): float(row["rate"]) for row in rows}


def _convert_groups(
    groups_dir: Path,
    currency_map: dict[str, tuple[str, float]],
    rates: dict[tuple[str, str], float],
) -> None:
    for path in sorted(groups_dir.glob("*/*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if not fieldnames or fieldnames[0] != "Date":
                raise ValueError(f"{path} does not have Date as its first column")
            rows = list(reader)
        for row in rows:
            date = row["Date"]
            for ticker in fieldnames[1:]:
                quote, minor_units = currency_map[ticker]
                rate = rates.get((date, quote))
                if rate is None:
                    raise ValueError(f"No EUR/{quote} rate for {date}, required by {ticker} in {path}")
                value = row[ticker].strip()
                if value:
                    row[ticker] = repr(float(value) / rate / minor_units)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def run(source_dir: Path, grouped_dir: Path) -> None:
    currency_map = _load_currency_map(source_dir)
    currencies = {quote for quote, _ in currency_map.values()}
    rates_path = grouped_dir / "fx_rates" / "frankfurter_eur_rates.csv"
    rates = _download_rates(currencies, rates_path, START_DATE, END_DATE)
    _convert_groups(grouped_dir / "groups", currency_map, rates)
    metadata = {
        "provider": "Frankfurter",
        "endpoint": API_URL,
        "base": "EUR",
        "start_date": START_DATE,
        "end_date": END_DATE,
        "currencies": sorted(currencies),
        "minor_unit_aliases": CURRENCY_ALIASES,
        "blank_currency_inference": {"SAU": "SAR", "SGO": "CLP", "CCS": "CLP"},
        "conversion": "EUR price = local price / (local currency per EUR) / minor units",
    }
    with (grouped_dir / "fx_rates" / "conversion_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).parent.parent / "yahoo_equities")
    parser.add_argument("--grouped", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    run(args.source, args.grouped)


if __name__ == "__main__":
    main()
