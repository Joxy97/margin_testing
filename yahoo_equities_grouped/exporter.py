"""Build and export region/date-signature groups."""

import csv
import json
from collections import defaultdict
from pathlib import Path

from .data import all_dates, price_path, read_dates, read_price_rows, read_substantial_symbols, read_universe
from .hashing import date_numbers, polynomial_hash


def _safe_name(value: str) -> str:
    return value or "UNKNOWN"


def run(source_dir: Path, output_dir: Path) -> None:
    universe = read_universe(source_dir / "universe.csv")
    symbols = read_substantial_symbols(source_dir / "substantial_tickers.csv")
    prices_dir = source_dir / "prices"
    available = [symbol for symbol in symbols if price_path(prices_dir, symbol).is_file()]
    missing = sorted(set(symbols) - set(available))

    global_dates = all_dates(iter(price_path(prices_dir, symbol) for symbol in available))
    date_to_number = {date: number for number, date in enumerate(global_dates, start=1)}

    groups: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    date_sets: dict[tuple[str, str], tuple[str, ...]] = {}
    for symbol in available:
        region = universe.get(symbol, "")
        dates = tuple(read_dates(price_path(prices_dir, symbol)))
        key = polynomial_hash(date_numbers(dates, date_to_number))
        groups[region][key].add(symbol)
        date_sets[(region, symbol)] = dates

    output_dir.mkdir(parents=True, exist_ok=True)
    groups_dir = output_dir / "groups"
    groups_dir.mkdir(exist_ok=True)
    # Group filenames are derived from the current ranking, so old generated
    # filenames must not remain alongside a new export.
    for path in groups_dir.glob("*/*.csv"):
        path.unlink()
    with (output_dir / "date_mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "number"])
        writer.writerows((date, date_to_number[date]) for date in global_dates)

    manifest = {
        "prime": 1_000_000_007,
        "base": 911_382_323,
        "hash_order": "sorted trading dates, chronological",
        "date_numbering": "global unique observed dates, one-based",
        "source_filter": "substantial_tickers.csv",
        "substantial_tickers": len(symbols),
        "exported_tickers": len(available),
        "missing_price_files": missing,
        "regions": {},
    }
    for region, region_groups in sorted(groups.items()):
        region_name = _safe_name(region)
        region_manifest = []
        region_dir = output_dir / "groups" / region_name
        region_dir.mkdir(exist_ok=True)
        ordered_groups = sorted(
            region_groups.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
        for rank, (key, group_symbols) in enumerate(ordered_groups, start=1):
            group_symbols = sorted(group_symbols)
            output_path = region_dir / f"group_{rank}_{len(group_symbols)}.csv"
            source_data = {
                symbol: read_price_rows(price_path(prices_dir, symbol))
                for symbol in group_symbols
            }
            first_fields, _ = source_data[group_symbols[0]]
            if "Close" not in first_fields:
                raise ValueError(f"{price_path(prices_dir, group_symbols[0])} does not contain Close")
            fieldnames = ["Date"] + group_symbols
            rows_by_symbol_and_date = {
                symbol: {row["Date"]: row for row in rows}
                for symbol, (fields, rows) in source_data.items()
            }
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for date in date_sets[(region, group_symbols[0])]:
                    merged = {"Date": date}
                    for symbol in group_symbols:
                        fields, _ = source_data[symbol]
                        if "Close" not in fields:
                            raise ValueError(f"Price columns differ for region {region!r}, hash {key}")
                        row = rows_by_symbol_and_date[symbol].get(date)
                        if row is None:
                            raise ValueError(f"Trading dates differ for region {region!r}, hash {key}")
                        merged[symbol] = row["Close"]
                    writer.writerow(merged)
            region_manifest.append(
                {
                    "rank": rank,
                    "number_of_stocks": len(group_symbols),
                    "hash": key,
                    "tickers": group_symbols,
                    "csv": str(output_path.relative_to(output_dir)),
                }
            )
        manifest["regions"][region_name] = region_manifest

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
