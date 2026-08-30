#!/usr/bin/env python3
"""Convert ATP option portfolios into runnable application YAML files."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
CONFIG_ROOT = GENERATED / "configs"
MARKET_ROOT = GENERATED / "market"
MANIFEST = GENERATED / "manifest.csv"


def _sources() -> list[Path]:
    return sorted(ROOT.glob("*/*/atp/*/*.csv"))


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _date(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def _number(value: str) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def _quote(source: Path, row: dict[str, str]) -> dict[str, object]:
    product = row["Product Group"]
    return {
        "date": _date(source.parent.name),
        "symbol": row["Symbol"],
        "instrument_type": "future" if product == "FU" else "futures_option",
        "expiration_date": _date(row["Expiration Date"]),
        "strike": _number(row["Strike Price"]),
        "option_type": row["Option Type"],
        "exercise_style": "" if product == "FU" else "E",
        "multiplier": _number(row["Contract Size"]),
        "price": _number(row["Closing"]),
        "dividend_yield": 0,
    }


def _position(row: dict[str, str]) -> dict[str, object]:
    product = row["Product Group"]
    position = {
        "instrumentType": "future" if product == "FU" else "futures_option",
        "symbol": row["Symbol"],
        "expirationDate": _date(row["Expiration Date"]),
        "quantity": _number(row["Position"]),
        "multiplier": _number(row["Contract Size"]),
        "currency": row["Currency"],
    }
    if product == "OP":
        position.update({
            "strike": _number(row["Strike Price"]),
            "optionType": row["Option Type"],
            "exerciseStyle": "E",
        })
    return position


def _config(source: Path, rows: list[dict[str, str]]) -> dict:
    symbol, category, _, margin_date = source.parts[-5:-1]
    return {
        "marginDate": _date(margin_date),
        "portfolio": {
            "positions": [
                _position(row) for row in rows if float(row["Position"]) != 0.0
            ],
            "metadata": {
                "benchmarkCategory": category,
                "source": str(source.relative_to(ROOT)),
            },
        },
        "engine": {
            "downloadManager": {
                "providers": {"quotes": "derivative_csv"},
                "requestParameters": {
                    "location": f"../../../../market/{symbol}.csv"
                },
            },
            "dataManager": {"type": "derivative_quotes"},
            "riskStateGenerator": {
                "type": "option_scenarios",
                "historyDays": 365,
                "riskFreeRate": 0.04,
                "marginPeriodDays": 5,
                "confidenceLevel": 0.99,
                "priceScenarioSteps": 9,
                "minimumPriceShock": 0.03,
                "maximumPriceShock": 0.15,
                "volatilityShifts": [-0.03, 0.0, 0.03],
            },
            "marginCalculator": {
                "type": "state_aware_greedy",
                "pnlAnchor": "market",
            },
        },
    }


def main() -> None:
    sources = _sources()
    if not sources:
        raise RuntimeError("No ATP benchmark portfolios were found")
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    MARKET_ROOT.mkdir(parents=True, exist_ok=True)

    quotes: dict[str, dict[tuple, dict[str, object]]] = {}
    manifest = []
    for source in sources:
        rows = _read(source)
        symbol, category, _, margin_date = source.parts[-5:-1]
        for row in rows:
            quote = _quote(source, row)
            key = tuple(quote[column] for column in (
                "date", "symbol", "instrument_type", "expiration_date",
                "strike", "option_type", "exercise_style",
            ))
            existing = quotes.setdefault(symbol, {}).get(key)
            if existing is not None and existing["price"] != quote["price"]:
                raise ValueError(f"Conflicting quote for {key}")
            quotes[symbol][key] = quote

        output = CONFIG_ROOT / symbol / category / margin_date / f"{source.stem}.yaml"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            yaml.safe_dump(_config(source, rows), sort_keys=False),
            encoding="utf-8",
        )
        manifest.append({
            "config": str(output.relative_to(GENERATED)),
            "source": str(source.relative_to(ROOT)),
            "symbol": symbol,
            "category": category,
            "date": _date(margin_date),
            "portfolio": source.stem,
        })

    quote_fields = (
        "date", "symbol", "instrument_type", "expiration_date", "strike",
        "option_type", "exercise_style", "multiplier", "price",
        "dividend_yield",
    )
    for symbol, items in quotes.items():
        with (MARKET_ROOT / f"{symbol}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=quote_fields)
            writer.writeheader()
            writer.writerows(items.values())

    with MANIFEST.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest[0])
        writer.writeheader()
        writer.writerows(manifest)
    print(f"generated {len(manifest)} YAML files and {len(quotes)} market files")


if __name__ == "__main__":
    main()
