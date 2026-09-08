from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from yfinance import EquityQuery


# ============================================================
# Configuration
# ============================================================

START_DATE = "2025-01-01"
END_DATE = "2026-09-08"   # yf.download end is exclusive

BATCH_SIZE = 25           # history-download batch size
SCREEN_PAGE_SIZE = 250    # Yahoo screener maximum

MAX_RETRIES = 8
INITIAL_BACKOFF = 5.0     # seconds
MAX_BACKOFF = 300.0       # 5 minutes
JITTER = 0.25             # ±25%

INTER_BATCH_SLEEP = 2.0
INTER_SCREEN_SLEEP = 1.0

OUTPUT_DIR = Path("yahoo_equities")
DATA_DIR = OUTPUT_DIR / "prices"

UNIVERSE_FILE = OUTPUT_DIR / "universe.csv"
STATUS_FILE = OUTPUT_DIR / "status.csv"

OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# ============================================================
# Retry helper
# ============================================================

def sleep_with_backoff(attempt: int) -> None:
    """
    Exponential backoff:
        delay = min(MAX_BACKOFF, INITIAL_BACKOFF * 2^attempt)
    plus random jitter.
    """
    base = min(MAX_BACKOFF, INITIAL_BACKOFF * (2 ** attempt))
    multiplier = random.uniform(1.0 - JITTER, 1.0 + JITTER)
    delay = base * multiplier

    print(f"    Backing off for {delay:.1f} seconds...")
    time.sleep(delay)


def retry_call(func, *args, **kwargs):
    last_exception = None

    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)

        except Exception as exc:
            last_exception = exc
            message = str(exc).lower()

            # Things commonly associated with rate limiting / temporary
            # Yahoo failures.
            retryable = any(
                x in message
                for x in [
                    "429",
                    "too many requests",
                    "rate limit",
                    "timed out",
                    "timeout",
                    "connection",
                    "temporarily unavailable",
                    "502",
                    "503",
                    "504",
                ]
            )

            if not retryable:
                raise

            print(
                f"    Temporary failure "
                f"(attempt {attempt + 1}/{MAX_RETRIES}): {exc}"
            )

            if attempt < MAX_RETRIES - 1:
                sleep_with_backoff(attempt)

    raise RuntimeError(
        f"Operation failed after {MAX_RETRIES} retries"
    ) from last_exception


# ============================================================
# 1. Discover Yahoo equity symbols
# ============================================================

from yfinance.const import EQUITY_SCREENER_EQ_MAP

def get_regions() -> list[str]:
    return sorted(set(EQUITY_SCREENER_EQ_MAP["region"]))


def fetch_region(region: str) -> list[dict]:
    """
    Enumerate all equities reported by Yahoo for one region.
    """
    query = EquityQuery("eq", ["region", region])

    records = []
    offset = 0
    total = None

    while total is None or offset < total:
        print(
            f"[SCREEN] region={region:>4} "
            f"offset={offset}"
            + (f"/{total}" if total is not None else "")
        )

        response = retry_call(
            yf.screen,
            query,
            offset=offset,
            size=SCREEN_PAGE_SIZE,
            sortField="ticker",
            sortAsc=True,
        )

        quotes = response.get("quotes", [])
        total = response.get("total", len(quotes))

        if not quotes:
            break

        for q in quotes:
            symbol = q.get("symbol")
            if not symbol:
                continue

            records.append(
                {
                    "symbol": symbol,
                    "region": q.get("region", region),
                    "exchange": q.get("exchange"),
                    "quoteType": q.get("quoteType"),
                    "shortName": q.get("shortName"),
                    "longName": q.get("longName"),
                    "currency": q.get("currency"),
                }
            )

        offset += len(quotes)

        if len(quotes) < SCREEN_PAGE_SIZE:
            break

        time.sleep(INTER_SCREEN_SLEEP)

    return records


def build_universe() -> pd.DataFrame:
    """
    Enumerate Yahoo equities region-by-region and deduplicate symbols.
    """
    regions = get_regions()

    print(f"Found {len(regions)} Yahoo region codes.")

    all_records = []

    for i, region in enumerate(regions, start=1):
        print(f"\nRegion {i}/{len(regions)}: {region}")

        try:
            records = fetch_region(region)
            all_records.extend(records)

        except Exception as exc:
            print(f"FAILED region {region}: {exc}")

        # Save continuously so a crash does not lose the universe.
        if all_records:
            df = pd.DataFrame(all_records)
            df = df.drop_duplicates(subset=["symbol"])
            df.to_csv(UNIVERSE_FILE, index=False)

    df = pd.DataFrame(all_records)

    if df.empty:
        raise RuntimeError("No equities discovered.")

    # Yahoo's EquityQuery should already be equity-specific,
    # but preserve quoteType for later inspection.
    df = (
        df.drop_duplicates(subset=["symbol"])
        .sort_values("symbol")
        .reset_index(drop=True)
    )

    df.to_csv(UNIVERSE_FILE, index=False)

    print()
    print(f"Unique Yahoo equity symbols: {len(df):,}")

    return df


# ============================================================
# 2. History download
# ============================================================

def load_status() -> pd.DataFrame:
    if STATUS_FILE.exists():
        return pd.read_csv(STATUS_FILE)

    return pd.DataFrame(
        columns=[
            "symbol",
            "status",
            "rows_total",
            "rows_2025",
            "rows_2026",
        ]
    )


def save_status(status: pd.DataFrame) -> None:
    status = (
        status.drop_duplicates(subset=["symbol"], keep="last")
        .sort_values("symbol")
    )
    status.to_csv(STATUS_FILE, index=False)


def split_download(
    data: pd.DataFrame,
    symbols: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Convert yf.download() output into:
        symbol -> DataFrame
    """

    result = {}

    if len(symbols) == 1:
        result[symbols[0]] = data.dropna(how="all")
        return result

    if not isinstance(data.columns, pd.MultiIndex):
        return result

    # yfinance commonly returns:
    #
    # level 0: Price
    # level 1: Ticker
    #
    # e.g.
    # Close AAPL MSFT ...
    ticker_level = None

    for level in range(data.columns.nlevels):
        values = set(
            str(x)
            for x in data.columns.get_level_values(level)
        )

        matches = len(values.intersection(symbols))

        if matches:
            ticker_level = level
            break

    if ticker_level is None:
        return result

    for symbol in symbols:
        try:
            sub = data.xs(
                symbol,
                axis=1,
                level=ticker_level,
                drop_level=True,
            )
            sub = sub.dropna(how="all")

            if not sub.empty:
                result[symbol] = sub

        except KeyError:
            pass

    return result


def download_batch(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """
    Download one ticker batch.

    threads=False is deliberate: it reduces request bursts and makes
    rate-limiting behavior substantially easier to control.
    """

    data = retry_call(
        yf.download,
        tickers=symbols,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        actions=False,
        threads=False,
        progress=False,
        group_by="column",
        timeout=30,
    )

    if data is None or data.empty:
        return {}

    return split_download(data, symbols)


# ============================================================
# Fallback: retry failed batch ticker-by-ticker
# ============================================================

def download_single(symbol: str) -> pd.DataFrame | None:
    try:
        result = download_batch([symbol])
        return result.get(symbol)

    except Exception as exc:
        print(f"    {symbol}: final failure: {exc}")
        return None


# ============================================================
# Save individual ticker
# ============================================================

def safe_filename(symbol: str) -> str:
    return (
        symbol
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def save_ticker(symbol: str, df: pd.DataFrame) -> tuple[int, int, int]:
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    rows_total = len(df)
    rows_2025 = int((df.index.year == 2025).sum())
    rows_2026 = int((df.index.year == 2026).sum())

    path = DATA_DIR / f"{safe_filename(symbol)}.csv"
    df.to_csv(path)

    return rows_total, rows_2025, rows_2026


# ============================================================
# Main history pipeline
# ============================================================

def download_all(universe: pd.DataFrame) -> None:

    status = load_status()

    finished = set(
        status.loc[
            status["status"].isin(
                [
                    "valid",
                    "insufficient",
                    "no_data",
                ]
            ),
            "symbol",
        ].astype(str)
    )

    symbols = [
        s
        for s in universe["symbol"].astype(str)
        if s not in finished
    ]

    print(f"\nRemaining symbols: {len(symbols):,}")

    batches = [
        symbols[i : i + BATCH_SIZE]
        for i in range(0, len(symbols), BATCH_SIZE)
    ]

    for batch_no, batch in enumerate(batches, start=1):

        print(
            f"\n[BATCH {batch_no}/{len(batches)}] "
            f"{len(batch)} symbols"
        )

        try:
            downloaded = download_batch(batch)

        except Exception as exc:
            print(f"Batch failed entirely: {exc}")
            downloaded = {}

        # Any symbols missing from the batch result get one slower,
        # individual attempt.
        missing = [
            symbol
            for symbol in batch
            if symbol not in downloaded
        ]

        if missing:
            print(
                f"  {len(missing)} symbols missing from batch; "
                "retrying individually..."
            )

        for symbol in missing:
            df = download_single(symbol)

            if df is not None and not df.empty:
                downloaded[symbol] = df

            # Slow down individual fallback requests.
            time.sleep(1.0)

        # Process results
        for symbol in batch:

            df = downloaded.get(symbol)

            if df is None or df.empty:
                row = {
                    "symbol": symbol,
                    "status": "no_data",
                    "rows_total": 0,
                    "rows_2025": 0,
                    "rows_2026": 0,
                }

            else:
                rows_total, rows_2025, rows_2026 = save_ticker(
                    symbol,
                    df,
                )

                # Definition #3:
                #
                # at least one observation in BOTH 2025 and 2026
                if rows_2025 > 0 and rows_2026 > 0:
                    result_status = "valid"
                else:
                    result_status = "insufficient"

                row = {
                    "symbol": symbol,
                    "status": result_status,
                    "rows_total": rows_total,
                    "rows_2025": rows_2025,
                    "rows_2026": rows_2026,
                }

            status = pd.concat(
                [status, pd.DataFrame([row])],
                ignore_index=True,
            )

        save_status(status)

        n_valid = (
            status.drop_duplicates("symbol", keep="last")["status"]
            == "valid"
        ).sum()

        print(f"  Valid so far: {n_valid:,}")

        # Deliberately avoid hammering Yahoo between batches.
        time.sleep(INTER_BATCH_SLEEP)


# ============================================================
# Final report
# ============================================================

def make_report() -> None:
    status = pd.read_csv(STATUS_FILE)
    status = status.drop_duplicates("symbol", keep="last")

    total = len(status)
    valid = status["status"].eq("valid").sum()
    insufficient = status["status"].eq("insufficient").sum()
    no_data = status["status"].eq("no_data").sum()

    substantial = (
        (status["rows_2025"] >= 100)
        & (status["rows_2026"] >= 100)
    ).sum()

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)

    print(f"Tested:                  {total:,}")
    print(f"Valid 2025 + 2026:       {valid:,}")
    print(f"100+ obs in both years:  {substantial:,}")
    print(f"Insufficient history:    {insufficient:,}")
    print(f"No usable data:          {no_data:,}")

    valid_df = status[status["status"] == "valid"]
    valid_df.to_csv(
        OUTPUT_DIR / "valid_tickers.csv",
        index=False,
    )

    substantial_df = status[
        (status["rows_2025"] >= 100)
        & (status["rows_2026"] >= 100)
    ]

    substantial_df.to_csv(
        OUTPUT_DIR / "substantial_tickers.csv",
        index=False,
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    if UNIVERSE_FILE.exists():
        print(f"Loading existing universe: {UNIVERSE_FILE}")
        universe = pd.read_csv(UNIVERSE_FILE)
    else:
        universe = build_universe()

    download_all(universe)
    make_report()
