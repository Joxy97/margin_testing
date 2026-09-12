#!/usr/bin/env python3

"""
download_massive_options_flatfiles.py

Download Massive Options daily aggregate Flat Files and retain only
options whose underlying ticker is in our target universe.

Date range:
    2025-01-01 through 2026-09-10

Input:
    tickers.txt

Environment:
    export MASSIVE_ACCESS_KEY="..."
    export MASSIVE_SECRET_KEY="..."

Output:
    massive_options_filtered/
        year=2025/
            month=01/
                2025-01-02.parquet
                ...
        year=2026/
            ...

Requirements:
    pip install boto3 pandas pyarrow tqdm
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import re
import sys
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

START_DATE = date(2025, 1, 1)
END_DATE = date(2026, 9, 10)

TICKER_FILE = Path("yahoo_equities_grouped/groups/US/group_1_8589.txt")

OUTPUT_DIR = Path("massive_options_filtered")

# Massive Flat Files / S3 configuration.
#
# Massive currently exposes flat files through its S3-compatible endpoint.
S3_ENDPOINT = "https://files.massive.com"
S3_BUCKET = "flatfiles"

# Remote dataset path; the local ticker file controls portfolio filtering.
PREFIX = "us_options_opra/day_aggs_v1"

# Parallel file downloads.
#
# Flat-file access is not equivalent to hammering the REST API. Downloading
# several daily objects concurrently substantially improves throughput.
MAX_WORKERS = 16

# Parquet compression.
PARQUET_COMPRESSION = "zstd"

# Skip files already successfully generated.
RESUME = True


# ============================================================
# Tickers
# ============================================================

def load_underlyings(path: Path) -> set[str]:
    """
    Load desired underlying symbols.

    Expected format:
        AAPL
        MSFT
        BRK.B
        ...
    """

    if not path.exists():
        raise FileNotFoundError(f"Ticker file not found: {path}")

    tickers = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ticker = line.strip().upper()

            if not ticker:
                continue

            if ticker.startswith("#"):
                continue

            tickers.add(ticker)

    print(f"Loaded {len(tickers):,} underlying tickers.")

    return tickers


# ============================================================
# Massive flat-file client
# ============================================================

def create_s3_client():

    access_key = os.environ.get("MASSIVE_ACCESS_KEY")
    secret_key = os.environ.get("MASSIVE_SECRET_KEY")

    if not access_key or not secret_key:
        raise RuntimeError(
            "\nMissing Massive Flat Files credentials.\n\n"
            "Set:\n"
            '    export MASSIVE_ACCESS_KEY="..."\n'
            '    export MASSIVE_SECRET_KEY="..."\n'
        )

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(
            signature_version="s3v4",
            retries={
                "max_attempts": 10,
                "mode": "adaptive",
            },
            max_pool_connections=max(32, MAX_WORKERS * 2),
        ),
    )


# ============================================================
# Flat-file discovery
# ============================================================

def parse_date_from_key(key: str) -> date | None:
    """
    Massive aggregate filenames conventionally contain YYYY-MM-DD.

    Extract the date defensively rather than relying on an exact filename
    suffix.
    """

    match = re.search(
        r"(\d{4})-(\d{2})-(\d{2})",
        key,
    )

    if not match:
        return None

    try:
        return date(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )

    except ValueError:
        return None


def list_daily_files(s3) -> list[tuple[date, str, int]]:
    """
    List all Massive day aggregate objects intersecting our date range.

    Return:
        [(date, S3 key, compressed bytes), ...]
    """

    paginator = s3.get_paginator("list_objects_v2")

    files = []

    for year in tqdm(
        range(START_DATE.year, END_DATE.year + 1),
        desc="Discovering files",
        unit="year",
    ):

        prefix = f"{PREFIX}/{year}/"

        for page in paginator.paginate(
            Bucket=S3_BUCKET,
            Prefix=prefix,
        ):
            for obj in page.get("Contents", []):

                key = obj["Key"]

                d = parse_date_from_key(key)

                if d is None:
                    continue

                if START_DATE <= d <= END_DATE:
                    files.append(
                        (
                            d,
                            key,
                            obj.get("Size", 0),
                        )
                    )

    files.sort(key=lambda x: x[0])

    return files


# ============================================================
# Option symbol parsing
# ============================================================

#
# OCC-style option symbol:
#
#     AAPL250117C00150000
#
# underlying = AAPL
# expiry     = 2025-01-17
# type       = C
# strike     = 150.000
#
# Massive frequently prepends:
#
#     O:AAPL250117C00150000
#

OCC_RE = re.compile(
    r"^(?:O:)?"
    r"(?P<underlying>.+?)"
    r"(?P<expiry>\d{6})"
    r"(?P<option_type>[CP])"
    r"(?P<strike>\d{8})$"
)


def parse_option_symbol(symbol: str):
    """
    Return

        underlying,
        expiration_date,
        option_type,
        strike

    or None if the symbol does not match OCC format.
    """

    if not isinstance(symbol, str):
        return None

    match = OCC_RE.match(symbol)

    if not match:
        return None

    underlying = match.group("underlying")

    expiry_raw = match.group("expiry")

    expiration = datetime.strptime(
        expiry_raw,
        "%y%m%d",
    ).date()

    option_type = (
        "call"
        if match.group("option_type") == "C"
        else "put"
    )

    strike = int(match.group("strike")) / 1000.0

    return (
        underlying,
        expiration,
        option_type,
        strike,
    )


def extract_underlying_fast(symbol: str) -> str | None:
    """
    Faster parser when we only need the underlying.

    OCC suffix has:

        YYMMDD  = 6
        C/P     = 1
        strike  = 8

    Therefore the final 15 characters aren't part of the underlying.
    """

    if not isinstance(symbol, str):
        return None

    if symbol.startswith("O:"):
        symbol = symbol[2:]

    if len(symbol) <= 15:
        return None

    return symbol[:-15]


# ============================================================
# Reading Massive CSV
# ============================================================

def read_massive_csv(raw_bytes: bytes) -> pd.DataFrame:

    #
    # Most Massive flat files are gzip-compressed CSVs.
    #
    # pandas can infer gzip if given a filename but we're reading directly
    # from memory, so decompress explicitly.
    #

    try:
        raw_bytes = gzip.decompress(raw_bytes)

    except (gzip.BadGzipFile, OSError):
        # It might already be an uncompressed CSV.
        pass

    return pd.read_csv(
        io.BytesIO(raw_bytes),
        low_memory=False,
    )


# ============================================================
# Column detection
# ============================================================

def find_symbol_column(df: pd.DataFrame) -> str:

    candidates = [
        "ticker",
        "symbol",
        "option_ticker",
    ]

    lower_to_original = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    raise RuntimeError(
        "Couldn't identify the option symbol column.\n"
        f"Columns returned by Massive:\n{list(df.columns)}"
    )


# ============================================================
# One-day processing
# ============================================================

def output_path_for_date(d: date) -> Path:

    return (
        OUTPUT_DIR
        / f"year={d.year:04d}"
        / f"month={d.month:02d}"
        / f"{d.isoformat()}.parquet"
    )


def process_file(
    s3,
    d: date,
    key: str,
    underlyings: set[str],
    on_download: Callable[[int], object] | None = None,
) -> dict:

    output_path = output_path_for_date(d)

    if RESUME and output_path.exists():

        return {
            "date": d,
            "status": "skipped",
            "input_rows": None,
            "output_rows": None,
            "bytes": None,
        }

    #
    # Download object into memory.
    #
    response = s3.get_object(
        Bucket=S3_BUCKET,
        Key=key,
    )

    body = response["Body"]
    try:
        with io.BytesIO() as buffer:
            for chunk in body.iter_chunks(chunk_size=1024 * 1024):
                buffer.write(chunk)
                if on_download is not None:
                    on_download(len(chunk))
            raw = buffer.getvalue()
    finally:
        body.close()

    compressed_bytes = len(raw)

    df = read_massive_csv(raw)

    if df.empty:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_parquet(
            output_path,
            index=False,
            compression=PARQUET_COMPRESSION,
        )

        return {
            "date": d,
            "status": "empty",
            "input_rows": 0,
            "output_rows": 0,
            "bytes": compressed_bytes,
        }

    ticker_col = find_symbol_column(df)

    #
    # Recover underlying from OCC symbol.
    #
    under = df[ticker_col].map(
        extract_underlying_fast
    )

    mask = under.isin(underlyings)

    filtered = df.loc[mask].copy()

    #
    # Add useful decoded contract metadata.
    #
    if not filtered.empty:

        parsed = filtered[ticker_col].map(
            parse_option_symbol
        )

        filtered["underlying_ticker"] = [
            x[0] if x else None
            for x in parsed
        ]

        filtered["expiration_date"] = [
            x[1] if x else None
            for x in parsed
        ]

        filtered["option_type"] = [
            x[2] if x else None
            for x in parsed
        ]

        filtered["strike_price"] = [
            x[3] if x else None
            for x in parsed
        ]

        #
        # The file itself corresponds to trading date d.
        #
        filtered["trading_date"] = d.isoformat()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    #
    # Atomic output:
    #
    # Don't create the final filename until writing succeeds.
    #
    temp_path = output_path.with_suffix(
        ".parquet.tmp"
    )

    filtered.to_parquet(
        temp_path,
        index=False,
        compression=PARQUET_COMPRESSION,
    )

    temp_path.replace(output_path)

    return {
        "date": d,
        "status": "ok",
        "input_rows": len(df),
        "output_rows": len(filtered),
        "bytes": compressed_bytes,
    }


# ============================================================
# Main download
# ============================================================

def download_all():

    underlyings = load_underlyings(
        TICKER_FILE
    )

    s3 = create_s3_client()

    print(
        "\nLooking for Massive options daily aggregate files..."
    )

    files = list_daily_files(s3)

    if not files:
        raise RuntimeError(
            "No Massive flat files found in the requested date range."
        )

    total_bytes = sum(
        size
        for _, _, size in files
    )

    print()
    print(f"Trading-day files : {len(files):,}")
    print(
        "Compressed input  : "
        f"{total_bytes / 1024**3:.2f} GiB"
    )
    print(f"Date range        : {files[0][0]} -> {files[-1][0]}")
    print(f"Workers           : {MAX_WORKERS}")
    print()

    total_input_rows = 0
    total_output_rows = 0
    total_downloaded_bytes = 0
    skipped = 0
    failed = 0
    pending_bytes = sum(
        size for d, _, size in files
        if not (RESUME and output_path_for_date(d).exists())
    )

    #
    # boto3 clients are thread-safe after construction.
    #
    with tqdm(
        total=pending_bytes,
        unit="B",
        unit_scale=True,
        desc="Downloading",
        position=1,
    ) as download_progress, ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {}

        for d, key, size in files:

            future = executor.submit(
                process_file,
                s3,
                d,
                key,
                underlyings,
                download_progress.update,
            )

            futures[future] = d

        with tqdm(
            total=len(futures),
            unit="day",
            desc="Processing",
            position=0,
        ) as progress:

            for future in as_completed(futures):

                d = futures[future]

                try:
                    result = future.result()

                except Exception as exc:

                    failed += 1
                    tqdm.write(
                        f"FAILED {d}: {exc}",
                        file=sys.stderr,
                    )

                    progress.set_postfix(
                        kept=f"{total_output_rows:,}", skipped=skipped, failed=failed,
                    )
                    progress.update(1)
                    continue

                if result["status"] == "skipped":
                    skipped += 1

                else:

                    total_input_rows += (
                        result["input_rows"] or 0
                    )

                    total_output_rows += (
                        result["output_rows"] or 0
                    )

                    total_downloaded_bytes += (
                        result["bytes"] or 0
                    )

                progress.set_postfix(
                    kept=f"{total_output_rows:,}",
                    skipped=skipped,
                    failed=failed,
                )

                progress.update(1)

    print()
    print("Done." if not failed else "Finished with failures.")
    print()
    print(
        f"Input rows processed : {total_input_rows:,}"
    )
    print(
        f"Rows retained        : {total_output_rows:,}"
    )
    print(
        f"Downloaded           : "
        f"{total_downloaded_bytes / 1024**3:.2f} GiB"
    )
    print(
        f"Files skipped        : {skipped:,}"
    )
    print(f"Files failed         : {failed:,}")
    print(
        f"Output directory     : {OUTPUT_DIR.resolve()}"
    )


# ============================================================
# Inspection utility
# ============================================================

def inspect():

    s3 = create_s3_client()

    files = list_daily_files(s3)

    total = sum(
        size
        for _, _, size in files
    )

    print(f"Files: {len(files):,}")

    if files:
        print(f"First: {files[0][0]}")
        print(f"Last : {files[-1][0]}")

    print(
        f"Compressed size: "
        f"{total / 1024**3:.3f} GiB"
    )


# ============================================================
# CLI
# ============================================================

def main():
    global MAX_WORKERS

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["download", "inspect"],
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
    )

    args = parser.parse_args()

    MAX_WORKERS = args.workers

    if args.command == "inspect":
        inspect()
    else:
        download_all()


if __name__ == "__main__":
    main()
