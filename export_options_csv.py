"""Build wide option-close CSVs with exactly the Group 1 stock date rows."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


def export_options(source: Path, stocks: Path, output: Path, *, max_columns: int = 2000) -> dict:
    """Partition by expiry/type, pivot closes, and round-trip verify every cell."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if isinstance(max_columns, bool) or not isinstance(max_columns, int) or max_columns < 1:
        raise ValueError("max_columns must be a positive integer")
    with stocks.open(newline="") as stream:
        raw_header = next(csv.reader(stream))
    if len(raw_header) < 2 or len(set(raw_header)) != len(raw_header):
        raise ValueError("Stock CSV must have a date column and unique instrument columns")
    dates = pd.read_csv(stocks, usecols=[raw_header[0]]).iloc[:, 0].astype(str).tolist()
    parsed = pd.to_datetime(dates, format="%Y-%m-%d", errors="raise")
    if not dates or parsed.has_duplicates or not parsed.is_monotonic_increasing:
        raise ValueError("Stock dates must be nonempty, unique and increasing")
    if any(value != timestamp.strftime("%Y-%m-%d") for value, timestamp in zip(dates, parsed)):
        raise ValueError("Stock dates must use YYYY-MM-DD")
    universe = pa.array(raw_header[1:], type=pa.large_string())
    inputs = {}
    for path in source.rglob("*.parquet"):
        if path.stem in inputs:
            raise ValueError(f"Duplicate source day: {path.stem}")
        inputs[path.stem] = path
    missing = sorted(set(dates)-inputs.keys())
    if missing:
        raise ValueError(f"Missing downloaded option dates: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary)
        result_dir = stage/"result"
        result_dir.mkdir()
        groups_dir = stage/"groups"
        groups_dir.mkdir()
        writers, calendar, snapshots = {}, [], []
        total_rows = 0
        with ExitStack() as stack:
            for day_index, day in enumerate(dates):
                path = inputs[day]
                before = path.stat()
                required = ["ticker", "underlying_ticker", "expiration_date", "option_type",
                            "strike_price", "close", "trading_date"]
                table = pq.ParquetFile(path).read(columns=required)
                if table.column_names != required or any(table[name].null_count for name in required):
                    raise ValueError(f"Missing/null option fields: {path}")
                if len(table) and not pc.all(pc.equal(table["trading_date"], day)).as_py():
                    raise ValueError(f"Source trading_date disagrees with filename: {path}")
                input_rows = len(table)
                table = table.filter(pc.is_in(table["underlying_ticker"], value_set=universe))
                kinds = set(pc.unique(table["option_type"]).to_pylist())
                if not kinds.issubset({"call", "put"}):
                    raise ValueError(f"Unexpected option_type in {path}: {kinds}")
                if len(pc.unique(table["ticker"])) != len(table):
                    raise ValueError(f"Duplicate contract/date observations: {path}")
                for name in ("strike_price", "close"):
                    if len(table) and not pc.all(pc.is_finite(table[name])).as_py():
                        raise ValueError(f"Nonfinite {name}: {path}")
                if pc.any(pc.less_equal(table["strike_price"], 0)).as_py():
                    raise ValueError(f"Nonpositive strike: {path}")
                if pc.any(pc.less(table["close"], 0)).as_py():
                    raise ValueError(f"Negative close: {path}")
                frame = table.to_pandas()
                frame["date_index"] = np.int32(day_index)
                for (expiry, kind), part in frame.groupby(["expiration_date", "option_type"], sort=True):
                    group = f"{expiry.isoformat()}_{kind}s"
                    compact = pa.Table.from_pandas(part[["date_index", "ticker", "close",
                        "underlying_ticker", "strike_price"]], preserve_index=False)
                    if group not in writers:
                        writers[group] = stack.enter_context(pq.ParquetWriter(groups_dir/f"{group}.parquet",
                            compact.schema, compression="zstd"))
                    writers[group].write_table(compact)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"Downloaded file changed during export: {path}")
                snapshots.append({"date": day, "path": str(path.resolve()),
                                  "bytes": before.st_size, "mtime_ns": before.st_mtime_ns})
                calendar.append({"Date": day, "call_observations": int((frame.option_type == "call").sum()),
                    "put_observations": int((frame.option_type == "put").sum()),
                    "excluded_underlying_rows": input_rows-len(table)})
                total_rows += len(table)
                if (day_index+1) % 20 == 0 or day_index+1 == len(dates):
                    print(f"Read {day_index+1}/{len(dates)} dates; {total_rows:,} observations", flush=True)
        statistics = {}
        metadata_columns = ["ticker", "file", "underlying_ticker", "expiration_date", "option_type",
                            "strike_price", "first_observed_date", "last_observed_date", "observed_dates"]
        with (result_dir/"contracts.csv").open("w", newline="") as stream:
            metadata_writer = csv.DictWriter(stream, fieldnames=metadata_columns)
            metadata_writer.writeheader()
            paths = sorted(groups_dir.glob("*.parquet"))
            for group_number, path in enumerate(paths, 1):
                frame = pq.read_table(path).to_pandas()
                codes, tickers = pd.factorize(frame.ticker, sort=True)
                rows = frame.date_index.to_numpy()
                values = frame.close.to_numpy(dtype=np.float64)
                metadata = frame[["ticker", "underlying_ticker", "strike_price"]].drop_duplicates()
                if metadata.ticker.duplicated().any():
                    raise ValueError(f"Inconsistent contract metadata: {path.stem}")
                metadata = metadata.set_index("ticker")
                observed = frame.groupby("ticker").date_index.agg(["min", "max", "count"])
                expiry, kind = path.stem.split("_")
                parts = (len(tickers)+max_columns-1)//max_columns
                for number, start in enumerate(range(0, len(tickers), max_columns), 1):
                    stop = min(start+max_columns, len(tickers))
                    selected = (codes >= start) & (codes < stop)
                    matrix = np.full((len(dates), stop-start), np.nan)
                    matrix[rows[selected], codes[selected]-start] = values[selected]
                    if np.count_nonzero(~np.isnan(matrix)) != np.count_nonzero(selected):
                        raise ValueError(f"Duplicate contract/date in partition: {path.stem}")
                    columns = tickers[start:stop].tolist()
                    wide = pd.DataFrame(matrix, columns=columns)
                    wide.insert(0, "Date", dates)
                    csv_table = pa.Table.from_pandas(wide, preserve_index=False)
                    suffix = f"_part{number:03d}" if parts > 1 else ""
                    name = f"{path.stem}{suffix}.csv"
                    destination = result_dir/name
                    pacsv.write_csv(csv_table, destination)
                    check = pacsv.read_csv(destination,
                        convert_options=pacsv.ConvertOptions(column_types=csv_table.schema))
                    if not csv_table.equals(check):
                        raise AssertionError(f"CSV round-trip differs from source prices: {name}")
                    statistics[name] = {"contracts": len(columns), "dates": len(dates),
                        "observed_closes": int(np.count_nonzero(selected)), "bytes": destination.stat().st_size}
                    for ticker in columns:
                        item = metadata.loc[ticker]
                        coverage = observed.loc[ticker]
                        metadata_writer.writerow({"ticker": ticker, "file": name,
                            "underlying_ticker": item.underlying_ticker, "expiration_date": expiry,
                            "option_type": kind[:-1], "strike_price": item.strike_price,
                            "first_observed_date": dates[int(coverage["min"])],
                            "last_observed_date": dates[int(coverage["max"])],
                            "observed_dates": int(coverage["count"])})
                path.unlink()
                if group_number % 10 == 0 or group_number == len(paths):
                    print(f"Wrote and verified {group_number}/{len(paths)} expiry/type groups; "
                          f"{len(statistics)} wide CSVs", flush=True)
        if sum(info["observed_closes"] for info in statistics.values()) != total_rows:
            raise AssertionError("Export observation count differs from retained source count")
        with (result_dir/"calendar.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(calendar[0]))
            writer.writeheader()
            writer.writerows(calendar)
        with stocks.open("rb") as stream:
            stock_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest = {"stock_csv": str(stocks.resolve()), "stock_csv_sha256": stock_digest,
            "date_count": len(dates), "first_date": dates[0], "last_date": dates[-1],
            "stock_instrument_count": len(raw_header)-1,
            "excluded_source_dates": sorted(inputs.keys()-set(dates)),
            "observations": total_rows, "contracts": sum(info["contracts"] for info in statistics.values()),
            "layout": "Date rows, option ticker columns, close prices; partitioned by expiry and call/put",
            "max_contract_columns": max_columns, "files": statistics, "sources": snapshots,
            "verification": "Every CSV cell round-tripped against its source pivot; retained observation count reconciled.",
            "missing_observations": "Blank; no filling or interpolation, including before listing and after expiry.",
            "contract_metadata": "Exercise style, multiplier and currency are absent from the source; none inferred.",
            "price_units": "Original quoted option close units, without contract multipliers."}
        (result_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        (result_dir/"README.md").write_text(
            "# Group 1 option close prices\n\n"
            f"Every price CSV has exactly {len(dates)} stock-calendar rows ({dates[0]} through {dates[-1]}). "
            "The first column is `Date`; remaining columns are full option tickers and contain quoted close prices. "
            "Files are grouped by expiry and calls/puts, with large groups split into numbered parts.\n\n"
            "Blank cells mean no observed close on that date, including before listing and after expiry. "
            "No values are filled or interpolated. The source's later dates are excluded.\n\n"
            "`contracts.csv` maps each option column to its file, underlying, strike, expiry and type. "
            "`calendar.csv` records the exact stock dates and source observation counts. "
            "`manifest.json` records source identity and verification. Exercise style, multiplier and "
            "currency are unavailable in these aggregate files and are not inferred.\n")
        result_dir.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("massive_options_filtered"))
    parser.add_argument("--stocks", type=Path, default=Path("yahoo_equities_grouped/groups/US/group_1_8589.csv"))
    parser.add_argument("--output", type=Path, default=Path("massive_options_csv_group1_8589"))
    parser.add_argument("--max-columns", type=int, default=2000)
    args = parser.parse_args()
    result = export_options(args.source, args.stocks, args.output, max_columns=args.max_columns)
    print(json.dumps({"output": str(args.output.resolve()), "observations": result["observations"],
                      "contracts": result["contracts"], "dates": result["date_count"],
                      "files": len(result["files"])}), flush=True)


if __name__ == "__main__":
    main()
