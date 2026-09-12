"""Split the wide option tables using verified per-contract exercise metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv


def load_styles(reference: Path) -> dict[str, str]:
    """Read explicit contract styles; reject conflicts and unsupported labels."""
    if reference.suffix.lower() == ".parquet":
        frame = pd.read_parquet(reference, columns=["ticker", "exercise_style"])
    else:
        frame = pd.read_csv(reference, usecols=["ticker", "exercise_style"], dtype=str).fillna("")
    aliases = {"e": "european", "a": "american", "b": "bermudan",
               "european": "european", "american": "american", "bermudan": "bermudan",
               "unknown": "unknown", "": "unknown"}
    result = {}
    for row in frame.itertuples(index=False):
        if not isinstance(row.ticker, str) or not row.ticker.strip():
            raise ValueError("Reference ticker must be a nonempty string")
        ticker = row.ticker.strip()
        if not ticker.startswith("O:"):
            ticker = "O:"+ticker
        style = "" if pd.isna(row.exercise_style) else str(row.exercise_style).strip().lower()
        if style not in aliases:
            raise ValueError(f"Unrecognized exercise_style for {ticker}: {style}")
        normalized = aliases[style]
        if ticker in result and result[ticker] != normalized:
            raise ValueError(f"Conflicting exercise styles for {ticker}")
        result[ticker] = normalized
    return result


def split_by_style(source: Path, reference: Path, output: Path) -> dict:
    """Preserve price rows and contract columns; unmatched contracts stay unknown."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    styles = load_styles(reference)
    original = json.loads((source/"manifest.json").read_text())
    contracts = pd.read_csv(source/"contracts.csv", keep_default_na=False)
    if contracts.ticker.duplicated().any():
        raise ValueError("Contract index contains duplicate tickers")
    contracts["exercise_style"] = contracts.ticker.map(styles).fillna("unknown")
    del styles
    file_indices = contracts.groupby("file", sort=False).indices
    dates = pd.read_csv(source/"calendar.csv", dtype={"Date": str}).Date.tolist()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary)
        counts = {style: 0 for style in ("american", "european", "bermudan", "unknown")}
        for style in counts:
            (stage/style).mkdir()
        statistics = {}
        filenames = list(original["files"])
        for number, name in enumerate(filenames, 1):
            members = contracts.iloc[file_indices[name]]
            column_types = {ticker: pa.float64() for ticker in members.ticker}
            column_types["Date"] = pa.string()
            table = pacsv.read_csv(source/name,
                convert_options=pacsv.ConvertOptions(column_types=column_types))
            if table["Date"].to_pylist() != dates or set(table.column_names[1:]) != set(members.ticker):
                raise ValueError(f"Input table differs from its calendar/contract index: {name}")
            for style, group in members.groupby("exercise_style", sort=True):
                selected = set(group.ticker)
                columns = ["Date"]+[ticker for ticker in table.column_names[1:] if ticker in selected]
                subset = table.select(columns)
                relative = f"{style}/{name}"
                destination = stage/relative
                destination.parent.mkdir(exist_ok=True)
                if len(columns) == len(table.column_names):
                    shutil.copyfile(source/name, destination)
                    with (source/name).open("rb") as first, destination.open("rb") as second:
                        if hashlib.file_digest(first, "sha256").digest() != hashlib.file_digest(second, "sha256").digest():
                            raise AssertionError(f"Style split changed file bytes: {relative}")
                else:
                    pacsv.write_csv(subset, destination)
                    check = pacsv.read_csv(destination,
                        convert_options=pacsv.ConvertOptions(column_types=subset.schema))
                    if not subset.equals(check):
                        raise AssertionError(f"Style split changed prices: {relative}")
                contracts.loc[group.index, "file"] = relative
                counts[style] += len(group)
                statistics[relative] = {"contracts": len(group), "dates": len(dates),
                    "observed_closes": sum(len(subset[c])-subset[c].null_count for c in columns[1:]),
                    "bytes": destination.stat().st_size, "exercise_style": style}
            if number % 25 == 0 or number == len(filenames):
                print(f"Split and verified {number}/{len(filenames)} input CSVs", flush=True)
        if sum(counts.values()) != original["contracts"]:
            raise AssertionError("Exercise-style split lost contract columns")
        if sum(item["observed_closes"] for item in statistics.values()) != original["observations"]:
            raise AssertionError("Exercise-style split lost observations")
        contracts.to_csv(stage/"contracts.csv", index=False)
        (stage/"calendar.csv").write_bytes((source/"calendar.csv").read_bytes())
        with reference.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest = {**original, "files": statistics, "exercise_style_counts": counts,
            "exercise_style_reference": str(reference.resolve()), "reference_sha256": digest,
            "layout": "Exercise-style folders, expiry/call-put files, stock dates as rows, option tickers as columns",
            "contract_metadata": "Exercise style comes from the supplied reference; unmatched contracts are unknown.",
            "source_export": str(source.resolve())}
        (stage/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        (stage/"README.md").write_text(
            "# Option closes split by verified exercise style\n\n"
            "`american/` and `european/` contain contracts explicitly labelled by the supplied reference. "
            "`bermudan/` retains that separate style when present. `unknown/` contains contracts without "
            "verified style metadata. Empty categories have no price files.\n\n"
            "Every price file retains the exact stock calendar. Blank prices remain missing; no interpolation "
            "or filling is applied. `contracts.csv` maps ticker, style, parameters and file. "
            "`manifest.json` records reference identity and counts.\n")
        stage.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("massive_options_csv_group1_8589"))
    parser.add_argument("--reference", type=Path, required=True,
                        help="CSV or Parquet with ticker and exercise_style columns")
    parser.add_argument("--output", type=Path, default=Path("massive_options_csv_group1_8589_by_style"))
    args = parser.parse_args()
    result = split_by_style(args.source, args.reference, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "contracts_by_style": result["exercise_style_counts"]}))


if __name__ == "__main__":
    main()
