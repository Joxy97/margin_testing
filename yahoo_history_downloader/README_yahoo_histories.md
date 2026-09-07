# Real Yahoo histories for unchanged numeric portfolios

## Run (Windows, macOS, Linux)

Unzip this package into a folder. The two original portfolio CSVs are included byte-for-byte.
Use Python 3.11 or newer. Open a terminal in that folder:

```sh
python -m pip install "yfinance>=0.2.65,<2" "pandas>=2,<3" tzdata
python build_yahoo_histories.py --portfolio-200 portfolio.csv --portfolio-1000 "portfolio(1).csv"
```

On systems where Python is named `python3`, substitute that command. Downloading and
checking 1000+ securities may take hours. A 1.5-second pause is applied before each
request; rate limits remain possible. No credentials or API key are required.

The script never writes to the input portfolios, and checks their SHA-256 hashes
before completion. Their IDs, row order, client IDs and weight strings remain intact.

## Output

Inside `yahoo_output/`:

- `historical_close_200.csv`: `date` followed by the 200 original numeric IDs, sorted numerically.
- `historical_close_1000.csv`: `date` followed by the 1000 original numeric IDs, sorted numerically.
- `historical_adjusted_close_200.csv` and `historical_adjusted_close_1000.csv`: the same structure, using Yahoo Adj Close.
- `ticker_mapping_200.csv` and `ticker_mapping_1000.csv`: numeric ID, Yahoo symbol, company name, unchanged weight, market cap, snapshot timestamp, coverage and source URL.
- `manifest.json`: completion flag, input/output checksums, dates, versions, exclusions and caveats.

The history layout is exactly a wide table: one actual trading date per row and one
numeric ticker column per stock. Markdown `**` markers from the pasted example are
not included. Prices are downloaded, not generated. No weights or portfolio columns
are inserted into the history CSVs.

Use outputs only when `manifest.json` exists with `status: complete`. A failed run
may leave intermediate files, which are not a completed dataset.

## Selection and mapping

1. Collect a Yahoo US-region equity screener snapshot, ordered by market capitalization.
2. Keep USD EQUITY records; ETFs such as SPY are not portfolio candidates.
3. Check candidate Close and Adj Close against the SPY trading-date calendar for the requested window.
4. Reject any security with missing, nonpositive or nonfinite prices. No interpolation,
   forward fill, zero padding, pre-IPO backfilling or fabricated history is performed.
5. Keep the first 1000 eligible securities by recorded market capitalization; the 200-stock
   universe is the first 200 of those. Ties use Yahoo symbol order.
6. Independently for each portfolio, sort its existing IDs by descending numerical
   weight (tie: ascending numeric ID), then assign descending-cap stocks.

Mappings are **portfolio-specific**. An ID such as `17` may mean different companies
in the two datasets because its weights/ranks can differ. Always load the mapping
associated with the corresponding portfolio/history. A single global mapping would
not generally satisfy both weight-ranking constraints.

These are securities, not necessarily distinct issuers: GOOG/GOOGL, other share
classes and ADRs may coexist. The script does not guarantee the world's largest
companies; it ranks eligible securities in the returned USD US-region universe.

## Dates and price interpretation

The requested range defaults to 2020-01-01 through 2026-12-31 inclusive. The end is
capped at the day before execution in New York, excluding a possibly incomplete
current session. Thus a run in September 2026 cannot obtain December 2026 prices.
January 1 and other exchange holidays do not appear as rows. Use `--end 2026-09-03`
for an explicit fixed inclusive cutoff. The exact first/last dates appear in the manifest.

`Close` is Yahoo's Close with yfinance automatic adjustment disabled, not a promise
of the original tape price before every historical split. Yahoo commonly reports
split-adjusted history. `Adj Close` additionally reflects provider distribution
adjustments and is normally the useful series for equity-return risk modelling.
Do not mix Close and Adj Close when calculating a return series. Yahoo histories
can be revised by the provider; save the cached data and manifest for reproducibility.

Using current surviving companies and current caps to populate a 2020–2026 history
introduces survivorship/look-ahead bias. These datasets support today's synthetic
portfolio/risk experiments, not unbiased historical investment-performance claims.

## Resume and failure behaviour

If Yahoo rate-limits requests, the script stops rather than repeatedly hammering
the service or silently omitting high-cap candidates. Wait and rerun the same
command with the same output directory. Successfully downloaded histories and
the universe snapshot are reused. The script does not circumvent Yahoo restrictions.
Network/provider errors stop the run; inspect the error instead of treating such
an error as proof that a stock has no history.

If there are insufficient eligible candidates, start in a new directory with a
larger `--max-candidates` (default 6000), or provide a larger Yahoo-sourced candidate
CSV via `--candidate-file candidates.csv` with these exact columns:

```text
symbol,name,market_cap_usd
```

This optional input must contain real, positive USD market caps from one documented
snapshot. All price data still come from Yahoo. No candidate list is fabricated.
When supplying it, retain the source/timestamp separately for audit.

Completed output directories are not overwritten: choose `--out yahoo_output_new`
for a fresh universe/mapping snapshot. Do not use the cache from a different date
window. SPY determines the common US trading calendar; international holidays
are deliberately not filled in.

## Verification performed for this delivery

The supplied script's syntax and offline mapping/input-preservation logic are tested.
Live end-to-end downloads have NOT been verified here because Yahoo returned HTTP 429.
No real history CSV is claimed to have been downloaded in this package.
Yahoo/yfinance API changes or access restrictions may require adjustments on your machine.

```sh
python -m unittest -v test_yahoo_histories.py
```

References: https://ranaroussi.github.io/yfinance/ and
https://finance.yahoo.com/research-hub/screener/equity/
