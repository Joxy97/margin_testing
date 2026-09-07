#!/usr/bin/env python3
"""Download real Yahoo daily closes; never modify input portfolios.
Install: python -m pip install 'yfinance>=0.2.65,<2' 'pandas>=2,<3'
Run: python build_yahoo_histories.py --portfolio-200 portfolio.csv --portfolio-1000 'portfolio(1).csv'
"""
import argparse
import csv
import hashlib
import json
import math
import time
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_portfolio(path, count):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['client_id', 'ticker', 'weight']:
            raise ValueError(f'{path}: expected client_id,ticker,weight')
        rows = list(reader)
    if len(rows) != count or len({r['ticker'] for r in rows}) != count:
        raise ValueError(f'{path}: expected {count} unique holdings')
    if any(not r['ticker'].isdigit() for r in rows):
        raise ValueError('Ticker IDs must be numeric strings')
    if any(not Decimal(r['weight']).is_finite() for r in rows):
        raise ValueError('Nonfinite weight')
    return rows


def assignments(rows, stocks):
    """Descending numerical weight, deterministic numeric-ID tie break."""
    ranked = sorted(rows, key=lambda r: (-Decimal(r['weight']), int(r['ticker'])))
    return [(row, stock) for row, stock in zip(ranked, stocks, strict=True)]


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
    temp.replace(path)


def rate_limited(exc):
    return any(x in str(exc).lower() for x in ['429', 'rate limit', 'too many requests'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--portfolio-200', type=Path, default=Path('portfolio.csv'))
    parser.add_argument('--portfolio-1000', type=Path, default=Path('portfolio(1).csv'))
    parser.add_argument('--start', default='2020-01-01')
    parser.add_argument('--end', default='2026-12-31', help='Inclusive requested end; future/current partial days excluded')
    parser.add_argument('--out', type=Path, default=Path('yahoo_output'))
    parser.add_argument('--pause', type=float, default=1.5, help='Seconds between Yahoo calls')
    parser.add_argument('--max-candidates', type=int, default=6000)
    parser.add_argument('--candidate-file', type=Path, help='Optional Yahoo-sourced snapshot: symbol,name,market_cap_usd')
    args = parser.parse_args()
    import pandas as pd
    import yfinance as yf

    paths = [args.portfolio_200.resolve(), args.portfolio_1000.resolve()]
    portfolios = [read_portfolio(paths[0], 200), read_portfolio(paths[1], 1000)]
    original_hashes = {str(p): digest(p) for p in paths}
    start = date.fromisoformat(args.start)
    # Avoid an incomplete current trading session, even when the market is currently open.
    exclusive_end = min(date.fromisoformat(args.end) + timedelta(days=1),
                        datetime.now(ZoneInfo('America/New_York')).date())
    if start >= exclusive_end:
        raise ValueError('Empty requested date interval')
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / f'cache_{start}_{exclusive_end}'
    cache.mkdir(exist_ok=True)
    manifest_file = args.out / 'manifest.json'
    if manifest_file.exists():
        raise RuntimeError('Completed output exists. Use a new --out directory; mappings will not be silently reassigned.')

    def call(fn):
        time.sleep(max(0, args.pause))
        try:
            return fn()
        except Exception as exc:
            if rate_limited(exc):
                raise RuntimeError('Yahoo rate limit. Stop and rerun later with the same --out to resume cache.') from exc
            raise

    def history(symbol):
        key = hashlib.sha256(symbol.encode()).hexdigest()[:20]
        path = cache / f'{key}.csv'
        if path.exists():
            h = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            h = call(lambda: yf.Ticker(symbol).history(
                start=str(start), end=str(exclusive_end), interval='1d',
                auto_adjust=False, back_adjust=False, actions=False,
                repair=False, raise_errors=True))
            if h.empty:
                raise ValueError('No history returned')
            h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
            h = h[['Close', 'Adj Close']].sort_index()
            if h.index.has_duplicates:
                raise ValueError('Duplicate history dates')
            temp = path.with_suffix('.tmp')
            h.to_csv(temp, index_label='date', date_format='%Y-%m-%d')
            temp.replace(path)
        return h

    benchmark = history('SPY')
    calendar = benchmark.index
    if len(calendar) < 2 or calendar.min().date() > start + timedelta(days=7):
        raise RuntimeError('Benchmark does not cover requested start')
    if calendar.max().date() < exclusive_end - timedelta(days=7):
        raise RuntimeError('Benchmark history is stale or incomplete at end')

    snapshot_path = cache / 'universe.json'
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
    else:
        if args.candidate_file:
            with args.candidate_file.open(newline='', encoding='utf-8-sig') as f:
                stocks = [dict(symbol=r['symbol'], name=r['name'],
                               market_cap_usd=float(r['market_cap_usd'])) for r in csv.DictReader(f)]
            source = str(args.candidate_file.resolve())
        else:
            stocks = []
            query = yf.EquityQuery('eq', ['region', 'us'])
            for offset in range(0, args.max_candidates, 250):
                page = call(lambda offset=offset: yf.screen(query, offset=offset, size=250,
                    sortField='intradaymarketcap', sortAsc=False))
                quotes = page.get('quotes', [])
                if not quotes:
                    break
                for q in quotes:
                    cap = q.get('marketCap')
                    if (q.get('quoteType') == 'EQUITY' and q.get('currency') == 'USD'
                            and q.get('symbol') and cap and math.isfinite(float(cap)) and float(cap) > 0):
                        stocks.append(dict(symbol=q['symbol'], name=q.get('longName', q.get('shortName', q['symbol'])),
                                           market_cap_usd=float(cap)))
                print(f'Universe: {len(stocks)} candidate USD equities collected', flush=True)
                if offset + len(quotes) >= page.get('total', 0):
                    break
            source = 'Yahoo Finance equity screener; region=us; quoteType=EQUITY; currency=USD'
        dedup = {s['symbol']: s for s in stocks}
        stocks = sorted(dedup.values(), key=lambda s: (-s['market_cap_usd'], s['symbol']))
        if len(stocks) < 1000:
            raise RuntimeError(f'Only {len(stocks)} candidates. Need >1000, including replacements for incomplete histories.')
        snapshot = dict(retrieved_at=datetime.now().astimezone().isoformat(), source=source, stocks=stocks)
        write_json(snapshot_path, snapshot)

    accepted = []
    rejected = []
    histories = {}
    for stock in snapshot['stocks']:
        symbol = stock['symbol']
        try:
            h = history(symbol).reindex(calendar)
            if h[['Close', 'Adj Close']].isna().any().any():
                raise ValueError('Missing a required trading date or price field; no filling allowed')
            if not ((h[['Close', 'Adj Close']] > 0) & (h[['Close', 'Adj Close']] < float('inf'))).all().all():
                raise ValueError('Nonpositive or nonfinite price')
        except Exception as exc:
            if rate_limited(exc):
                raise
            # Network/provider failures must not silently exclude a high-cap candidate.
            if not isinstance(exc, ValueError):
                raise RuntimeError(f'{symbol}: provider error; rerun later to resume. {exc}') from exc
            rejected.append(dict(symbol=symbol, reason=str(exc)))
            write_json(cache / 'rejections.json', rejected)
            print(f'Reject {symbol}: {exc}', flush=True)
            continue
        accepted.append(stock)
        histories[symbol] = h
        print(f'Accepted {len(accepted)}/1000: {symbol}', flush=True)
        if len(accepted) == 1000:
            break
    if len(accepted) < 1000:
        raise RuntimeError(f'Only {len(accepted)} complete histories; increase --max-candidates in a new output directory or supply --candidate-file.')

    files = []
    for count, rows in zip([200, 1000], portfolios):
        pairs = assignments(rows, accepted[:count])
        for field, prefix in [('Close', 'historical_close'), ('Adj Close', 'historical_adjusted_close')]:
            wide = pd.DataFrame({r['ticker']: histories[s['symbol']][field] for r, s in pairs}, index=calendar)
            wide = wide[sorted(wide.columns, key=int)]
            target = args.out / f'{prefix}_{count}.csv'
            wide.to_csv(target, index_label='date', date_format='%Y-%m-%d', float_format='%.10g')
            files.append(target)
        mapping = args.out / f'ticker_mapping_{count}.csv'
        with mapping.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['ticker','yahoo_ticker','company_name','weight',
                'market_cap_usd','market_cap_snapshot','first_date','last_date','observations','source_url'])
            writer.writeheader()
            for r, s in sorted(pairs, key=lambda pair: int(pair[0]['ticker'])):
                writer.writerow(dict(ticker=r['ticker'], yahoo_ticker=s['symbol'], company_name=s['name'],
                    weight=r['weight'], market_cap_usd=s['market_cap_usd'], market_cap_snapshot=snapshot['retrieved_at'],
                    first_date=str(calendar.min().date()), last_date=str(calendar.max().date()),
                    observations=len(calendar), source_url=f'https://finance.yahoo.com/quote/{s["symbol"]}/history/'))
        files.append(mapping)
    assert all(digest(p) == original_hashes[str(p)] for p in paths), 'Input file changed!'
    write_json(manifest_file, dict(status='complete', input_sha256=original_hashes,
        start=str(start), end_exclusive=str(exclusive_end), first_date=str(calendar.min().date()),
        last_date=str(calendar.max().date()), rows=len(calendar), yfinance_version=yf.__version__,
        universe_source=snapshot['source'], ranking_snapshot=snapshot['retrieved_at'],
        selection='Top 1000 complete-history USD US-screened equities by snapshot market cap; top 200 subset',
        mapping_scope='Separate per portfolio; same numeric ID can map to different equities',
        caveats=['Current-universe selection has survivorship and look-ahead bias; not a historical backtest universe.',
                 'Yahoo Close follows provider split-adjustment conventions; Adj Close also reflects distributions.',
                 'Share classes and ADRs may coexist; 1000 securities does not guarantee 1000 unique issuers.',
                 'Source pages fetched sequentially; market caps are a retrieval snapshot, not simultaneous ticks.'],
        outputs={p.name: digest(p) for p in files}, rejected=rejected))
    print(f'Done: {args.out.resolve()}. Input portfolios unchanged.')


if __name__ == '__main__':
    main()
