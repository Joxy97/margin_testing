"""Fetch and cache Massive reference metadata for the exported option contracts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import pandas as pd


# Reuse certificate/context preparation across requests and keep aggregate
# request starts below Massive's published 100/second recommendation.
_TLS_CONTEXT = ssl.create_default_context()
_REQUEST_LOCK = threading.Lock()
_NEXT_REQUEST = 0.0


def pace_request() -> None:
    global _NEXT_REQUEST
    with _REQUEST_LOCK:
        now = time.monotonic()
        wait = max(0., _NEXT_REQUEST-now)
        _NEXT_REQUEST = max(now, _NEXT_REQUEST)+.025
    if wait:
        time.sleep(wait)


def safe_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in {"api.massive.com", "api.polygon.io"}:
        raise ValueError("Unexpected contract-reference pagination host")
    query = [(k, v) for k, v in parse_qsl(parts.query) if k.lower() != "apikey"]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def read_api_key(path: Path) -> str:
    """Accept a raw key or extract apiKey from saved curl text, without executing it."""
    value = path.read_text().strip()
    match = re.search(r'''(?i)[?&]apiKey=([^&\s"']+)''', value)
    if match:
        value = unquote(match.group(1))
    if not value or any(c.isspace() for c in value):
        raise ValueError("Key file must contain a REST key or a curl URL with an apiKey parameter")
    return value


def request_page(url: str, key: str) -> dict:
    for attempt in range(12):
        try:
            pace_request()
            request = Request(safe_url(url), headers={"Authorization": "Bearer "+key})
            with urlopen(request, timeout=45, context=_TLS_CONTEXT) as response:
                result = json.load(response)
            if result.get("status") != "OK":
                raise RuntimeError("Contract-reference API did not return OK")
            return result
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"Contract-reference HTTP status {error.code}") from None
            delay = min(60., max(15., 2.**attempt))
            print(f"Reference HTTP {error.code}; retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)
        except (URLError, TimeoutError):
            time.sleep(min(30., 2.**attempt))
    raise RuntimeError("Contract-reference retries exhausted")


def fetch_group(group, *, wanted, key, cache, as_of):
    expiry, kind, expired = group
    prefix = f"{expiry}_{kind}_{expired}"
    url = "https://api.massive.com/v3/reference/options/contracts?"+urlencode({
        "expiration_date": expiry, "contract_type": kind, "expired": str(expired).lower(),
        "as_of": as_of, "limit": 1000, "sort": "ticker", "order": "asc"})
    page = 0
    while url:
        path = cache/f"{prefix}_{page:05d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["url"] != safe_url(url):
                raise ValueError("Reference cache pagination identity changed")
        else:
            result = request_page(url, key)
            matches = [item for item in result.get("results", []) if item.get("ticker") in wanted]
            saved = {"url": safe_url(url), "next_url": safe_url(result["next_url"]) if result.get("next_url") else None,
                     "results": matches}
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(saved, separators=(",", ":")))
            temporary.replace(path)
            time.sleep(.1)
        url = saved["next_url"]
        page += 1
    return prefix, page


def fetch_styles(source: Path, output: Path, key: str, workers: int = 4) -> dict:
    if not key or any(c.isspace() for c in key):
        raise ValueError("Provide the REST key only, without a curl command or whitespace")
    if workers < 1 or workers > 64:
        raise ValueError("workers must be between 1 and 64")
    manifest = json.loads((source/"manifest.json").read_text())
    contracts = source/"contracts.csv"
    with contracts.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    wanted = set(pd.read_csv(contracts, usecols=["ticker"]).ticker)
    groups = sorted({(name[:10], name[11:].split("_")[0].split(".")[0][:-1], expired)
                     for name in manifest["files"] for expired in (False, True)})
    output.mkdir(parents=True, exist_ok=True)
    cache = output/"pages"
    cache.mkdir(exist_ok=True)
    identity = {"contracts_sha256": digest, "as_of": manifest["last_date"], "groups": [list(g) for g in groups]}
    identity_path = output/"identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Reference output belongs to a different export")
    identity_path.write_text(json.dumps(identity, indent=2)+"\n")
    completed, pages = 0, 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_group, group, wanted=wanted, key=key,
                    cache=cache, as_of=manifest["last_date"]) for group in groups]
        for future in as_completed(futures):
            prefix, count = future.result()
            completed += 1
            pages += count
            if completed % 10 == 0 or completed == len(groups):
                print(f"Reference groups {completed}/{len(groups)}; {pages} cached pages", flush=True)
    del wanted
    # A contract can occur in both status queries; retain one consistent style.
    seen = {}
    counts = {"american": 0, "european": 0, "bermudan": 0, "unknown": 0}
    temporary = output/"exercise_styles.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ticker", "exercise_style", "underlying_ticker",
            "expiration_date", "contract_type", "strike_price", "shares_per_contract", "as_of"])
        writer.writeheader()
        for path in sorted(cache.glob("*.json")):
            for item in json.loads(path.read_text())["results"]:
                ticker = item["ticker"]
                style = item.get("exercise_style") or "unknown"
                if style not in counts:
                    raise ValueError("Unexpected exercise style returned by Massive")
                if ticker in seen:
                    if seen[ticker] != style:
                        raise ValueError(f"Conflicting reference styles for {ticker}")
                    continue
                seen[ticker] = style
                counts[style] += 1
                writer.writerow({"ticker": ticker, "exercise_style": style,
                    "underlying_ticker": item.get("underlying_ticker", ""),
                    "expiration_date": item.get("expiration_date", ""),
                    "contract_type": item.get("contract_type", ""), "strike_price": item.get("strike_price", ""),
                    "shares_per_contract": item.get("shares_per_contract", ""), "as_of": manifest["last_date"]})
    temporary.replace(output/"exercise_styles.csv")
    summary = {"matched_contracts": len(seen), "requested_contracts": manifest["contracts"],
               "unmatched_contracts": manifest["contracts"]-len(seen), "styles": counts, "pages": pages,
               "as_of": manifest["last_date"]}
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("massive_options_csv_group1_8589"))
    parser.add_argument("--output", type=Path, default=Path("massive_options_csv_group1_8589/reference"))
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    key = read_api_key(args.key_file) if args.key_file else os.environ.get("MASSIVE_API_KEY", "")
    fetch_styles(args.source, args.output, key, args.workers)


if __name__ == "__main__":
    main()
