# Substantial ticker groups

This folder contains the generated grouping pipeline and its CSV output. The
pipeline reads only from `../yahoo_equities` and uses
`substantial_tickers.csv` as the inclusion list.

Run from the repository root with:

```bash
python -m yahoo_equities_grouped.main
```

`manifest.json` records the hash parameters, counts, regions, group members,
group ranks, and output paths. `date_mapping.csv` records the one-based global
date mapping. Each `groups/<region>/group_<rank>_<number_of_stocks>.csv`
contains one `Date` column followed by one `<ticker>` column for every ticker in
that group. Groups are ranked by ticker count in descending order, with the
date hash used as a deterministic tie-breaker. A group is one region and one
trading-day hash, so all tickers in a group have identical trading dates.

To download historical EUR exchange rates and convert every grouped price to
EUR, run:

```bash
python -m yahoo_equities_grouped.fx_converter
```

Rates are downloaded from Frankfurter with retry/backoff handling for transient
HTTP and network failures. The downloaded observations are stored in
`fx_rates/frankfurter_eur_rates.csv`, with the conversion assumptions in
`fx_rates/conversion_metadata.json`. Conversion requires an exact FX rate for
each price date; it does not forward-fill missing rates.
