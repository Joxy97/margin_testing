# Option margin benchmarks

`prepare_benchmarks.py` converts the 205 ATP portfolio CSVs into strict
application YAML files and normalized derivative-quote files under `generated/`.
The paired CME CSVs are not used as inputs because they do not contain market
prices; the ATP files contain both the positions and closing prices.

The source data does not specify exercise style. Generated futures-option
contracts and quotes therefore use European exercise (`E`). Each YAML is a
single-date margin calculation, not a backtest; derivative backtesting is not
currently supported by the application.

Run and plot the complete set from the repository root:

```bash
PYTHONPATH=src .venv/bin/python options_margin_benchmark/prepare_benchmarks.py
PYTHONPATH=src .venv/bin/python options_margin_benchmark/run_benchmarks.py
PYTHONPATH=src .venv/bin/python options_margin_benchmark/plot_results.py
```

Results are written to `generated/results.csv`. Matplotlib charts are written
to `generated/plots/`. Each symbol/category has a connected portfolio-order
chart for this application and a connected comparison chart containing CME,
CoH, CoH with new parameters, and this application's result.

The workbook and ATP folders contain different portfolio counts and do not
share a stable portfolio identifier. Comparison charts therefore show sorted
margin against the percentile within each source rather than claiming an
unreliable row-by-row match. This preserves all available observations and
makes the different result distributions directly comparable.

Blank CME margin cells in the source workbook are interpreted as zero. Blank
CoH fields remain missing because the source does not define the same rule for
those series.
