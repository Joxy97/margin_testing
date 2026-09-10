"""Plot saved daily factor margins and realized losses, with gains clipped to zero."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def plot(experiment: Path) -> None:
    daily = pd.read_csv(experiment/"daily_margins.csv", parse_dates=["date"])
    reference = pd.read_csv(experiment/"results/daily_references.csv", parse_dates=["date"])
    margins = daily.pivot(index="date", columns="method", values="margin").sort_index()
    references = reference.pivot(index="date", columns="method", values="margin").sort_index()
    if not margins.index.equals(references.index):
        raise ValueError("solver and benchmark dates differ")
    if daily.groupby("date")["realized_loss"].nunique().max() != 1:
        raise ValueError("methods disagree on realized loss")
    loss = daily.groupby("date")["realized_loss"].first().reindex(margins.index).clip(lower=0.)
    plotted = margins.join(references)
    days = json.loads((experiment/"results/days.json").read_text())["days"]
    lattice = pd.Series({pd.Timestamp(d["date"]): d["lattice"]["8"]["margin"] for d in days})
    plotted["quadratic_lattice_8"] = lattice.reindex(margins.index)
    plotted["realized_loss_clipped"] = loss
    if not np.isfinite(plotted.to_numpy()).all() or np.any(plotted.to_numpy() < 0):
        raise ValueError("plotted margins and clipped losses must be finite and nonnegative")
    # CSV keeps the original fractional exposure units; the chart shows percent.
    plotted.to_csv(experiment/"margin_loss_plot_data.csv", index_label="date")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    panels = (
        ("QUBO methods after repair", (
            ("torch_sbm", "SBM", "#2563eb", "-"),
            ("torch_svl", "SVL", "#e57900", "--"),
            ("torch_transverse_route", "TRF", "#7c3aed", "-."),
            ("combined", "Combined maximum", "#15803d", ":"))),
        ("Benchmark methods", (
            ("linear_analytic", "Linear analytic", "#2563eb", "-"),
            ("quadratic_continuous", "Quadratic continuous", "#e57900", "--"),
            ("exact_repricing_continuous", "Exact-repricing continuous", "#7c3aed", "-."),
            ("quadratic_lattice_8", "Quadratic lattice (8-bit)", "#15803d", ":"))),
    )
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, sharey=True, layout="constrained")
    timeline = plotted.index
    for axis, (title, methods) in zip(axes, panels):
        for column, label, color, style in methods:
            axis.plot(timeline, plotted[column]*100, label=label, color=color, linestyle=style, lw=1.8)
        axis.fill_between(timeline, loss*100, 0., color="#334155", alpha=.16)
        axis.plot(timeline, loss*100, color="#111827", lw=1.1, label="Realized loss (clipped at 0)", zorder=5)
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=12)
        axis.set_ylabel("Margin / loss (% of portfolio exposure)")
        axis.set_ylim(0., float(plotted.max().max())*100*1.22)
        axis.grid(alpha=.18)
        axis.legend(loc="upper center", bbox_to_anchor=(.5, 1.01), ncol=3, fontsize=9, framealpha=.95)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].set_xlabel("Evaluation date")
    fig.suptitle("Group 1 · 8,590 stocks · 125-day EW window · 8-bit coordinates\n"
                 "Daily margins and realized loss · Jul 2025–Sep 2026", fontsize=15)
    for suffix in ("png", "svg"):
        fig.savefig(experiment/f"margins_and_realized_loss.{suffix}", dpi=180)
    plt.close(fig)
    print(json.dumps(dict(dates=len(plotted), margin_methods=len(plotted.columns)-1,
        min_plotted_loss=float(loss.min()), max_plotted_loss=float(loss.max()),
        clipped_dates=int((daily.groupby("date")["realized_loss"].first() < 0).sum()),
        image=str(experiment/"margins_and_realized_loss.png"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    plot(parser.parse_args().experiment.resolve())
