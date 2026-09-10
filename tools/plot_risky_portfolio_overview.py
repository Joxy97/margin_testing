"""Compare completed risky-portfolio margins, clipped losses and realized P&L."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from report_risky_factor_backtests import loadGreedy


def plot(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    settings = json.loads((root/"settings.json").read_text())
    moves_path = root/"large_asset_moves.csv"
    large_moves = pd.read_csv(moves_path) if moves_path.exists() else pd.DataFrame(columns=["portfolio", "date", "pnl_contribution"])
    methods = (("torch_sbm", "SBM", "#2563eb"), ("torch_svl", "SVL", "#d97706"),
               ("torch_transverse_route", "TRF", "#7c3aed"), ("combined", "Combined", "#15803d"))
    fig, axes = plt.subplots(5, 2, figsize=(17, 16), layout="constrained")
    metrics, breach_details = [], []
    for axis, metadata in zip(axes.flat, settings["portfolios"]):
        directory = root/metadata["id"]
        axis.set_title(f"{metadata['id']}: {metadata['name']}", loc="left", fontsize=10)
        if not (directory/"local_verification.json").exists():
            axis.text(.5, .5, "Pending completion", transform=axis.transAxes, ha="center", color="gray")
            axis.set_axis_off()
            continue
        data = pd.read_csv(directory/"daily_margins.csv", parse_dates=["date"])
        data = data[data.analysis == "primary_best3"]
        greedy = loadGreedy(directory, data)
        for method, label, color in methods:
            selected = data[data.solver == method].sort_values("date")
            axis.plot(selected.date, selected.margin*100, label=label, color=color, lw=1.)
        if greedy is not None:
            axis.plot(pd.to_datetime(greedy.date), greedy.margin*100,
                      label="Greedy PCA", color="#be185d", lw=1.2, linestyle="--")
            greedy_breaches = greedy[greedy.breach]
            axis.scatter(pd.to_datetime(greedy_breaches.date), greedy_breaches.realized_loss*100,
                         edgecolors="#be185d", facecolors="none", marker="s", s=25,
                         label="Greedy breach", zorder=6)
        combined = data[data.solver == "combined"].sort_values("date")
        loss = combined.realized_loss.clip(lower=0)
        axis.plot(combined.date, loss*100, color="#111827", lw=.8, label="Realized loss")
        breaches = combined[combined.breach]
        days = {d["date"]: d for d in json.loads((directory/"days.json").read_text())["days"]}
        for row in breaches.itertuples():
            day = row.date.strftime("%Y-%m-%d")
            moves = large_moves[(large_moves.portfolio == metadata["id"]) & (large_moves.date == day)]
            reference = next(r for r in days[day]["references"] if r["method"] == "exact_repricing_continuous" and r["coverage"] == .9995)
            lower = reference.get("lower_bound")
            continuous_upper_loss = max(0., -lower) if lower is not None else None
            breach_details.append(dict(portfolio=metadata["id"], date=day, margin=row.margin,
                realized_loss=row.realized_loss, breach_excess=row.breach_excess,
                large_asset_moves=len(moves),
                flagged_move_pnl_contribution=float(moves.pnl_contribution.sum()),
                local_reference_margin=reference.get("margin"),
                local_reference_success=reference["success"],
                continuous_upper_loss_bound=continuous_upper_loss,
                loss_above_continuous_bound=None if continuous_upper_loss is None else bool(row.realized_loss > continuous_upper_loss+1e-12),
                reference_covers_loss=None if reference.get("margin") is None else bool(row.realized_loss <= reference["margin"])))
        axis.scatter(breaches.date, breaches.realized_loss*100, color="#dc2626", s=13, label="Combined breach", zorder=5)
        symlog = max(loss.max(), combined.margin.max(), 0 if greedy is None else greedy.margin.max()) > 10*combined.margin.median()
        if symlog:
            axis.set_yscale("symlog", linthresh=1.)
        axis.set_ylabel("Margin / loss (%)"+(" · symlog" if symlog else ""), fontsize=9)
        axis.set_ylim(bottom=0)
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axis.tick_params(labelsize=8)
        axis.grid(alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
        counts = f"Combined: {len(breaches)} breaches"
        if greedy is not None:
            counts += f" · Greedy: {int(greedy.breach.sum())}"
        axis.text(.02, .96, counts+" / 294 dates", transform=axis.transAxes,
                  va="top", fontsize=9, bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
        metrics.append(dict(portfolio=metadata["id"], name=metadata["name"], positions=metadata["positions"],
            gross=metadata["gross"], net=metadata["net"], dates=len(combined), breaches=len(breaches),
            mean_margin=float(combined.margin.mean()), max_realized_loss=float(loss.max()),
            max_realized_gain=max(0., float(combined.realized_pnl.max())),
            daily_pnl_std=float(np.std(combined.realized_pnl)), loss_days=int((loss>0).sum()),
            gain_days=int((combined.realized_pnl>0).sum()),
            greedy_breaches=None if greedy is None else int(greedy.breach.sum()),
            greedy_mean_margin=None if greedy is None else float(greedy.margin.mean())))
    if metrics:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=10)
        pd.DataFrame(metrics).to_csv(root/"realized_risk_comparison.csv", index=False)
        if breach_details:
            pd.DataFrame(breach_details).to_csv(root/"breach_diagnostics.csv", index=False)
    has_greedy = any(row["greedy_breaches"] is not None for row in metrics)
    fig.suptitle(f"Group 1 · {len(metrics)}/10 portfolios complete · joint-factor nominal 99.95%\n"
                 +("Greedy PCA: original 105-scenario region · realized loss clipped at zero" if has_greedy else
                   "8-bit coordinates · penalty multiplier 1 · realized loss clipped at zero"), fontsize=16)
    fig.savefig(root/"portfolio_overview.png", dpi=160)
    plt.close(fig)
    print(json.dumps(dict(completed=len(metrics), image=str(root/"portfolio_overview.png"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    plot(parser.parse_args().root.resolve())
