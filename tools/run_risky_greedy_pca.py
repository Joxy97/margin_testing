"""Run the original greedy PCA grid for the saved ten-portfolio experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait
from datetime import date
from decimal import Decimal
import hashlib
import json
import multiprocessing
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

from benchmark_factor_stress import prepareApplication, writeJson
from margin_calculator.state_aware_greedy_risk_state_visitor import StateAwareGreedyRiskStateVisitor
from portfolio import Portfolio
from risk_state_generator import PCAGridProvider, ReturnsVolaGridRiskStateGenerator, ReturnsPCAKey, RiskStateGenerationContext


GRID_PARAMETERS = dict(ew_window=125, ew_lambda=.93, components=2,
    scenariosPerComponents=(21, 5), tailDensityGamma=1., nZBins=21,
    nNearest=None, residualSigmaRange=5., allowEmptyBinFallback=True,
    distanceInflationAlpha=.5, distanceInflationPower=2., maxInflationFactor=5.)


def marginsFromBounds(bounds, weights):
    """Independent signed-exposure reference for a saved scenario grid."""
    bounds = np.asarray(bounds, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if bounds.ndim != 3 or bounds.shape[2] != 2 or weights.ndim != 2 or bounds.shape[1] != weights.shape[1]:
        raise ValueError("expected bounds (scenarios, assets, 2) and weights (portfolios, assets)")
    if not len(bounds) or not np.isfinite(bounds).all() or not np.isfinite(weights).all():
        raise ValueError("scenario bounds and weights must be nonempty and finite")
    if np.any(bounds[:, :, 0] > bounds[:, :, 1]):
        raise ValueError("return lower bounds must not exceed upper bounds")
    pnl = bounds[:, :, 0] @ np.maximum(weights, 0).T + bounds[:, :, 1] @ np.minimum(weights, 0).T
    return np.maximum(0., -pnl.min(axis=0)), pnl


def loadPortfolios(root, settings):
    portfolios = []
    for item in settings["portfolios"]:
        table = pd.read_csv(root/item["id"]/"portfolio.csv", dtype={"ticker": str, "weight": str})
        if table.ticker.duplicated().any():
            raise ValueError("duplicate portfolio ticker")
        portfolios.append(Portfolio(dict(zip(table.ticker, map(Decimal, table.weight)))))
    return portfolios


def runShard(root, group, output, shard, indices):
    settings = json.loads((root/"settings.json").read_text())
    work = output/f"worker_{shard}"
    work.mkdir()
    app = prepareApplication(SimpleNamespace(group=group, date=None, window=125,
        ew_lambda=.93, seed=20260910), work)
    portfolios = loadPortfolios(root, settings)
    universe = app.portfolio.instruments
    if len(universe) != 8590 or any(p.instruments != universe for p in portfolios):
        raise ValueError("portfolios must retain the same 8590-stock PCA universe")
    weights = np.array([[float(p.weights[t]) for t in universe] for p in portfolios])
    active = np.any(weights != 0, axis=0)
    expected = [json.loads((root/p["id"]/"days.json").read_text())["days"] for p in settings["portfolios"]]
    engine = app.createEngine()
    dates = [date.fromisoformat(settings["dates"][i]) for i in indices]
    before = perf_counter()
    engine.prepareBacktest(app.portfolio, dates)
    preparation = perf_counter()-before
    provider = PCAGridProvider(memorySize=1)
    generator = ReturnsVolaGridRiskStateGenerator(pcaGridProvider=provider, **GRID_PARAMETERS)
    visitor = StateAwareGreedyRiskStateVisitor()
    for index, day in zip(indices, dates):
        begin = perf_counter()
        before = perf_counter()
        data = engine.getPortfolioMarketData(app.portfolio, day)
        acquisition = perf_counter()-before
        before = perf_counter()
        key = ReturnsPCAKey(universe, 125, day, .93, 2)
        grid = provider.getOrCreate(key, data)
        pca_seconds = perf_counter()-before
        if str(grid.calibrationEndDate) >= str(day) or len(grid.factors) != 125:
            raise ValueError("PCA calibration contains look-ahead or has wrong window")
        for days in expected:
            row = days[index]
            if (row["date"] != str(day) or row["calibration_start"] != str(grid.calibrationStartDate)
                    or row["calibration_end"] != str(grid.calibrationEndDate)):
                raise ValueError("greedy dates differ from the completed experiment")
        prices = data.copy()
        if "date" in prices:
            prices = prices.set_index("date")
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index().loc[:, list(universe)]
        previous = prices.loc[prices.index < pd.Timestamp(day)].iloc[-1].to_numpy(float)
        current = prices.loc[pd.Timestamp(day)].to_numpy(float)
        realized = weights @ (current/previous-1)
        np.testing.assert_allclose(realized, [days[index]["realized_pnl"] for days in expected], rtol=1e-12, atol=1e-12)
        context = RiskStateGenerationContext(data, generator.createDataRequest(app.portfolio, day), day)
        states = iter(generator.getRiskStates(context))
        bounds, pnls, fallback = [], [], []
        generation = selection = 0.
        while True:
            before = perf_counter()
            try:
                state = next(states)
            except StopIteration:
                generation += perf_counter()-before
                break
            generation += perf_counter()-before
            before = perf_counter()
            pnls.append([visitor.portfolioPnl(state, p) for p in portfolios])
            selection += perf_counter()-before
            bounds.append(state.returnsVolaGrid.returnBounds[active].copy())
            fallback.append(int(state.returnsVolaGrid.fallbackAssetMask.sum()))
        if len(bounds) != 105:
            raise ValueError("expected exactly 105 PCA scenarios")
        before = perf_counter()
        margins, reference_pnls = marginsFromBounds(bounds, weights[:, active])
        np.testing.assert_allclose(pnls, reference_pnls, rtol=1e-12, atol=1e-12)
        np.savez_compressed(output/"days"/f"{index:03d}.npz", bounds=np.array(bounds),
            weights=weights[:, active], instruments=np.array(universe)[active],
            scenario_pnl=np.array(pnls), margins=margins, realized_pnl=realized)
        audit_seconds = perf_counter()-before
        rows = [dict(portfolio=p["id"], date=str(day), margin=float(margins[j]),
            realized_pnl=float(realized[j]), realized_loss=max(0., -float(realized[j])),
            breach=bool(max(0., -float(realized[j])) > margins[j]),
            winning_scenario=int(np.argmin(reference_pnls[:, j]))) for j, p in enumerate(settings["portfolios"])]
        timing = dict(acquisition_seconds=acquisition, pca_seconds=pca_seconds,
            scenario_generation_seconds=generation, ten_portfolio_selection_seconds=selection,
            audit_and_save_seconds=audit_seconds, day_wall_seconds=perf_counter()-begin)
        writeJson(output/"days"/f"{index:03d}.json", dict(index=index, date=str(day), rows=rows,
            calibration_start=str(grid.calibrationStartDate), calibration_end=str(grid.calibrationEndDate),
            observations=125, universe_stocks=len(universe), scenarios=105,
            fallback_asset_scenarios=sum(fallback), max_fallback_assets=max(fallback), timing=timing))
    return dict(worker=shard, dates=len(dates), preparation_seconds=preparation)


def verifyAndReport(root):
    output = root/"greedy_pca"
    settings = json.loads((root/"settings.json").read_text())
    config = json.loads((output/"settings.json").read_text())
    if config["group_sha256"] != settings["group_sha256"] or config["dates"] != settings["dates"]:
        raise ValueError("greedy dataset/date identity differs")
    portfolios = loadPortfolios(root, settings)
    expected = [json.loads((root/p["id"]/"days.json").read_text())["days"] for p in settings["portfolios"]]
    rows, stages = [], []
    for index, day in enumerate(settings["dates"]):
        record = json.loads((output/"days"/f"{index:03d}.json").read_text())
        with np.load(output/"days"/f"{index:03d}.npz", allow_pickle=False) as saved:
            weights = np.array([[float(p.weights[t]) for t in saved["instruments"]] for p in portfolios])
            np.testing.assert_array_equal(weights, saved["weights"])
            margins, pnl = marginsFromBounds(saved["bounds"], weights)
            np.testing.assert_allclose(pnl, saved["scenario_pnl"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(margins, saved["margins"], rtol=1e-12, atol=1e-12)
            if len(pnl) != 105 or record["date"] != day or len(record["rows"]) != len(portfolios):
                raise ValueError("unexpected greedy scenario/date/portfolio coverage")
            for j, row in enumerate(record["rows"]):
                original = expected[j][index]
                if row["portfolio"] != settings["portfolios"][j]["id"] or row["date"] != day:
                    raise ValueError("greedy portfolio/date ordering differs")
                if record["calibration_start"] != original["calibration_start"] or record["calibration_end"] != original["calibration_end"]:
                    raise ValueError("greedy calibration window differs")
                np.testing.assert_allclose([row["margin"], row["realized_pnl"], row["realized_loss"], saved["realized_pnl"][j]],
                    [margins[j], original["realized_pnl"], original["realized_loss"], original["realized_pnl"]], rtol=1e-12, atol=1e-12)
                if row["breach"] != (row["realized_loss"] > margins[j]):
                    raise ValueError("incorrect greedy breach flag")
        rows.extend(record["rows"])
        stages.append(dict(date=day, **record["timing"]))
    frame = pd.DataFrame(rows)
    frame.to_csv(output/"daily_margins.csv", index=False)
    pd.DataFrame(stages).to_csv(output/"daily_timings.csv", index=False)
    summary = []
    for identifier, group in frame.groupby("portfolio"):
        group.to_csv(root/identifier/"greedy_pca.csv", index=False)
        summary.append(dict(portfolio=identifier, dates=len(group), breaches=int(group.breach.sum()),
            mean_margin=float(group.margin.mean()), max_margin=float(group.margin.max()),
            max_realized_loss=float(group.realized_loss.max())))
    pd.DataFrame(summary).to_csv(output/"summary.csv", index=False)
    writeJson(output/"verification.json", dict(status="passed", portfolios=len(portfolios), dates=len(stages),
        scenario_portfolio_pnls=len(stages)*105*len(portfolios), margin_rows=len(rows),
        group_sha256=config["group_sha256"], validation="saved return bounds independently repriced with original signed weights; dates, windows, P&L and breaches matched"))
    return summary


def run(args):
    root, group = args.root.resolve(), args.group.resolve()
    settings = json.loads((root/"settings.json").read_text())
    if hashlib.sha256(group.read_bytes()).hexdigest() != settings["group_sha256"]:
        raise ValueError("source dataset differs from completed backtest")
    if not 1 <= args.workers <= len(settings["dates"]):
        raise ValueError("workers must be between one and the number of dates")
    output = root/"greedy_pca"
    output.mkdir()
    (output/"days").mkdir()
    writeJson(output/"settings.json", dict(group_sha256=settings["group_sha256"], dates=settings["dates"],
        grid=GRID_PARAMETERS, workers=args.workers, backend="numpy float64 CPU",
        scenario_region="original 21x5 grid (PC1 +/-10 sigma, PC2 +/-2 sigma), independent asset residual bounds; no 99.95% coverage claim",
        shared_work="one full-universe PCA and scenario stream per date, evaluated for all ten portfolios"))
    started = perf_counter()
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(runShard, root, group, output, shard, list(range(shard, len(settings["dates"]), args.workers)))
                   for shard in range(args.workers)]
        pending = set(futures)
        previous = -1
        while pending:
            done, pending = wait(pending, timeout=5)
            for future in done:
                future.result()
            completed = len(list((output/"days").glob("*.json")))
            if completed != previous:
                state = dict(stage="greedy_pca", completed_dates=completed, total_dates=len(settings["dates"]),
                    elapsed_seconds=perf_counter()-started)
                writeJson(output/"status.json", state)
                print(json.dumps(state), flush=True)
                previous = completed
        workers = [f.result() for f in futures]
    compute_seconds = perf_counter()-started
    summary = verifyAndReport(root)
    writeJson(output/"timings.json", dict(total_wall_seconds=perf_counter()-started, compute_wall_seconds=compute_seconds,
        verification_report_seconds=perf_counter()-started-compute_seconds, workers=workers,
        note="daily stage times are summed work across concurrent dates; preparation is once per worker"))
    writeJson(output/"status.json", dict(stage="complete", dates=len(settings["dates"]), portfolios=len(summary)))
    print(json.dumps(dict(stage="complete", summary=summary)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--group", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verifyAndReport(args.root.resolve()), indent=2))
    else:
        if args.group is None:
            parser.error("--group is required for computation")
        run(args)
