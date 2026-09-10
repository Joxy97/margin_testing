"""Ten sequential portfolios with separate confidence-radius and penalty probes."""

from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

from benchmark_factor_stress import prepareApplication, writeJson, referenceRow
from report_factor_sweep import writeCsv
from sweep_factor_stress import SOLVERS, execute, loadModel
from risky_factor_portfolios import constructPortfolios, configurations, compactModel, confidenceRadius
from margin_calculator.optimization.factor_stress import QuadraticStressObjective
from margin_calculator.optimization.factor_stress_reference import (
    solveLinearReference, solveQuadraticReference, solveRepricedReference,
)
from portfolio import Portfolio
from risk_state_generator import ReturnsPCAGrid, ReturnsPCAKey
from risk_state_generator.factor_stress_model import FactorStressModel


def prepare(args):
    started = perf_counter()
    root = args.output
    shared = root/"shared"
    shared.mkdir(parents=True)
    app = prepareApplication(SimpleNamespace(group=args.group, date=None, window=125,
        ew_lambda=.93, seed=20260910), shared)
    date_column = pd.read_csv(args.group, nrows=0).columns[0]
    dates = sorted(pd.to_datetime(pd.read_csv(args.group, usecols=[date_column])[date_column]).dt.date)
    selected = dates[126:]
    if len(selected) != 294 or selected[0] != date(2025, 7, 8) or selected[-1] != date(2026, 9, 4):
        raise ValueError("input dates differ from the requested full backtest")
    if len(app.portfolio.instruments) != 8590:
        raise ValueError("expected the full 8590-stock universe")
    engine = app.createEngine()
    before = perf_counter()
    engine.prepareBacktest(app.portfolio, selected)
    prefetch_seconds = perf_counter()-before
    portfolio_rows, portfolios, prepared = None, None, {}
    stages, market_diagnostics = [], []
    for index, day in enumerate(selected):
        begin = perf_counter()
        before = perf_counter()
        data = engine.getPortfolioMarketData(app.portfolio, day)
        acquisition_seconds = perf_counter()-before
        before = perf_counter()
        grid = ReturnsPCAGrid.construct(ReturnsPCAKey(app.portfolio.instruments, 125, day, .93, 2), data)
        pca_seconds = perf_counter()-before
        assert grid.calibrationEndDate < day
        prices = data.copy()
        if "date" in prices:
            prices = prices.set_index("date")
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index().loc[:, list(grid.instruments)]
        prior = prices.loc[prices.index < pd.Timestamp(day)].iloc[-126:]
        current = prices.loc[pd.Timestamp(day)].to_numpy(dtype=float)
        previous = prior.iloc[-1].to_numpy(dtype=float)
        if not np.isfinite(current).all() or np.any(current <= 0) or np.any(previous <= 0):
            raise ValueError("market prices must be finite and positive")
        returns = current/previous-1
        if portfolios is None:
            portfolio_rows, weights, eligible = constructPortfolios(prior, grid)
            portfolios = [Portfolio({name: Decimal(str(weight)) for name, weight in zip(grid.instruments, row)})
                          for row in weights]
            writeCsv(root/"portfolios.csv", portfolio_rows)
            for metadata, portfolio in zip(portfolio_rows, portfolios):
                directory = root/metadata["id"]
                (directory/"models").mkdir(parents=True)
                (directory/"samples").mkdir()
                (directory/"batches").mkdir()
                prepared[metadata["id"]] = []
                writeCsv(directory/"portfolio.csv", [dict(ticker=name, weight=str(weight))
                    for name, weight in portfolio.weights.items()])
            settings = dict(window=125, ew_lambda=.93, bits=8, target_coverage=.9995,
                target_radius=confidenceRadius(.9995), primary_multiplier=1.,
                runs=args.runs, steps=args.steps, devices=args.devices, primary_repeats=3,
                dates=[str(d) for d in selected], portfolios=portfolio_rows,
                eligible_design_stocks=int(eligible.sum()), design_cutoff=str(prior.index[-1].date()),
                design_start=str(prior.index[0].date()), group_sha256=hashlib.sha256(args.group.read_bytes()).hexdigest(),
                confidence_interpretation="nominal standard Gaussian 3D scenario-region content; not certified loss coverage",
                portfolio_selection="fixed using only pre-period observations; min prior price 1, last price 5, max absolute prior log return <= log(2)",
                precision="8-bit fixed point coordinates; float64 model and solver arithmetic",
                settings=configurations(), expected_trials_per_portfolio=294*3*sum(r["repeats"] for r in configurations()),
                sequence="shared PCA preparation, then complete each portfolio's references and GPU solves before starting the next",
                benchmark_note="exhaustive quadratic lattice benchmark omitted; continuous exact-repricing reference retained")
            writeJson(root/"settings.json", settings)
        before = perf_counter()
        for metadata, portfolio in zip(portfolio_rows, portfolios):
            full = FactorStressModel.fromPCAGrid(grid, portfolio)
            model = compactModel(full)
            objective = QuadraticStressObjective(*model.quadraticCoefficients())
            # Zero-exposure removal must preserve both repricing and Taylor energy.
            for point in (np.zeros(3), np.array([.5, -.25, .125])):
                np.testing.assert_allclose(model.pnl(point), full.pnl(point), rtol=1e-12, atol=1e-12)
            realized = float(returns @ full.exposures)
            row = dict(index=index, date=str(day), previous_close_date=str(prior.index[-1].date()),
                calibration_start=str(grid.calibrationStartDate), calibration_end=str(grid.calibrationEndDate),
                calibration_observations=len(grid.factors), universe_stocks=len(grid.instruments),
                stocks=len(model.instruments), realized_pnl=realized, realized_loss=max(0., -realized),
                references=[], lattice={}, pca_explained=grid.explained.tolist())
            prepared[metadata["id"]].append(row)
            directory = root/metadata["id"]
            np.savez(directory/"models"/f"{index:03d}.npz", instruments=np.array(model.instruments),
                     exposures=model.exposures, center=model.center, directions=model.directions,
                     constant=objective.constant, gradient=objective.gradient, hessian=objective.hessian)
            active = full.exposures != 0
            for asset in np.flatnonzero(active & (np.abs(np.log(current/previous)) > np.log(2.))):
                market_diagnostics.append(dict(portfolio=metadata["id"], date=str(day), ticker=grid.instruments[asset],
                    previous_close=float(previous[asset]), current_close=float(current[asset]),
                    exposure=float(full.exposures[asset]), asset_return=float(returns[asset]),
                    pnl_contribution=float(returns[asset]*full.exposures[asset])))
        stages.append(dict(date=str(day), acquisition_seconds=acquisition_seconds, pca_seconds=pca_seconds,
                           ten_models_seconds=perf_counter()-before, day_wall_seconds=perf_counter()-begin))
        if (index+1) % 20 == 0 or index == len(selected)-1:
            state = dict(stage="shared_preparation", completed_dates=index+1, total_dates=len(selected),
                         elapsed_seconds=perf_counter()-started)
            writeJson(root/"status.json", state)
            print(json.dumps(state), flush=True)
    for metadata in portfolio_rows:
        writeJson(root/metadata["id"]/"days.json", dict(days=prepared[metadata["id"]]))
    if market_diagnostics:
        writeCsv(root/"large_asset_moves.csv", market_diagnostics)
    writeCsv(root/"shared_preparation_stages.csv", stages)
    writeJson(root/"shared_timing.json", dict(wall_seconds=perf_counter()-started,
        prefetch_seconds=prefetch_seconds, pca_seconds=sum(r["pca_seconds"] for r in stages),
        portfolio_models_seconds=sum(r["ten_models_seconds"] for r in stages)))


def references(directory, settings):
    started = perf_counter()
    days = json.loads((directory/"days.json").read_text())["days"]
    for day in days:
        before = perf_counter()
        model, objective = loadModel(directory/"models"/f"{day['index']:03d}.npz")
        rows = []
        for coverage in sorted({r["coverage"] for r in settings["settings"]}):
            radius = confidenceRadius(coverage)
            for method, solve in (("exact_repricing_continuous", lambda: solveRepricedReference(model, radius)),
                                  ("linear_analytic", lambda: solveLinearReference(objective, radius)),
                                  ("quadratic_continuous", lambda: solveQuadraticReference(objective, radius))):
                begin = perf_counter()
                try:
                    result = solve()
                    row = referenceRow(method, result, model, objective, perf_counter()-begin)
                    row.update(coverage=coverage, radius=radius,
                               global_certificate=result.lowerBound is not None,
                               breach=bool(day["realized_loss"] > row["margin"]))
                except (FloatingPointError, RuntimeError, ValueError) as error:
                    row = dict(method=method, coverage=coverage, radius=radius, success=False,
                               error=str(error), margin=None, seconds=perf_counter()-begin)
                rows.append(row)
        day["references"] = rows
        day["reference_seconds"] = perf_counter()-before
    writeJson(directory/"days.json", dict(days=days))
    return perf_counter()-started


def schedule(settings):
    grouped = {solver: [] for solver in SOLVERS}
    for day in range(len(settings["dates"])):
        for config in settings["settings"]:
            for repeat in range(config["repeats"]):
                for solver in SOLVERS:
                    grouped[solver].append(dict(id=f"d{day:03d}_{config['id']}_r{repeat}_{solver}", day=day,
                        bits=8, repeat=repeat, solver=solver, seed_offset=day*100000+8000+repeat,
                        multiplier=config["multiplier"], radius=config["radius"], coverage=config["coverage"],
                        setting=config["id"], primary=config["primary"]))
    work = []
    for start in range(0, len(grouped[SOLVERS[0]]), 32):
        for solver in SOLVERS:
            work.append(dict(id=len(work), solver=solver, trials=grouped[solver][start:start+32]))
    return work


def run(args):
    args.output = args.output.resolve()
    args.group = args.group.resolve()
    if args.output.exists() and not args.resume:
        raise FileExistsError("choose a new experiment directory")
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    if not args.resume:
        prepare(args)
    elif not (args.output/"shared_timing.json").exists():
        raise ValueError("shared preparation must be complete before resuming")
    settings = json.loads((args.output/"settings.json").read_text())
    if (settings["group_sha256"] != hashlib.sha256(args.group.read_bytes()).hexdigest()
            or settings["runs"] != args.runs or settings["steps"] != args.steps or settings["devices"] != args.devices):
        raise ValueError("resume settings or dataset differ")
    from report_risky_factor_backtests import reportPortfolio, reportAll
    work = schedule(settings)
    timings = []
    for metadata in settings["portfolios"]:
        directory = args.output/metadata["id"]
        if args.resume and (directory/"verification.json").exists():
            timings.append(json.loads((directory/"timings.json").read_text()))
            continue
        begin = perf_counter()
        writeJson(args.output/"status.json", dict(stage="portfolio", portfolio=metadata["id"],
            name=metadata["name"], completed_portfolios=len(timings), total_portfolios=10))
        reference_seconds = references(directory, settings)
        options = SimpleNamespace(output=directory, steps=args.steps, runs=args.runs, seed=20260910,
                                  radius=confidenceRadius(.9995), devices=args.devices, repair_samples=True)
        writeJson(directory/"schedule.json", dict(batches=work))
        timing = execute(options, work)
        timing.update(reference_seconds=reference_seconds, portfolio_wall_seconds=perf_counter()-begin,
                      portfolio=metadata["id"])
        writeJson(directory/"timings.json", timing)
        before = perf_counter()
        reportPortfolio(args.output, metadata["id"])
        timing["verification_report_seconds"] = perf_counter()-before
        timings.append(timing)
        writeJson(args.output/"portfolio_timings.json", dict(portfolios=timings))
        reportAll(args.output)
        print(json.dumps(dict(stage="portfolio_complete", **timing)), flush=True)
    writeJson(args.output/"timings.json", dict(total_wall_seconds=perf_counter()-started, portfolios=timings))
    reportAll(args.output)
    writeJson(args.output/"status.json", dict(stage="complete", portfolios=10, dates=294,
        trials=10*settings["expected_trials_per_portfolio"], total_wall_seconds=perf_counter()-started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--devices", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.runs < 1 or arguments.steps < 1 or not arguments.devices or len(set(arguments.devices)) != len(arguments.devices):
        parser.error("positive runs/steps and distinct devices are required")
    run(arguments)
