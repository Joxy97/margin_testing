"""Offline European/American PCA-QUBO example with independent CRR comparisons."""

import argparse
from dataclasses import asdict
from datetime import timedelta
from decimal import Decimal
import json
from pathlib import Path

import numpy as np
import pandas as pd

from option_pricing import AmericanEquityBinomialPricingModel, OptionBook, VanillaPriceContext
from portfolio import DerivativePosition, EquityOptionContract, Portfolio
from risk_state_generator import OptionFactorStressModel, ReturnsPCAGrid, ReturnsPCAKey
from margin_calculator.optimization import FactorStressQUBOConfig, solveOptionFactorStress
from margin_calculator.optimization.optimization_solver.bqm_solver import BQMSolverFactory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(19)
    instruments = ("AAA", "BBB", "CCC", "DDD")
    prices = pd.DataFrame(100*np.exp(np.vstack((np.zeros(4),
        np.cumsum(rng.normal(0., .015, (60, 4)), axis=0)))),
        index=pd.bdate_range("2025-01-01", periods=61), columns=instruments)
    today = prices.index[-1].date()
    horizon, expiry = today+timedelta(days=1), today+timedelta(days=180)
    spots = prices.iloc[-1].to_numpy()
    positions = tuple(DerivativePosition(EquityOptionContract(symbol, expiry, Decimal("100"), kind,
        style, Decimal("100"), dividendYield=.03), Decimal(quantity))
        for symbol, kind, style, quantity in zip(instruments, ("C", "P", "C", "P"),
            ("E", "E", "A", "A"), ("-2", "1", "-2", "-1")))
    marks = [float(VanillaPriceContext(100., 180/365., .04, .03, .27,
        p.contract.optionType, p.contract.exerciseStyle).evaluate(spots[i])) for i, p in enumerate(positions)]
    options = OptionBook(instruments, spots, positions, marks, today, horizon, .04)
    portfolio = Portfolio({symbol: Decimal("10000") for symbol in instruments})
    grid = ReturnsPCAGrid.construct(ReturnsPCAKey(instruments, 60, horizon, .93, 2), prices)
    model = OptionFactorStressModel.fromPCAGrid(grid, portfolio, options)
    config = FactorStressQUBOConfig(bitsPerCoordinate=3, radius=3.)
    parameters = {"num_reads": 16, "num_sweeps": 500, "seed": 19}
    result = solveOptionFactorStress(model, BQMSolverFactory.create("simulated_annealing"),
        config=config, solverParameters=parameters)
    references = []
    for index, (position, context) in enumerate(zip(options.positions, options.contexts)):
        if position.contract.exerciseStyle != "A":
            continue
        spot = float(result.stressedSpots[index])
        tree = [AmericanEquityBinomialPricingModel(steps).price(spot, context.strike,
            context.timeToExpiry, context.riskFreeRate, context.volatility, context.optionType,
            context.dividendYield) for steps in (800, 1600)]
        ju = float(context.evaluate(spot))
        references.append({"position": index, "ju_zhong_price": ju, "crr_800": tree[0],
                           "crr_1600": tree[1], "ju_minus_crr_1600": ju-tree[1]})
    payload = {
        "scope": "synthetic frozen-book equity/index vanilla example; greatest model loss found",
        "valuation_date": str(today), "horizon_date": str(horizon),
        "calibration_end": str(grid.calibrationEndDate),
        "configuration": asdict(config), "solver": "simulated_annealing", "solver_parameters": parameters,
        "calibrations": [asdict(item) for item in options.calibrations],
        "horizon_domain": model.domainDiagnostics(config.radius), "residual_alignment": model.residualAlignment,
        "margin_found": result.margin, "full_model_pnl": result.pnl,
        "coordinates": result.coordinates.tolist(), "stressed_spots": result.stressedSpots.tolist(),
        "pnl_by_type": {"stock": result.stockPnL, "European": result.europeanPnL, "American": result.americanPnL},
        "stress_dimensions": model.dimension, "smooth_anchors": result.smoothAnchors,
        "nonsmooth_anchors": result.nonsmoothAnchors, "candidate_count": len(result.candidates),
        "raw_solver_feasibility": result.solverFeasibility, "boundary_lattice_side_coverage": result.boundaryCoverage,
        "all_repaired_candidates_feasible": all(np.sum(c.integers*c.integers) <= (2**(config.bitsPerCoordinate-1)-1)**2
                                                for c in result.candidates),
        "repair_cap_hits": sum(not c.converged for c in result.candidates),
        "same_iv_pricing_references_at_worst_scenario": references,
    }
    rendered = json.dumps(payload, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
