"""Offline example: underlying PCA, European marks, binary solve and exact repair.

Run with PYTHONPATH=src python tools/example_european_factor_stress.py.
Synthetic marks demonstrate the calibration round trip; replace them with dated
market marks for research on actual portfolios.
"""

from datetime import timedelta
from decimal import Decimal
import json

import numpy as np
import pandas as pd

from option_pricing import EuropeanOptionBook, EquityBlackScholesPricingModel
from portfolio import DerivativePosition, EquityOptionContract, Portfolio
from risk_state_generator import EuropeanOptionFactorStressModel, ReturnsPCAGrid, ReturnsPCAKey
from margin_calculator.optimization.factor_stress import FactorStressQUBO, FactorStressQUBOConfig, QuadraticStressObjective
from margin_calculator.optimization.factor_stress_reference import solveRepricedReference
from margin_calculator.optimization.factor_stress_repair import FactorStressRepair
from margin_calculator.optimization.optimization_solver.bqm_solver import BQMSolverFactory


def main() -> None:
    rng = np.random.default_rng(19)
    instruments = ("AAA", "BBB", "CCC", "DDD")
    prices = pd.DataFrame(
        100*np.exp(np.vstack((np.zeros(4), np.cumsum(rng.normal(0., .015, (60, 4)), axis=0)))),
        index=pd.bdate_range("2025-01-01", periods=61), columns=instruments,
    )
    valuation_date = prices.index[-1].date()
    horizon_date = valuation_date+timedelta(days=1)
    spots = prices.iloc[-1].to_numpy()
    expiry = valuation_date+timedelta(days=180)
    portfolio = Portfolio({i: Decimal("10000") for i in instruments})
    positions = (
        DerivativePosition(EquityOptionContract("AAA", expiry, Decimal("100"), "C",
            exerciseStyle="E", multiplier=Decimal("100")), Decimal("-2")),
        DerivativePosition(EquityOptionContract("BBB", expiry, Decimal("100"), "P",
            exerciseStyle="E", multiplier=Decimal("100")), Decimal("1")),
    )
    pricer = EquityBlackScholesPricingModel()
    marks = [pricer.price(spots[instruments.index(p.contract.symbol)], float(p.contract.strike),
             180/365., .04, .25, p.contract.optionType) for p in positions]
    options = EuropeanOptionBook(instruments, spots, positions, marks, valuation_date, horizon_date, .04)
    grid = ReturnsPCAGrid.construct(ReturnsPCAKey(instruments, 60, horizon_date, .93, 2), prices)
    model = EuropeanOptionFactorStressModel.fromPCAGrid(grid, portfolio, options)
    encoding = FactorStressQUBO.build(QuadraticStressObjective(*model.quadraticCoefficients()),
                                     FactorStressQUBOConfig(bitsPerCoordinate=3, radius=3.))
    solver = BQMSolverFactory.create("simulated_annealing")
    result = solver.solve(encoding.problem, {"num_reads": 32, "num_sweeps": 2000, "seed": 19})
    repaired = FactorStressRepair(model, encoding).repair(result.sample)
    reference = solveRepricedReference(model, 3., boundCoordinates=True)
    print(json.dumps({
        "valuation_date": str(valuation_date), "horizon_date": str(horizon_date),
        "calibration_end": str(grid.calibrationEndDate),
        "implied_volatilities": options.impliedVolatilities.tolist(),
        "stress_dimensions": model.dimension, "qubo_variables": encoding.problem.variableCount,
        "raw_encoding_feasible": encoding.diagnostics(result.sample)["encoding_feasible"],
        "repaired_encoding_feasible": encoding.diagnostics(repaired.sample)["encoding_feasible"],
        "coordinates": encoding.coordinates(repaired.sample).tolist(),
        "quadratic_pnl": float(encoding.objective.value(encoding.coordinates(repaired.sample))),
        "exact_pnl": repaired.pnl, "margin": max(0., -repaired.pnl),
        "repair_converged": repaired.converged,
        "continuous_local_reference_pnl": reference.objective,
        "continuous_reference_success": reference.success,
        "continuous_reference_lower_bound": reference.lowerBound,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
