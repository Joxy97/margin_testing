"""Numerical failures must be visible before SBM binary rounding hides them."""

import itertools

import pytest

from qubo_solvers import create_bqm_solver
from qubo_solvers.backends.problem import QUBOProblem


@pytest.mark.parametrize("dynamics", ["standard", "adaptive"])
@pytest.mark.parametrize("mode", ["discrete", "ballistic"])
@pytest.mark.parametrize("track_best", [False, True])
def test_sbm_rejects_nonfinite_state_before_rounding(dynamics, mode, track_best):
    problem = QUBOProblem([0., 0.], [], [], [], offset=5.)
    solver = create_bqm_solver("lib_simulated_bifurcation", {"device": "cpu"})
    # Individually finite inputs overflow the position update's a0 * dt.
    # Previously, sign(NaN) and the integer cast silently produced valid bits.
    with pytest.raises(FloatingPointError, match="dynamics became nonfinite"):
        solver.solve(problem, {
            "steps": 1, "runs": 1, "dtype": "float64", "initial_scale": 0.,
            "dt": 1e308, "a0": 1e308, "dynamics": dynamics, "mode": mode,
            "track_best": track_best, "local_search_sweeps": 0,
        })


def test_sbm_finite_zero_ties_keep_existing_rounding_policy():
    problem = QUBOProblem([0., 0.], [], [], [], offset=5.)
    solver = create_bqm_solver("lib_simulated_bifurcation", {"device": "cpu"})
    for dynamics, mode, track in itertools.product(
        ("standard", "adaptive"), ("discrete", "ballistic"), (False, True),
    ):
        result = solver.solve(problem, {
            "steps": 1, "runs": 1, "dtype": "float64", "initial_scale": 0.,
            "dynamics": dynamics, "mode": mode, "track_best": track,
            "local_search_sweeps": 0,
        })
        bit = 0 if mode == "ballistic" and not track else 1
        assert result.sample == (bit, bit)
        assert result.energy == problem.energy(result.sample) == 5.


def test_sbm_rejects_overflowed_momentum_even_when_positions_remain_finite():
    problem = QUBOProblem([0., 0.], [], [], [])
    solver = create_bqm_solver("lib_simulated_bifurcation", {"device": "cpu"})
    # The finite position update precedes heating; gamma * dt then overflows
    # the momentum buffer before another position update can consume it.
    with pytest.raises(FloatingPointError, match="dynamics became nonfinite"):
        solver.solve(problem, {
            "steps": 1, "runs": 1, "dtype": "float64", "initial_scale": .05,
            "dt": 2., "gamma": 1e308,
        })


@pytest.mark.parametrize("name, preparation", [
    ("lib_random_search", "device"),
    ("lib_categorical", "source_snapshot"),
])
def test_resident_report_identifies_solver_preparation_mode(tmp_path, name, preparation):
    from margin_engine import MarginApplicationConfig

    (tmp_path / "prices.csv").write_text(
        "date,A\n2024-01-01,100\n2024-01-02,101\n2024-01-03,98\n"
        "2024-01-04,103\n2024-01-05,101\n2024-01-06,105\n"
        "2024-01-07,104\n2024-01-08,108\n2024-01-09,107\n"
        "2024-01-10,110\n2024-01-11,111\n", encoding="utf-8",
    )
    application = MarginApplicationConfig.fromYamlText(f"""
marginDate: '2024-01-11'
portfolio:
  weights: {{A: 10}}
engine:
  numericalExecution: {{type: torch, device: cpu}}
  downloadManager:
    providers: {{local: local_csv}}
    requestParameters: {{location: prices.csv}}
  riskStateGenerator:
    ew_window: 5
    components: 1
    scenariosPerComponents: [3]
    nZBins: 2
    allowEmptyBinFallback: true
  marginCalculator:
    type: bqm
    solver:
      type: {name}
      constructorParameters: {{device: cpu}}
      solverParameters: {{steps: 2, runs: 2, dtype: float64}}
""", tmp_path)
    report = application.generateReport()
    assert report.numericalDiagnostics["solverPreparationMode"] == preparation
    assert report.margin >= 0.
