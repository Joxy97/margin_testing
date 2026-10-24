"""Adapters for existing single-anchor and boundary-aware search behavior."""

from time import perf_counter
from typing import Protocol, TYPE_CHECKING

import numpy as np

from margin_calculator.optimization import (
    FactorStressQUBO, QuadraticStressObjective, solveOptionFactorStress,
    FactorStressRepair,
)
from .values import SearchResult

if TYPE_CHECKING:
    from .scenarios import StressModel


class InfeasibleSolutionError(ValueError):
    """Raw solve failed the factor encoding and repair was explicitly disabled."""


class StressSearch(Protocol):
    def search(self, model: "StressModel") -> SearchResult: ...


class SingleAnchorSearch:
    """Existing Taylor objective, solver result, optional factor repair and repricing."""

    def __init__(self, solver, encoding, repair, repairEnabled=True, solverParameters=None, executionPolicy=None):
        self.solver, self.encoding, self.repair = solver, encoding, repair
        self.repairEnabled = repairEnabled
        self.solverParameters = dict(solverParameters or {})
        self.executionPolicy = executionPolicy

    def search(self, model: "StressModel") -> SearchResult:
        timings = {}
        started = perf_counter()
        encoding = FactorStressQUBO.build(QuadraticStressObjective(*model.quadraticCoefficients()), self.encoding)
        timings["encodingSeconds"] = perf_counter()-started
        started = perf_counter()
        if self.executionPolicy is None:
            solved = self.solver.solve(encoding.problem, dict(self.solverParameters))
        else:
            items = list(self.executionPolicy.execute(self.solver, [(None, encoding.problem)], self.solverParameters))
            if len(items) != 1:
                raise RuntimeError("execution policy must return one result per problem")
            _, solved = items[0]
        timings["solveSeconds"] = perf_counter()-started
        raw = encoding.diagnostics(solved.sample)
        started = perf_counter()
        repair_diagnostics = {"enabled": self.repairEnabled}
        if self.repairEnabled:
            repaired = FactorStressRepair(model, encoding, self.repair).repair(solved.sample)
            sample = repaired.sample
            repair_diagnostics.update(converged=repaired.converged, steps=repaired.steps,
                                      neighborScores=repaired.neighborScores,
                                      projectedPnL=repaired.projectedPnL)
        else:
            if not raw["encoding_feasible"]:
                raise InfeasibleSolutionError("raw factor-stress encoding is infeasible; no margin accepted")
            sample = solved.sample
        point = encoding.coordinates(sample)
        pnl = float(model.pnl(point))
        timings["selectionSeconds"] = perf_counter()-started
        return SearchResult(pnl, point, float(encoding.objective.value(point)), {
            "type": "single_anchor", "raw": raw, "selected": encoding.diagnostics(sample),
            "sourceEnergy": float(encoding.problem.energy(solved.sample)),
            "variables": encoding.problem.variableCount, "repair": repair_diagnostics,
            "timings": timings,
        })


class BoundaryAwareSearch:
    """Delegate all anchors, seeds, projection and exact ranking to existing search."""

    def __init__(self, solver, encoding, repair, solverParameters=None, executionPolicy=None):
        self.solver, self.encoding, self.repair = solver, encoding, repair
        self.solverParameters = dict(solverParameters or {})
        self.executionPolicy = executionPolicy

    def search(self, model: "StressModel") -> SearchResult:
        started = perf_counter()
        result = solveOptionFactorStress(model, self.solver, config=self.encoding,
            repairConfig=self.repair, solverParameters=dict(self.solverParameters), executionPolicy=self.executionPolicy)
        return SearchResult(result.pnl, result.coordinates, None, {
            "type": "boundary_aware", "candidateCount": len(result.candidates),
            "smoothAnchors": result.smoothAnchors, "nonsmoothAnchors": result.nonsmoothAnchors,
            "rawSolverFeasibility": result.solverFeasibility, "boundaryCoverage": result.boundaryCoverage,
            "repairCapHits": sum(not candidate.converged for candidate in result.candidates),
            "selectedScenarioFeasible": bool(np.linalg.norm(result.coordinates) <= self.encoding.radius+1e-12),
            "stressedSpots": result.stressedSpots,
            "pnlByType": {"stock": result.stockPnL, "European": result.europeanPnL, "American": result.americanPnL},
            "timings": {"anchorSearchSeconds": perf_counter()-started},
        })
